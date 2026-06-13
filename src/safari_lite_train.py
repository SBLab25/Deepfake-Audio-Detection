from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class Sample:
    feature_path: Path
    label: int
    language_idx: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_split_paths(
    project_root: Path,
    split_folder: str,
    csv_override: str | None,
    feature_dir_override: str | None,
) -> Tuple[Path, Path]:
    if csv_override:
        csv_path = Path(csv_override).expanduser().resolve()
    else:
        split_root = (project_root / split_folder).resolve()
        csv_candidates = sorted(split_root.rglob("balanced_index.csv"))
        if not csv_candidates:
            csv_candidates = sorted(split_root.rglob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(
                f"No metadata CSV found under split folder '{split_root}'."
            )
        csv_path = csv_candidates[0]

    if feature_dir_override:
        feature_dir = Path(feature_dir_override).expanduser().resolve()
    else:
        feature_dir = csv_path.parent.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")
    return csv_path, feature_dir


def parse_label(raw: str) -> int:
    v = raw.strip().lower()
    if v == "real":
        return 0
    if v == "fake":
        return 1
    raise ValueError(f"Unsupported label value: '{raw}'")


def load_samples(
    csv_path: Path,
    feature_dir: Path,
    lang_to_idx: Dict[str, int],
    add_new_languages: bool,
) -> Tuple[List[Sample], int, int]:
    samples: List[Sample] = []
    missing_files = 0
    unknown_lang_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"feature_path", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

        for row in reader:
            rel = row["feature_path"].strip()
            file_path = feature_dir / rel
            if not file_path.exists():
                missing_files += 1
                continue

            label = parse_label(row["label"])
            lang = row.get("language", "<unk>").strip().lower() or "<unk>"
            if lang not in lang_to_idx:
                if add_new_languages:
                    lang_to_idx[lang] = len(lang_to_idx)
                else:
                    lang = "<unk>"
                    unknown_lang_rows += 1

            samples.append(
                Sample(
                    feature_path=file_path,
                    label=label,
                    language_idx=lang_to_idx[lang],
                )
            )

    if not samples:
        raise RuntimeError(f"No usable samples loaded from {csv_path}")
    return samples, missing_files, unknown_lang_rows


def infer_input_dim(samples: Iterable[Sample]) -> int:
    first = next(iter(samples), None)
    if first is None:
        raise RuntimeError("Cannot infer input dimension from an empty sample list.")
    arr = np.load(first.feature_path, mmap_mode="r")
    return int(arr.reshape(-1).shape[0])


def estimate_train_stats(
    samples: List[Sample], input_dim: int, max_samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(samples) > max_samples:
        indices = rng.choice(len(samples), size=max_samples, replace=False)
        chosen = [samples[int(i)] for i in indices]
    else:
        chosen = samples

    n = 0
    mean = np.zeros(input_dim, dtype=np.float64)
    m2 = np.zeros(input_dim, dtype=np.float64)

    for s in tqdm(chosen, desc="estimating feature stats", leave=False):
        x = np.load(s.feature_path).astype(np.float32).reshape(-1)
        if x.shape[0] != input_dim:
            if x.shape[0] > input_dim:
                x = x[:input_dim]
            else:
                x = np.pad(x, (0, input_dim - x.shape[0]))
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2

    var = m2 / max(1, n - 1)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class AudioFeatureDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        input_dim: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.samples = samples
        self.input_dim = input_dim
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        x = np.load(sample.feature_path).astype(np.float32).reshape(-1)
        if x.shape[0] != self.input_dim:
            if x.shape[0] > self.input_dim:
                x = x[: self.input_dim]
            else:
                x = np.pad(x, (0, self.input_dim - x.shape[0]))

        x = (x - self.mean) / self.std
        feature = torch.from_numpy(x)
        language_idx = torch.tensor(sample.language_idx, dtype=torch.long)
        label = torch.tensor(sample.label, dtype=torch.float32)
        return feature, language_idx, label


class AFUM(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 384) -> None:
        super().__init__()
        self.semantic_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.acoustic_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.30),
        )

    def forward(
        self, semantic: torch.Tensor, acoustic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.semantic_proj(semantic)
        a = self.acoustic_proj(acoustic)
        g = self.gate(torch.cat([s, a], dim=1))
        fused = g * s + (1.0 - g) * a
        diff = torch.abs(s - a)
        prod = s * a
        out = self.fusion(torch.cat([fused, diff, prod], dim=1))
        return out, s, a, diff


class ReasoningModule(nn.Module):
    def __init__(
        self,
        token_dim: int = 384,
        nhead: int = 8,
        ff_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.encoder(tokens)
        x = self.norm(x)
        return x.mean(dim=1)


class SafariLitePlus(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        language_emb_dim: int = 32,
        num_languages: int = 3,
    ) -> None:
        super().__init__()
        self.afum = AFUM(input_dim=input_dim, hidden_dim=hidden_dim)
        self.reasoning = ReasoningModule(token_dim=hidden_dim)
        self.language_emb = nn.Embedding(num_languages, language_emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + language_emb_dim, 256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 1),
        )

    def forward(
        self, semantic: torch.Tensor, acoustic: torch.Tensor, language_idx: torch.Tensor
    ) -> torch.Tensor:
        fused, s, a, diff = self.afum(semantic, acoustic)
        tokens = torch.stack([fused, s, a, diff], dim=1)
        context = self.reasoning(tokens)
        l = self.language_emb(language_idx)
        return self.classifier(torch.cat([context, l], dim=1)).squeeze(1)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    best_acc = -1.0
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 1001),
                np.quantile(y_prob, np.linspace(0.0, 1.0, 1001)),
            ]
        )
    )
    for t in candidates:
        y_pred = (y_prob >= t).astype(np.int64)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)
        if (float(f1), float(acc)) > (best_f1, best_acc):
            best_f1 = float(f1)
            best_acc = float(acc)
            best_t = float(t)
    return best_t


def make_two_views(
    features: torch.Tensor, train_mode: bool, noise_std: float, drop_prob: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not train_mode:
        return features, features

    semantic = features
    if noise_std > 0:
        semantic = semantic + torch.randn_like(semantic) * noise_std

    acoustic = features.clone()
    if drop_prob > 0:
        mask = torch.rand_like(acoustic) < drop_prob
        acoustic = acoustic.masked_fill(mask, 0.0)
    return semantic, acoustic


def maybe_mixup(
    x_sem: torch.Tensor,
    x_aco: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x_sem, x_aco, y
    lam = np.random.beta(alpha, alpha)
    lam = float(max(lam, 1.0 - lam))
    idx = torch.randperm(y.size(0), device=y.device)
    x_sem_mix = lam * x_sem + (1.0 - lam) * x_sem[idx]
    x_aco_mix = lam * x_aco + (1.0 - lam) * x_aco[idx]
    y_mix = lam * y + (1.0 - lam) * y[idx]
    return x_sem_mix, x_aco_mix, y_mix


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    view_noise_std: float,
    view_drop_prob: float,
    mixup_alpha: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_lang: List[np.ndarray] = []

    loop = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    for feature, lang_idx, label in loop:
        feature = feature.to(device, non_blocking=True)
        lang_idx = lang_idx.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        sem, aco = make_two_views(
            feature,
            train_mode=is_train,
            noise_std=view_noise_std,
            drop_prob=view_drop_prob,
        )

        if is_train and mixup_alpha > 0:
            sem, aco, label_mix = maybe_mixup(sem, aco, label, mixup_alpha)
        else:
            label_mix = label

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(sem, aco, lang_idx)
            loss = criterion(logits, label_mix)

        if is_train:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        running_loss += float(loss.item()) * feature.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label.detach().cpu().numpy().astype(np.int64))
        all_lang.append(lang_idx.detach().cpu().numpy().astype(np.int64))

    avg_loss = running_loss / max(1, len(loader.dataset))
    return (
        avg_loss,
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_lang, axis=0),
    )


def evaluate_by_language(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lang_idx: np.ndarray,
    idx_to_lang: Dict[int, str],
    threshold: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i in sorted(np.unique(lang_idx).tolist()):
        m = lang_idx == i
        if int(m.sum()) == 0:
            continue
        lang_name = idx_to_lang.get(int(i), str(i))
        out[lang_name] = compute_metrics(y_true[m], y_prob[m], threshold)
        out[lang_name]["samples"] = int(m.sum())
    return out


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = device_from_arg(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv, train_feat_dir = resolve_split_paths(
        project_root, args.train_split, args.train_csv, args.train_feature_dir
    )
    val_csv, val_feat_dir = resolve_split_paths(
        project_root, args.val_split, args.val_csv, args.val_feature_dir
    )
    test_csv, test_feat_dir = resolve_split_paths(
        project_root, args.test_split, args.test_csv, args.test_feature_dir
    )

    lang_to_idx = {"<unk>": 0}
    train_samples, train_missing, train_unk = load_samples(
        train_csv, train_feat_dir, lang_to_idx, add_new_languages=True
    )
    val_samples, val_missing, val_unk = load_samples(
        val_csv, val_feat_dir, lang_to_idx, add_new_languages=False
    )
    test_samples, test_missing, test_unk = load_samples(
        test_csv, test_feat_dir, lang_to_idx, add_new_languages=False
    )
    idx_to_lang = {v: k for k, v in lang_to_idx.items()}

    input_dim = infer_input_dim(train_samples)
    mean, std = estimate_train_stats(
        train_samples, input_dim=input_dim, max_samples=args.stats_samples, seed=args.seed
    )

    print("Dataset setup:")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val CSV:   {val_csv}")
    print(f"  Test CSV:  {test_csv}")
    print("Loaded samples:")
    print(
        f"  Train={len(train_samples)} (missing={train_missing}, unknown-lang={train_unk})"
    )
    print(f"  Val={len(val_samples)} (missing={val_missing}, unknown-lang={val_unk})")
    print(
        f"  Test={len(test_samples)} (missing={test_missing}, unknown-lang={test_unk})"
    )
    print(f"Input dim: {input_dim}")
    print(f"Languages: {idx_to_lang}")
    print(f"Device: {device} | AMP={'on' if amp_enabled else 'off'}")

    train_ds = AudioFeatureDataset(
        train_samples, input_dim=input_dim, mean=mean, std=std
    )
    val_ds = AudioFeatureDataset(val_samples, input_dim=input_dim, mean=mean, std=std)
    test_ds = AudioFeatureDataset(test_samples, input_dim=input_dim, mean=mean, std=std)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = SafariLitePlus(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        language_emb_dim=args.language_emb_dim,
        num_languages=len(lang_to_idx),
    ).to(device)

    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed model weights from: {resume_path}")

    label_arr = np.array([s.label for s in train_samples], dtype=np.int64)
    pos = int(label_arr.sum())
    neg = int(label_arr.shape[0] - pos)
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=max(1.0, args.max_lr / max(args.lr, 1e-7)),
        final_div_factor=100.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_model_path = out_dir / "best_safari_lite_plus.pt"
    history: List[Dict[str, float]] = []
    best_score = (-math.inf, -math.inf)
    bad_epochs = 0
    best_record: Dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_prob, train_y, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            view_noise_std=args.view_noise_std,
            view_drop_prob=args.view_drop_prob,
            mixup_alpha=args.mixup_alpha,
        )
        val_loss, val_prob, val_y, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scheduler=None,
            scaler=None,
            amp_enabled=amp_enabled,
            view_noise_std=0.0,
            view_drop_prob=0.0,
            mixup_alpha=0.0,
        )

        threshold = best_threshold(val_y, val_prob)
        train_metrics = compute_metrics(train_y, train_prob, threshold)
        val_metrics = compute_metrics(val_y, val_prob, threshold)
        score_auc = val_metrics["roc_auc"]
        score_auc = -math.inf if math.isnan(score_auc) else score_auc
        score = (score_auc, val_metrics["f1"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_auc": float(train_metrics["roc_auc"]),
                "train_f1": float(train_metrics["f1"]),
                "val_auc": float(val_metrics["roc_auc"]),
                "val_f1": float(val_metrics["f1"]),
                "threshold": float(threshold),
                "lr": lr_now,
            }
        )

        print(
            "  "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_auc={val_metrics['roc_auc']:.5f} val_f1={val_metrics['f1']:.5f} "
            f"threshold={threshold:.3f} lr={lr_now:.7f}"
        )

        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_record = {
                "epoch": epoch,
                "threshold": float(threshold),
                "input_dim": input_dim,
                "lang_to_idx": lang_to_idx,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "args": vars(args),
                "best_val_metrics": val_metrics,
                "model_state_dict": model.state_dict(),
            }
            torch.save(best_record, best_model_path)
            print(f"  Saved best model to: {best_model_path}")
        else:
            bad_epochs += 1
            print(f"  No improvement ({bad_epochs}/{args.patience})")
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_record is None:
        raise RuntimeError("No valid checkpoint produced.")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_t = float(checkpoint["threshold"])

    val_loss, val_prob, val_y, val_lang = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scheduler=None,
        scaler=None,
        amp_enabled=amp_enabled,
        view_noise_std=0.0,
        view_drop_prob=0.0,
        mixup_alpha=0.0,
    )
    test_loss, test_prob, test_y, test_lang = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scheduler=None,
        scaler=None,
        amp_enabled=amp_enabled,
        view_noise_std=0.0,
        view_drop_prob=0.0,
        mixup_alpha=0.0,
    )

    val_metrics = compute_metrics(val_y, val_prob, best_t)
    test_metrics = compute_metrics(test_y, test_prob, best_t)

    results = {
        "best_epoch": int(checkpoint["epoch"]),
        "threshold": best_t,
        "input_dim": input_dim,
        "languages": idx_to_lang,
        "splits": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "train_feature_dir": str(train_feat_dir),
            "val_feature_dir": str(val_feat_dir),
            "test_feature_dir": str(test_feat_dir),
        },
        "missing_files": {
            "train": train_missing,
            "val": val_missing,
            "test": test_missing,
        },
        "val": {
            "loss": float(val_loss),
            "metrics": val_metrics,
            "by_language": evaluate_by_language(
                val_y, val_prob, val_lang, idx_to_lang, best_t
            ),
        },
        "test": {
            "loss": float(test_loss),
            "metrics": test_metrics,
            "by_language": evaluate_by_language(
                test_y, test_prob, test_lang, idx_to_lang, best_t
            ),
        },
        "history": history,
    }

    metrics_path = out_dir / "metrics_safari_lite_plus.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nFinal metrics:")
    print(f"  Best epoch: {results['best_epoch']}")
    print(f"  Threshold:  {results['threshold']:.3f}")
    print(f"  Val AUC/F1: {val_metrics['roc_auc']:.5f} / {val_metrics['f1']:.5f}")
    print(f"  Test AUC/F1:{test_metrics['roc_auc']:.5f} / {test_metrics['f1']:.5f}")
    print(f"  Model: {best_model_path}")
    print(f"  Metrics: {metrics_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SafariLite+ deepfake detector with robust split loading."
    )

    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--train-split", type=str, default="Balanced_train")
    parser.add_argument("--val-split", type=str, default="Balanced_val")
    parser.add_argument("--test-split", type=str, default="Balanced_test")
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--train-feature-dir", type=str, default=None)
    parser.add_argument("--val-feature-dir", type=str, default=None)
    parser.add_argument("--test-feature-dir", type=str, default=None)

    parser.add_argument("--out-dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--language-emb-dim", type=int, default=32)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)

    parser.add_argument("--view-noise-std", type=float, default=0.012)
    parser.add_argument("--view-drop-prob", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument(
        "--stats-samples",
        type=int,
        default=50000,
        help="How many train examples to use for normalization statistics (0 = full train).",
    )

    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional checkpoint path to warm-start model weights.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
