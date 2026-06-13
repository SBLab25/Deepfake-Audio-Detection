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
class DualSample:
    base_path: Path
    wavlm_path: Path
    label: int
    language_idx: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_label(raw: str) -> int:
    v = raw.strip().lower()
    if v == "real":
        return 0
    if v == "fake":
        return 1
    raise ValueError(f"Unsupported label: {raw}")


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
            raise FileNotFoundError(f"No CSV found under split folder '{split_root}'.")
        csv_path = csv_candidates[0]

    if feature_dir_override:
        feature_dir = Path(feature_dir_override).expanduser().resolve()
    else:
        feature_dir = csv_path.parent.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature dir not found: {feature_dir}")
    return csv_path, feature_dir


def load_dual_samples(
    csv_path: Path,
    base_feature_dir: Path,
    wavlm_feature_dir: Path | None,
    lang_to_idx: Dict[str, int],
    add_new_languages: bool,
) -> Tuple[List[DualSample], int, int, int]:
    samples: List[DualSample] = []
    missing_base = 0
    missing_wavlm_fallback = 0
    unknown_lang_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = row["feature_path"].strip()
            base_path = base_feature_dir / rel
            if not base_path.exists():
                missing_base += 1
                continue

            if wavlm_feature_dir is None:
                wavlm_path = base_path
                missing_wavlm_fallback += 1
            else:
                candidate = wavlm_feature_dir / rel
                if candidate.exists():
                    wavlm_path = candidate
                else:
                    wavlm_path = base_path
                    missing_wavlm_fallback += 1

            label = parse_label(row["label"])
            lang = row.get("language", "<unk>").strip().lower() or "<unk>"
            if lang not in lang_to_idx:
                if add_new_languages:
                    lang_to_idx[lang] = len(lang_to_idx)
                else:
                    lang = "<unk>"
                    unknown_lang_rows += 1

            samples.append(
                DualSample(
                    base_path=base_path,
                    wavlm_path=wavlm_path,
                    label=label,
                    language_idx=lang_to_idx[lang],
                )
            )

    if not samples:
        raise RuntimeError(f"No valid samples loaded from: {csv_path}")

    return samples, missing_base, missing_wavlm_fallback, unknown_lang_rows


def infer_dim(path: Path) -> int:
    arr = np.load(path, mmap_mode="r").reshape(-1)
    return int(arr.shape[0])


def infer_dims(samples: Iterable[DualSample]) -> Tuple[int, int]:
    first = next(iter(samples), None)
    if first is None:
        raise RuntimeError("Cannot infer dimensions from empty samples.")
    return infer_dim(first.base_path), infer_dim(first.wavlm_path)


def estimate_stats(
    paths: List[Path], input_dim: int, max_samples: int, seed: int, desc: str
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(paths) > max_samples:
        idx = rng.choice(len(paths), size=max_samples, replace=False)
        chosen = [paths[int(i)] for i in idx]
    else:
        chosen = paths

    n = 0
    mean = np.zeros(input_dim, dtype=np.float64)
    m2 = np.zeros(input_dim, dtype=np.float64)

    for p in tqdm(chosen, desc=desc, leave=False):
        x = np.load(p).astype(np.float32).reshape(-1)
        if x.shape[0] > input_dim:
            x = x[:input_dim]
        elif x.shape[0] < input_dim:
            x = np.pad(x, (0, input_dim - x.shape[0]))
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2

    var = m2 / max(1, n - 1)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class DualFeatureDataset(Dataset):
    def __init__(
        self,
        samples: List[DualSample],
        base_dim: int,
        wavlm_dim: int,
        base_mean: np.ndarray,
        base_std: np.ndarray,
        wavlm_mean: np.ndarray,
        wavlm_std: np.ndarray,
    ) -> None:
        self.samples = samples
        self.base_dim = base_dim
        self.wavlm_dim = wavlm_dim
        self.base_mean = base_mean
        self.base_std = base_std
        self.wavlm_mean = wavlm_mean
        self.wavlm_std = wavlm_std

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_and_fit(path: Path, dim: int) -> np.ndarray:
        x = np.load(path).astype(np.float32).reshape(-1)
        if x.shape[0] > dim:
            x = x[:dim]
        elif x.shape[0] < dim:
            x = np.pad(x, (0, dim - x.shape[0]))
        return x

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        xb = self._load_and_fit(s.base_path, self.base_dim)
        xw = self._load_and_fit(s.wavlm_path, self.wavlm_dim)

        xb = (xb - self.base_mean) / self.base_std
        xw = (xw - self.wavlm_mean) / self.wavlm_std

        return (
            torch.from_numpy(xb),
            torch.from_numpy(xw),
            torch.tensor(s.language_idx, dtype=torch.long),
            torch.tensor(s.label, dtype=torch.float32),
        )


class AFUM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.s_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.a_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )

    def forward(
        self, s_in: torch.Tensor, a_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.s_proj(s_in)
        a = self.a_proj(a_in)
        g = self.gate(torch.cat([s, a], dim=1))
        mixed = g * s + (1.0 - g) * a
        diff = torch.abs(s - a)
        prod = s * a
        fused = self.fuse(torch.cat([mixed, diff, prod], dim=1))
        return fused, s, a


class SafariBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.afum = AFUM(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 3,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.reasoning = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sem: torch.Tensor, aco: torch.Tensor) -> torch.Tensor:
        fused, s_proj, a_proj = self.afum(sem, aco)
        tokens = torch.stack([fused, s_proj, a_proj, torch.abs(s_proj - a_proj)], dim=1)
        ctx = self.reasoning(tokens)
        return self.norm(ctx.mean(dim=1))


class WavLMBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SafariWavLMEnsemble(nn.Module):
    def __init__(
        self,
        base_dim: int,
        wavlm_dim: int,
        hidden_dim: int,
        language_emb_dim: int,
        num_languages: int,
    ) -> None:
        super().__init__()
        self.safari = SafariBackbone(base_dim, hidden_dim)
        self.wavlm = WavLMBranch(wavlm_dim, hidden_dim)
        self.lang_emb = nn.Embedding(num_languages, language_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + language_emb_dim, 320),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(320, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, 1),
        )

    def forward(
        self,
        base_sem: torch.Tensor,
        base_aco: torch.Tensor,
        wavlm_feat: torch.Tensor,
        language_idx: torch.Tensor,
    ) -> torch.Tensor:
        z_safari = self.safari(base_sem, base_aco)
        z_wavlm = self.wavlm(wavlm_feat)
        z_lang = self.lang_emb(language_idx)
        return self.head(torch.cat([z_safari, z_wavlm, z_lang], dim=1)).squeeze(1)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
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
        "roc_auc": safe_auc(y_true, y_prob),
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


def make_views(x: torch.Tensor, train_mode: bool, noise_std: float, drop_prob: float):
    if not train_mode:
        return x, x
    sem = x
    if noise_std > 0:
        sem = sem + torch.randn_like(sem) * noise_std
    aco = x.clone()
    if drop_prob > 0:
        mask = torch.rand_like(aco) < drop_prob
        aco = aco.masked_fill(mask, 0.0)
    return sem, aco


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
) -> Tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)

    running_loss = 0.0
    all_prob: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    loop = tqdm(loader, desc="train" if train_mode else "eval", leave=False)
    for xb, xw, lang, y in loop:
        xb = xb.to(device, non_blocking=True)
        xw = xw.to(device, non_blocking=True)
        lang = lang.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sem, aco = make_views(xb, train_mode, view_noise_std, view_drop_prob)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(sem, aco, xw, lang)
            loss = criterion(logits, y)

        if train_mode:
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

        running_loss += float(loss.item()) * xb.size(0)
        all_prob.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy().astype(np.int64))

    avg_loss = running_loss / max(1, len(loader.dataset))
    return avg_loss, np.concatenate(all_prob), np.concatenate(all_y)


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

    train_csv, train_base_dir = resolve_split_paths(
        project_root, args.train_split, args.train_csv, args.train_feature_dir
    )
    val_csv, val_base_dir = resolve_split_paths(
        project_root, args.val_split, args.val_csv, args.val_feature_dir
    )
    test_csv, test_base_dir = resolve_split_paths(
        project_root, args.test_split, args.test_csv, args.test_feature_dir
    )

    train_wavlm_dir = (
        Path(args.wavlm_train_feature_dir).expanduser().resolve()
        if args.wavlm_train_feature_dir
        else None
    )
    val_wavlm_dir = (
        Path(args.wavlm_val_feature_dir).expanduser().resolve()
        if args.wavlm_val_feature_dir
        else None
    )
    test_wavlm_dir = (
        Path(args.wavlm_test_feature_dir).expanduser().resolve()
        if args.wavlm_test_feature_dir
        else None
    )

    lang_to_idx = {"<unk>": 0}
    train_samples, miss_train_base, miss_train_wavlm, unk_train = load_dual_samples(
        train_csv, train_base_dir, train_wavlm_dir, lang_to_idx, add_new_languages=True
    )
    val_samples, miss_val_base, miss_val_wavlm, unk_val = load_dual_samples(
        val_csv, val_base_dir, val_wavlm_dir, lang_to_idx, add_new_languages=False
    )
    test_samples, miss_test_base, miss_test_wavlm, unk_test = load_dual_samples(
        test_csv, test_base_dir, test_wavlm_dir, lang_to_idx, add_new_languages=False
    )

    base_dim, wavlm_dim = infer_dims(train_samples)
    base_mean, base_std = estimate_stats(
        [s.base_path for s in train_samples],
        base_dim,
        args.stats_samples,
        args.seed,
        desc="stats base",
    )
    wavlm_mean, wavlm_std = estimate_stats(
        [s.wavlm_path for s in train_samples],
        wavlm_dim,
        args.stats_samples,
        args.seed + 7,
        desc="stats wavlm",
    )

    print("Dataset setup:")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val CSV:   {val_csv}")
    print(f"  Test CSV:  {test_csv}")
    print(f"  Base dim: {base_dim} | WavLM dim: {wavlm_dim}")
    print(f"  Device: {device} | AMP={'on' if amp_enabled else 'off'}")
    print("Loaded samples:")
    print(
        f"  Train={len(train_samples)} (base-missing={miss_train_base}, wavlm-fallback={miss_train_wavlm}, unk-lang={unk_train})"
    )
    print(
        f"  Val={len(val_samples)} (base-missing={miss_val_base}, wavlm-fallback={miss_val_wavlm}, unk-lang={unk_val})"
    )
    print(
        f"  Test={len(test_samples)} (base-missing={miss_test_base}, wavlm-fallback={miss_test_wavlm}, unk-lang={unk_test})"
    )
    if train_wavlm_dir is None:
        print("  Note: wavlm feature dir not provided, wavlm branch currently reuses base features.")

    train_ds = DualFeatureDataset(
        train_samples,
        base_dim,
        wavlm_dim,
        base_mean,
        base_std,
        wavlm_mean,
        wavlm_std,
    )
    val_ds = DualFeatureDataset(
        val_samples, base_dim, wavlm_dim, base_mean, base_std, wavlm_mean, wavlm_std
    )
    test_ds = DualFeatureDataset(
        test_samples, base_dim, wavlm_dim, base_mean, base_std, wavlm_mean, wavlm_std
    )

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

    model = SafariWavLMEnsemble(
        base_dim=base_dim,
        wavlm_dim=wavlm_dim,
        hidden_dim=args.hidden_dim,
        language_emb_dim=args.language_emb_dim,
        num_languages=len(lang_to_idx),
    ).to(device)

    y_train = np.array([s.label for s in train_samples], dtype=np.int64)
    pos = int(y_train.sum())
    neg = int(y_train.shape[0] - pos)
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.1,
        div_factor=max(1.0, args.max_lr / max(args.lr, 1e-7)),
        final_div_factor=100.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_path = out_dir / "best_safari_wavlm_ensemble.pt"
    best_score = (-math.inf, -math.inf)
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_prob, train_y = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            view_noise_std=args.view_noise_std,
            view_drop_prob=args.view_drop_prob,
        )
        val_loss, val_prob, val_y = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scheduler=None,
            scaler=None,
            amp_enabled=amp_enabled,
            view_noise_std=0.0,
            view_drop_prob=0.0,
        )

        threshold = best_threshold(val_y, val_prob)
        train_m = metrics(train_y, train_prob, threshold)
        val_m = metrics(val_y, val_prob, threshold)
        score_auc = val_m["roc_auc"]
        score_auc = -math.inf if math.isnan(score_auc) else score_auc
        score = (score_auc, val_m["f1"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_auc": float(train_m["roc_auc"]),
                "train_f1": float(train_m["f1"]),
                "val_auc": float(val_m["roc_auc"]),
                "val_f1": float(val_m["f1"]),
                "threshold": float(threshold),
                "lr": lr_now,
            }
        )
        print(
            f"  train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_auc={val_m['roc_auc']:.5f} val_f1={val_m['f1']:.5f} "
            f"threshold={threshold:.3f} lr={lr_now:.7f}"
        )

        if score > best_score:
            best_score = score
            bad_epochs = 0
            checkpoint = {
                "epoch": epoch,
                "threshold": float(threshold),
                "base_dim": base_dim,
                "wavlm_dim": wavlm_dim,
                "lang_to_idx": lang_to_idx,
                "base_mean": base_mean.tolist(),
                "base_std": base_std.tolist(),
                "wavlm_mean": wavlm_mean.tolist(),
                "wavlm_std": wavlm_std.tolist(),
                "args": vars(args),
                "model_state_dict": model.state_dict(),
                "best_val_metrics": val_m,
            }
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")
        else:
            bad_epochs += 1
            print(f"  No improvement ({bad_epochs}/{args.patience})")
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_t = float(best_ckpt["threshold"])

    val_loss, val_prob, val_y = run_epoch(
        model,
        val_loader,
        criterion,
        device,
        optimizer=None,
        scheduler=None,
        scaler=None,
        amp_enabled=amp_enabled,
        view_noise_std=0.0,
        view_drop_prob=0.0,
    )
    test_loss, test_prob, test_y = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        scheduler=None,
        scaler=None,
        amp_enabled=amp_enabled,
        view_noise_std=0.0,
        view_drop_prob=0.0,
    )
    val_m = metrics(val_y, val_prob, best_t)
    test_m = metrics(test_y, test_prob, best_t)

    results = {
        "best_epoch": int(best_ckpt["epoch"]),
        "threshold": best_t,
        "val": {"loss": float(val_loss), "metrics": val_m},
        "test": {"loss": float(test_loss), "metrics": test_m},
        "history": history,
    }
    metrics_path = out_dir / "metrics_safari_wavlm_ensemble.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nFinal metrics:")
    print(f"  Best epoch: {results['best_epoch']}")
    print(f"  Val AUC/F1: {val_m['roc_auc']:.5f} / {val_m['f1']:.5f}")
    print(f"  Test AUC/F1:{test_m['roc_auc']:.5f} / {test_m['f1']:.5f}")
    print(f"  Model: {best_path}")
    print(f"  Metrics: {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SafariLite + WavLM late-fusion ensemble trainer."
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--train-split", type=str, default="Balanced_train")
    p.add_argument("--val-split", type=str, default="Balanced_val")
    p.add_argument("--test-split", type=str, default="Balanced_test")
    p.add_argument("--train-csv", type=str, default=None)
    p.add_argument("--val-csv", type=str, default=None)
    p.add_argument("--test-csv", type=str, default=None)
    p.add_argument("--train-feature-dir", type=str, default=None)
    p.add_argument("--val-feature-dir", type=str, default=None)
    p.add_argument("--test-feature-dir", type=str, default=None)

    p.add_argument("--wavlm-train-feature-dir", type=str, default=None)
    p.add_argument("--wavlm-val-feature-dir", type=str, default=None)
    p.add_argument("--wavlm-test-feature-dir", type=str, default=None)

    p.add_argument("--out-dir", type=str, default="./artifacts")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--language-emb-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--view-noise-std", type=float, default=0.01)
    p.add_argument("--view-drop-prob", type=float, default=0.03)
    p.add_argument("--stats-samples", type=int, default=50000)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--amp", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
