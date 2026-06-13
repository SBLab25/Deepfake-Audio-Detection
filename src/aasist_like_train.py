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


def parse_label(raw: str) -> int:
    v = raw.strip().lower()
    if v == "real":
        return 0
    if v == "fake":
        return 1
    raise ValueError(f"Unsupported label value: {raw}")


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
        candidates = sorted(split_root.rglob("balanced_index.csv"))
        if not candidates:
            candidates = sorted(split_root.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV found under '{split_root}'.")
        csv_path = candidates[0]

    if feature_dir_override:
        feature_dir = Path(feature_dir_override).expanduser().resolve()
    else:
        feature_dir = csv_path.parent.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature dir not found: {feature_dir}")
    return csv_path, feature_dir


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
        for row in reader:
            rel = row["feature_path"].strip()
            path = feature_dir / rel
            if not path.exists():
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
                    feature_path=path,
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
        raise RuntimeError("Cannot infer input dim from empty list.")
    return int(np.load(first.feature_path, mmap_mode="r").reshape(-1).shape[0])


def estimate_stats(
    samples: List[Sample], input_dim: int, max_samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(samples) > max_samples:
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        chosen = [samples[int(i)] for i in idx]
    else:
        chosen = samples

    n = 0
    mean = np.zeros(input_dim, dtype=np.float64)
    m2 = np.zeros(input_dim, dtype=np.float64)
    for s in tqdm(chosen, desc="estimating stats", leave=False):
        x = np.load(s.feature_path).astype(np.float32).reshape(-1)
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


class FeatureDataset(Dataset):
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

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = np.load(s.feature_path).astype(np.float32).reshape(-1)
        if x.shape[0] > self.input_dim:
            x = x[: self.input_dim]
        elif x.shape[0] < self.input_dim:
            x = np.pad(x, (0, self.input_dim - x.shape[0]))
        x = (x - self.mean) / self.std
        return (
            torch.from_numpy(x),
            torch.tensor(s.language_idx, dtype=torch.long),
            torch.tensor(s.label, dtype=torch.float32),
        )


class GraphAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        f = self.ffn(x)
        return self.norm(x + self.dropout(f))


class AASISTLike(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_languages: int,
        token_dim: int = 96,
        num_tokens: int = 12,
        layers: int = 3,
        language_emb_dim: int = 24,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.tokenizer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_tokens * token_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [GraphAttentionBlock(token_dim, heads=4, dropout=dropout) for _ in range(layers)]
        )
        self.post = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lang_emb = nn.Embedding(num_languages, language_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(192 + language_emb_dim, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor, lang_idx: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x).reshape(-1, self.num_tokens, self.token_dim)
        for blk in self.blocks:
            tokens = blk(tokens)
        pooled = tokens.mean(dim=1)
        z = self.post(pooled)
        l = self.lang_emb(lang_idx)
        return self.head(torch.cat([z, l], dim=1)).squeeze(1)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    noise_std: float,
    feature_drop: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)

    running_loss = 0.0
    all_prob: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for x, lang, y in tqdm(loader, desc="train" if train_mode else "eval", leave=False):
        x = x.to(device, non_blocking=True)
        lang = lang.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std
            if feature_drop > 0:
                m = torch.rand_like(x) < feature_drop
                x = x.masked_fill(m, 0.0)
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x, lang)
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

        running_loss += float(loss.item()) * x.size(0)
        all_prob.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy().astype(np.int64))

    return (
        running_loss / max(1, len(loader.dataset)),
        np.concatenate(all_prob),
        np.concatenate(all_y),
    )


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

    train_csv, train_dir = resolve_split_paths(
        project_root, args.train_split, args.train_csv, args.train_feature_dir
    )
    val_csv, val_dir = resolve_split_paths(
        project_root, args.val_split, args.val_csv, args.val_feature_dir
    )
    test_csv, test_dir = resolve_split_paths(
        project_root, args.test_split, args.test_csv, args.test_feature_dir
    )

    lang_to_idx = {"<unk>": 0}
    train_samples, miss_train, unk_train = load_samples(
        train_csv, train_dir, lang_to_idx, add_new_languages=True
    )
    val_samples, miss_val, unk_val = load_samples(
        val_csv, val_dir, lang_to_idx, add_new_languages=False
    )
    test_samples, miss_test, unk_test = load_samples(
        test_csv, test_dir, lang_to_idx, add_new_languages=False
    )

    input_dim = infer_input_dim(train_samples)
    mean, std = estimate_stats(train_samples, input_dim, args.stats_samples, args.seed)

    print("Dataset setup:")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val CSV:   {val_csv}")
    print(f"  Test CSV:  {test_csv}")
    print(f"  Input dim: {input_dim}")
    print(f"  Device: {device} | AMP={'on' if amp_enabled else 'off'}")
    print("Loaded samples:")
    print(f"  Train={len(train_samples)} (missing={miss_train}, unknown-lang={unk_train})")
    print(f"  Val={len(val_samples)} (missing={miss_val}, unknown-lang={unk_val})")
    print(f"  Test={len(test_samples)} (missing={miss_test}, unknown-lang={unk_test})")

    train_ds = FeatureDataset(train_samples, input_dim, mean, std)
    val_ds = FeatureDataset(val_samples, input_dim, mean, std)
    test_ds = FeatureDataset(test_samples, input_dim, mean, std)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = AASISTLike(
        input_dim=input_dim,
        num_languages=len(lang_to_idx),
        token_dim=args.token_dim,
        num_tokens=args.num_tokens,
        layers=args.layers,
        language_emb_dim=args.language_emb_dim,
        dropout=args.dropout,
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

    best_path = out_dir / "best_aasist_like.pt"
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
            noise_std=args.noise_std,
            feature_drop=args.feature_drop,
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
            noise_std=0.0,
            feature_drop=0.0,
        )

        threshold = best_threshold(val_y, val_prob)
        train_m = compute_metrics(train_y, train_prob, threshold)
        val_m = compute_metrics(val_y, val_prob, threshold)
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
            ckpt = {
                "epoch": epoch,
                "threshold": float(threshold),
                "input_dim": input_dim,
                "lang_to_idx": lang_to_idx,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "args": vars(args),
                "best_val_metrics": val_m,
                "model_state_dict": model.state_dict(),
            }
            torch.save(ckpt, best_path)
            print(f"  Saved best model: {best_path}")
        else:
            bad_epochs += 1
            print(f"  No improvement ({bad_epochs}/{args.patience})")
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_t = float(ckpt["threshold"])

    val_loss, val_prob, val_y = run_epoch(
        model,
        val_loader,
        criterion,
        device,
        optimizer=None,
        scheduler=None,
        scaler=None,
        amp_enabled=amp_enabled,
        noise_std=0.0,
        feature_drop=0.0,
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
        noise_std=0.0,
        feature_drop=0.0,
    )
    val_m = compute_metrics(val_y, val_prob, best_t)
    test_m = compute_metrics(test_y, test_prob, best_t)

    results = {
        "best_epoch": int(ckpt["epoch"]),
        "threshold": best_t,
        "val": {"loss": float(val_loss), "metrics": val_m},
        "test": {"loss": float(test_loss), "metrics": test_m},
        "history": history,
    }
    metrics_path = out_dir / "metrics_aasist_like.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nFinal metrics:")
    print(f"  Best epoch: {results['best_epoch']}")
    print(f"  Val AUC/F1: {val_m['roc_auc']:.5f} / {val_m['f1']:.5f}")
    print(f"  Test AUC/F1:{test_m['roc_auc']:.5f} / {test_m['f1']:.5f}")
    print(f"  Model: {best_path}")
    print(f"  Metrics: {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AASIST-inspired embedding trainer.")
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
    p.add_argument("--out-dir", type=str, default="./artifacts")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-dim", type=int, default=96)
    p.add_argument("--num-tokens", type=int, default=12)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--language-emb-dim", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--feature-drop", type=float, default=0.03)
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
