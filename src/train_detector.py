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
    if csv_override is not None:
        csv_path = Path(csv_override).expanduser().resolve()
    else:
        split_root = (project_root / split_folder).resolve()
        csv_candidates = sorted(split_root.rglob("balanced_index.csv"))
        if not csv_candidates:
            csv_candidates = sorted(split_root.rglob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(
                f"No CSV file found under split folder '{split_root}'. "
                "Pass --*-csv explicitly."
            )
        csv_path = csv_candidates[0]

    if feature_dir_override is not None:
        feature_dir = Path(feature_dir_override).expanduser().resolve()
    else:
        feature_dir = csv_path.parent.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    return csv_path, feature_dir


def parse_label(raw_label: str) -> int:
    v = raw_label.strip().lower()
    if v == "fake":
        return 1
    if v == "real":
        return 0
    raise ValueError(f"Unsupported label value: '{raw_label}'")


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
        missing_cols = required.difference(set(reader.fieldnames or []))
        if missing_cols:
            raise ValueError(
                f"{csv_path} missing required columns: {sorted(missing_cols)}"
            )

        for row in reader:
            relative_path = row["feature_path"].strip()
            abs_path = feature_dir / relative_path
            if not abs_path.exists():
                missing_files += 1
                continue

            label = parse_label(row["label"])
            language = row.get("language", "<unk>").strip().lower() or "<unk>"

            if language not in lang_to_idx:
                if add_new_languages:
                    lang_to_idx[language] = len(lang_to_idx)
                else:
                    unknown_lang_rows += 1
                    language = "<unk>"

            samples.append(
                Sample(
                    feature_path=abs_path,
                    label=label,
                    language_idx=lang_to_idx[language],
                )
            )

    if not samples:
        raise RuntimeError(f"No usable samples loaded from {csv_path}")

    return samples, missing_files, unknown_lang_rows


class FeatureDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        input_dim: int,
        training: bool = False,
        noise_std: float = 0.01,
        feature_dropout: float = 0.02,
        scale_jitter_std: float = 0.02,
    ) -> None:
        self.samples = samples
        self.input_dim = input_dim
        self.training = training
        self.noise_std = noise_std
        self.feature_dropout = feature_dropout
        self.scale_jitter_std = scale_jitter_std

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

        if self.training:
            if self.noise_std > 0:
                x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(
                    np.float32
                )
            if self.feature_dropout > 0:
                mask = np.random.random_sample(x.shape) < self.feature_dropout
                x[mask] = 0.0
            if self.scale_jitter_std > 0:
                scale = np.random.normal(1.0, self.scale_jitter_std)
                x = x * np.float32(scale)

        feature = torch.from_numpy(x)
        language_idx = torch.tensor(sample.language_idx, dtype=torch.long)
        label = torch.tensor(sample.label, dtype=torch.float32)
        return feature, language_idx, label


class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_languages: int,
        hidden_dim: int = 512,
        language_emb_dim: int = 32,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        half_hidden = max(128, hidden_dim // 2)
        quarter_hidden = max(64, hidden_dim // 4)

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half_hidden),
            nn.BatchNorm1d(half_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.language_embedding = nn.Embedding(num_languages, language_emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(half_hidden + language_emb_dim, quarter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(quarter_hidden, 1),
        )

    def forward(self, x: torch.Tensor, language_idx: torch.Tensor) -> torch.Tensor:
        h = self.feature_encoder(x)
        l = self.language_embedding(language_idx)
        out = torch.cat([h, l], dim=1)
        logits = self.classifier(out).squeeze(1)
        return logits


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_languages: List[np.ndarray] = []

    progress = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    for features, language_idx, labels in progress:
        features = features.to(device, non_blocking=True)
        language_idx = language_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(features, language_idx)
            loss = criterion(logits, labels)

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

        running_loss += float(loss.item()) * features.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy().astype(np.int64))
        all_languages.append(language_idx.detach().cpu().numpy().astype(np.int64))

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return (
        epoch_loss,
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_languages, axis=0),
    )


def evaluate_by_language(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lang_idx: np.ndarray,
    idx_to_lang: Dict[int, str],
    threshold: float,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for i in sorted(np.unique(lang_idx).tolist()):
        mask = lang_idx == i
        if int(mask.sum()) == 0:
            continue
        lang_name = idx_to_lang.get(int(i), str(i))
        metrics[lang_name] = compute_metrics(y_true[mask], y_prob[mask], threshold)
        metrics[lang_name]["samples"] = int(mask.sum())
    return metrics


def first_feature_dim(samples: Iterable[Sample]) -> int:
    first = next(iter(samples), None)
    if first is None:
        raise RuntimeError("Cannot infer feature dimension from empty sample list.")
    return int(np.load(first.feature_path, mmap_mode="r").reshape(-1).shape[0])


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = device_from_arg(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

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

    train_samples, train_missing, train_unknown_lang = load_samples(
        train_csv, train_feat_dir, lang_to_idx, add_new_languages=True
    )
    val_samples, val_missing, val_unknown_lang = load_samples(
        val_csv, val_feat_dir, lang_to_idx, add_new_languages=False
    )
    test_samples, test_missing, test_unknown_lang = load_samples(
        test_csv, test_feat_dir, lang_to_idx, add_new_languages=False
    )

    input_dim = first_feature_dim(train_samples)
    idx_to_lang = {v: k for k, v in lang_to_idx.items()}

    print("Resolved dataset:")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val CSV:   {val_csv}")
    print(f"  Test CSV:  {test_csv}")
    print("Loaded samples:")
    print(
        f"  Train={len(train_samples)} (missing files={train_missing}, unknown language rows={train_unknown_lang})"
    )
    print(
        f"  Val={len(val_samples)} (missing files={val_missing}, unknown language rows={val_unknown_lang})"
    )
    print(
        f"  Test={len(test_samples)} (missing files={test_missing}, unknown language rows={test_unknown_lang})"
    )
    print(f"Detected input feature dimension: {input_dim}")
    print(f"Languages: {idx_to_lang}")
    print(f"Device: {device} (AMP={'on' if amp_enabled else 'off'})")

    train_ds = FeatureDataset(
        train_samples,
        input_dim=input_dim,
        training=True,
        noise_std=args.noise_std,
        feature_dropout=args.feature_dropout,
        scale_jitter_std=args.scale_jitter_std,
    )
    val_ds = FeatureDataset(val_samples, input_dim=input_dim, training=False)
    test_ds = FeatureDataset(test_samples, input_dim=input_dim, training=False)

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

    model = DeepfakeDetector(
        input_dim=input_dim,
        num_languages=len(lang_to_idx),
        hidden_dim=args.hidden_dim,
        language_emb_dim=args.language_emb_dim,
        dropout=args.dropout,
    ).to(device)

    train_labels = np.array([s.label for s in train_samples], dtype=np.int64)
    positive = int(train_labels.sum())
    negative = int(train_labels.shape[0] - positive)
    pos_weight_value = negative / max(1, positive)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, args.lr_patience),
        min_lr=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_checkpoint_path = out_dir / "best_model.pt"
    best_state = None
    best_score = (-math.inf, -math.inf)
    no_improve_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_prob, train_y, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_loss, val_prob, val_y, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )

        threshold = best_threshold(val_y, val_prob)
        train_metrics = compute_metrics(train_y, train_prob, threshold)
        val_metrics = compute_metrics(val_y, val_prob, threshold)
        score_auc = val_metrics["roc_auc"]
        score_auc = -math.inf if math.isnan(score_auc) else score_auc
        score = (score_auc, val_metrics["f1"])

        scheduler.step(score_auc if score_auc > -math.inf else val_metrics["f1"])

        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            "  "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_auc={val_metrics['roc_auc']:.5f} val_f1={val_metrics['f1']:.5f} "
            f"threshold={threshold:.3f} lr={current_lr:.7f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_threshold": float(threshold),
                "train_auc": float(train_metrics["roc_auc"]),
                "train_f1": float(train_metrics["f1"]),
                "val_auc": float(val_metrics["roc_auc"]),
                "val_f1": float(val_metrics["f1"]),
                "lr": current_lr,
            }
        )

        if score > best_score:
            best_score = score
            no_improve_epochs = 0
            best_state = {
                "epoch": epoch,
                "input_dim": input_dim,
                "threshold": float(threshold),
                "model_state_dict": model.state_dict(),
                "lang_to_idx": lang_to_idx,
                "args": vars(args),
                "best_val_metrics": val_metrics,
            }
            torch.save(best_state, best_checkpoint_path)
            print(f"  Saved new best checkpoint to: {best_checkpoint_path}")
        else:
            no_improve_epochs += 1
            print(
                f"  No improvement ({no_improve_epochs}/{args.early_stopping_patience})"
            )

        if no_improve_epochs >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    print("\nLoading best checkpoint for final validation/test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_t = float(checkpoint["threshold"])

    val_loss, val_prob, val_y, val_lang = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp_enabled=amp_enabled,
    )
    test_loss, test_prob, test_y, test_lang = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp_enabled=amp_enabled,
    )

    val_metrics = compute_metrics(val_y, val_prob, best_t)
    test_metrics = compute_metrics(test_y, test_prob, best_t)
    val_by_lang = evaluate_by_language(val_y, val_prob, val_lang, idx_to_lang, best_t)
    test_by_lang = evaluate_by_language(
        test_y, test_prob, test_lang, idx_to_lang, best_t
    )

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
            "by_language": val_by_lang,
        },
        "test": {
            "loss": float(test_loss),
            "metrics": test_metrics,
            "by_language": test_by_lang,
        },
        "history": history,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nFinal results:")
    print(f"  Best epoch: {results['best_epoch']}")
    print(f"  Threshold:  {results['threshold']:.3f}")
    print(f"  Val AUC/F1: {val_metrics['roc_auc']:.5f} / {val_metrics['f1']:.5f}")
    print(f"  Test AUC/F1:{test_metrics['roc_auc']:.5f} / {test_metrics['f1']:.5f}")
    print(f"  Metrics JSON: {metrics_path}")
    print(f"  Best model:   {best_checkpoint_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train deepfake multilingual audio detector on precomputed .npy embeddings."
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--language-emb-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.35)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=6)

    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--feature-dropout", type=float, default=0.02)
    parser.add_argument("--scale-jitter-std", type=float, default=0.02)

    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training (CUDA only).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
