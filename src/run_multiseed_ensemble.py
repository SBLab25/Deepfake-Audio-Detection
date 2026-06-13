from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_label(v: str) -> int:
    return 1 if v.strip().lower() == "fake" else 0


def auc_score(labels: List[int], scores: List[float]) -> float:
    n = len(labels)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        rank += j - i + 1
        i = j + 1

    pos = [idx for idx, y in enumerate(labels) if y == 1]
    neg = [idx for idx, y in enumerate(labels) if y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos_ranks = sum(ranks[idx] for idx in pos)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def f1_score(labels: List[int], preds: List[int]) -> float:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def accuracy(labels: List[int], preds: List[int]) -> float:
    return sum(1 for y, p in zip(labels, preds) if y == p) / max(1, len(labels))


def read_prediction_csv(path: Path) -> Dict[str, Tuple[int, float]]:
    out: Dict[str, Tuple[int, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feature_path = row["feature_path"].strip()
            label = parse_label(row["label"])
            prob = float(row["fake_probability"])
            out[feature_path] = (label, prob)
    return out


def aggregate_mean_prob(paths: List[Path]) -> Tuple[List[str], List[int], List[float]]:
    tables = [read_prediction_csv(p) for p in paths]
    keys = sorted(set.intersection(*(set(t.keys()) for t in tables)))
    labels: List[int] = []
    probs: List[float] = []
    for k in keys:
        label = tables[0][k][0]
        mean_prob = sum(t[k][1] for t in tables) / len(tables)
        labels.append(label)
        probs.append(mean_prob)
    return keys, labels, probs


def best_threshold(labels: List[int], probs: List[float]) -> Tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    best_acc = -1.0
    candidates = sorted(set([i / 1000.0 for i in range(1001)] + probs))
    for t in candidates:
        preds = [1 if p >= t else 0 for p in probs]
        f1 = f1_score(labels, preds)
        acc = accuracy(labels, preds)
        if (f1, acc) > (best_f1, best_acc):
            best_f1 = f1
            best_acc = acc
            best_t = t
    return best_t, best_f1


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train multiple Safari+WavLM seeds and evaluate ensemble average."
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="./artifacts/multiseed_ensemble")
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 77, 123, 202])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--language-emb-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--view-noise-std", type=float, default=0.01)
    p.add_argument("--view-drop-prob", type=float, default=0.03)
    p.add_argument("--stats-samples", type=int, default=50000)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--amp", action="store_true")

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
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.project_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def has_npy_files(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            next(path.rglob("*.npy"))
            return True
        except StopIteration:
            return False

    auto_wavlm_root = root / "WavLM_embeddings"
    auto_train_wavlm = auto_wavlm_root / "train"
    auto_val_wavlm = auto_wavlm_root / "val"
    auto_test_wavlm = auto_wavlm_root / "test"

    wavlm_train_dir = (
        Path(args.wavlm_train_feature_dir).expanduser().resolve()
        if args.wavlm_train_feature_dir
        else (auto_train_wavlm.resolve() if has_npy_files(auto_train_wavlm) else None)
    )
    wavlm_val_dir = (
        Path(args.wavlm_val_feature_dir).expanduser().resolve()
        if args.wavlm_val_feature_dir
        else (auto_val_wavlm.resolve() if has_npy_files(auto_val_wavlm) else None)
    )
    wavlm_test_dir = (
        Path(args.wavlm_test_feature_dir).expanduser().resolve()
        if args.wavlm_test_feature_dir
        else (auto_test_wavlm.resolve() if has_npy_files(auto_test_wavlm) else None)
    )

    val_csv = (
        Path(args.val_csv).expanduser().resolve()
        if args.val_csv
        else (root / args.val_split / args.val_split / "balanced_index.csv").resolve()
    )
    test_csv = (
        Path(args.test_csv).expanduser().resolve()
        if args.test_csv
        else (root / args.test_split / args.test_split / "balanced_index.csv").resolve()
    )
    val_base_root = (
        Path(args.val_feature_dir).expanduser().resolve()
        if args.val_feature_dir
        else val_csv.parent.resolve()
    )
    test_base_root = (
        Path(args.test_feature_dir).expanduser().resolve()
        if args.test_feature_dir
        else test_csv.parent.resolve()
    )

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "safari_wavlm_ensemble_train.py"
    predict_script = script_dir / "predict_safari_wavlm_ensemble.py"

    val_pred_paths: List[Path] = []
    test_pred_paths: List[Path] = []
    seed_runs: List[dict] = []

    for seed in args.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            str(train_script),
            "--project-root",
            str(root),
            "--out-dir",
            str(seed_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--hidden-dim",
            str(args.hidden_dim),
            "--language-emb-dim",
            str(args.language_emb_dim),
            "--lr",
            str(args.lr),
            "--max-lr",
            str(args.max_lr),
            "--weight-decay",
            str(args.weight_decay),
            "--patience",
            str(args.patience),
            "--view-noise-std",
            str(args.view_noise_std),
            "--view-drop-prob",
            str(args.view_drop_prob),
            "--stats-samples",
            str(args.stats_samples),
            "--device",
            str(args.device),
            "--seed",
            str(seed),
            "--train-split",
            str(args.train_split),
            "--val-split",
            str(args.val_split),
            "--test-split",
            str(args.test_split),
        ]
        if args.amp:
            train_cmd.append("--amp")
        if args.train_csv:
            train_cmd += ["--train-csv", str(Path(args.train_csv).expanduser().resolve())]
        if args.val_csv:
            train_cmd += ["--val-csv", str(val_csv)]
        if args.test_csv:
            train_cmd += ["--test-csv", str(test_csv)]
        if args.train_feature_dir:
            train_cmd += [
                "--train-feature-dir",
                str(Path(args.train_feature_dir).expanduser().resolve()),
            ]
        if args.val_feature_dir:
            train_cmd += ["--val-feature-dir", str(val_base_root)]
        if args.test_feature_dir:
            train_cmd += ["--test-feature-dir", str(test_base_root)]

        if wavlm_train_dir is not None:
            train_cmd += ["--wavlm-train-feature-dir", str(wavlm_train_dir)]
        if wavlm_val_dir is not None:
            train_cmd += ["--wavlm-val-feature-dir", str(wavlm_val_dir)]
        if wavlm_test_dir is not None:
            train_cmd += ["--wavlm-test-feature-dir", str(wavlm_test_dir)]

        run_cmd(train_cmd, root)

        model_path = seed_dir / "best_safari_wavlm_ensemble.pt"
        val_pred = seed_dir / "val_predictions.csv"
        test_pred = seed_dir / "test_predictions.csv"

        val_cmd = [
            sys.executable,
            str(predict_script),
            "--model-path",
            str(model_path),
            "--input-csv",
            str(val_csv),
            "--base-feature-root",
            str(val_base_root),
            "--output-csv",
            str(val_pred),
            "--device",
            str(args.device),
        ]
        test_cmd = [
            sys.executable,
            str(predict_script),
            "--model-path",
            str(model_path),
            "--input-csv",
            str(test_csv),
            "--base-feature-root",
            str(test_base_root),
            "--output-csv",
            str(test_pred),
            "--device",
            str(args.device),
        ]
        if wavlm_val_dir is not None:
            val_cmd += ["--wavlm-feature-root", str(wavlm_val_dir)]
        if wavlm_test_dir is not None:
            test_cmd += ["--wavlm-feature-root", str(wavlm_test_dir)]

        run_cmd(val_cmd, root)
        run_cmd(test_cmd, root)

        val_pred_paths.append(val_pred)
        test_pred_paths.append(test_pred)
        seed_runs.append(
            {
                "seed": seed,
                "model_path": str(model_path),
                "val_predictions": str(val_pred),
                "test_predictions": str(test_pred),
                "metrics_path": str(seed_dir / "metrics_safari_wavlm_ensemble.json"),
            }
        )

    val_keys, val_labels, val_probs = aggregate_mean_prob(val_pred_paths)
    test_keys, test_labels, test_probs = aggregate_mean_prob(test_pred_paths)

    threshold, val_f1 = best_threshold(val_labels, val_probs)
    val_pred = [1 if p >= threshold else 0 for p in val_probs]
    test_pred = [1 if p >= threshold else 0 for p in test_probs]

    val_metrics = {
        "roc_auc": auc_score(val_labels, val_probs),
        "f1": val_f1,
        "accuracy": accuracy(val_labels, val_pred),
        "threshold": threshold,
    }
    test_metrics = {
        "roc_auc": auc_score(test_labels, test_probs),
        "f1": f1_score(test_labels, test_pred),
        "accuracy": accuracy(test_labels, test_pred),
        "threshold": threshold,
    }

    blend_dir = out_dir / "ensemble_predictions"
    blend_dir.mkdir(parents=True, exist_ok=True)
    val_out_csv = blend_dir / "val_ensemble_predictions.csv"
    test_out_csv = blend_dir / "test_ensemble_predictions.csv"

    with val_out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_path", "label", "fake_probability", "predicted_label"],
        )
        writer.writeheader()
        for k, y, p in zip(val_keys, val_labels, val_probs):
            writer.writerow(
                {
                    "feature_path": k,
                    "label": "fake" if y == 1 else "real",
                    "fake_probability": f"{p:.6f}",
                    "predicted_label": "fake" if p >= threshold else "real",
                }
            )

    with test_out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_path", "label", "fake_probability", "predicted_label"],
        )
        writer.writeheader()
        for k, y, p in zip(test_keys, test_labels, test_probs):
            writer.writerow(
                {
                    "feature_path": k,
                    "label": "fake" if y == 1 else "real",
                    "fake_probability": f"{p:.6f}",
                    "predicted_label": "fake" if p >= threshold else "real",
                }
            )

    summary = {
        "seeds": args.seeds,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "seed_runs": seed_runs,
        "ensemble_val_predictions": str(val_out_csv),
        "ensemble_test_predictions": str(test_out_csv),
    }
    summary_path = out_dir / "multiseed_ensemble_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nMulti-seed ensemble complete.")
    print(f"Val metrics:  {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
