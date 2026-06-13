import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def collect_records(raw_root: Path) -> list[dict]:
    source_map = {
        "eng fake": ("en", "fake"),
        "eng real": ("en", "real"),
        "hindi fake": ("hi", "fake"),
        "hindi real": ("hi", "real"),
    }

    records: list[dict] = []
    feature_seen: dict[str, int] = defaultdict(int)

    for folder_name, (language, label) in source_map.items():
        folder = raw_root / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing expected folder: {folder}")

        for audio_path in sorted(folder.rglob("*")):
            if not audio_path.is_file():
                continue
            if audio_path.suffix.lower() not in AUDIO_EXTS:
                continue

            stem = audio_path.stem
            base = f"{language}/{label}/{stem}.npy"
            dup_idx = feature_seen[base]
            if dup_idx == 0:
                feature_path = base
            else:
                feature_path = f"{language}/{label}/{stem}_{dup_idx:03d}.npy"
            feature_seen[base] += 1

            records.append(
                {
                    "feature_path": feature_path,
                    "label": label,
                    "language": language,
                    "audio_path": str(audio_path.resolve()),
                }
            )

    if not records:
        raise RuntimeError(f"No audio files found under {raw_root}")
    return records


def stratified_split(
    records: list[dict], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["label"], row["language"])].append(row)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for key, rows in grouped.items():
        rows = list(rows)
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)

        train_rows.extend(rows[:n_train])
        val_rows.extend(rows[n_train : n_train + n_val])
        test_rows.extend(rows[n_train + n_val :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["feature_path", "label", "language", "audio_path"]
        )
        w.writeheader()
        w.writerows(rows)


def class_summary(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[f"{r['language']}_{r['label']}"] += 1
    return dict(sorted(out.items()))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build train/val/test CSV splits from extracted raw audio dataset."
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--raw-root", type=str, default="Raw Dataset")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    raw_root = (project_root / args.raw_root).resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be >0 and sum to <1")

    rows = collect_records(raw_root)
    train_rows, val_rows, test_rows = stratified_split(
        rows, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    train_csv = project_root / "Raw_train" / "Raw_train" / "balanced_index.csv"
    val_csv = project_root / "Raw_val" / "Raw_val" / "balanced_index.csv"
    test_csv = project_root / "Raw_test" / "Raw_test" / "balanced_index.csv"

    write_csv(train_csv, train_rows)
    write_csv(val_csv, val_rows)
    write_csv(test_csv, test_rows)

    summary = {
        "raw_root": str(raw_root),
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "train_breakdown": class_summary(train_rows),
        "val_breakdown": class_summary(val_rows),
        "test_breakdown": class_summary(test_rows),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
    }
    summary_path = project_root / "Raw_splits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Built raw splits:")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
