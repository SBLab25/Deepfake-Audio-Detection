import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_rows(csv_path: Path, origin_split: str) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_origin_split"] = origin_split
    return rows


def parse_source(audio_path: str) -> tuple[str, str]:
    p = Path(audio_path)
    parts = [x.lower() for x in p.parts]
    if "raw dataset" in parts:
        i = parts.index("raw dataset")
        cls = parts[i + 1] if i + 1 < len(parts) else "<cls>"
        sub = parts[i + 2] if i + 2 < len(parts) else cls
        return cls, sub
    parent = p.parent.name.lower() if p.parent else "<parent>"
    return parent, parent


def en_real_speaker_id(audio_path: str) -> str:
    stem = Path(audio_path).stem
    # e.g. 1089_134686_000002_000001 -> speaker 1089
    return stem.split("_")[0] if "_" in stem else stem


def class_summary(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[f"{r['language']}_{r['label']}"] += 1
    return dict(sorted(out.items()))


def source_summary(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        cls, sub = parse_source(r["audio_path"])
        out[f"{r['language']}_{r['label']}|{cls}|{sub}"] += 1
    return dict(sorted(out.items()))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["feature_path", "label", "language", "audio_path"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "feature_path": r["feature_path"],
                    "label": r["label"],
                    "language": r["language"],
                    "audio_path": r["audio_path"],
                }
            )


def overlap_count(a: list[dict], b: list[dict], key: str) -> int:
    return len(set(r[key] for r in a) & set(r[key] for r in b))


def assign_groups_to_targets(
    grouped_rows: dict[str, list[dict]],
    targets: dict[str, int],
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    groups = list(grouped_rows.items())
    rng.shuffle(groups)
    groups.sort(key=lambda kv: len(kv[1]), reverse=True)

    counts = {"train": 0, "val": 0, "test": 0}
    out = {"train": [], "val": [], "test": []}
    for _, rows in groups:
        size = len(rows)
        best = None
        best_score = None
        for split in ("train", "val", "test"):
            temp = dict(counts)
            temp[split] += size
            score = sum((temp[s] - targets[s]) ** 2 for s in ("train", "val", "test"))
            if best_score is None or score < best_score:
                best_score = score
                best = split
        out[best].extend(rows)
        counts[best] += size
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build challenge splits with unseen-generator English fake eval."
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument(
        "--strict-train-csv",
        type=str,
        default="RawStrict_train/RawStrict_train/balanced_index.csv",
    )
    p.add_argument(
        "--strict-val-csv",
        type=str,
        default="RawStrict_val/RawStrict_val/balanced_index.csv",
    )
    p.add_argument(
        "--strict-test-csv",
        type=str,
        default="RawStrict_test/RawStrict_test/balanced_index.csv",
    )
    p.add_argument("--prefix", type=str, default="RawCh")
    p.add_argument(
        "--en-fake-val-sources",
        type=str,
        nargs="+",
        default=["seedtts_files"],
        help="Subsources held out for EN fake validation split.",
    )
    p.add_argument(
        "--en-fake-test-sources",
        type=str,
        nargs="+",
        default=["openai", "xtts"],
        help="Subsources held out for EN fake test split.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    strict_train = (root / args.strict_train_csv).resolve()
    strict_val = (root / args.strict_val_csv).resolve()
    strict_test = (root / args.strict_test_csv).resolve()

    rows = (
        read_rows(strict_train, "train")
        + read_rows(strict_val, "val")
        + read_rows(strict_test, "test")
    )

    # Start with existing strict split assignment for all rows.
    split_rows = {"train": [], "val": [], "test": []}
    for r in rows:
        split_rows[r["_origin_split"]].append(r)

    # Rebuild English subset with generator-held-out protocol.
    en_fake = []
    en_real = []
    others = {"train": [], "val": [], "test": []}
    for r in rows:
        lang = r["language"].strip().lower()
        label = r["label"].strip().lower()
        if lang == "en" and label == "fake":
            en_fake.append(r)
        elif lang == "en" and label == "real":
            en_real.append(r)
        else:
            others[r["_origin_split"]].append(r)

    val_sources = {s.strip().lower() for s in args.en_fake_val_sources}
    test_sources = {s.strip().lower() for s in args.en_fake_test_sources}
    if val_sources & test_sources:
        raise ValueError("Validation and test source sets overlap.")

    en_fake_split = {"train": [], "val": [], "test": []}
    for r in en_fake:
        _, sub = parse_source(r["audio_path"])
        if sub in test_sources:
            en_fake_split["test"].append(r)
        elif sub in val_sources:
            en_fake_split["val"].append(r)
        else:
            en_fake_split["train"].append(r)

    if min(len(en_fake_split["train"]), len(en_fake_split["val"]), len(en_fake_split["test"])) == 0:
        raise RuntimeError(
            "EN fake source split produced an empty split. Adjust --en-fake-val-sources/--en-fake-test-sources."
        )

    # Assign EN real by speaker groups to match EN fake counts in each split.
    targets = {
        "train": len(en_fake_split["train"]),
        "val": len(en_fake_split["val"]),
        "test": len(en_fake_split["test"]),
    }
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in en_real:
        by_speaker[en_real_speaker_id(r["audio_path"])].append(r)
    en_real_split = assign_groups_to_targets(by_speaker, targets=targets, seed=args.seed)

    final = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        final[split].extend(others[split])
        final[split].extend(en_fake_split[split])
        final[split].extend(en_real_split[split])

    rng = random.Random(args.seed)
    for split in ("train", "val", "test"):
        rng.shuffle(final[split])

    prefix = args.prefix.strip()
    if not prefix:
        raise ValueError("prefix cannot be empty")
    train_out = root / f"{prefix}_train" / f"{prefix}_train" / "balanced_index.csv"
    val_out = root / f"{prefix}_val" / f"{prefix}_val" / "balanced_index.csv"
    test_out = root / f"{prefix}_test" / f"{prefix}_test" / "balanced_index.csv"
    write_csv(train_out, final["train"])
    write_csv(val_out, final["val"])
    write_csv(test_out, final["test"])

    summary = {
        "inputs": {
            "strict_train_csv": str(strict_train),
            "strict_val_csv": str(strict_val),
            "strict_test_csv": str(strict_test),
        },
        "policy": {
            "en_fake_val_sources": sorted(val_sources),
            "en_fake_test_sources": sorted(test_sources),
            "en_real_assignment": "speaker-grouped to match EN fake split counts",
            "hi_splits": "kept from strict split assignment",
        },
        "outputs": {
            "train_csv": str(train_out),
            "val_csv": str(val_out),
            "test_csv": str(test_out),
        },
        "counts": {
            "train": len(final["train"]),
            "val": len(final["val"]),
            "test": len(final["test"]),
        },
        "class_breakdown": {
            "train": class_summary(final["train"]),
            "val": class_summary(final["val"]),
            "test": class_summary(final["test"]),
        },
        "source_breakdown": {
            "train": source_summary(final["train"]),
            "val": source_summary(final["val"]),
            "test": source_summary(final["test"]),
        },
        "overlap_checks": {
            "audio_path_train_val": overlap_count(final["train"], final["val"], "audio_path"),
            "audio_path_train_test": overlap_count(final["train"], final["test"], "audio_path"),
            "audio_path_val_test": overlap_count(final["val"], final["test"], "audio_path"),
            "feature_path_train_val": overlap_count(final["train"], final["val"], "feature_path"),
            "feature_path_train_test": overlap_count(final["train"], final["test"], "feature_path"),
            "feature_path_val_test": overlap_count(final["val"], final["test"], "feature_path"),
        },
    }
    summary_path = root / f"{prefix}_splits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Challenge splits created.")
    print(json.dumps(summary["counts"], indent=2))
    print("Overlap checks:")
    print(json.dumps(summary["overlap_checks"], indent=2))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
