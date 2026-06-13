import argparse
import csv
import filecmp
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"feature_path", "label", "language", "audio_path"}
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return rows


def audio_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dedup_by_audio_hash(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    hash_seen: dict[str, dict] = {}
    keep: list[dict] = []
    dropped: list[dict] = []

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r["audio_path"]).lower(),
            str(r["feature_path"]).lower(),
            str(r["language"]).lower(),
            str(r["label"]).lower(),
        ),
    )

    for idx, row in enumerate(rows_sorted, 1):
        p = Path(row["audio_path"])
        if not p.exists():
            raise FileNotFoundError(f"Missing audio file in CSV: {p}")
        h = audio_md5(p)
        row2 = dict(row)
        row2["_audio_hash"] = h
        if h in hash_seen:
            prev = hash_seen[h]
            dropped.append(
                {
                    "hash": h,
                    "kept_audio_path": prev["audio_path"],
                    "kept_feature_path": prev["feature_path"],
                    "dropped_audio_path": row2["audio_path"],
                    "dropped_feature_path": row2["feature_path"],
                }
            )
            continue
        hash_seen[h] = row2
        keep.append(row2)
        if idx % 1000 == 0:
            print(f"Hashed {idx}/{len(rows_sorted)} rows...")
    return keep, dropped


def infer_source_parts(audio_path: str) -> tuple[str, str]:
    p = Path(audio_path)
    parts = list(p.parts)
    lower = [x.lower() for x in parts]
    if "raw dataset" in lower:
        i = lower.index("raw dataset")
        cls = lower[i + 1] if i + 1 < len(lower) else "<cls>"
        # keep subsource to preserve generator/domain grouping
        sub = lower[i + 2] if i + 2 < len(lower) else cls
        return cls, sub
    parent = p.parent.name.lower() if p.parent else "<parent>"
    return parent, parent


def infer_family_key(stem: str) -> str:
    s = stem.lower()
    us = s.split("_")
    if len(us) >= 4 and all(tok.isdigit() for tok in us):
        # e.g., 1089_134686_000002_000001 -> 1089_134686_000002
        return "_".join(us[:-1])
    ds = s.split("-")
    if len(ds) >= 3 and all(tok.isdigit() for tok in ds):
        # e.g., 02-12676-02 -> 02-12676
        return "-".join(ds[:-1])
    m = re.match(r"^(.+)_chunk\d+$", s)
    if m:
        return m.group(1)
    return s


def group_id(row: dict) -> str:
    cls, sub = infer_source_parts(row["audio_path"])
    stem = Path(row["audio_path"]).stem
    fam = infer_family_key(stem)
    lang = row["language"].strip().lower()
    label = row["label"].strip().lower()
    return f"{lang}|{label}|{cls}|{sub}|{fam}"


def compute_targets(n: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
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
    return {"train": n_train, "val": n_val, "test": n_test}


def grouped_stratified_split(
    rows: list[dict], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[dict], list[dict], list[dict], dict]:
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[(r["language"], r["label"])].append(r)

    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    stats = {}

    for stratum, stratum_rows in sorted(by_stratum.items()):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for r in stratum_rows:
            grouped[group_id(r)].append(r)
        groups = list(grouped.values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)

        targets = compute_targets(len(stratum_rows), train_ratio, val_ratio)
        counts = {"train": 0, "val": 0, "test": 0}
        alloc = {"train": [], "val": [], "test": []}

        for g in groups:
            size = len(g)
            best_split = None
            best_score = None
            for split in ("train", "val", "test"):
                temp = dict(counts)
                temp[split] += size
                # minimize squared distance from targets
                score = sum((temp[s] - targets[s]) ** 2 for s in ("train", "val", "test"))
                if best_score is None or score < best_score:
                    best_score = score
                    best_split = split
            alloc[best_split].extend(g)
            counts[best_split] += size

        for split in ("train", "val", "test"):
            out[split].extend(alloc[split])
            rng.shuffle(out[split])

        stats[f"{stratum[0]}_{stratum[1]}"] = {
            "rows": len(stratum_rows),
            "groups": len(groups),
            "target": targets,
            "actual": counts,
        }

    return out["train"], out["val"], out["test"], stats


def class_summary(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[f"{r['language']}_{r['label']}"] += 1
    return dict(sorted(out.items()))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
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
    sa = set(r[key] for r in a)
    sb = set(r[key] for r in b)
    return len(sa & sb)


def group_overlap_count(a: list[dict], b: list[dict]) -> int:
    sa = set(group_id(r) for r in a)
    sb = set(group_id(r) for r in b)
    return len(sa & sb)


def build_unified_embedding_root(source_root: Path, out_root: Path) -> dict:
    copied = 0
    skipped_same = 0
    conflicts = 0
    missing_split_roots = []
    for split in ("train", "val", "test"):
        src_split = source_root / split
        if not src_split.exists():
            missing_split_roots.append(str(src_split))
            continue
        for src in src_split.rglob("*.npy"):
            rel = src.relative_to(src_split)
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if filecmp.cmp(src, dst, shallow=False):
                    skipped_same += 1
                else:
                    conflicts += 1
                continue
            shutil.copy2(src, dst)
            copied += 1

    return {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "copied": copied,
        "skipped_same": skipped_same,
        "conflicts": conflicts,
        "missing_split_roots": missing_split_roots,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build strict raw splits with hash dedup + group-aware stratification."
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--train-csv", type=str, default="Raw_train/Raw_train/balanced_index.csv")
    p.add_argument("--val-csv", type=str, default="Raw_val/Raw_val/balanced_index.csv")
    p.add_argument("--test-csv", type=str, default="Raw_test/Raw_test/balanced_index.csv")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict-prefix", type=str, default="RawStrict")
    p.add_argument("--build-unified-embeddings", action="store_true")
    p.add_argument("--wavlm-source-root", type=str, default="WavLM_embeddings")
    p.add_argument("--wavlm-unified-root", type=str, default="WavLM_embeddings_unified")
    args = p.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be >0 and sum to <1")

    root = Path(args.project_root).expanduser().resolve()
    train_csv = (root / args.train_csv).resolve()
    val_csv = (root / args.val_csv).resolve()
    test_csv = (root / args.test_csv).resolve()
    for c in (train_csv, val_csv, test_csv):
        if not c.exists():
            raise FileNotFoundError(f"Missing CSV: {c}")

    print("Loading rows from existing raw splits...")
    train_rows = read_rows(train_csv)
    val_rows = read_rows(val_csv)
    test_rows = read_rows(test_csv)
    all_rows = train_rows + val_rows + test_rows
    print(f"Loaded rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} total={len(all_rows)}")

    print("Deduplicating by audio hash (this may take ~1 minute)...")
    dedup_rows, dropped = dedup_by_audio_hash(all_rows)
    print(f"Dedup complete: kept={len(dedup_rows)} dropped={len(dropped)}")

    print("Building group-aware stratified strict splits...")
    strict_train, strict_val, strict_test, stratum_stats = grouped_stratified_split(
        dedup_rows, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    prefix = args.strict_prefix.strip()
    if not prefix:
        raise ValueError("strict-prefix cannot be empty.")
    train_out = root / f"{prefix}_train" / f"{prefix}_train" / "balanced_index.csv"
    val_out = root / f"{prefix}_val" / f"{prefix}_val" / "balanced_index.csv"
    test_out = root / f"{prefix}_test" / f"{prefix}_test" / "balanced_index.csv"

    write_csv(train_out, strict_train)
    write_csv(val_out, strict_val)
    write_csv(test_out, strict_test)

    summary = {
        "project_root": str(root),
        "inputs": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
        },
        "dedup": {
            "original_total": len(all_rows),
            "kept_total": len(dedup_rows),
            "dropped_duplicates": len(dropped),
            "dropped_examples": dropped[:20],
        },
        "strict_splits": {
            "train_csv": str(train_out),
            "val_csv": str(val_out),
            "test_csv": str(test_out),
            "train": len(strict_train),
            "val": len(strict_val),
            "test": len(strict_test),
            "train_breakdown": class_summary(strict_train),
            "val_breakdown": class_summary(strict_val),
            "test_breakdown": class_summary(strict_test),
        },
        "stratum_group_stats": stratum_stats,
        "overlap_checks": {
            "audio_path_train_val": overlap_count(strict_train, strict_val, "audio_path"),
            "audio_path_train_test": overlap_count(strict_train, strict_test, "audio_path"),
            "audio_path_val_test": overlap_count(strict_val, strict_test, "audio_path"),
            "hash_train_val": overlap_count(strict_train, strict_val, "_audio_hash"),
            "hash_train_test": overlap_count(strict_train, strict_test, "_audio_hash"),
            "hash_val_test": overlap_count(strict_val, strict_test, "_audio_hash"),
            "group_train_val": group_overlap_count(strict_train, strict_val),
            "group_train_test": group_overlap_count(strict_train, strict_test),
            "group_val_test": group_overlap_count(strict_val, strict_test),
        },
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
    }

    if args.build_unified_embeddings:
        src = (root / args.wavlm_source_root).resolve()
        dst = (root / args.wavlm_unified_root).resolve()
        print(f"Building unified embedding root: {dst}")
        emb_stats = build_unified_embedding_root(src, dst)
        # verify strict split features exist in unified root
        missing_features = []
        for split_name, rows in (
            ("train", strict_train),
            ("val", strict_val),
            ("test", strict_test),
        ):
            miss = 0
            for r in rows:
                fp = dst / r["feature_path"]
                if not fp.exists():
                    miss += 1
            missing_features.append({split_name: miss})
        summary["unified_embeddings"] = emb_stats
        summary["unified_embeddings"]["missing_features_in_strict_splits"] = missing_features

    out_summary = root / f"{prefix}_splits_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Strict splits created.")
    print(json.dumps(summary["strict_splits"], indent=2))
    print("Overlap checks:")
    print(json.dumps(summary["overlap_checks"], indent=2))
    print(f"Summary written to: {out_summary}")


if __name__ == "__main__":
    main()
