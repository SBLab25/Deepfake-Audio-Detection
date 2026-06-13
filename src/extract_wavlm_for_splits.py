from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from extract_wavlm_embeddings import SUPPORTED_EXTS, extract_embeddings


def resolve_default_csv(project_root: Path, split_name: str) -> Path:
    split_root = (project_root / split_name).resolve()
    candidates = sorted(split_root.rglob("balanced_index.csv"))
    if not candidates:
        candidates = sorted(split_root.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found for split '{split_name}' under {split_root}")
    return candidates[0]


def run_one_split(
    split_label: str,
    csv_path: Path,
    audio_root: Path,
    output_root: Path,
    model_name: str,
    audio_col: str | None,
    audio_exts: list[str],
    device: str,
    overwrite: bool,
) -> None:
    print(f"\n=== {split_label} split ===")
    args = Namespace(
        csv_path=str(csv_path),
        audio_root=str(audio_root),
        output_root=str(output_root),
        report_path=str(output_root / "wavlm_extraction_report.json"),
        feature_col="feature_path",
        audio_col=audio_col,
        audio_exts=audio_exts,
        model_name=model_name,
        device=device,
        overwrite=overwrite,
    )
    extract_embeddings(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract WavLM embeddings for train/val/test splits."
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--out-root", type=str, default="./WavLM_embeddings")

    p.add_argument("--train-csv", type=str, default=None)
    p.add_argument("--val-csv", type=str, default=None)
    p.add_argument("--test-csv", type=str, default=None)

    p.add_argument("--train-audio-root", type=str, required=True)
    p.add_argument("--val-audio-root", type=str, required=True)
    p.add_argument("--test-audio-root", type=str, required=True)

    p.add_argument("--audio-col", type=str, default=None)
    p.add_argument("--audio-exts", nargs="+", default=SUPPORTED_EXTS)
    p.add_argument("--model-name", type=str, default="microsoft/wavlm-base-plus")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    train_csv = (
        Path(args.train_csv).expanduser().resolve()
        if args.train_csv
        else resolve_default_csv(project_root, "Balanced_train")
    )
    val_csv = (
        Path(args.val_csv).expanduser().resolve()
        if args.val_csv
        else resolve_default_csv(project_root, "Balanced_val")
    )
    test_csv = (
        Path(args.test_csv).expanduser().resolve()
        if args.test_csv
        else resolve_default_csv(project_root, "Balanced_test")
    )

    run_one_split(
        "train",
        csv_path=train_csv,
        audio_root=Path(args.train_audio_root).expanduser().resolve(),
        output_root=out_root / "train",
        model_name=args.model_name,
        audio_col=args.audio_col,
        audio_exts=args.audio_exts,
        device=args.device,
        overwrite=bool(args.overwrite),
    )
    run_one_split(
        "val",
        csv_path=val_csv,
        audio_root=Path(args.val_audio_root).expanduser().resolve(),
        output_root=out_root / "val",
        model_name=args.model_name,
        audio_col=args.audio_col,
        audio_exts=args.audio_exts,
        device=args.device,
        overwrite=bool(args.overwrite),
    )
    run_one_split(
        "test",
        csv_path=test_csv,
        audio_root=Path(args.test_audio_root).expanduser().resolve(),
        output_root=out_root / "test",
        model_name=args.model_name,
        audio_col=args.audio_col,
        audio_exts=args.audio_exts,
        device=args.device,
        overwrite=bool(args.overwrite),
    )

    print("\nAll split extractions completed.")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
