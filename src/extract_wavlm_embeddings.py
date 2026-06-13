from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


SUPPORTED_EXTS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".wma"]


@dataclass
class Stats:
    total_rows: int = 0
    extracted: int = 0
    skipped_existing: int = 0
    missing_audio: int = 0
    ambiguous_audio: int = 0
    failed_audio_load: int = 0
    failed_forward: int = 0


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def build_audio_index(audio_root: Path, exts: List[str]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for p in audio_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        key = p.stem.lower()
        index.setdefault(key, []).append(p)
    return index


def try_load_with_soundfile(path: Path):
    import soundfile as sf

    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def try_load_with_librosa(path: Path):
    import librosa

    wav, sr = librosa.load(path, sr=None, mono=True)
    return wav.astype(np.float32), int(sr)


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    wav = None
    sr = None

    try:
        wav, sr = try_load_with_soundfile(path)
    except Exception:
        wav, sr = try_load_with_librosa(path)

    if wav is None or wav.size == 0:
        raise RuntimeError(f"Empty audio: {path}")

    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        wav = resample_poly(wav, up=up, down=down).astype(np.float32)

    if wav.ndim != 1:
        wav = wav.reshape(-1)
    return wav.astype(np.float32)


def resolve_audio_path(
    row: dict,
    feature_col: str,
    audio_col: str | None,
    audio_root: Path,
    audio_index: Dict[str, List[Path]],
) -> tuple[Path | None, bool]:
    if audio_col and row.get(audio_col):
        raw = row[audio_col].strip()
        if raw:
            cand = Path(raw)
            if not cand.is_absolute():
                cand = (audio_root / cand).resolve()
            return cand if cand.exists() else None, False

    feat_name = Path(row[feature_col]).stem.lower()
    matches = audio_index.get(feat_name, [])
    if not matches:
        return None, False
    if len(matches) > 1:
        return matches[0], True
    return matches[0], False


def extract_embeddings(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv_path).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else None

    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.audio_exts]
    feature_col = args.feature_col
    audio_col = args.audio_col
    overwrite = bool(args.overwrite)

    device = device_from_arg(args.device)
    print(f"Device: {device}")
    print(f"CSV: {csv_path}")
    print(f"Audio root: {audio_root}")
    print(f"Output root: {output_root}")
    print(f"Audio ext filter: {exts}")

    print("Loading model...")
    extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    target_sr = int(extractor.sampling_rate)
    print(f"Model: {args.model_name} | target_sr={target_sr}")

    print("Indexing audio files...")
    audio_index = build_audio_index(audio_root, exts)
    print(f"Indexed stems: {len(audio_index)}")

    stats = Stats()
    failed_items: List[dict] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if feature_col not in (reader.fieldnames or []):
            raise ValueError(f"CSV missing '{feature_col}' column.")
        if audio_col and audio_col not in (reader.fieldnames or []):
            raise ValueError(f"CSV missing audio column '{audio_col}'.")

        rows = list(reader)

    stats.total_rows = len(rows)
    for row in tqdm(rows, desc="extracting wavlm embeddings"):
        rel_feat = Path(row[feature_col])
        out_path = (output_root / rel_feat).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            stats.skipped_existing += 1
            continue

        audio_path, ambiguous = resolve_audio_path(
            row=row,
            feature_col=feature_col,
            audio_col=audio_col,
            audio_root=audio_root,
            audio_index=audio_index,
        )
        if ambiguous:
            stats.ambiguous_audio += 1

        if audio_path is None:
            stats.missing_audio += 1
            failed_items.append(
                {"feature_path": str(rel_feat), "reason": "missing_audio"}
            )
            continue

        try:
            wav = load_audio(audio_path, target_sr=target_sr)
        except Exception as exc:
            stats.failed_audio_load += 1
            failed_items.append(
                {
                    "feature_path": str(rel_feat),
                    "audio_path": str(audio_path),
                    "reason": f"audio_load_error: {exc}",
                }
            )
            continue

        try:
            with torch.no_grad():
                inputs = extractor(
                    wav,
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=False,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                emb = emb.astype(np.float32)
        except Exception as exc:
            stats.failed_forward += 1
            failed_items.append(
                {
                    "feature_path": str(rel_feat),
                    "audio_path": str(audio_path),
                    "reason": f"model_forward_error: {exc}",
                }
            )
            continue

        np.save(out_path, emb)
        stats.extracted += 1

    summary = {
        "csv_path": str(csv_path),
        "audio_root": str(audio_root),
        "output_root": str(output_root),
        "model_name": args.model_name,
        "target_sampling_rate": target_sr,
        "device": str(device),
        "stats": stats.__dict__,
        "failed_examples": failed_items[:200],
    }

    print("\nExtraction complete:")
    for k, v in stats.__dict__.items():
        print(f"  {k}: {v}")

    if report_path is None:
        report_path = output_root / "wavlm_extraction_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract WavLM embeddings aligned to feature_path names in CSV."
    )
    p.add_argument("--csv-path", required=True, type=str)
    p.add_argument("--audio-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--report-path", type=str, default=None)
    p.add_argument("--feature-col", type=str, default="feature_path")
    p.add_argument(
        "--audio-col",
        type=str,
        default=None,
        help="Optional CSV column that stores audio file path per row.",
    )
    p.add_argument(
        "--audio-exts",
        nargs="+",
        default=SUPPORTED_EXTS,
        help="Audio extensions to index under audio-root.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="microsoft/wavlm-base-plus",
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    extract_embeddings(args)


if __name__ == "__main__":
    main()
