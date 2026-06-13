from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from train_detector import DeepfakeDetector


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    lang_to_idx = checkpoint.get("lang_to_idx", {"<unk>": 0})
    args = checkpoint.get("args", {})
    model = DeepfakeDetector(
        input_dim=int(checkpoint["input_dim"]),
        num_languages=len(lang_to_idx),
        hidden_dim=int(args.get("hidden_dim", 512)),
        language_emb_dim=int(args.get("language_emb_dim", 32)),
        dropout=float(args.get("dropout", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_feature_vector(path: Path, input_dim: int) -> np.ndarray:
    x = np.load(path).astype(np.float32).reshape(-1)
    if x.shape[0] > input_dim:
        x = x[:input_dim]
    elif x.shape[0] < input_dim:
        x = np.pad(x, (0, input_dim - x.shape[0]))
    return x


def predict_single(
    model: torch.nn.Module,
    feature_vec: np.ndarray,
    language_idx: int,
    device: torch.device,
) -> float:
    with torch.no_grad():
        x = torch.from_numpy(feature_vec).unsqueeze(0).to(device)
        l = torch.tensor([language_idx], dtype=torch.long, device=device)
        logits = model(x, l)
        prob = torch.sigmoid(logits).item()
    return float(prob)


def run_single_mode(
    model: torch.nn.Module,
    checkpoint: dict,
    feature_path: Path,
    language: str,
    device: torch.device,
) -> None:
    lang_to_idx: Dict[str, int] = checkpoint.get("lang_to_idx", {"<unk>": 0})
    threshold = float(checkpoint.get("threshold", 0.5))
    input_dim = int(checkpoint["input_dim"])

    lang_key = language.strip().lower()
    language_idx = lang_to_idx.get(lang_key, lang_to_idx.get("<unk>", 0))
    vec = load_feature_vector(feature_path, input_dim)
    fake_prob = predict_single(model, vec, language_idx, device)
    label = "fake" if fake_prob >= threshold else "real"

    output = {
        "feature_path": str(feature_path.resolve()),
        "language": lang_key,
        "threshold": threshold,
        "fake_probability": fake_prob,
        "predicted_label": label,
    }
    print(json.dumps(output, indent=2))


def run_batch_mode(
    model: torch.nn.Module,
    checkpoint: dict,
    input_csv: Path,
    feature_root: Path | None,
    output_csv: Path,
    device: torch.device,
) -> None:
    lang_to_idx: Dict[str, int] = checkpoint.get("lang_to_idx", {"<unk>": 0})
    threshold = float(checkpoint.get("threshold", 0.5))
    input_dim = int(checkpoint["input_dim"])

    if feature_root is None:
        feature_root = input_csv.parent

    rows_out: List[dict] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "feature_path" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain a 'feature_path' column.")

        for row in reader:
            rel_path = row["feature_path"].strip()
            lang_key = row.get("language", "<unk>").strip().lower() or "<unk>"
            full_path = (feature_root / rel_path).resolve()
            if not full_path.exists():
                rows_out.append(
                    {
                        **row,
                        "fake_probability": "",
                        "predicted_label": "missing_file",
                    }
                )
                continue

            language_idx = lang_to_idx.get(lang_key, lang_to_idx.get("<unk>", 0))
            vec = load_feature_vector(full_path, input_dim)
            fake_prob = predict_single(model, vec, language_idx, device)
            label = "fake" if fake_prob >= threshold else "real"

            rows_out.append(
                {
                    **row,
                    "fake_probability": f"{fake_prob:.6f}",
                    "predicted_label": label,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    if rows_out:
        fieldnames = list(rows_out[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote predictions to: {output_csv.resolve()}")
    print(f"Rows: {len(rows_out)} | Threshold: {threshold:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained deepfake multilingual detector."
    )
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--feature-path", type=str, default=None)
    parser.add_argument("--language", type=str, default="<unk>")

    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--feature-root", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default="./artifacts/predictions.csv")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.feature_path is None and args.input_csv is None:
        raise ValueError("Pass --feature-path for single inference or --input-csv for batch inference.")

    device = device_from_arg(args.device)
    model_path = Path(args.model_path).expanduser().resolve()
    checkpoint = load_checkpoint(model_path, device)
    model = build_model_from_checkpoint(checkpoint, device)

    if args.feature_path is not None:
        run_single_mode(
            model=model,
            checkpoint=checkpoint,
            feature_path=Path(args.feature_path).expanduser().resolve(),
            language=args.language,
            device=device,
        )
    else:
        run_batch_mode(
            model=model,
            checkpoint=checkpoint,
            input_csv=Path(args.input_csv).expanduser().resolve(),
            feature_root=Path(args.feature_root).expanduser().resolve()
            if args.feature_root
            else None,
            output_csv=Path(args.output_csv).expanduser().resolve(),
            device=device,
        )


if __name__ == "__main__":
    main()
