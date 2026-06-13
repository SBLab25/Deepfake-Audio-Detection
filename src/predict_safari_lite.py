from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from safari_lite_train import SafariLitePlus


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def build_model(ckpt: dict, device: torch.device) -> SafariLitePlus:
    args = ckpt.get("args", {})
    lang_to_idx = ckpt.get("lang_to_idx", {"<unk>": 0})
    model = SafariLitePlus(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(args.get("hidden_dim", 384)),
        language_emb_dim=int(args.get("language_emb_dim", 32)),
        num_languages=len(lang_to_idx),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def normalize_feature(path: Path, input_dim: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.load(path).astype(np.float32).reshape(-1)
    if x.shape[0] > input_dim:
        x = x[:input_dim]
    elif x.shape[0] < input_dim:
        x = np.pad(x, (0, input_dim - x.shape[0]))
    return (x - mean) / std


def predict_one(
    model: SafariLitePlus,
    feature: np.ndarray,
    language_idx: int,
    device: torch.device,
) -> float:
    with torch.no_grad():
        x = torch.from_numpy(feature).unsqueeze(0).to(device)
        l = torch.tensor([language_idx], dtype=torch.long, device=device)
        logits = model(x, x, l)
        return float(torch.sigmoid(logits).item())


def run_single(
    model: SafariLitePlus,
    ckpt: dict,
    feature_path: Path,
    language: str,
    device: torch.device,
) -> None:
    input_dim = int(ckpt["input_dim"])
    threshold = float(ckpt.get("threshold", 0.5))
    lang_to_idx: Dict[str, int] = ckpt.get("lang_to_idx", {"<unk>": 0})

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    lang_key = language.strip().lower() or "<unk>"
    lang_idx = lang_to_idx.get(lang_key, lang_to_idx.get("<unk>", 0))
    x = normalize_feature(feature_path, input_dim, mean, std)
    p_fake = predict_one(model, x, lang_idx, device)
    pred = "fake" if p_fake >= threshold else "real"

    out = {
        "feature_path": str(feature_path.resolve()),
        "language": lang_key,
        "fake_probability": p_fake,
        "threshold": threshold,
        "predicted_label": pred,
    }
    print(json.dumps(out, indent=2))


def run_batch(
    model: SafariLitePlus,
    ckpt: dict,
    input_csv: Path,
    feature_root: Path | None,
    output_csv: Path,
    device: torch.device,
) -> None:
    input_dim = int(ckpt["input_dim"])
    threshold = float(ckpt.get("threshold", 0.5))
    lang_to_idx: Dict[str, int] = ckpt.get("lang_to_idx", {"<unk>": 0})
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    if feature_root is None:
        feature_root = input_csv.parent

    outputs: List[dict] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "feature_path" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must include 'feature_path' column.")
        for row in reader:
            rel = row["feature_path"].strip()
            path = (feature_root / rel).resolve()
            if not path.exists():
                outputs.append(
                    {
                        **row,
                        "fake_probability": "",
                        "predicted_label": "missing_file",
                    }
                )
                continue

            lang_key = row.get("language", "<unk>").strip().lower() or "<unk>"
            lang_idx = lang_to_idx.get(lang_key, lang_to_idx.get("<unk>", 0))
            x = normalize_feature(path, input_dim, mean, std)
            p_fake = predict_one(model, x, lang_idx, device)
            pred = "fake" if p_fake >= threshold else "real"
            outputs.append(
                {
                    **row,
                    "fake_probability": f"{p_fake:.6f}",
                    "predicted_label": pred,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(outputs[0].keys()) if outputs else []
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outputs)

    print(f"Wrote: {output_csv.resolve()}")
    print(f"Rows: {len(outputs)} | Threshold: {threshold:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference script for SafariLite+ checkpoints."
    )
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--feature-path", type=str, default=None)
    parser.add_argument("--language", type=str, default="<unk>")

    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--feature-root", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default="./artifacts/predictions_safari.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.feature_path is None and args.input_csv is None:
        raise ValueError("Provide --feature-path for single mode or --input-csv for batch mode.")

    device = device_from_arg(args.device)
    model_path = Path(args.model_path).expanduser().resolve()
    ckpt = load_checkpoint(model_path, device)
    model = build_model(ckpt, device)

    if args.feature_path:
        run_single(
            model=model,
            ckpt=ckpt,
            feature_path=Path(args.feature_path).expanduser().resolve(),
            language=args.language,
            device=device,
        )
    else:
        run_batch(
            model=model,
            ckpt=ckpt,
            input_csv=Path(args.input_csv).expanduser().resolve(),
            feature_root=Path(args.feature_root).expanduser().resolve()
            if args.feature_root
            else None,
            output_csv=Path(args.output_csv).expanduser().resolve(),
            device=device,
        )


if __name__ == "__main__":
    main()
