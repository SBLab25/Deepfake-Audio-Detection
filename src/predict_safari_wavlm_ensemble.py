from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from safari_wavlm_ensemble_train import SafariWavLMEnsemble


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_ckpt(path: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def build_model(ckpt: dict, device: torch.device) -> SafariWavLMEnsemble:
    args = ckpt.get("args", {})
    lang_to_idx = ckpt.get("lang_to_idx", {"<unk>": 0})
    model = SafariWavLMEnsemble(
        base_dim=int(ckpt["base_dim"]),
        wavlm_dim=int(ckpt["wavlm_dim"]),
        hidden_dim=int(args.get("hidden_dim", 384)),
        language_emb_dim=int(args.get("language_emb_dim", 32)),
        num_languages=len(lang_to_idx),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_vec(path: Path, dim: int) -> np.ndarray:
    x = np.load(path).astype(np.float32).reshape(-1)
    if x.shape[0] > dim:
        x = x[:dim]
    elif x.shape[0] < dim:
        x = np.pad(x, (0, dim - x.shape[0]))
    return x


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def predict_prob(
    model: SafariWavLMEnsemble,
    xb: np.ndarray,
    xw: np.ndarray,
    lang_idx: int,
    device: torch.device,
) -> float:
    with torch.no_grad():
        tb = torch.from_numpy(xb).unsqueeze(0).to(device)
        tw = torch.from_numpy(xw).unsqueeze(0).to(device)
        tl = torch.tensor([lang_idx], dtype=torch.long, device=device)
        logits = model(tb, tb, tw, tl)
        return float(torch.sigmoid(logits).item())


def run_batch(
    model: SafariWavLMEnsemble,
    ckpt: dict,
    input_csv: Path,
    base_feature_root: Path | None,
    wavlm_feature_root: Path | None,
    output_csv: Path,
    device: torch.device,
) -> None:
    if base_feature_root is None:
        base_feature_root = input_csv.parent

    base_dim = int(ckpt["base_dim"])
    wavlm_dim = int(ckpt["wavlm_dim"])
    threshold = float(ckpt.get("threshold", 0.5))
    lang_to_idx: Dict[str, int] = ckpt.get("lang_to_idx", {"<unk>": 0})

    base_mean = np.asarray(ckpt["base_mean"], dtype=np.float32)
    base_std = np.asarray(ckpt["base_std"], dtype=np.float32)
    wavlm_mean = np.asarray(ckpt["wavlm_mean"], dtype=np.float32)
    wavlm_std = np.asarray(ckpt["wavlm_std"], dtype=np.float32)

    rows_out: List[dict] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "feature_path" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain 'feature_path' column.")

        for row in reader:
            rel = row["feature_path"].strip()
            lang_key = row.get("language", "<unk>").strip().lower() or "<unk>"
            lang_idx = lang_to_idx.get(lang_key, lang_to_idx.get("<unk>", 0))

            base_path = (base_feature_root / rel).resolve()
            wavlm_path = (
                (wavlm_feature_root / rel).resolve()
                if wavlm_feature_root is not None
                else base_path
            )
            if not wavlm_path.exists():
                wavlm_path = base_path

            if not base_path.exists():
                rows_out.append(
                    {
                        **row,
                        "fake_probability": "",
                        "predicted_label": "missing_file",
                    }
                )
                continue

            xb = normalize(load_vec(base_path, base_dim), base_mean, base_std)
            xw = normalize(load_vec(wavlm_path, wavlm_dim), wavlm_mean, wavlm_std)

            p_fake = predict_prob(model, xb, xw, lang_idx, device)
            pred = "fake" if p_fake >= threshold else "real"
            rows_out.append(
                {
                    **row,
                    "fake_probability": f"{p_fake:.6f}",
                    "predicted_label": pred,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows_out[0].keys()) if rows_out else []
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote predictions: {output_csv.resolve()}")
    print(f"Rows: {len(rows_out)} | Threshold: {threshold:.3f}")


def run_single(
    model: SafariWavLMEnsemble,
    ckpt: dict,
    base_feature_path: Path,
    wavlm_feature_path: Path | None,
    language: str,
    device: torch.device,
) -> None:
    base_dim = int(ckpt["base_dim"])
    wavlm_dim = int(ckpt["wavlm_dim"])
    threshold = float(ckpt.get("threshold", 0.5))
    lang_to_idx: Dict[str, int] = ckpt.get("lang_to_idx", {"<unk>": 0})
    base_mean = np.asarray(ckpt["base_mean"], dtype=np.float32)
    base_std = np.asarray(ckpt["base_std"], dtype=np.float32)
    wavlm_mean = np.asarray(ckpt["wavlm_mean"], dtype=np.float32)
    wavlm_std = np.asarray(ckpt["wavlm_std"], dtype=np.float32)

    wavlm_path = wavlm_feature_path if wavlm_feature_path is not None else base_feature_path
    if not wavlm_path.exists():
        wavlm_path = base_feature_path

    lang_key = language.strip().lower() or "<unk>"
    lang_idx = lang_to_idx.get(lang_key, lang_to_idx.get("<unk>", 0))

    xb = normalize(load_vec(base_feature_path, base_dim), base_mean, base_std)
    xw = normalize(load_vec(wavlm_path, wavlm_dim), wavlm_mean, wavlm_std)
    p_fake = predict_prob(model, xb, xw, lang_idx, device)
    pred = "fake" if p_fake >= threshold else "real"

    out = {
        "base_feature_path": str(base_feature_path.resolve()),
        "wavlm_feature_path": str(wavlm_path.resolve()),
        "language": lang_key,
        "fake_probability": p_fake,
        "threshold": threshold,
        "predicted_label": pred,
    }
    print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference for Safari+WavLM ensemble checkpoints.")
    p.add_argument("--model-path", required=True, type=str)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--input-csv", type=str, default=None)
    p.add_argument("--base-feature-root", type=str, default=None)
    p.add_argument("--wavlm-feature-root", type=str, default=None)
    p.add_argument("--output-csv", type=str, default="./artifacts/predictions_safari_wavlm.csv")

    p.add_argument("--base-feature-path", type=str, default=None)
    p.add_argument("--wavlm-feature-path", type=str, default=None)
    p.add_argument("--language", type=str, default="<unk>")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.input_csv is None and args.base_feature_path is None:
        raise ValueError(
            "Provide --input-csv for batch mode or --base-feature-path for single mode."
        )

    device = device_from_arg(args.device)
    ckpt = load_ckpt(Path(args.model_path).expanduser().resolve(), device)
    model = build_model(ckpt, device)

    if args.input_csv:
        run_batch(
            model=model,
            ckpt=ckpt,
            input_csv=Path(args.input_csv).expanduser().resolve(),
            base_feature_root=Path(args.base_feature_root).expanduser().resolve()
            if args.base_feature_root
            else None,
            wavlm_feature_root=Path(args.wavlm_feature_root).expanduser().resolve()
            if args.wavlm_feature_root
            else None,
            output_csv=Path(args.output_csv).expanduser().resolve(),
            device=device,
        )
    else:
        run_single(
            model=model,
            ckpt=ckpt,
            base_feature_path=Path(args.base_feature_path).expanduser().resolve(),
            wavlm_feature_path=Path(args.wavlm_feature_path).expanduser().resolve()
            if args.wavlm_feature_path
            else None,
            language=args.language,
            device=device,
        )


if __name__ == "__main__":
    main()
