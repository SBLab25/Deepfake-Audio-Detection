from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required for sweep configs. Install with: pip install PyYAML"
    ) from exc


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Sweep config must be a YAML mapping/object.")
    return cfg


def as_cli_args(params: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in params.items():
        key = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                args.append(key)
        else:
            args.extend([key, str(v)])
    return args


def expand_search_space(search: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(search.keys())
    values = [search[k] for k in keys]
    combos = []
    for val_tuple in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, val_tuple)})
    return combos


def read_auc(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return float("-inf")
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return float(metrics["val"]["metrics"]["roc_auc"])


def run_trial(
    python_exec: str,
    script_path: Path,
    base_args: Dict[str, Any],
    trial_params: Dict[str, Any],
    out_dir: Path,
) -> int:
    merged = dict(base_args)
    merged.update(trial_params)
    merged["out_dir"] = str(out_dir)

    cmd = [python_exec, str(script_path)] + as_cli_args(merged)
    print(f"Running trial in: {out_dir}")
    print("Command:", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, cwd=str(Path.cwd()))
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid/random sweep runner for AUC target.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(Path(args.config).expanduser().resolve())
    random.seed(args.seed)

    script = Path(cfg["script"]).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    base_args = dict(cfg.get("base_args", {}))
    search = dict(cfg.get("search", {}))
    target_auc = float(cfg.get("target_auc", 0.96))
    max_trials_cfg = int(cfg.get("max_trials", 50))
    max_trials = args.max_trials if args.max_trials is not None else max_trials_cfg
    metrics_filename = str(cfg.get("metrics_filename", "metrics_safari_wavlm_ensemble.json"))
    sweep_out_root = Path(cfg.get("sweep_out_root", "./artifacts/sweeps")).expanduser().resolve()
    sweep_out_root.mkdir(parents=True, exist_ok=True)

    candidates = expand_search_space(search)
    random.shuffle(candidates)
    candidates = candidates[: max_trials]

    print(f"Total candidate trials: {len(candidates)}")
    print(f"Target val AUC: {target_auc:.4f}")

    leaderboard: List[Dict[str, Any]] = []
    best_auc = float("-inf")
    best_trial = None

    for i, trial_params in enumerate(candidates, start=1):
        trial_dir = sweep_out_root / f"trial_{i:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        returncode = run_trial(
            python_exec=sys.executable,
            script_path=script,
            base_args=base_args,
            trial_params=trial_params,
            out_dir=trial_dir,
        )
        metrics_path = trial_dir / metrics_filename
        auc = read_auc(metrics_path) if returncode == 0 else float("-inf")

        item = {
            "trial_index": i,
            "returncode": returncode,
            "val_auc": auc,
            "params": trial_params,
            "metrics_path": str(metrics_path),
        }
        leaderboard.append(item)
        leaderboard.sort(key=lambda x: x["val_auc"], reverse=True)

        if auc > best_auc:
            best_auc = auc
            best_trial = item

        print(f"Trial {i} complete | val_auc={auc:.5f} | returncode={returncode}")

        if best_auc >= target_auc:
            print(
                f"Target reached: best val AUC {best_auc:.5f} >= {target_auc:.5f}. Stopping sweep."
            )
            break

    leaderboard_path = sweep_out_root / "leaderboard.json"
    with leaderboard_path.open("w", encoding="utf-8") as handle:
        json.dump(leaderboard, handle, indent=2)

    if best_trial is None:
        print("No successful trial completed.")
        return

    summary = {
        "target_auc": target_auc,
        "best_val_auc": best_auc,
        "best_trial": best_trial,
        "leaderboard_path": str(leaderboard_path),
    }
    summary_path = sweep_out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nSweep summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
