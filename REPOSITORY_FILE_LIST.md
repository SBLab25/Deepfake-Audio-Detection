# Repository File List

This file documents what should be committed to Git and what should stay outside the repository.

## Keep In The Repository

### Root

| Path | Reason |
|---|---|
| `README.md` | Main project explanation and usage |
| `REPOSITORY_FILE_LIST.md` | Inclusion and exclusion policy |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Prevents large or useless files from being committed |

### Source Code

Keep all files under:

```text
src/
```

Included scripts:

| File | Purpose |
|---|---|
| `src/safari_wavlm_ensemble_train.py` | Final model trainer |
| `src/predict_safari_wavlm_ensemble.py` | Final model inference |
| `src/run_multiseed_ensemble.py` | Five-seed training and ensemble runner |
| `src/safari_lite_train.py` | SafariLite+ baseline |
| `src/predict_safari_lite.py` | SafariLite+ inference |
| `src/aasist_like_train.py` | AASIST-inspired graph baseline |
| `src/train_detector.py` | Basic MLP baseline |
| `src/predict_detector.py` | Basic MLP inference |
| `src/build_raw_audio_splits.py` | Initial raw split builder |
| `src/build_raw_strict_splits.py` | Deduplicated group-aware split builder |
| `src/build_raw_challenge_splits.py` | Held-out-generator challenge split builder |
| `src/extract_wavlm_embeddings.py` | WavLM extraction for CSV rows |
| `src/extract_wavlm_for_splits.py` | WavLM extraction wrapper for train/val/test |
| `src/build_bilingual_180k_from_hf.py` | Future large dataset builder |
| `src/run_hparam_sweep.py` | Hyperparameter sweep runner |
| `src/generate_publication_graphs.py` | Publication graph generation |

### Configs

Keep:

```text
configs/safari_wavlm_auc96.yaml
```

This is a lightweight example sweep configuration updated for the `src/` layout and RawCh split paths.

### Dataset Metadata

Keep split CSVs only:

```text
data/splits/rawch/train.csv
data/splits/rawch/val.csv
data/splits/rawch/test.csv
data/splits/rawstrict/train.csv
data/splits/rawstrict/val.csv
data/splits/rawstrict/test.csv
```

These files are small metadata files. They do not contain audio or embeddings.

### Documentation

Keep:

```text
docs/PROJECT_STUDY_GUIDE.md
docs/AUDIO_DEEPFAKE_RESEARCH_REPORT.md
docs/MULTISEED_ENSEMBLE_REPORT.md
docs/dataset_summaries/*.json
```

These explain the project, methodology, results, and split construction.

### Notebooks

Keep only final/high-value notebooks:

```text
notebooks/10_multiseed_ensemble_stepwise_challenge.ipynb
notebooks/11_publication_ready_challenge_improvements.ipynb
notebooks/12.Final_SafariWavLm_Multiseed_Ensemble.ipynb
```

Older scratch notebooks are intentionally not included.

### Lightweight Final Results

Keep:

```text
artifacts/publication_challenge_improved/publication_challenge_summary.json
artifacts/publication_challenge_improved/observation_metrics_compiled.json
artifacts/publication_challenge_improved/graphs/
```

These are small enough and useful for review.

## Do Not Commit

Never commit:

| Type | Examples |
|---|---|
| Virtual environments | `.venv/`, `venv/` |
| Python caches | `__pycache__/`, `*.pyc` |
| Raw audio | `*.wav`, `*.mp3`, `*.flac`, `Raw Dataset/` |
| Feature embeddings | `*.npy`, `WavLM_embeddings_unified/` |
| Model checkpoints | `*.pt`, `*.pth`, `models/*.pt` |
| Archives | `*.zip`, `*.rar`, `*.7z` |
| Large generated runs | `artifacts/runs/`, seed folders, prediction dumps |
| Temporary folders | `temporary_to_delete_*` |
| Old balanced datasets | `Balanced_train/`, `Balanced_val/`, `Balanced_test/` |

## Recommended Local-Only Folders

Use these paths for large files after cloning:

```text
data/features/WavLM_embeddings_unified/
models/
artifacts/runs/
```

These paths are ignored by Git.

