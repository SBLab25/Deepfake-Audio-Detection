# English-Hindi Audio Deepfake Detection

This repository contains a cleaned, source-first version of the English-Hindi audio deepfake detection project.

The final system detects whether an audio sample is real or fake using WavLM speech embeddings, a Safari-style fusion branch, a direct WavLM MLP branch, language conditioning, validation threshold calibration, and five-seed probability ensembling.

## Final Result

Final protocol:

```text
RawCh held-out-generator challenge
```

Final model:

```text
SafariWavLMEnsemble
```

Final test performance:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9668 |
| Accuracy | 0.9048 |
| Balanced accuracy | 0.9042 |
| Precision | 0.9403 |
| Recall | 0.8618 |
| F1 | 0.8993 |
| Threshold | 0.198 |

The test split contains English fake samples from OpenAI and xTTS, which were not used during training.

## Repository Layout

```text
.
├── README.md
├── REPOSITORY_FILE_LIST.md
├── requirements.txt
├── configs/
│   └── safari_wavlm_auc96.yaml
├── src/
│   ├── safari_wavlm_ensemble_train.py
│   ├── predict_safari_wavlm_ensemble.py
│   ├── run_multiseed_ensemble.py
│   ├── safari_lite_train.py
│   ├── aasist_like_train.py
│   ├── train_detector.py
│   ├── build_raw_strict_splits.py
│   ├── build_raw_challenge_splits.py
│   ├── extract_wavlm_embeddings.py
│   └── ...
├── data/
│   └── splits/
│       ├── rawch/
│       └── rawstrict/
├── docs/
│   ├── PROJECT_STUDY_GUIDE.md
│   ├── AUDIO_DEEPFAKE_RESEARCH_REPORT.md
│   ├── MULTISEED_ENSEMBLE_REPORT.md
│   └── dataset_summaries/
├── notebooks/
│   ├── 10_multiseed_ensemble_stepwise_challenge.ipynb
│   ├── 11_publication_ready_challenge_improvements.ipynb
│   └── 12.Final_SafariWavLm_Multiseed_Ensemble.ipynb
├── artifacts/
│   └── publication_challenge_improved/
└── models/
```

## What Is Included

This repository intentionally includes only lightweight, useful files:

- source code
- final notebooks
- requirements
- configuration files
- split CSV metadata
- dataset summary JSON files
- final result summaries
- final evaluation graphs
- project study documentation

## What Is Not Included

Large files are intentionally excluded from Git:

- raw audio files
- `.npy` WavLM embeddings
- `.pt` model checkpoints
- zip or rar archives
- virtual environments
- cache folders
- generated prediction dumps
- temporary folders

Place large local files in these ignored locations:

```text
data/features/WavLM_embeddings_unified/
models/
artifacts/runs/
```

## Setup

Use Python 3.10 to 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Needed To Reproduce Training

The repo includes split CSVs, but not the large WavLM `.npy` feature files.

Expected feature location:

```text
data/features/WavLM_embeddings_unified/
```

The files under this folder should match the relative paths in:

```text
data/splits/rawch/train.csv
data/splits/rawch/val.csv
data/splits/rawch/test.csv
```

Example feature path from the CSV:

```text
hi/fake/audio_1285.npy
```

Expected local file:

```text
data/features/WavLM_embeddings_unified/hi/fake/audio_1285.npy
```

## Train The Final Model

Single-seed training:

```powershell
python src/safari_wavlm_ensemble_train.py `
  --project-root . `
  --train-csv .\data\splits\rawch\train.csv `
  --val-csv .\data\splits\rawch\val.csv `
  --test-csv .\data\splits\rawch\test.csv `
  --train-feature-dir .\data\features\WavLM_embeddings_unified `
  --val-feature-dir .\data\features\WavLM_embeddings_unified `
  --test-feature-dir .\data\features\WavLM_embeddings_unified `
  --wavlm-train-feature-dir .\data\features\WavLM_embeddings_unified `
  --wavlm-val-feature-dir .\data\features\WavLM_embeddings_unified `
  --wavlm-test-feature-dir .\data\features\WavLM_embeddings_unified `
  --out-dir .\artifacts\runs\single_seed `
  --epochs 60 `
  --batch-size 512 `
  --num-workers 4 `
  --device cuda `
  --amp `
  --seed 42
```

Five-seed ensemble training:

```powershell
python src/run_multiseed_ensemble.py `
  --project-root . `
  --out-dir .\artifacts\runs\rawch_multiseed `
  --train-csv .\data\splits\rawch\train.csv `
  --val-csv .\data\splits\rawch\val.csv `
  --test-csv .\data\splits\rawch\test.csv `
  --train-feature-dir .\data\features\WavLM_embeddings_unified `
  --val-feature-dir .\data\features\WavLM_embeddings_unified `
  --test-feature-dir .\data\features\WavLM_embeddings_unified `
  --wavlm-train-feature-dir .\data\features\WavLM_embeddings_unified `
  --wavlm-val-feature-dir .\data\features\WavLM_embeddings_unified `
  --wavlm-test-feature-dir .\data\features\WavLM_embeddings_unified `
  --seeds 17 42 77 123 202 `
  --epochs 60 `
  --batch-size 512 `
  --num-workers 4 `
  --device cuda `
  --amp
```

## Run Inference

The final checkpoint is not committed. Put your checkpoint in:

```text
models/best_safari_wavlm_ensemble.pt
```

Batch prediction:

```powershell
python src/predict_safari_wavlm_ensemble.py `
  --model-path .\models\best_safari_wavlm_ensemble.pt `
  --input-csv .\data\splits\rawch\test.csv `
  --base-feature-root .\data\features\WavLM_embeddings_unified `
  --wavlm-feature-root .\data\features\WavLM_embeddings_unified `
  --output-csv .\artifacts\runs\test_predictions.csv `
  --device cuda
```

Single feature prediction:

```powershell
python src/predict_safari_wavlm_ensemble.py `
  --model-path .\models\best_safari_wavlm_ensemble.pt `
  --base-feature-path .\data\features\WavLM_embeddings_unified\hi\fake\audio_1285.npy `
  --language hi `
  --device cuda
```

## Architecture Summary

The final model has three branches:

| Branch | Role |
|---|---|
| Safari branch | Creates two views of the embedding, fuses them with AFUM-style gated fusion, and reasons over tokens with a Transformer |
| WavLM branch | Direct MLP over the WavLM feature vector |
| Language branch | Learned language embedding for English, Hindi, and unknown |

The final classifier concatenates:

```text
Safari representation + WavLM representation + language embedding
```

and outputs:

```text
fake probability
```

## Previous Models

Earlier models are preserved in `src/` for reproducibility:

| Model | Script | Why not final |
|---|---|---|
| Basic MLP detector | `train_detector.py` | Too simple, lower performance |
| SafariLite+ | `safari_lite_train.py` | Good baseline, but Hindi remained weak on balanced features |
| AASIST-like graph model | `aasist_like_train.py` | Useful ablation, did not outperform Safari plus WavLM |
| Safari plus WavLM single seed | `safari_wavlm_ensemble_train.py` | Stronger, but single seed and old balanced protocol were not enough |
| Final five-seed RawCh ensemble | `run_multiseed_ensemble.py` | Selected final system |

## Important Documentation

Start here:

```text
docs/PROJECT_STUDY_GUIDE.md
```

Then read:

```text
docs/AUDIO_DEEPFAKE_RESEARCH_REPORT.md
docs/MULTISEED_ENSEMBLE_REPORT.md
REPOSITORY_FILE_LIST.md
```

