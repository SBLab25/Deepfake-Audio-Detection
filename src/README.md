# Source Code

This folder contains all executable project scripts.

## Final Model

| File | Purpose |
|---|---|
| `safari_wavlm_ensemble_train.py` | Trains the final Safari plus WavLM model |
| `predict_safari_wavlm_ensemble.py` | Runs single-file or batch inference with the final model |
| `run_multiseed_ensemble.py` | Trains multiple seeds, predicts validation/test, and averages probabilities |

## Baselines And Ablations

| File | Purpose |
|---|---|
| `train_detector.py` | Basic MLP baseline |
| `predict_detector.py` | Inference for the MLP baseline |
| `safari_lite_train.py` | SafariLite+ baseline |
| `predict_safari_lite.py` | Inference for SafariLite+ |
| `aasist_like_train.py` | AASIST-inspired graph-attention baseline |

## Dataset And Feature Utilities

| File | Purpose |
|---|---|
| `build_raw_audio_splits.py` | Builds initial splits from raw audio folders |
| `build_raw_strict_splits.py` | Deduplicates and creates group-aware strict splits |
| `build_raw_challenge_splits.py` | Creates the held-out-generator RawCh protocol |
| `extract_wavlm_embeddings.py` | Extracts WavLM embeddings for one CSV |
| `extract_wavlm_for_splits.py` | Extracts WavLM embeddings for train/val/test |
| `build_bilingual_180k_from_hf.py` | Future larger English-Hindi dataset builder |

## Experiment Utilities

| File | Purpose |
|---|---|
| `run_hparam_sweep.py` | Runs config-driven hyperparameter sweeps |
| `generate_publication_graphs.py` | Regenerates final result figures |

