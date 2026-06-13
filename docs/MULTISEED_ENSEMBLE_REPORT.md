# Multi-Seed Safari+WavLM Ensemble: Methodology and Final Analysis

## 1) Objective
Build a stronger deepfake audio detector by training multiple Safari+WavLM ensemble models with different random seeds and averaging their probabilities.

## 2) Dataset
Balanced precomputed embedding splits were used:
- Train: 113,400 samples
- Validation: 12,600 samples
- Test: 14,000 samples
- Labels: `real` / `fake`
- Languages: `en`, `hi`
- Input feature size: 768

## 3) Model Family
Base model: `SafariWavLMEnsemble`
- Safari branch: AFUM-style fusion + Transformer reasoning
- WavLM branch: MLP encoder
- Language embedding branch: learned language embedding
- Fusion head: concatenated branch features -> classification head

Current run note:
- No separate WavLM feature directories were provided, so the WavLM branch used base features as fallback.

## 4) Training Methodology
Implementation path:
- Stepwise notebook: `notebooks/07_multiseed_ensemble_stepwise.ipynb`
- Trainer: `safari_wavlm_ensemble_train.py`
- Predictor: `predict_safari_wavlm_ensemble.py`

Per-seed workflow:
1. Train one seed model.
2. Save best checkpoint by validation objective.
3. Run inference on validation and test splits for that seed.
4. Store `val_predictions.csv` and `test_predictions.csv` under `artifacts/multiseed_stepwise/seed_<seed>/`.

Ensemble aggregation workflow:
1. Load prediction CSVs from all completed seeds.
2. Align rows by `feature_path`.
3. Average fake probabilities across seeds.
4. Select threshold on validation set by best F1.
5. Apply that threshold to test set.
6. Save:
   - `ensemble_predictions/val_ensemble_predictions.csv`
   - `ensemble_predictions/test_ensemble_predictions.csv`
   - `multiseed_ensemble_summary.json`

## 5) Seed Set and Run Summary
Requested seeds:
- 17, 42, 77, 123, 202

Completed seed metrics (from `metrics_safari_wavlm_ensemble.json`):

| Seed | Val AUC | Test AUC | Val F1 | Test F1 | Best Epoch |
|---|---:|---:|---:|---:|---:|
| 17  | 0.910554 | 0.909243 | 0.834553 | 0.830698 | 28 |
| 42  | 0.910972 | 0.908285 | 0.835718 | 0.832004 | 24 |
| 77  | 0.911894 | 0.908240 | 0.834651 | 0.831864 | 31 |
| 123 | 0.911938 | 0.909601 | 0.836746 | 0.831596 | 28 |
| 202 | 0.911564 | 0.908562 | 0.834106 | 0.831514 | 26 |

Single-seed test AUC statistics:
- Mean: **0.908786**
- Std: **0.000543**
- Best single seed: **0.909601**

## 6) Final Ensemble Result
From `artifacts/multiseed_stepwise/multiseed_ensemble_summary.json`:

- Validation ROC-AUC: **0.914511**
- Validation F1: **0.837507**
- Validation Accuracy: **0.812540**
- Test ROC-AUC: **0.912002**
- Test F1: **0.834984**
- Test Accuracy: **0.809571**
- Selected threshold: **0.395**

## 7) Conclusion
The multiseed probability-averaging ensemble outperformed all individual models in this run.

Key gains:
- Better than best single seed on test AUC:
  - `0.912002 - 0.909601 = +0.002401`
- Better than prior single-run method2 baseline (`~0.909603` test AUC):
  - `+0.002398` absolute AUC improvement

This confirms the expected stabilization and uplift from seed ensembling.

## 8) Reproducibility Artifacts
- Ensemble summary:
  - `artifacts/multiseed_stepwise/multiseed_ensemble_summary.json`
- Per-seed checkpoints + metrics:
  - `artifacts/multiseed_stepwise/seed_*/`
- Ensemble prediction CSVs:
  - `artifacts/multiseed_stepwise/ensemble_predictions/`
