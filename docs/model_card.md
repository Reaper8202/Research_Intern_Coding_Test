---
license: mit
language: en
tags:
  - electronic-health-records
  - synthea
  - transformer
  - lightgbm
  - multi-label-classification
  - pytorch
library_name: pytorch
---

# ehr-forecast: five-year first-diagnosis forecasting on Synthea timelines

For each of 40 target conditions, the probability that a patient is diagnosed with it for the
first time in the five years after an anchor date, from the patient's pre-anchor events in the
Synthea Coherent CSV tables (plus genotype tokens and one brain-MRI token where those files
exist). The submitted probability is a fixed blend of three models, then per-target Platt
scaling:

1. Fifteen small decoder-only transformers (2 layers, width 128, about 738k parameters each):
   three random seeds, five cross-validation folds each, self-pretrained on pre-anchor event
   sequences and fully fine-tuned on the 40 labels. Their probabilities are averaged.
2. LightGBM on log code counts, age, and sex (one classifier per target, 4 leaves, 100 trees).
3. LightGBM on the same counts plus each numeric observation's latest pre-anchor value.

The three are combined by the mean of their logits; a per-target logistic map fitted on
training out-of-fold predictions scales the result; cells for conditions the patient already
has are set to 0. Everything was trained from random weights on the training split of the
research intern coding test dataset. No external data, pretrained weights, or clinical
coefficients were used. The blend and the scaling were chosen by rules written before their
scores (`configs/budget_v3.toml` in the repository). Code, plan, decision log, and results:
https://github.com/Reaper8202/Research_Intern_Coding_Test

## What is in the package

```
package.json                      target order, prior-mask rule, input schema, stack recipe, every file hash
transformer/small-v2-base/fold<k>/tokenizer.json       the five fold tokenizers
transformer/small-v2-c1/fold<k>/F/{best.pt,run.json}   seed 1, five folds
transformer/small-v2-c1-seed2/fold<k>/F/...            seed 2
transformer/small-v2-c1-seed3/fold<k>/F/...            seed 3
lightgbm/counts/{feature_spec.json,<code>.txt}         40 boosters and the feature spec
lightgbm/counts_values/...                             40 boosters and the feature spec
calibration/platt_weights.json                         slope and intercept per target
configs/                                               the frozen configurations
```

Each `best.pt` holds the model weights, its full configuration, and training provenance (fold,
tokenizer hash, pretraining parent hash). Each `feature_spec.json` holds the code vocabulary,
the value keys with their fill-in medians, and the column order, so the LightGBM features can
be rebuilt without the fitting code. The transformer layout is the one the project's prediction
command reads.

## How to reproduce the predictions

```bash
git clone https://github.com/Reaper8202/Research_Intern_Coding_Test ehr-forecast && cd ehr-forecast
make setup                                   # Python 3.11+, exact versions from requirements-lock.txt
ln -s /path/to/research_intern_coding_test data/raw
make dataset                                 # anchors, leak-free histories (about 20 s)
make modalities                              # genotype and DICOM cache (about 20 s)
hf download Reaper8202/ehr-forecast --local-dir models/hf
.venv/bin/python -m ehrforecast.hf_package verify --package models/hf \
    --dataset data/processed --predictions predictions.csv   # CPU; under a minute
```

`verify` checks every file hash in `package.json`, rebuilds the test probabilities from the
package alone (transformer members in-process, LightGBM boosters in their own process, then the
blend and the scaling), and compares them with the submitted `predictions.csv` (tolerance 1e-6).
The command reads no labels. It needs the processed event table because the tokenizer encodes
events, not raw CSV rows; `make dataset` builds that table from the supplied files with the same
anchor rules the test files follow.

## Results

Official validation split, 365 patients, scored by the frozen bundle (95% patient BCa bootstrap
intervals). Attempt 2 of 2: attempt 1 scored the transformer ensemble alone (0.751 / 0.209) before
the blend was adopted on training CV; both attempts are recorded in the repository.

| | Macro AUROC | mAP |
|---|---|---|
| Submitted blend | 0.767 [0.746, 0.788] | 0.258 [0.234, 0.268] |
| Transformer ensemble alone | 0.751 | 0.209 |
| LightGBM on code counts alone | 0.762 | 0.239 |
| Age-sex incidence table, at-risk patients (reference) | 0.727 | 0.141 |

Training-split nested five-fold CV, pooled out of fold and before Platt scaling: blend AUROC 0.774,
mAP 0.212; transformer ensemble 0.761 / 0.181; LightGBM counts 0.764 / 0.188. Test labels are held
by the organisers.
Full tables and per-condition results are in the repository under `docs/` and `reports/val/`.

## Limits

The data are synthetic. Anchors are five years before each patient's last encounter, so an early
anchor usually means the patient died, which any history model can read. Nine of the 40 targets
(acute injuries and infections) are near chance for the blend, five of them for every model tried. Having a genotype file or
a scan is itself correlated with outcomes in this dataset; the modality tokens showed no
measurable effect in CV. This is not a clinical tool.

## Data and licence

The model was trained only on the training split of the dataset released for the task, which is
derived from the Coherent Synthetic Data Set generated by Synthea (Copyright 2022 The MITRE
Corporation, MITRE Public Release Case Number 21-1917, Creative Commons Attribution 4.0
International). No external data or pretrained weights were used. The code and these weights are
released under the MIT licence (`LICENSE` in the repository).
