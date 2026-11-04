# Model results

The submitted model, its training-CV evidence, and the two official validation runs. Numbers
come from the committed reports named in each section. Every metric is labelled by split.
Test labels are held by the organisers; no test score appears here.

## 1. The submitted model

A fixed blend (decision D38, rules written in `configs/budget_v3.toml` before their scores):

1. **Transformer ensemble** (D36): `small-v2-c1` with three seeds, a decoder-only transformer with
   2 layers, width 128, 4 heads, rotary positions, continuous age features, a `last_mean` readout,
   and a 40-logit head, about 738,000 parameters per model. Inputs are the pre-anchor event tokens
   of the CSV tables (without the simulator scores DALY, QALY, and QOLS), genotype tokens, and one
   brain-MRI token from a 3-D CNN. Each seed runs its own self-pretraining (next-event prediction
   on fit patients' pre-anchor histories, target diagnoses weighted 5) and then full fine-tuning
   with dropout 0.3 and one extra earlier-anchor example per fit patient per epoch. Fifteen fold
   models (3 seeds x 5 folds) are averaged.
2. **LightGBM on code counts** (E11 setting: 4 leaves, 100 trees, one classifier per target on
   at-risk patients), fitted on all training patients.
3. **LightGBM on counts plus the latest value of each numeric observation**, same setting.

The three probabilities are combined by the mean of their logits, a per-target Platt map fitted on
training out-of-fold predictions scales the result, and prior-condition cells are set to 0.
Frozen artifacts: `reports/final/manifest.json`, `configs/final.toml`, `models/lightgbm_final/`,
`reports/calibration/stack/platt_weights.json`.

## 2. Training-split cross-validation (development evidence)

Five folds of the 2,791 training patients. All model choices were made here. Held-out fold
labels chose each run's best epoch and each round's configuration, so these scores carry
selection effects (model plan section 5.1): 28 transformer configurations (`models/transformer/`,
counting seeds and pretraining parents) and 3 stackers were scored on the same folds.

| Stage | Mean fold AUROC | Mean fold mAP | Source |
|---|---|---|---|
| S/H/F on the CSV-only small model (`small-v1`): F selected | 0.760 / 0.744 / 0.742 | 0.186 / 0.163 / 0.161 | `reports/transformer/small-v1/` |
| Multimodal `small-v1-mm` (D34) | 0.759 | 0.197 | `reports/transformer/small-v1-mm/` |
| Round 1 (D35): dropout 0.3 | 0.766 | 0.203 | `reports/transformer/experiments.csv` |
| Clock codes removed, A1 reference (D36) | 0.752 | 0.188 | `reports/transformer/round2/` |
| Round 2 C1 (seed 1 / 2 / 3) | 0.760 / 0.758 / 0.760 | 0.198 / 0.192 / 0.189 | same |
| Round 3 A1, 4 x 192 (fold-0 screen, against C1 0.772) | 0.758 | 0.211 | `reports/transformer/round3/` |

Pooled out of fold (one prediction per patient from models that did not fit on them), with the
nested-fold stacking of round 3 (`reports/stacking/clean/`):

| Model | Pooled AUROC | Pooled mAP | Mean fold AUROC | Mean fold mAP |
|---|---|---|---|---|
| Transformer ensemble (T) | 0.761 | 0.181 | 0.765 | 0.204 |
| LightGBM counts (L1) | 0.764 | 0.188 | | |
| LightGBM counts + values (L2) | 0.761 | 0.202 | | |
| Oracle: better of T and LightGBM per target | 0.778 | | | |
| **Blend, mean of logits (submitted)** | **0.774** | **0.212** | **0.780** | **0.236** |
| Blend, per-target logistic stacker | 0.769 | 0.216 | 0.781 | 0.244 |

Paired patient BCa 95% (1,000 resamples), pooled out of fold: blend minus T, AUROC +0.014
[+0.009, +0.019] and mAP +0.032 [+0.025, +0.040]; blend minus L1, AUROC +0.011 [+0.007, +0.014]
and mAP +0.024 [+0.018, +0.031]. Platt scaling (`reports/calibration/stack/`): macro Brier 0.0366 to 0.0365,
macro expected calibration error 0.013 to 0.010, lower Brier on 4 of 5 folds, per-fold AUROC
unchanged. Pooled values are lower than mean fold values because pooling mixes the folds'
probability scales. The blend rows are before Platt scaling; after the per-fold maps the pooled
OOF scores are 0.767 AUROC and 0.206 mAP while no per-fold score changes
(`reports/calibration/stack/calibration.json`).

## 3. Official validation (two runs, both recorded)

365 patients, 685 positive labels, common prior mask, 1,000 patient BCa resamples.
`reports/val/` holds attempt 2 (bundle `093a88a1f5b5`, 3 November 2026, 22:28); attempt 1 (bundle
`168b78f5b5d8`, 20:46 the same day) is under `reports/val/attempts/`. Attempt 1 scored the
transformer ensemble alone before the blend existed. Its numbers were known when round 3 was
written; round 3 used training CV only, and the bundle refuses a run without a documented reason.
Attempt 1 had shown LightGBM ahead on val mAP, and round 3 added LightGBM to the submission, so
attempt 2 is a second look at the same patients rather than an independent check. Before the
bundle existed, the step-2 baselines had scored val once (28 and 29 October,
`reports/baseline/metrics.json`): three reads of the val labels in all. The comparison models'
own intervals below were computed after the bundle from its saved prediction files
(`reports/val/post_hoc_single_model_intervals.json`).

| Model | Macro AUROC | mAP | Brier | At-risk AUROC |
|---|---|---|---|---|
| **Blend (submitted), attempt 2** | **0.767** [0.746, 0.788] | **0.258** [0.234, 0.268] | 0.0365 | 0.739 |
| Transformer ensemble (attempts 1 and 2, identical) | 0.751 [0.730, 0.773] | 0.209 [0.192, 0.211] | 0.0381 | 0.722 |
| LightGBM counts | 0.762 [0.740, 0.781] | 0.239 [0.209, 0.249] | 0.0370 | 0.733 |
| LightGBM counts + latest values | 0.758 [0.740, 0.780] | 0.257 [0.228, 0.267] | 0.0367 | 0.729 |
| Logistic regression, code counts + age and sex | 0.764 [0.742, 0.782] | 0.222 [0.195, 0.229] | 0.0384 | 0.733 |
| Age-sex incidence, at-risk patients | 0.727 [0.705, 0.751] | 0.141 [0.125, 0.146] | 0.0390 | 0.696 |

Paired differences, blend minus comparison (same patients, same resamples):

| Comparison | AUROC | mAP | Brier |
|---|---|---|---|
| transformer ensemble | +0.016 [+0.002, +0.030] | +0.049 [+0.033, +0.072] | -0.0016 [-0.0022, -0.0011] |
| LightGBM counts | +0.005 [-0.005, +0.016] | +0.019 [+0.002, +0.040] | -0.0005 [-0.0008, -0.0002] |
| LightGBM counts + values | +0.009 [-0.004, +0.019] | +0.001 [-0.017, +0.019] | -0.0002 [-0.0006, +0.0002] |
| logistic regression | +0.003 [-0.010, +0.020] | +0.036 [+0.011, +0.062] | -0.0019 [-0.0027, -0.0012] |
| at-risk incidence | +0.040 [+0.016, +0.062] | +0.117 [+0.078, +0.139] | -0.0025 [-0.0034, -0.0017] |

Val agrees with CV. The blend beats its transformer part on every metric with intervals above
zero. Its AUROC lead over LightGBM counts is inside the interval on 365 patients; its mAP and
Brier leads are not. The at-risk column scores each target only among patients who did not already
have it; the "already has it" information is worth about 0.03 AUROC to every model.

### Subgroups (descriptive, no intervals)

| Group | Patients | Blend AUROC / mAP | Transformer | LightGBM counts |
|---|---|---|---|---|
| Anchor before 2015 (mostly patients who later died) | 155 | 0.748 / 0.281 | 0.722 / 0.239 | 0.752 / 0.260 |
| Anchor from 2015 | 210 | 0.746 / 0.183 | 0.730 / 0.129 | 0.739 / 0.192 |
| Genotype file or pre-anchor scan | 21 | 0.707 / 0.176 | 0.719 / 0.171 | 0.711 / 0.170 |

### Per condition (`reports/val/val_per_target.csv`)

The blend is above the transformer on 29 of 40 targets and above LightGBM counts on 17. Most val
targets have 5 to 20 positives, so single-target differences are noisy.

* **Near-certain** (AUROC above 0.95 for the blend; six of the seven for the transformer and
  LightGBM counts too): prostate neoplasm, carcinoma in situ and
  metastasis of the prostate, normal pregnancy, pneumonia, obesity, and the pregnancy
  complication code. Each follows a visible Synthea pathway (PSA tests, contraceptives,
  dementia care plans, BMI observations) that any history model reads.
* **Where the blend repairs the transformer**: the count-driven conditions where the transformer
  trailed LightGBM. Myocardial infarction 0.927 (transformer 0.827, LightGBM 0.904), osteoporosis
  0.883 (0.804, 0.895), recurrent rectal polyp 0.931 (0.860, 0.946), pathological fracture 0.911
  (0.847, 0.909), colon polyp 0.876 (0.812, 0.867). In Synthea their risk grows with the count of
  specific earlier events, which a count feature states directly and the transformer's
  1,024-token window partly drops.
* **Where the transformer's lead survives**: the paired brain targets, silent micro-haemorrhage and
  abnormal gait (blend 0.892, LightGBM 0.832; the two labels are identical for every patient),
  heart failure (0.753 against 0.707), and stroke (0.806 against 0.783).
* **Still below LightGBM**: the three lung-cancer codes (0.81 to 0.82 against 0.83 to 0.85).
* **Near chance** (below 0.6 for the blend; seven of the nine for the transformer and LightGBM
  counts, five for all six models scored): whiplash, lacerations of forearm and foot, sprains
  of wrist and ankle, streptococcal sore throat, clavicle fracture, sinusitis, and wrist
  fracture-subluxation. These are acute events that Synthea generates with little dependence on
  the prior record.

### Validation curves (`reports/val/val_curves.csv`, wandb group `val-bundle-093a88a1f5b5`)

All 209 saved epoch checkpoints of the 15 transformer members were scored on val after training
(identical in both attempts, since the members did not change). The epoch selected on each
member's held-out training fold scores a mean val AUROC of 0.743 (range 0.733 to 0.756) and mAP
0.200. Averaging the 15 members adds 0.008 AUROC; the blend adds 0.016 more. Val reaches its mean
peak around epochs 9 to 12 and declines only slightly after; the held-out fold chose an earlier
epoch than val would have in 13 of 15 members, by 1 to 5 epochs. Choosing epochs on val would
have raised the mean member AUROC by 0.008; that selection was not made.

## 4. What the evidence supports

* The pipeline is leak-free as far as the tests, the audits, and an independent read-only audit
  of every data boundary can show: train, val, and test histories follow the same cut, a future
  record changes no prediction, every fitted object (vocabulary, bins, folds, pretraining windows,
  LightGBM statistics, stack, Platt map) is fitted inside its fold, and official val was read by a
  bundle that records each attempt.
* Self-pretraining helps: F beat S on every fold in both the CSV-only and the multimodal model.
* The modalities add nothing measurable, and their file presence correlates with outcomes in the
  dataset: genotype files exist for 14.2% of train patients but 0.6% of val and test, and patients
  with a genotype or imaging file have more positive labels (2.11 and 2.35 against 1.83 per
  patient; `reports/dataset/modality_presence.json`). Disclosed; the tokens stay in.
* A 2-layer transformer on 2,791 synthetic patients ranks patients about as well as a
  gradient-boosted model on code counts and worse at the top of the ranking; a larger body (4 x
  192) overfits faster. The two models fail on different targets, so a fixed blend beats both. The LightGBM setting
  sits on the edge of its grid on all three axes (4 leaves, 100 trees, minimum child size 20;
  `grid_edge` in `reports/lightgbm_clean/cv_summary.json`) and the grid was not extended.
* Removing the clock codes cost about 0.014 AUROC in the transformer and 0.005 in the blend in
  CV, and probably costs test score too; the model should not learn how the anchor was built.
* A macro AUROC of 0.80 is out of reach here: nine targets are near chance for the blend (five
  for every model scored), and
  an oracle choice per target reaches 0.778 in CV.

## 5. Reproduction

```bash
make setup && ln -s /path/to/research_intern_coding_test data/raw
make dataset && make modalities && make folds && make tokenize CONFIG=configs/v2_base.toml
# training (about 20 minutes per fold and seed on an M-series Mac):
for f in 0 1 2 3 4; do make pretrain CONFIG=configs/experiments/v2_ptw_pre.toml FOLD=$f; \
  make finetune CONFIG=configs/experiments/v2_c1.toml FOLD=$f METHOD=F; done   # and the two seeds
make lightgbm && make lightgbm-clean && make stacking && make calibration && make lightgbm-final
make freeze && make val-score && make predict && make submission && make package
```

The public package (`ehrforecast.hf_package`) carries the 15 checkpoints, 5 tokenizers, 80
LightGBM boosters with their feature specs, and the Platt weights; `hf_package verify`
reproduces `predictions.csv` from it on CPU (`reports/final/package_verify_*.json`).
