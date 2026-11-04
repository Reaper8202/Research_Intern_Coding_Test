# Model results

The submitted model, its training-CV evidence, and the one official validation run. Numbers
come from the committed reports named in each section. Every metric is labelled by split.
Test labels are held by the organisers; no test score appears here.

## 1. The submitted model

`small-v2-c1` with three seeds (decision D36): a decoder-only transformer with 2 layers, width
128, 4 heads, rotary positions, continuous age features, a `last_mean` readout, and a 40-logit
head; about 738,000 parameters per model. Inputs are the pre-anchor event tokens of the CSV
tables (without the simulator scores DALY, QALY, and QOLS), genotype tokens, and one brain-MRI
token from a 3-D CNN. Each seed runs its own self-pretraining (next-event prediction on fit
patients' pre-anchor histories, target diagnoses weighted 5) and then full fine-tuning (F) with
dropout 0.3 and one extra earlier-anchor example per fit patient per epoch. Fifteen fold models
(3 seeds x 5 folds) are averaged; prior-condition cells are set to 0.

Frozen artifacts: `reports/final/manifest.json` (294 hashes), `configs/final.toml`.

## 2. Training-split cross-validation (development evidence)

Five folds of the 2,791 training patients. All model choices were made here. Held-out fold
labels chose each run's best epoch and each round's configuration, so these scores carry
selection effects (model plan section 5.1).

| Stage | Mean fold AUROC | Mean fold mAP | Source |
|---|---|---|---|
| S/H/F on the CSV-only small model (`small-v1`): F selected | 0.760 / 0.744 / 0.742 | 0.186 / 0.163 / 0.161 | plan status, `reports/transformer/small-v1/` |
| Multimodal `small-v1-mm` (D34) | 0.759 | 0.197 | `reports/transformer/small-v1-mm/` |
| Tuning round (D35): dropout 0.3 | 0.766 | 0.203 | `reports/transformer/experiments.csv` |
| Clock codes removed, A1 reference (D36) | 0.752 | 0.188 | `reports/transformer/round2/` |
| Round 2 C1 (seed 1 / 2 / 3) | 0.760 / 0.758 / 0.760 | 0.198 / 0.192 / 0.189 | same |
| C1 three-seed ensemble, pooled out of fold | 0.761 | 0.181 | `reports/lightgbm_clean/ensemble_comparison.json` |
| LightGBM counts, pooled out of fold | 0.764 | 0.188 | `reports/lightgbm_clean/cv_summary.json` |

Pooled out-of-fold, paired patient BCa 95%: ensemble minus LightGBM counts, AUROC -0.003
[-0.010, +0.005], mAP -0.007 [-0.017, +0.002]; ensemble minus its first member, AUROC +0.006
[+0.001, +0.011]. Pooled values are lower than mean fold values because pooling mixes the folds'
probability scales.

## 3. Official validation (one run, bundle `168b78f5b5d8`, 3 November 2026)

`reports/val/`. 365 patients, 685 positive labels, common prior mask, 1,000 patient BCa resamples.
The step-2 baselines are the models fitted on train in October; LightGBM was refitted on all
training patients with its CV-selected setting inside the bundle (`lightgbm_val.json`).

| Model | Macro AUROC | mAP | Brier | At-risk AUROC |
|---|---|---|---|---|
| **Transformer ensemble** | **0.751** [0.730, 0.773] | **0.209** [0.192, 0.211] | 0.0381 | 0.722 |
| Logistic regression, code counts + age and sex | 0.764 | 0.222 | 0.0384 | 0.733 |
| LightGBM counts (4 leaves, 100 trees) | 0.762 | 0.239 | 0.0370 | 0.733 |
| LightGBM counts + latest values | 0.758 | 0.257 | 0.0367 | 0.729 |
| Age-sex incidence, at-risk patients | 0.727 | 0.141 | 0.0390 | 0.696 |

Paired differences, transformer minus comparison (same patients, same resamples):

| Comparison | AUROC | mAP | Brier |
|---|---|---|---|
| at-risk incidence | +0.025 [+0.001, +0.045] | +0.068 [+0.036, +0.086] | -0.0009 [-0.0017, -0.0001] |
| logistic regression | -0.013 [-0.031, +0.006] | -0.013 [-0.039, +0.013] | -0.0003 [-0.0010, +0.0004] |
| LightGBM counts | -0.010 [-0.031, +0.010] | -0.030 [-0.060, -0.011] | +0.0011 [+0.0005, +0.0017] |
| LightGBM counts + values | -0.007 [-0.031, +0.012] | -0.048 [-0.080, -0.024] | +0.0014 [+0.0007, +0.0022] |

The transformer is better than the epidemiological reference, indistinguishable in AUROC from
the two count-based models, and behind them on average precision and Brier score. This is the
same picture as training CV. The at-risk column scores each target only among patients who did
not already have it; the "already has it" information is worth about 0.03 AUROC to every model.

### Subgroups (descriptive, no intervals)

| Group | Patients | Transformer AUROC / mAP | LightGBM counts AUROC / mAP |
|---|---|---|---|
| Anchor before 2015 (mostly patients who later died) | 155 | 0.722 / 0.239 | 0.752 / 0.260 |
| Anchor from 2015 | 210 | 0.730 / 0.129 | 0.739 / 0.192 |
| Genotype file or pre-anchor scan | 21 | 0.719 / 0.171 | 0.711 / 0.170 |

### Per condition (`reports/val/val_per_target.csv`)

The transformer is higher than LightGBM counts on 16 of 40 targets. Most val targets have 5 to
20 positives, so single-target differences are noisy.

* **Near-certain for every model** (AUROC above 0.95): prostate neoplasm, carcinoma in situ and
  metastasis of the prostate, normal pregnancy, pneumonia, obesity, and the pregnancy
  complication code. Each follows a visible Synthea pathway (PSA tests, contraceptives,
  dementia care plans, BMI observations) that any history model reads.
* **Where the transformer loses most**: osteoporosis (0.804 against 0.895), recurrent rectal
  polyp (0.860 against 0.946), myocardial infarction (0.827 against 0.904), the three lung-cancer
  codes (0.76 to 0.79 against 0.83 to 0.85), and pathological fracture (0.847 against 0.909).
  These were also the largest CV losses. They are conditions whose risk in Synthea grows with
  the count of specific earlier events (screening visits, bone-density tests, lung nodules); a
  count-based model represents that directly, and the transformer's 1,024-token window drops
  15 to 18 percent of event tokens for the densest histories.
* **Where the transformer wins most**: the paired brain targets, silent micro-haemorrhage and
  abnormal gait (0.905 against 0.832; these two labels are identical for every patient), and
  concussion (0.686 against 0.540, 7 positives). Stroke is 0.801 against 0.783.
* **Near chance for every model** (below 0.6): whiplash, lacerations of forearm and foot, sprain
  of wrist, streptococcal sore throat, clavicle fracture, sinusitis, wrist fracture-subluxation,
  and ankle fracture. These are acute events that Synthea generates with little dependence on
  the prior record.

### Validation curves (`reports/val/val_curves.csv`, wandb group `val-bundle-168b78f5b5d8`)

All 209 saved epoch checkpoints of the 15 members were scored on val after training. The epoch
selected on each member's held-out training fold scores a mean val AUROC of 0.743 (range 0.733
to 0.756) and mAP 0.200. Averaging the 15 members adds 0.008 AUROC. Val reaches its mean peak
around epochs 9 to 12 and declines only slightly after; the held-out fold chose an earlier epoch
than val would have in 13 of 15 members, by 1 to 5 epochs. Choosing epochs on val would have
raised the mean member AUROC by 0.008; that selection was not made and the submission is
unchanged. These curves are reported, as the task asked, and were not used for any choice.

## 4. What the evidence supports

* The pipeline is leak-free as far as the tests and audits can show: train, val, and test
  histories follow the same cut, a future record changes no prediction, and official val was
  read once by a bundle that refuses to run again.
* Self-pretraining helps: F beat S on every fold in both the CSV-only and the multimodal model.
* The modalities add nothing measurable: genomics reach 2 of 358 test patients, and the MRI
  volumes are near-identical across patients.
* A 2-layer transformer on 2,791 synthetic patients ranks patients about as well as a
  gradient-boosted model on code counts, and worse at the top of the ranking. The remaining gap
  is on count-driven conditions, and the count branch tested in round 2 did not close it.
* Removing the clock codes cost about 0.014 AUROC in CV and probably costs test score too, since
  the test files contain them; the model should not learn how the anchor was built.

## 5. Reproduction

```bash
make setup && ln -s /path/to/research_intern_coding_test data/raw
make dataset && make modalities && make folds && make tokenize CONFIG=configs/v2_base.toml
# training (about 20 minutes per fold and seed on an M-series Mac):
for f in 0 1 2 3 4; do make pretrain CONFIG=configs/experiments/v2_ptw_pre.toml FOLD=$f; \
  make finetune CONFIG=configs/experiments/v2_c1.toml FOLD=$f METHOD=F; done   # and the two seeds
make freeze && make val-score && make predict
```

The public package (`ehrforecast.hf_package`) carries the 15 checkpoints and 5 tokenizers;
`hf_package verify` reproduces `predictions.csv` from it on CPU
(`reports/final/package_verify_*.json`).
