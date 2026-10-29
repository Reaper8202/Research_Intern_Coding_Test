# Step 2 results: logistic-regression baseline

All numbers are on the validation split (365 patients). The test labels are hidden. `C` was
chosen by 5-fold cross-validation on train, so val was never used for a model choice. Intervals
are 95% BCa bootstrap intervals over 1,000 patient resamples.

| Model | Features | C | Val macro AUROC | Val mAP | At-risk macro AUROC |
|---|---|---|---|---|---|
| Demographics | age, age bands, sex (8) | 1.0 | 0.698 [0.675, 0.722] | 0.113 [0.102, 0.118] | 0.700 |
| Codes | log code counts, whole history + last 2 years, plus demographics (1,070) | 0.003 | 0.756 [0.735, 0.775] | 0.217 [0.185, 0.225] | 0.733 |

Chance-level mAP (mean prevalence) is 0.047. The cross-validated macro AUROC on train at the
chosen `C` is 0.728. For the demographics model the CV curve is flat (0.6925 to 0.6942 over the
whole grid), so the "C on grid edge" warning has no practical effect.

Reproduce: `make dataset && make baseline`. Full per-target numbers are in
`artifacts/baseline/per_condition_val.csv`.

## Leakage review of the highest scores

The run flags targets with val AUROC above 0.98. Each one has a pre-anchor cause in the Synthea
disease modules, not a leak:

| Target | Val AUROC | Top features (all before the anchor) |
|---|---|---|
| Neoplasm of prostate | 0.992 | recent PSA test, recent digital rectal exam, age 50–65 |
| Metastasis / carcinoma in situ of prostate | 0.988 / 0.986 | same prostate work-up pathway; male sex |
| Normal pregnancy | 0.987 | oral contraceptive prescriptions, female, age (demographics alone: 0.963) |
| Pneumonia | 0.980 | dementia care plan, MMSE scores, Alzheimer's disease |

## What the per-target results already show

* Recurrent acute conditions (viral sinusitis, pharyngitis, bronchitis) score well overall
  (0.64 to 0.75) but near chance among patients who never had them (0.48 to 0.55). The model
  mainly learns that a condition the patient already had cannot be "newly diagnosed".
* Most injuries (sprains, lacerations, whiplash, concussion, wrist and clavicle fractures) are
  near chance (0.43 to 0.58). These events look random with respect to the history in Synthea.
  Ankle and forearm fractures are the exception (0.74 and 0.72).
* "Silent micro-hemorrhage of brain" and "Abnormal gait" have identical labels for all 3,156
  train and val patients (52 positives each). They come from the same module and always occur
  together, so their scores are identical.
