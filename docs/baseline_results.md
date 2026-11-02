# Step 2 results: reference baselines

All numbers are on the validation split (365 patients). The test labels are hidden.
Hyperparameters were chosen by 5-fold cross-validation on train (the same folds for every
model), so val was never used for a model choice. Intervals are 95% BCa bootstrap intervals
over 1,000 patient resamples.

## Models

| Model | What it uses | Chosen setting |
|---|---|---|
| Age-sex incidence | observed 5-year rate of first diagnosis in the patient's sex and age band | 5-year bands, pseudo-count 10 |
| Age-sex incidence, at risk | same, but the denominator counts only patients without the condition; patients who already have it get 0 | 5-year bands, pseudo-count 10 |
| Logistic regression, demographics | age, 6 age bands, sex (8 features) | C = 1.0 |
| Logistic regression, codes | log code counts over the whole history and the last 2 years, plus demographics (1,070 features) | C = 0.003 |

The incidence tables are the epidemiological reference: they need no model at all. The at-risk
version adds one fact from the history per target, namely whether the patient already has it.

## Results

| Model | Val macro AUROC | Val mAP | At-risk macro AUROC | CV macro AUROC |
|---|---|---|---|---|
| Age-sex incidence | 0.692 [0.667, 0.716] | 0.111 [0.097, 0.116] | 0.695 | 0.692 |
| Age-sex incidence, at risk | 0.727 [0.703, 0.749] | 0.141 [0.120, 0.148] | 0.696 | 0.730 |
| Logistic regression, demographics | 0.698 [0.675, 0.722] | 0.113 [0.102, 0.118] | 0.700 | 0.694 |
| Logistic regression, codes | **0.756** [0.735, 0.775] | **0.217** [0.185, 0.225] | **0.733** | 0.727 |

Chance-level mAP (mean prevalence) is 0.047. "At-risk" scores count, for each target, only
patients who did not already have it before the anchor.

### Paired comparison: codes model minus each baseline

| Baseline | Macro AUROC difference | mAP difference |
|---|---|---|
| Age-sex incidence | +0.064 [+0.036, +0.090] | +0.105 [+0.069, +0.124] |
| Age-sex incidence, at risk | +0.029 [+0.003, +0.054] | +0.076 [+0.039, +0.098] |
| Logistic regression, demographics | +0.057 [+0.033, +0.082] | +0.103 [+0.067, +0.129] |

Every interval is above zero, so the codes model is better than each baseline. But an age-sex
table plus the single "already has it" flag reaches 0.727 macro AUROC. The full history with
1,070 features adds only 0.029 on top of that. The gain is larger in mAP (+0.076), so the
history helps most at the top of the ranking.

## Where the clinical history helps, and where it does not

Per-target AUROC differences, codes model minus at-risk incidence (val; most targets have 5 to
20 positives, so single-target differences are noisy):

* **History helps most** (14 targets better by more than 0.05): myocardial infarction (+0.26),
  pneumonia (+0.19), congestive heart failure (+0.13), the lung and prostate cancer codes
  (+0.07 to +0.11), and pathological osteoporotic fracture (+0.09). These conditions follow a
  visible pathway in Synthea: risk factors, screening, work-up.
* **History hurts** (8 targets worse by more than 0.05): osteoporosis (−0.12), atrial
  fibrillation (−0.07), anemia (−0.07), Alzheimer's disease (−0.05), and several injuries. For
  these targets the risk is mostly age and sex. The logistic regression models age with one
  linear term and 6 bands, and the strong regularisation (C = 0.003) shrinks that signal among
  1,070 features. The 5-year age-sex table captures it better.
* **Recurrent acute conditions** (viral sinusitis, pharyngitis, bronchitis): the plain age-sex
  incidence is at chance (0.50 to 0.57). The at-risk version gets 0.67 to 0.77 only from the
  "already has it" flag, and the codes model does not beat it.

For step 4 this means that the sequence model must keep a strong age-sex signal and use the
prior-condition information. The at-risk incidence table is the bar that shows whether the
history adds value.

## Leakage review of the highest scores

The run flags targets with val AUROC above 0.98 for the codes model. Each one has a pre-anchor
cause in the Synthea disease modules, not a leak:

| Target | Val AUROC | Top features (all before the anchor) |
|---|---|---|
| Neoplasm of prostate | 0.992 | recent PSA test, recent digital rectal exam, age 50–65 |
| Metastasis / carcinoma in situ of prostate | 0.988 / 0.986 | same prostate work-up pathway; male sex |
| Normal pregnancy | 0.987 | oral contraceptive prescriptions, female, age (age-sex table alone: 0.968) |
| Pneumonia | 0.980 | dementia care plan, MMSE scores, Alzheimer's disease |

## Other observations

* Most injuries (sprains, lacerations, whiplash, concussion, wrist and clavicle fractures) are
  near chance for every model. These events look random with respect to the history in
  Synthea.
* "Silent micro-hemorrhage of brain" and "Abnormal gait" have identical labels for all 3,156
  train and val patients (52 positives each). They come from the same module and always occur
  together, so their scores are identical.
* For the demographics logistic regression the CV curve is flat over the whole C grid (0.6925 to
  0.6942), so the "C on grid edge" warning has no practical effect.

Reproduce: `make dataset && make baseline`. Full per-target numbers:
`reports/baseline/per_condition_val.csv`. Model settings, CV curves, and paired comparisons:
`reports/baseline/metrics.json` and `reports/baseline/cv_results.csv`.
