# Model plan: steps 3 and 4

| | |
|---|---|
| Version | 1, 30 October 2026 |
| Status | Planning complete. Open items (section 18) do not block P3a. No implementation yet. |
| Scope | Step 3 (representation) and step 4 (models, training, evaluation on val). Step 5 (test predictions, report, publishing) is only covered where it constrains steps 3 and 4. |
| Related | `plan.md` (project overview), `decisions.md` (D1–D8, implemented), `baseline_results.md` (step 2), `clinical_reference_feasibility.md`, `endpoint_risk_scores.md` |

## Contents

0. How to read this plan
1. Summary
2. Task rules and how the plan follows them
3. Data facts (F1–F8)
4. Decision register (D9–D22)
5. Evaluation protocol
6. Phase overview
7. P3a: baseline extensions
8. P3b–P3c: representation
9. P4a–P4c: model, pretraining, fine-tuning
10. P4d: zero-shot generation
11. P4e: final models and the val run
12. P4f: clinical reference scores
13. Experiment register (E0–E12)
14. Test register (T1–T18)
15. Risk register (R1–R15)
16. Logging, reproducibility, and file layout
17. Report hooks
18. Open items
19. References

---

## 0. How to read this plan

* IDs: **F** = measured fact, **D** = decision, **P** = phase, **M** = measurement to take before a
  rule is frozen, **E** = experiment, **T** = test, **R** = risk.
* Decision numbers continue from `decisions.md`. A planned decision moves to `decisions.md`, with
  its evidence, when it is implemented.
* "Fit patients" are the training patients of one cross-validation fold (4 of 5 folds).
  "Held-out fold" is the remaining fold. "Val" always means the official validation split.
* Every phase lists its outputs, its checks, and a "done when" condition. A phase is not done
  until `make check` passes and the outputs exist.
* Time estimates are for implementation and review. Compute times are estimates until the
  throughput benchmark in P4a replaces them with measured values.

## 1. Summary

1. **Representation.** One token is one clinical event (`source:kind:code`). Numeric lab values
   are fused into the code as a per-code decile. Time is relative only: gap tokens between event
   days and a continuous age embedding on every token. There is no calendar time, because the
   anchor year encodes whether the patient dies in the label window (F1).
2. **Models.** A small GPT-style decoder (about 1–3 M parameters) pretrained with next-token
   loss on full training timelines, then fine-tuned with 40 sigmoid outputs. A zero-shot
   variant (ETHOS method) samples 5-year futures from the pretrained model. New baselines:
   LightGBM and selected clinical risk scores.
3. **Protocol.** All choices are made by 5-fold cross-validation on the training split, with the
   same folds as step 2. Val is scored once per final model, after the configuration is frozen
   and committed. Test is never used for anything except the final predictions.
4. **Leak controls.** A runtime assert and unit tests for events on or after the anchor, a
   time-shift invariance test, train-only vocabularies refitted in every fold, and a probe that
   measures how much calendar era the codes still carry.
5. **Effort.** About 27 hours of implementation and 13–23 hours of compute (section 6).

## 2. Task rules and how the plan follows them

| README rule | How this plan follows it |
|---|---|
| "Do not make your own [splits]." | The train, val, and test assignments are not changed. The 5 folds are an internal resampling of the training split only, as in step 2. |
| "Do not train on validation or test." | No val or test patient contributes to any fitted quantity: not to weights, vocabulary, value bins, gap statistics, imputation, or pretraining (pretraining is training). |
| "Validation is for monitoring and model selection only." | Val is used less than allowed: no selection, only the final scoring and the curves of the frozen final runs (D9, D17). |
| "Do not use external datasets or pretrained weights." | All models train from scratch. No external code ontology (SNOMED hierarchy, UMLS) and no published ETHOS weights (D11, D19). |
| "Score every test patient we give you." | 358 rows; checked by T17. |
| Columns = condition codes from `target_conditions.csv`. | Checked by T17. |
| Track training with wandb; link in the report. | Section 16. |
| Report test macro AUROC and mAP. | Test labels are hidden. The report gives val and cross-validated numbers and says that the test numbers come from the organisers. |
| "Prioritize a correct, leak-free, well-evaluated pipeline over a large or clever model." | Leak tests (section 14) are part of the done criteria of P3c. The model stays small. |

**ETHOS check (D11).** The README lists Renc et al. (2024) as recommended reading, says "the
modeling approach is your choice", and forbids only external datasets and pretrained weights.
PubMed confirms that ETHOS predicts by generating future timelines from a model trained on
tokenized patient timelines, with no labelled data and no fine-tuning (PMID 39300208). This
plan implements the method from scratch on the training split. It is allowed. Downloading
published ETHOS weights or tokenizer files fitted on other data would not be allowed.

## 3. Data facts (F1–F8)

Measured on 30 October 2026 from `data/processed/` and the raw tables. P3a and P3c turn the
numbers that the plan depends on into committed reports.

### F1. The anchor year tells whether the patient dies in the label window

| Train + val patients | Alive at end of record | Dead |
|---|---|---|
| Anchor year 2015 or later | 1,731 | 51 |
| Anchor year before 2015 | 71 | 1,303 |

The latest encounter in the data is 24 July 2021. A patient whose record runs to the end of the
data gets an anchor in 2015 or 2016. An earlier anchor means that the record stopped early, in
95% of cases because the patient died. Label rates for anchors before 2015 are 20 to 60 times
higher for pneumonia, chronic heart failure, and the prostate and lung cancer codes (prostate
metastasis has no positive at all with a later anchor), and 3 to 7 times lower for normal
pregnancy and obesity. 46.4% of test anchors are before 2015 (43.5% in
train + val). `DEATHDATE` was read only for this table. It never enters a model (D3).

### F2. Sequence length

Tokens per patient with the planned rules (one token per event, values fused, same-day
duplicates removed, one gap token per event day; refill collapse not yet applied):

| Split | Median | p90 | p99 | Max | Over 512 | Over 1,024 |
|---|---|---|---|---|---|---|
| train | 353 | 934 | 5,735 | 15,449 | 33% | 8% |
| val | 356 | 933 | 4,712 | 13,795 | 33% | 9% |
| test | 364 | 884 | 4,196 | 13,454 | 32% | 7% |

Observations give 42% and medications 37% of these tokens.

### F3. Pretraining data

The raw train + val tables hold 2.14 M rows over the full timelines. The pre-anchor event table
for all three splits holds 1.43 M events (start and stop rows counted separately). So the
post-anchor part of the training timelines is a large share of the data that pretraining can
use. P3c measures the exact token counts per fold.

### F4. Time between event days (pre-anchor, all splits)

| Gap | 1–6 d | 7–29 d | 30–90 d | 91–181 d | 182–364 d | 365–729 d | 730–1,825 d | ≥1,826 d |
|---|---|---|---|---|---|---|---|---|
| Share | 14.7% | 23.5% | 15.2% | 9.0% | 10.5% | 24.1% | 0.9% | 2.1% |

The 365–729 day bucket holds the yearly wellness visits.

### F5. Medication refills

Synthea writes a new medication row for each refill. Start rows per patient and drug: median 1,
p90 55, maximum 588.

### F6. Observation codes

113 numeric codes (5 have more than one unit; 17 have fewer than 50 values; 55 have fewer than
200) and 37 text codes.

### F7. Prior conditions

About 9.5% of patient × target cells have the target condition before the anchor (train 9.48%,
val 9.09%, test 9.48%). These cells are negative by the task definition.

### F8. Positives per target

Train: 29 to 741 per target. Val: 5 to 99 per target (`reports/dataset/label_prevalence.csv`).
"Silent micro-hemorrhage of brain" and "Abnormal gait" have identical labels (step-2 finding).

## 4. Decision register (D9–D22)

"Derek" = decided by Derek on 30 October 2026. "Plan" = proposed by this plan; Derek can
override it.

| ID | Decision | Alternatives considered | Reason | By |
|---|---|---|---|---|
| D9 | Make every model choice by 5-fold CV on the training split, with the step-2 folds. Score val once per final model. | Select on val directly (README allows it). | Selection on val makes the val score optimistic, and test labels are hidden, so no unbiased number would remain. The step-2 baselines never used val for a choice, so a fair comparison needs the same rule. | Derek |
| D10 | No absolute calendar time in any model input: no calendar year, birth year, or date embedding. Quantify the effect with one side run (E8) that is never submitted. | Use calendar time; leave it out without measuring it. | F1: the anchor year carries post-anchor information (death in the window). The gain would come from the anchor definition, not from clinical signal. | Derek |
| D11 | ETHOS-style zero-shot generation is in scope, implemented from scratch. | Leave it out. | Allowed by the README (section 2). It also gives a second use of the pretrained model for the report. | Derek |
| D12 | Optional extras in scope: LightGBM baseline, clinical risk scores, extra anchors per training patient, zero-shot generation. | Any subset. | Derek's selection. | Derek |
| D13 | Prior-condition mask: a cell where the patient had the target before the anchor gets prediction 0 in every model. Newly trained models leave these cells out of the loss. | Let each model learn it. | The label is 0 by definition. Moving known negatives to the bottom of the ranking can only keep or raise AUROC and AP, and it removes a difference between models that has nothing to do with the history. | Plan |
| D14 | Vocabulary, value bins, and gap statistics are refitted in each fold on that fold's fit patients. Each fold model keeps its own tokenizer. | One tokenizer fitted on all training patients. | Same rule as the step-2 features (D6). A tokenizer fitted on all training patients would leak held-out-fold codes into the CV estimate. | Plan |
| D15 | Demographic inputs: sex and age only. Race and ethnicity are not model inputs. | Add them as prefix tokens. | Same inputs as the baselines, so differences come from the history. No measured need. | Plan |
| D16 | Default pretraining data: full timelines (before and after the anchor) of the fold's fit patients. Held-out-fold patients are left out completely. | Pre-anchor history only (tested in E2). | More data, and the post-anchor part teaches what follows a history, which is the task. Held-out patients must be excluded, because the model would otherwise see their futures. | Plan |
| D17 | Val metrics are computed only in P4e, after the final configuration is committed. Val curves of the frozen runs go to wandb for the report. | Monitor val during development. | The README asks for validation curves. Logging them only for frozen runs gives the curves without letting them influence a choice. | Plan |
| D18 | `predictions.csv` comes from the transformer variant (fine-tuned ensemble or zero-shot ensemble) with the higher 5-fold CV macro AUROC. When the difference is below 0.005, the fine-tuned ensemble is used. Baselines are reported beside it but not submitted. | Best of all models by CV, even if LightGBM; a CV-chosen transformer + LightGBM average. | The task assesses sequence modelling. The rule uses CV only, so val cannot influence the submission. | Derek |
| D19 | No external knowledge sources: no ontology, code hierarchy, or embeddings from outside the dataset. Code descriptions from the dataset itself may be used in the report only. | Use the SNOMED hierarchy to group codes. | README forbids external datasets. | Plan |
| D20 | Time representation: gap tokens plus a continuous age embedding on every token. | Gap tokens only (ETHOS); age embedding only (BEHRT). | Step 2 showed that age is the main risk factor for many targets, so age must be explicit. Gap tokens are needed for zero-shot generation to advance a clock. E6 tests the age embedding. | Plan |
| D21 | Head input: the `[ANCHOR]` state joined with age at anchor and sex. | `[ANCHOR]` state only; mean pooling. | Keeps the demographic signal in reach of the head (step-2 lesson: the logistic regression lost it under strong regularisation). | Plan |
| D22 | Final model: the 5 fold models as an ensemble (mean probability). | One model refitted on all training patients with a fixed epoch count. | No epoch guess is needed, and the same runs give the out-of-fold CV score. Each fold model sees 80% of the training patients. | Plan |

## 5. Evaluation protocol

This section is fixed before any transformer result exists. Changes after that need a dated
entry in section 18 with the reason.

### 5.1 Folds

* `KFold(n_splits=5, shuffle=True, random_state=20261028)` on the training rows in the order
  used by `train_baseline.load_data`. P3a writes the assignment to
  `data/processed/folds.parquet` (`patient_id`, `fold`). All later code reads this file.
* Check: the step-2 CV table recomputed from the file equals `reports/baseline/cv_results.csv`.
* Extra-anchor samples (E3) belong to their patient's fold. Held-out folds contain real anchors
  only.

### 5.2 Metrics

| Role | Metric | Definition |
|---|---|---|
| Primary | Macro AUROC | Mean over the 40 targets of the AUROC over all patients, with the D13 mask applied. |
| Secondary | mAP | Mean over the 40 targets of the average precision, mask applied. |
| Secondary | At-risk macro AUROC and mAP | Same, per target only over patients without that target before the anchor. |
| Secondary | Brier score and calibration | Macro Brier score; one reliability plot of all cells pooled (10 quantile bins). The task asks for probabilities, so calibration is reported. |
| Descriptive | Per-target AUROC and AP | With positive counts. Not used for claims about single targets with fewer than 10 positives. |

* CV score: the mean of the 5 per-fold macro AUROCs (same definition as `cv_macro_auroc` in
  step 2). The pooled out-of-fold macro AUROC is also reported.
* Intervals: BCa bootstrap, 1,000 patient resamples (D7). Comparisons: paired BCa bootstrap of
  the difference, with the same resample indices for every model.

### 5.3 Pre-declared comparisons on val

1. Final transformer minus logistic regression (codes).
2. Final transformer minus age-sex incidence, at risk.
3. Final transformer minus LightGBM.
4. Zero-shot minus fine-tuned transformer.
5. Logistic regression with anchor year (E8) minus logistic regression (codes). Reported as the
   size of the calendar-time artifact.

Per-target differences are descriptive. No claim rests on one target alone.

### 5.4 Selection rules

* **Tier 1** decisions (E1 pretraining, E2 pretraining data, E3 extra anchors): run all 5 folds.
  Adopt the variant when the mean CV macro AUROC gain is at least 0.005 and at least 4 of 5
  folds favour it. Otherwise keep the default in section 13.
* **Tier 2** decisions (E4–E6 hyperparameters, context, age embedding): fold 0 only. Adopt a
  variant only when its fold-0 gain is larger than the seed spread measured in E0 (maximum minus
  minimum macro AUROC over 3 seeds). Otherwise keep the default.
* The fold-0 score is slightly optimistic after the search. The 5-fold CV score and the val
  score are the numbers that the report uses.

### 5.5 Val run procedure

1. Commit `configs/final.toml` and the code. Record the commit hash.
2. Run `make val-score`. The script refuses to run if the working tree has uncommitted changes.
3. The script appends one line to `reports/val_runs.log`: timestamp, commit hash, config hash,
   models scored.
4. A second val run is allowed only to fix a bug. The log line states the bug. The report states
   the number of val runs.

## 6. Phase overview

| Phase | Content | Depends on | Implementation | Compute |
|---|---|---|---|---|
| P3a | Folds file, prior-condition mask, LightGBM, anchor-year side run, era probe | — | 3 h | 0.5 h |
| P3b | Tokenizer | P3a | 3 h | — |
| P3c | Sequence builder, extra anchors, parity report, leak tests | P3b | 3 h | 0.2 h |
| P4a | Transformer, training loop, sanity checks, throughput benchmark | P3c | 3 h | 0.3 h |
| P4b | Pretraining and its search (fold 0), E0 | P4a | 2 h | 3–6 h |
| P4c | Fine-tuning, E1–E6 | P4b | 3 h | 4–8 h |
| P4d | Zero-shot generation, E7 | P4b | 3 h | 2–3 h |
| P4e | Final 5-fold models, freeze, val run | P4c, P4d | 1 h | 3–5 h |
| P4f | Clinical reference scores, E12 | P3a, P4e | 6 h | 0.2 h |
| | **Total** | | **27 h** | **13–23 h** |

Order: P3a → P3b → P3c → P4a → P4b → (P4c and P4d) → P4e → P4f. P4f can start after P3a for the
input mapping, but its val numbers wait for P4e so that all val scoring happens in one place.

Each phase ends with: `make check` green, outputs present, a short entry in `plan.md`, and one
commit. Derek reviews at each phase boundary.

## 7. P3a: baseline extensions

**Goal.** Fix the folds, apply the mask to every model, add LightGBM, and measure the
calendar-time artifact before any transformer exists.

**Outputs.** `data/processed/folds.parquet`; `reports/baseline/masked_vs_unmasked.csv`;
`reports/lightgbm/` (CV table, val metrics, per-target table); `reports/artifacts/anchor_year.json`;
`reports/artifacts/era_probe.json`; updated `docs/baseline_results.md`.

### 7.1 Folds file

As in 5.1.

### 7.2 Prior-condition mask (D13)

1. Build the mask matrix from `prior_conditions.parquet` (patients × 40, in
   `target_conditions.csv` order).
2. Apply it at prediction time to all step-2 models.
3. Write one table with each step-2 model's val and CV numbers with and without the mask.
4. Check: every masked number is greater than or equal to its unmasked number (T13 tests the
   function; this table shows the size).

Why the mask cannot lower AUROC or AP: masked cells are always negatives. Setting a negative to
the lowest score can only turn a wrongly ordered positive-negative pair into a correct one, and
it can only raise the precision at each positive's rank.

### 7.3 LightGBM baseline (E11)

| Item | Setting |
|---|---|
| Features | Identical to the logistic regression (codes): `BagOfCodes` refitted in each fold, plus age, age bands, sex. |
| Model | One `LGBMClassifier` per target, objective `binary`. |
| Training rows | Per target, patients without that target before the anchor (D13). |
| Fixed settings | learning rate 0.05, `feature_fraction` 0.5, `bagging_fraction` 0.8 with `bagging_freq` 1, `lambda_l2` 1.0, `deterministic=True`, fixed `num_threads`, seed 20261028. |
| Grid | `num_leaves` {4, 8, 16} × `n_estimators` {100, 300, 600} × `min_child_samples` {20, 50} = 18 settings. |
| Selection | One setting shared by all 40 targets, by mean CV macro AUROC (same rule as `C` in step 2). Grid-edge warning as in step 2. |
| Final | Refit on all training patients; val is scored in P4e with the other models. |

Shared settings are used because most targets have fewer than 150 training positives, so
per-target tuning would select on noise. Dependency: `lightgbm` needs `libomp` on macOS; the
README setup section must say so.

### 7.4 Anchor-year side run (E8)

1. Features: the logistic-regression (codes) features plus `anchor_before_2015` (0/1) and the
   standardised anchor year.
2. Same C grid, same folds, same selection rule.
3. Report: CV and val macro AUROC and mAP, the paired difference against the logistic
   regression without these features, and the 10 targets with the largest gain.
4. The model is saved under `models/artifacts/` and is never used for submission.

### 7.5 Era probe (E9)

Two numbers show how much calendar era the codes still carry after D10:

1. AUROC of a logistic regression on the codes features that predicts `anchor_before_2015`
   (CV on train). This mixes era and illness, because sick patients die sooner.
2. Among training patients with an anchor before 2015, Spearman correlation between the true
   anchor year and a ridge regression's CV prediction from the same features. Almost all of
   these patients died at the window end, so this isolates era.

The report states both numbers. A high value in (2) means that drug and procedure codes date
the record, and the transformer can partly recover the artifact even without calendar tokens.

**Done when.** All outputs exist; the folds check passes; `baseline_results.md` has the masked
table, LightGBM CV results, and E8/E9 results.

## 8. P3b–P3c: representation

### 8.1 Measurements before the tokenizer is frozen

| ID | Measurement | Decides |
|---|---|---|
| M1 | For each patient and drug: days from a `stop` row to the next `start` row of the same drug. | The refill threshold N in 8.4. |
| M2 | The 5 numeric codes with more than one unit: units, counts, value ranges per unit. | Per code: convert to one unit, or make the unit part of the token. |
| M3 | Token length distribution after all rules, per split and per fold (real anchors and full timelines). | Confirms the context length 1,024 (E5 tests 512). |
| M4 | `[UNK]` rate per source and split. | Confirms the minimum count of 5 patients. |
| M5 | Distribution of the last-event-to-anchor gap per split. | Parity check (should match D1 evidence). |
| M6 | Number of extra-anchor samples per patient and in total. | The cap K in 8.7. |

Each measurement goes to `reports/representation/` and its result to `decisions.md`.

### 8.2 Token types

| Source | Token | Notes |
|---|---|---|
| conditions | `cond:start:<code>`, `cond:stop:<code>` | `stop` = condition resolved before the anchor. |
| medications | `med:start:<code>`, `med:stop:<code>` | After refill collapse (8.4). |
| careplans | `careplan:start:<code>`, `careplan:stop:<code>` | |
| allergies | `allergy:start:<code>`, `allergy:stop:<code>` | |
| devices | `device:start:<code>`, `device:stop:<code>` | Pre-anchor histories have no device stop (blank-stop share 1.0); stops occur only in full timelines. |
| procedures | `proc:<code>` | |
| immunizations | `imm:<code>` | |
| observations, numeric | `obs:<code>:q<k>` | k = per-code quantile bin (8.5). |
| observations, text | `obs:<code>:<value>` | Value lower-cased, spaces to `_`. Values seen in fewer than 5 fit patients become `obs:<code>:other`. |
| observations, no usable value | `obs:<code>` | Includes the numeric values that fail to parse (logged by step 1). |
| imaging studies | `img:<modality>\|<body site>` | Study events only; no DICOM (D5). |
| encounters | `enc:<class>:<code>` | Encounter length is not used. |
| time | `[GAP:<bucket>]` | 8.6. |
| structure | `[PAD]`, `[SEP]`, `[ANCHOR]`, `[ANCHOR:EARLY]` (E3 only), `[SEX:F]`, `[SEX:M]`, `hx:<condition code>`, `[UNK:<source>]` | One `[UNK]` per source. |

### 8.3 Sequence layout

```
[SEX:F] hx:<c1> hx:<c2> ... [SEP] [GAP:..] <events of day 1> [GAP:..] <events of day 2> ... [GAP:..] [ANCHOR]
└ prefix ┘└── condition summary ──┘    └──────────────── events, oldest first ────────────────┘└ end ┘
```

Example (codes from the data; values illustrative):

```
[SEX:F] hx:53741008 [SEP] [GAP:365-729d] enc:wellness:410620009 obs:8480-6:q6 obs:8462-4:q4
obs:72166-2:never_smoker [GAP:7-29d] ... [GAP:1-6d] [ANCHOR]
```

* **Condition summary.** One `hx:` token for each distinct condition code whose first start is
  before the first event day in the window, in order of first diagnosis. Conditions that start
  inside the window appear as normal events. Every pre-anchor condition is therefore visible,
  even after left truncation. The same definition is used in pretraining windows (8.8), so the
  summary never contains a condition that is in the window's future (T5).
* **Within a day.** Events keep the order of `events.parquet` (source, kind, code), which is
  deterministic. Identical tokens on the same day are kept once.
* **End.** A gap token for the time from the last event day to the anchor, then `[ANCHOR]`.
  The fine-tuning head reads the `[ANCHOR]` position.
* **Truncation.** Context length L = 1,024 tokens including prefix, summary, and end. When the
  events do not fit, the oldest events are dropped, and the cut moves forward to the next gap
  token so that the window starts at a day boundary. The summary is then computed for the new
  window start.

### 8.4 Medication refill collapse

A `med:start:X` that follows a `med:stop:X` of the same patient within N days is a refill. The
collapse drops that stop and that start, so a continuous course becomes one start and one final
stop. N is set from M1 (default 1 day). Check (T7): the set of (patient, drug) pairs and the
first start date of each pair do not change.

### 8.5 Numeric value bins

* Per numeric code (or code and unit, per M2), quantile edges are fitted on all values of the
  fold's fit patients.
* Number of bins: k = min(10, floor(n / 20)), where n is the number of fit values; at least 1.
  Duplicate edges are merged, so a code can have fewer bins.
* Values below the first or above the last edge go to the first or last bin.
* Edges are stored in the tokenizer file. T4 checks that they come from fit rows only.

### 8.6 Time

* **Gap buckets** (F4): 1–6 d, 7–29 d, 30–90 d, 91–181 d, 182–364 d, 365–729 d, 730–1,825 d,
  ≥1,826 d. Gaps are counted in calendar days between distinct event days. No token for the same
  day.
* **Representative duration** per bucket: the median training gap in that bucket (fitted per
  fold). Only zero-shot generation (section 10) uses it.
* **Age embedding** (D20): age a in years at the token's day, (day − birth date) / 365.25.
  Features: a / 100 and sin(2πa/P), cos(2πa/P) for 8 periods P spaced geometrically from 0.5 to
  200 years. A linear layer maps them to the model width; the result is added to the token and
  position embeddings.
  * Event tokens: age at the event day.
  * Gap tokens: age at the day that follows the gap.
  * Prefix, summary, and `[SEP]`: age at the first event day in the window.
  * `[ANCHOR]`: age at the anchor.
* No calendar time anywhere (D10). T2 checks this.

### 8.7 Extra anchors (E3)

* Training patients only. Extra anchors a_k = real anchor − 5k calendar years, for k = 1 … K,
  as long as the patient has at least one event before a_k. K is capped from M6 (default 3).
  Each extra window [a_k, a_k + 5 years] ends at the next anchor, so the windows of one patient
  do not overlap.
* History and labels for a_k are built by the step-1 functions (truncation, labels, prior
  conditions) called with a_k. There is no second implementation (T8).
* E3 tests two forms: (a) extra samples with the same `[ANCHOR]` token; (b) extra samples end
  with `[ANCHOR:EARLY]` instead. All val and test samples are real anchors, so in (b) the model
  sees `[ANCHOR]` at every evaluation. The indicator therefore adds no ranking information
  between evaluation patients. It only stops the earlier anchors, which are not followed by an
  end of record, from changing what the model learns for real anchors.

### 8.8 Sample types

| Sample | Patients | Content | Used by |
|---|---|---|---|
| Real-anchor sample | train, val, test | History before the anchor, layout 8.3. | Fine-tuning, evaluation, zero-shot input |
| Extra-anchor sample | fit patients | Same layout at a_k. | Fine-tuning (E3) |
| Pretraining sample | fit patients | Full timeline to the last encounter, no `[ANCHOR]`, no end-of-record token. If longer than L, one random window per epoch, starting at a day boundary, with its own prefix and summary. | Pretraining |

No end-of-record token: by construction every record ends exactly 5 years after the real
anchor, so the end of record carries no clinical meaning that the model should learn. Zero-shot
generation always simulates the full 5 years.

### 8.9 Fitting and outputs

* The tokenizer is fitted per fold on the fit patients' full timelines (D14). A token is in the
  vocabulary when it occurs in at least 5 fit patients; otherwise it becomes `[UNK:<source>]`.
  IDs are sorted by frequency, then alphabetically.
* Output: `models/transformer/fold<k>/tokenizer.json` (vocabulary, bin edges, gap medians,
  version, SHA-256 of the sorted fit-patient IDs) and
  `data/processed/sequences/fold<k>/{pretrain,finetune,heldout,val,test}.parquet` (token IDs,
  ages, patient ID, anchor date, sample type).
* The final ensemble needs val and test sequences from each of the 5 tokenizers.

### 8.10 Parity report (P3c)

`reports/representation/parity.csv`, per split and per held-out fold: length quantiles, share of
truncated samples, `[UNK]` rate per source, token share per source, last-event-to-anchor gap
distribution. A warning (not a failure) is raised when a test value is outside the range of the
5 held-out folds.

**Done when (P3b + P3c).** M1–M6 recorded; T1–T9 pass; parity report written; F2 re-measured
with all rules.

## 9. P4a–P4c: model, pretraining, fine-tuning

### 9.1 Architecture (P4a)

| Item | Default | Search (E4, fold 0) |
|---|---|---|
| Type | GPT-style decoder, pre-LayerNorm, causal self-attention (PyTorch `scaled_dot_product_attention`) | — |
| Layers | 4 | {2, 4} |
| Width | 192 | {128, 256} |
| Heads | 4 | — |
| MLP | 4 × width, GELU | — |
| Dropout | 0.1 | {0.1, 0.2} |
| Embeddings | token + learned position + age (8.6) | E6: without age |
| Output | LM head tied to the token embedding | — |
| Parameters | about 2–3 M (vocabulary size from M4) | — |

Sanity checks before any real run:

* T10: changing token t+1 does not change any output at positions up to t.
* T11: the model overfits one batch of 8 sequences to a loss below 0.05.
* Throughput benchmark: tokens per second for training and for generation with a key-value
  cache, on MPS and on CPU, for the default size. The results replace the compute estimates in
  section 6.
* Budget gate: if one pretraining epoch takes more than 15 minutes, the default width drops to
  128, and E5 (context 512) is run first.

### 9.2 Pretraining (P4b)

| Item | Setting |
|---|---|
| Data | Pretraining samples of the fold's fit patients (D16). |
| Loss | Cross-entropy on next event and gap tokens. Prefix, summary, `[SEP]`, and padding are not targets. |
| Optimiser | AdamW, betas (0.9, 0.95), weight decay 0.1 on weight matrices only, gradient clipping 1.0. |
| Schedule | Peak learning rate 1e-3, 200 warm-up steps, cosine decay to 10%. |
| Batch | 32 sequences, length-bucketed, right-padded. |
| Precision | float32 (MPS support for lower precision is limited; float32 keeps runs comparable). |
| Stopping | Evaluate held-out-fold next-token loss (full timelines of held-out patients) every epoch; stop after 3 evaluations without improvement; maximum 30 epochs; keep the best checkpoint. |
| Search | E4 on fold 0: the 8 settings of 9.1. Keep the 2 with the lowest held-out loss; fine-tune both with default settings; choose by fold-0 macro AUROC (Tier 2 rule). |

Held-out-fold patients are training-split patients. Using their futures for this stopping rule
is part of cross-validation, not val use.

### 9.3 Fine-tuning (P4c)

| Item | Setting |
|---|---|
| Data | Real-anchor samples of fit patients (plus extra anchors if E3 adopts them). |
| Head | Linear layer from [`[ANCHOR]` state, age at anchor / 100, sex] to 40 logits (D21). Dropout 0.1 before it. |
| Loss | Binary cross-entropy, mean over unmasked cells (D13). No class weights: the metrics are ranking-based, and weights would distort the probabilities. |
| Optimiser | AdamW; learning rate 1e-3 for the head, body learning rate from the search; weight decay 0.01. |
| Search (E4, fold 0) | Body learning rate {3e-5, 1e-4, 3e-4} × body frozen for the first 2 epochs {yes, no} = 6 settings. |
| Stopping | Held-out-fold masked macro AUROC each epoch; patience 5; maximum 30 epochs; best checkpoint. |
| Inference | Sigmoid, then the D13 mask. |

Watch item (R4): pretraining has seen the futures of the fit patients, so the fine-tuning
training loss can fall very fast. The gap between training and held-out AUROC is logged every
epoch. E2 measures whether history-only pretraining generalises better.

**Done when (P4a–P4c).** T10–T12 pass; E0–E6 recorded in `reports/transformer/experiments.csv`
with wandb links; the chosen configuration written to `configs/final.toml` (not yet committed as
final).

## 10. P4d: zero-shot generation (ETHOS method)

| Item | Setting |
|---|---|
| Model | The fold's pretrained model, without fine-tuning. |
| Input | The real-anchor sample without `[ANCHOR]`: prefix, summary, history, and the last-event-to-anchor gap token. |
| Clock | Starts at 0 at the anchor. Each generated gap token advances it by the bucket's representative duration (8.6). Generated tokens get age = age at anchor + clock. |
| Sampling | Temperature 1.0, full distribution (no top-k: it would bias the event frequencies). Structure tokens (`[SEP]`, `hx:`, `[SEX:*]`, `[ANCHOR]`, `[PAD]`) get probability 0. |
| Stop | Clock ≥ 5 years, or 3,000 generated tokens. The share of trajectories that hit the token cap is reported. |
| Context overflow | When the sequence reaches L, rebuild it with the standard truncation (8.3, summary recomputed) and continue. |
| Count | Target k counts in a trajectory when `cond:start:<k>` is generated before the clock reaches 5 years. |
| Probability | Share of N = 64 trajectories that contain target k; then the D13 mask. |
| Evaluation | Held-out folds (CV) in P4d; val in P4e. |

Clock check: on held-out-fold full timelines, sum the representative durations of the real gap
tokens in each true 5-year window and compare with 5 years. The report gives the mean and the
spread of this error, because a biased clock changes how many events a trajectory has time for.

Cost: about 558 held-out patients × 64 trajectories × a few hundred tokens per fold. The P4a
generation benchmark decides whether N = 64 is feasible on all folds; if not, N = 32 for CV and
N = 64 for val and test, and the report says so.

**Done when.** T14 passes; clock check reported; zero-shot CV numbers in
`reports/transformer/experiments.csv`.

## 11. P4e: final models and the val run

1. Freeze: the D18 choice is computed from the CV scores; `configs/final.toml` and the code are committed; the commit hash is
   recorded (5.5, step 1). Nothing that touches val runs before this commit.
2. For each fold k = 0 … 4, with the frozen configuration: fit tokenizer k, pretrain, fine-tune,
   stop on held-out fold k, and run zero-shot.
3. Log the val curves of these frozen runs to wandb (D17). They do not change any setting.
4. Report the 5-fold CV score (mean of fold scores) and the pooled out-of-fold score. These are
   comparable with the step-2 CV column.
5. Run the val procedure (5.5, steps 2–4). Each fold model scores val with its own tokenizer;
   probabilities are averaged over the 5 models; the mask is applied; metrics and the
   comparisons in 5.3 are computed for the transformer, zero-shot, LightGBM, the step-2 models,
   and E8.
6. Write `docs/model_results.md` in the style of `baseline_results.md`.

Test predictions use the same ensemble in step 5. Nothing about test is computed in step 4.

**Done when.** `reports/val_runs.log` has exactly one entry, or one more with a stated bug fix;
`docs/model_results.md` is written.

## 12. P4f: clinical reference scores (E12)

The design follows `clinical_reference_feasibility.md`. Summary of what the plan adds:

* Scope: coronary heart disease (Framingham CHD), myocardial infarction (component-outcome
  comparison), stroke (revised Framingham stroke risk profile), atrial fibrillation (CHARGE-AF).
  Chronic heart failure (PREVENT-HF) only if kidney-function coverage allows it.
* Each formula and its coefficients are copied from the primary publication, and the source is
  cited with a DOI or PubMed ID. Nothing is taken from memory.
* Input mapping, eligibility, and imputation follow the 10 steps in the feasibility document.
  Imputation is fitted on training patients only.
* Each score is compared with the logistic regression, LightGBM, and the transformer on the same
  eligible val patients, with the same bootstrap resamples. The table gives the patient count
  and positive count per row.
* If a score's published inputs include race, race is used inside that score only. It does not
  change D15.
* The Synthea version caveat in the feasibility document goes into the report: shared formulas
  make strong score performance plausible, but they are not proven for this dataset.

The formulas and eligibility rules are fixed and committed before the val numbers are computed,
and the val numbers are computed in the P4e val run or in one documented extra run.

**Done when.** The clinical reference table is in `docs/model_results.md` with sources.

## 13. Experiment register (E0–E12)

| ID | Question | Variants (default first) | Data | Rule | Phase |
|---|---|---|---|---|---|
| E0 | How large is seed noise? | default config × 3 seeds | fold 0 | Sets the Tier 2 noise band | P4b |
| E1 | Does pretraining help? | pretrained; random initialisation | 5 folds | Tier 1 | P4c |
| E2 | Which pretraining data? | full timelines; pre-anchor history only | 5 folds | Tier 1 | P4c |
| E3 | Do extra anchors help? | none; extra with `[ANCHOR]`; extra with `[ANCHOR:EARLY]` | 5 folds | Tier 1 | P4c |
| E4 | Size and optimisation | grids in 9.1–9.3 | fold 0 | Tier 2 | P4b–P4c |
| E5 | Context length | 1,024; 512 | fold 0 | Tier 2 | P4c |
| E6 | Age embedding | with; without (gap tokens only) | fold 0 | Tier 2 | P4c |
| E7 | Zero-shot versus fine-tuned | fine-tuned; zero-shot | 5 folds, then val | Reported, not a selection | P4d–P4e |
| E8 | Size of the calendar-time artifact | LR codes; LR codes + anchor year | 5 folds, then val | Reported, never submitted | P3a |
| E9 | Era signal in the codes | probes in 7.5 | 5 folds | Reported | P3a |
| E10 | Which sources matter per target (optional) | final model with one source's tokens replaced by `[UNK]` at inference | val, after E7 | Descriptive, for the interpretation section | P4e |
| E11 | Stronger tabular baseline | LightGBM grid in 7.3 | 5 folds, then val | Baseline | P3a |
| E12 | Clinical reference scores | section 12 | val subsets | Reference | P4f |

When a Tier 1 result is a tie under 5.4, the default variant stays. All runs are listed in
`reports/transformer/experiments.csv` with: ID, variant, fold, seed, CV metrics, wandb URL,
commit hash.

## 14. Test register (T1–T18)

| ID | Test | Phase |
|---|---|---|
| T1 | No input token comes from an event on or after its sample's anchor (real and extra anchors, all splits). Unit test with a synthetic patient that has events on the anchor day and later; runtime assert in the sequence builder. | P3c |
| T2 | Time-shift invariance: shift birth date, all event times, and the anchor by 3,652 days and by −3,652 days; token IDs and ages must be identical. Whole days are used because a shift in calendar years changes day counts across leap years. | P3c |
| T3 | A code that occurs only in held-out patients maps to `[UNK:<source>]`. | P3b |
| T4 | Value-bin edges come from fit rows only, are increasing, and clip out-of-range values. | P3b |
| T5 | Summary: a condition first diagnosed inside the window is an event, not an `hx:` token; a condition before the window is one `hx:` token. Holds for pretraining windows too. | P3b |
| T6 | Determinism: building sequences twice gives byte-identical files. | P3c |
| T7 | Refill collapse keeps every (patient, drug) pair and its first start date. | P3b |
| T8 | Extra anchors: a_k + 5 years ≤ last encounter; labels and prior conditions equal the step-1 functions' output for a_k; held-out folds contain no extra-anchor samples. | P3c |
| T9 | Folds: no patient in both fit and held-out sets; val and test patients in no fit set; extra samples in their patient's fold. | P3c |
| T10 | Causal mask: outputs at positions ≤ t do not depend on token t+1. | P4a |
| T11 | One batch overfits to loss < 0.05. | P4a |
| T12 | Masked cells give zero gradient in the fine-tuning loss. | P4c |
| T13 | The prediction mask sets prior-condition cells to 0 and changes nothing else. | P3a |
| T14 | Zero-shot: generation stops at 5 simulated years or the token cap; only `cond:start` tokens of target codes count; structure tokens are never sampled. | P4d |
| T15 | Every metric function used for new models equals the sklearn value (extends the existing tests). | P3a |
| T16 | The val script refuses to run with uncommitted changes and writes a `val_runs.log` line. | P4e |
| T17 | Submission check: 358 rows, IDs equal `test_anchors.csv`, 40 columns named and ordered as `target_conditions.csv`, values in [0, 1], no missing values. | step 5 |
| T18 | A saved fold model, reloaded from disk with its tokenizer, gives the same predictions as in memory (needed for Hugging Face). | P4e |

## 15. Risk register (R1–R15)

| ID | Risk | Effect | Control |
|---|---|---|---|
| R1 | Calendar time reaches the model (F1). | Inflated scores without clinical meaning. | D10; T2; E8 and E9 measure what remains. |
| R2 | Train and test sequences are built differently. | Val and CV scores do not transfer to test. | One code path for all splits; test files truncated by the organisers already match ours (D1, D2); parity report 8.10. |
| R3 | Overfitting: 2,791 patients, many targets with fewer than 100 positives. | Unstable model, optimistic fold-0 score. | Small model, dropout, weight decay, early stopping, Tier rules, 5-fold ensemble. |
| R4 | Pretraining memorises the futures of fit patients. | Fine-tuning overfits quickly. | Early stopping on held-out folds; train–held-out gap logged; E2. |
| R5 | Selection noise on fold 0. | Wrong hyperparameters adopted. | E0 noise band; Tier 1 decisions on 5 folds. |
| R6 | MPS results are not deterministic, or an MPS kernel is wrong. | Runs cannot be repeated exactly. | Fixed seeds; one tiny configuration compared on MPS and CPU in P4a; E0 measures the spread; versions logged. |
| R7 | The summary or prefix shows future events in pretraining windows. | The model learns a shortcut that does not exist at inference. | Summary defined relative to the window start; no age-at-end prefix token; T5. |
| R8 | The zero-shot clock drifts. | Too few or too many generated events. | Clock check in section 10. |
| R9 | Extra anchors change the learned distribution. | Lower scores on real anchors. | `[ANCHOR:EARLY]` variant; Tier 1 rule; held-out folds use real anchors only. |
| R10 | Codes carry era information. | Part of the F1 artifact returns. | E9 measures it; the report states it. |
| R11 | Val is used more than once without a record. | Optimistic val score. | Val procedure 5.5; T16. |
| R12 | Compute is too slow on the M2. | Plan does not fit the time. | Benchmark and budget gate in P4a. |
| R13 | Two targets have identical labels (F8). | One module counts twice in the macro mean. | Kept, because the task defines the targets; stated in the report. |
| R14 | Refill rule removes real information. | Lost medication history. | M1 before the rule is frozen; T7. |
| R15 | Environment: `lightgbm` needs `libomp`; MPS needs a recent PyTorch. | Setup fails for reviewers. | Lock file; README setup notes; CPU fallback in config. |

## 16. Logging, reproducibility, and file layout

### 16.1 wandb

* Project `ehr-forecast`. Entity: open (section 18). Until it is chosen, runs use
  `WANDB_MODE=offline` and are synced later with `wandb sync`.
* Run name `<experiment>-fold<k>-seed<s>`, for example `E3b-fold2-seed1`. Group = experiment ID.
  Job type = `pretrain`, `finetune`, or `zeroshot`.
* Config logged: the full TOML configuration, commit hash, SHA-256 of each file in
  `data/processed/`, tokenizer hash, PyTorch version, device.
* Every step: training loss, learning rate, gradient norm. Every epoch: held-out loss,
  held-out macro AUROC and mAP, train macro AUROC (R4). Final: per-target table as a
  `wandb.Table`.
* Val metrics appear only in P4e runs (D17).

### 16.2 Reproducibility

* Seeds for Python, NumPy, and PyTorch come from the configuration and are logged.
* All configuration lives in `configs/*.toml` (read with the standard-library `tomllib`). No
  setting is hard-coded in a training script.
* New dependencies: `torch`, `wandb`, `lightgbm`. Exact versions go to `requirements-lock.txt`.
* `make` targets: `folds`, `baselines-extra`, `tokenize`, `pretrain`, `finetune`, `zeroshot`,
  `final`, `val-score`, `predict`. Each writes a log to `logs/`.

### 16.3 Proposed layout (new parts)

```
configs/                      default.toml, experiments/*.toml, final.toml
src/ehrforecast/
  folds.py                    fold file and fold-aware loaders
  masking.py                  prior-condition mask (D13)
  lgbm.py                     LightGBM baseline
  artifacts.py                anchor-year side run, era probe
  tokenizer.py                vocabulary, bins, gap buckets (8.2-8.6, 8.9)
  sequences.py                sample building, truncation, summary (8.3, 8.8)
  anchors.py                  extra anchors via step-1 functions (8.7)
  model.py                    decoder, age embedding, heads
  pretrain.py, finetune.py    training loops
  zeroshot.py                 generation (section 10)
  evaluate.py                 metrics, paired comparisons, val procedure (5.5)
  riskscores.py               clinical reference scores (section 12)
models/transformer/fold<k>/   tokenizer.json, pretrain.pt, finetune.pt, config.toml
reports/                      lightgbm/, artifacts/, representation/, transformer/, val_runs.log
docs/                         model_plan.md (this file), model_results.md, ai_workflow.md
```

The names are proposals. The implementation may merge small modules if that is clearer.

## 17. Report hooks

| Report section | Source in this plan |
|---|---|
| 1. Representation | Section 8; parity report; F1–F6; D10, D14, D15, D20 |
| 2. Model and training | Section 9; `configs/final.toml`; wandb links; E0–E6 |
| 3. AI workflow | `docs/ai_workflow.md`: start it with the first implementation session. Record which tool did what, what was reviewed, and which bugs the review found. |
| 4. Interpretation | Per-target tables against the baselines; E7 (zero-shot versus fine-tuned); E8 and E9 (artifact and era); E10 (sources); E12 (clinical scores) |
| Metrics requirement | Section 5; the test numbers come from the organisers |

## 18. Open items

| Item | Needed by | Note |
|---|---|---|
| wandb entity | P4b | Offline mode until then. |
| Name on the report; GitHub and Hugging Face account | Step 5 | Unchanged from `plan.md`. |

Changes to sections 4, 5, or 13 after the first transformer result are listed here with the date
and the reason.

## 19. References

* Renc P, et al. Zero shot health trajectory prediction using transformer. *npj Digit Med*
  7, 256 (2024). PMID 39300208. https://doi.org/10.1038/s41746-024-01235-0
* Li Y, et al. BEHRT: Transformer for Electronic Health Records. *Sci Rep* 10, 7155 (2020).
  PMID 32346050. https://doi.org/10.1038/s41598-020-62922-y
* Rasmy L, et al. Med-BERT. *npj Digit Med* 4, 86 (2021). PMID 34017034.
  https://doi.org/10.1038/s41746-021-00455-y
* Vaswani A, et al. Attention Is All You Need. NeurIPS (2017). https://arxiv.org/abs/1706.03762
* Karpathy A. nanoGPT. https://github.com/karpathy/nanoGPT
* Clinical score sources: see `clinical_reference_feasibility.md`; formula sources are added in
  P4f.

PubMed IDs were checked against PubMed on 30 October 2026.
