# Final audit criteria

Written 3 November 2026, before the final audit examined the codebase. The criteria come from
two sources only: the task brief (`ReadMe.md` in the dataset folder) and general practice for
machine-learning work that has to be correct, leak-free, and reproducible. They were fixed at
the commit that adds this file and are applied without change; anything found missing from
them during the audit goes in an appendix of the audit report, dated, and is not scored.

## Protocol

* The person who wrote these criteria also wrote most of the code (the AI sessions described
  in `docs/ai_workflow.md`). To limit that bias, the examination itself is done by fresh
  sessions that start with no knowledge of the codebase, receive only this file, the brief,
  and the repository path, and are told nothing about what the author expects to pass.
* The audit is read-only. Auditors may run existing commands, tests, and short read-only
  scripts kept outside the repository; they change no tracked file.
* Evidence is code, data, or command output with a file and line reference. A statement in a
  document is not evidence that the code behaves that way.
* Each criterion gets one verdict: **PASS**, **PARTIAL** (met with a gap that is named),
  **FAIL**, **N/A** (does not apply, with the reason), or **UNVERIFIED** (could not be checked
  within the audit; the reason is named). A verdict without evidence is UNVERIFIED.
* The consolidated report lists every criterion. Nothing is dropped because it failed.

## A. Fidelity to the brief

| ID | Criterion |
|---|---|
| A1 | The anchor of each train and val patient is the latest `STOP` in `encounters.csv` (or `START` where `STOP` is blank) minus five years, as the brief defines it. The arithmetic of "five years" (calendar years or a day count) is stated and applied to every patient the same way. |
| A2 | Test anchors are read from `test_anchors.csv`, never recomputed from the truncated test tables. |
| A3 | Split membership comes from `patient_splits.csv` unchanged: no patient is moved, dropped without a stated reason, or duplicated; counts per split match the file. |
| A4 | A label is positive only when the first-ever diagnosis of that target code, over the patient's whole record including pre-anchor events, falls inside the five-year window after the anchor. The boundary convention at both ends of the window is explicit and applied consistently. |
| A5 | Patients who already had a target condition before the anchor are negative (or masked) for that target, and the treatment of those cells in training and in scoring is stated. |
| A6 | Every input event lies before the anchor. Each table's date columns (`START`, `STOP`, `DATE`, and any other) are handled, including records that straddle the anchor, with the convention written down. |
| A7 | The truncation convention used for train and val matches the one the organisers applied to the released test tables (checked empirically against `test/` and `test_anchors.csv`, not assumed). |
| A8 | `predictions.csv` has a `patient_id` column, exactly one column per code in `target_conditions.csv` with the code as the header, one row per test patient in `patient_splits.csv`, probabilities in [0, 1], and no missing, duplicate, or non-numeric cells. |
| A9 | Only the provided data is used. No external dataset, no pretrained weights, no modality outside the provided folders. |
| A10 | Nothing is fitted on val or test patients: no model parameter, vocabulary, bin edge, normalisation statistic, stacking weight, calibration map, or threshold. Val is used for monitoring and model selection only, and that use is recorded. |
| A11 | The four submission items exist: a technical report PDF with the four required sections and a wandb link, a public GitHub repository with reproduction instructions, the trained model on the Hugging Face Hub, and `predictions.csv`. Items not yet public are listed with what remains. |
| A12 | Macro AUROC and mAP are computed as the brief states (unweighted mean over the target conditions). Targets with no positives in a split are handled explicitly. Where the organisers' convention is unknown (the prior-condition cells), the assumption is stated and its effect quantified. |

## B. Leakage control

| ID | Criterion |
|---|---|
| B1 | Every input field was reviewed. Fields that can carry post-anchor information (`STOP` dates after the anchor, encounter end times, costs or counts aggregated over the full record, death dates, age at the last encounter) are removed or truncated, and the kept fields are enumerated. |
| B2 | Anchor-relative features (age at anchor, time since an event) use pre-anchor data and the anchor date only. No feature uses the last-encounter date except through the anchor itself. |
| B3 | Artifacts of how the anchor is built (anything in the pre-anchor record that encodes calendar time, time to death, or the position of the anchor) are looked for, and their treatment is justified with numbers. |
| B4 | Cross-validation folds are assigned by patient, a patient is in one fold only, and the folds are persisted and reused by every experiment so that comparisons are paired. |
| B5 | Every fitted preprocessing object (vocabulary, value bins, scalers, imputation, feature selection, LightGBM statistics) is fitted on the training portion of its fold (or on the training split alone for the final model) and applied to held-out patients. |
| B6 | Self-supervised pretraining uses only training-split patients' pre-anchor histories. Any use of held-out fold, val, or test histories is disclosed and justified. |
| B7 | Early stopping and checkpoint selection use only the fold's held-out training patients, never val or test. |
| B8 | Hyperparameter, architecture, stacking, and calibration choices were made on training CV only. The number of official val evaluations is recorded, and none of them changed a modelling choice (or the change is disclosed). |
| B9 | Blend or stack weights and calibration maps are fitted on out-of-fold predictions in a nested way, never on predictions from models that trained on the same patients. |
| B10 | The code path that writes test predictions opens no label file. The val-label reader is confined to one module that can be audited. |
| B11 | Mutation-style tests exist: moving the anchor boundary (strict to inclusive) fails a test; a record added after the anchor changes no prediction; a label built from the wrong window fails a test. |
| B12 | For optional modalities, the presence of a patient's file (not its content) was checked for correlation with the outcomes and the split, and any dependence of predictions on presence is disclosed. |
| B13 | Pre-anchor history distributions of train, val, and test are compared in a report, so that a truncation mismatch would surface. |
| B14 | No label-derived information enters the inputs. The prior-condition mask and any "at-risk" definition use pre-anchor conditions only. |

## C. Evaluation rigour

| ID | Criterion |
|---|---|
| C1 | The CV protocol is stated: fold count, patient-level assignment, seed, stratification, and whether metrics are pooled or averaged over folds. Every configuration that is compared was scored on the same folds. |
| C2 | Headline metrics carry confidence intervals from a patient-level bootstrap, and model-to-model differences are paired on the same patients. |
| C3 | Comparison baselines include a trivial one (prevalence or incidence), a linear one, and a strong tabular one, trained on the same folds with comparable tuning effort. |
| C4 | Adoption rules were fixed before the results they judge; the count of configurations evaluated is reported; an estimate of selection optimism is given; the final choice follows the rules. |
| C5 | The number of val evaluations is recorded, every attempt's artifacts are kept, and val metrics are reported with intervals and labelled as val, never as test. |
| C6 | Per-target results and at least one subgroup breakdown are reported, and targets near chance are identified with an explanation. |
| C7 | Calibration is assessed (Brier score, expected calibration error, a reliability table) since the deliverable is probabilities. |
| C8 | Metric code is correct: AUROC and average precision come from a standard library or are tested against one; macro averaging is unweighted; all-negative targets are handled; the prior-condition convention is applied the same way to every model compared. |
| C9 | Training and validation curves are logged to wandb for the submitted members, and the run links are given. |
| C10 | Numbers in the report, README, model card, and results documents agree with the committed report files (spot-check the headline values and at least five others). |
| C11 | Ensembles and blends are evaluated under the same out-of-fold protocol as single models, not on the folds that fitted them. |

## D. Representation, model, and training

| ID | Criterion |
|---|---|
| D1 | The unit of input is defined. Token construction is documented, the vocabulary is built on the training portion of the fold, out-of-vocabulary handling is defined, and vocabulary size and coverage are reported. |
| D2 | Continuous values (labs, vitals) are handled by a stated method whose parameters are fitted on fold-training data; missing values are handled; the brief's question about continuous values is answered. |
| D3 | Time between events is encoded by a stated method; the context-length policy (what is dropped when a history is too long) is stated, its retention is measured, and the choice is justified for the task. |
| D4 | The architecture is documented with a parameter count, and the size is justified against the number of training patients, with evidence from at least one capacity comparison. |
| D5 | The supervised objective is stated (multi-label loss, masking of prior-condition cells or its absence, class-imbalance handling). The pretraining objective is stated. |
| D6 | Optimiser, learning rate, schedule, batch size, epoch budget, early-stopping rule, gradient clipping, and dropout live in configuration files, not in code constants, and the submitted values are recorded. |
| D7 | Seeds are set for initialisation and data order; a resumed run is tested to match an uninterrupted one; the run-to-run variance is reported. |
| D8 | Training and held-out curves were inspected; overfitting is addressed by named measures, and the evidence that they help is in the reports. |
| D9 | Ensembling across seeds and folds is documented, and blend weights were not fitted on val. |
| D10 | The comparison of pretraining against training from scratch is fair (same budget and selection rule) and its result is reported whichever way it went. |
| D11 | Encoders for optional modalities are trained only on training-split data, their contribution is measured against the same model without them, and the result is reported whichever way it went. |
| D12 | Model mechanics are tested: causal attention masks, padding masked out of the readout, position encoding applied, and output shapes checked. |

## E. Reproducibility and engineering

| ID | Criterion |
|---|---|
| E1 | Dependencies are pinned in a lock file, the Python version is stated, and one command builds the environment. |
| E2 | A documented command sequence runs from the raw data folder to `predictions.csv`. The Makefile targets match the documentation and there are no undocumented manual steps. |
| E3 | Reproduction was verified in a fresh environment at least for inference (the packaged model regenerates `predictions.csv` to within a stated tolerance), and what training reproduces to is stated. |
| E4 | A manifest records the hashes of every submitted artifact and the code commit, so the submitted predictions trace to a commit. |
| E5 | The test suite runs green in continuous integration, and covers the data boundaries, labels, metrics, tokenizer, model shapes, and resume. |
| E6 | Lint, format, and strict type checks pass. |
| E7 | No secrets (API keys, tokens), raw data, or trained weights are committed; the ignore rules cover data, models, and logs; the repository is of reasonable size. |
| E8 | No user-specific paths or hard-coded machine assumptions in the code; device selection is portable (CPU works). |
| E9 | Runs write logs and metric files; wandb runs are named so a reader can find the submitted members; offline runs were synced. |
| E10 | The Hugging Face package is self-contained (weights, tokenizers, configs, feature specifications, calibration) with a model card that states data, intended use, metrics by split, and how to run it, and the loader is verified. |
| E11 | The dataset licence (CC BY 4.0, MITRE) is acknowledged, and the repository has a licence file. |
| E12 | Modules have single responsibilities, the main path carries no dead experiment code, missing inputs raise clear errors, and command-line entry points are documented. |

## F. Documentation and honesty

| ID | Criterion |
|---|---|
| F1 | The report answers the four required questions explicitly, includes the wandb link, says that test metrics are computed by the organisers, and labels every reported number by split. |
| F2 | Every number in the report can be traced to a committed file. |
| F3 | Negative results and limitations are reported (what did not help, what failed, what the model cannot do). |
| F4 | A decision log records each data and protocol decision with its evidence and date, including the val attempts. |
| F5 | The AI-workflow section names the tools, the division of work, what was reviewed, and what the review caught. No placeholder text remains anywhere in the submitted documents. |
| F6 | The README describes the final model and the final run sequence; nothing in it describes an earlier model as the submission. |
| F7 | Public artifacts use one consistent identity, contain no personal data, and the GitHub, Hugging Face, and wandb references agree with each other. |
| F8 | Claims are proportionate: "leak-free" cites tests and audits; differences between models are claimed only with intervals; nothing is reported as a test score. |

## G. Final gates

| ID | Criterion |
|---|---|
| G1 | `predictions.csv` passes a structural validation written during the audit: IDs equal the test split, columns equal `target_conditions.csv`, values numeric in [0, 1], no gaps. |
| G2 | The committed `predictions.csv` is byte-equivalent (or equal within the stated tolerance) to a regeneration from the frozen artifacts at the committed commit. |
| G3 | The PDF opens, has the four sections, carries no placeholder text, and is named as the brief asks (or the rename is listed as remaining). |
| G4 | Publication status of the repository, the model, and the wandb project is recorded, with what remains to make each public. |
| G5 | Everything still to do before sending is listed in one place. |
