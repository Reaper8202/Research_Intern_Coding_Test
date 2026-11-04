# Final audit

Audit of the submission at commit `812393e` (branch `final-deliverable`), 3 and 4 November 2026,
against the 74 criteria of `docs/audit_criteria.md`. The criteria were written from the task
brief and general practice and committed (that same commit) before any code was examined for
the audit; they were applied without change.

## Protocol

* Four fresh sessions, each with no prior knowledge of the codebase, received only the criteria
  file, the task brief, and the repository path, and examined their assigned criteria read-only:
  part 1 data and leakage (A1 to A10, B1 to B5, B10 to B14), part 2 model and training (B6 to B9,
  D1 to D12), part 3 evaluation (A12, C1 to C11), part 4 engineering, documents, and final
  gates (A11, E1 to E12, F1 to F8, G1 to G5). Their full reports, with file and line evidence,
  are in `auditor_reports/`; the read-only scripts they wrote (anchor and label recomputation
  from the raw CSV files, the test-convention check, fitted-object recomputation, code mutations,
  metric and bootstrap reproduction, hash checks, the `predictions.csv` validator, the model
  mechanics probe) are in `auditor_scripts/`.
* The author of the code (the AI sessions described in `docs/ai_workflow.md`) wrote the criteria,
  verified every PARTIAL and FAIL verdict against the repository before acting on it, and made
  the fixes recorded below. No criterion was dropped or reworded; this file records every verdict
  as the auditors gave it. Timestamps are local time on the training machine.
* Verdicts: PASS, PARTIAL (met with a named gap), FAIL, N/A, UNVERIFIED. Disposition: what was
  done on this branch after the audit, or why the item was left as it was.

## Summary

| | PASS | PARTIAL | FAIL |
|---|---|---|---|
| At commit `812393e` (74 criteria) | 50 | 18 | 6 |
| After the fixes on this branch | 63 | 9 | 2 |

Every FAIL at `812393e` was a document or publishing item; no auditor found a leak, a wrong label,
a wrong metric, or a selection made on val. The two items still failing are the two placeholders
in the report (author name, Codex line) and the PDF file name, which only the author can resolve
(`docs/submission_checklist.md`). Of the remaining partials, three are the author's publishing
steps (A11, C9, G4 are tracked in the checklist), three were accepted with a reason (B10, C3, E6),
and the rest are the author's identity decision (F7) and the wandb project visibility.

What the auditors reproduced from scratch, with zero mismatch: all 3,156 train and val anchors
from the raw encounters table; all 126,240 label cells from the raw conditions table; the prior
mask; the fit sets of all 15 tokenizers, the final LightGBM statistics, and the Platt weights; the
390 manifest hashes; `predictions.csv` from the frozen base files and from the Hugging Face
package (difference 0.0); the val metrics of all seven scored files with scikit-learn to five
decimals; the committed BCa intervals with an independent scipy implementation; the nested
logistic stacker to 1e-13. Six code mutations (anchor boundary, STOP boundary, label window
start and end, first-ever rule, five years as days) each failed a test.

## Verdicts

### A. Fidelity to the brief

| ID | Verdict | Evidence (short) | Disposition |
|---|---|---|---|
| A1 | PASS | `cohort.py:21,60-64` calendar `DateOffset(years=5)`; all 3,156 anchors equal a recomputation from `encounters.csv`; 1,019 would differ under 1,826 days (`auditor_scripts/a_anchors_labels_splits.py`). | none needed |
| A2 | PASS | `cohort.py:43-50`, `build_dataset.py:97-103`: test anchors from `test_anchors.csv`; `compute_anchors` runs on train and val only. | none needed |
| A3 | PASS | 2,791 / 365 / 358 match `patient_splits.csv`; 0 duplicates; `cohort.py:24-32,69-83`. | none needed |
| A4 | PASS | `labels.py:14-21,49`: first-ever over the full record, closed window; 0 of 126,240 cells differ from the raw recomputation; 119 cells would flip under an exclusive end. | the 119 window-end positives are noted in D4's convention (nit) |
| A5 | PASS | prior from the truncated history (`labels.py:53-58`), loss mask (`training.py:327-332`), scoring (`evaluation.py:47-51`); `predictions.csv` has exactly the 1,358 prior cells at 0. | none needed |
| A6 | PASS | `truncation.py:32-40` keeps `< anchor`, blanks `STOP >= anchor`; 0 of 1,431,503 processed events on or after the anchor. | none needed |
| A7 | PASS | test tables: 0 rows on or after the anchor, 0 `STOP >= anchor`, 0 anchor-day events against 2.15% in raw train/val (so the organisers cut strictly before the day, as the pipeline does); blank-STOP shares match. `STOP == anchor midnight` has no test row to check. | the untestable edge is recorded in D2 (nit) |
| A8 | PASS | `patient_id` + 40 code columns in file order, 358 rows = test ids, no missing or duplicate cells, range [0, 0.912]. | none needed |
| A9 | PASS | `config.py:19-55`, `modalities.py:151-152` read the organisers' folder only; `model.py:97-113` encoder from default init; no hub or URL in `src/`; lock file has no pretrained-model package. | none needed |
| A10 | PASS | all 15 tokenizer fit-set hashes equal their fold's fit set; final LightGBM vocabulary and medians equal a train-only recomputation; Platt weights equal a recomputation from train OOF; val use recorded in the bundle. | none needed |
| A11 | PARTIAL | PDF (8 pages, four sections, wandb link) exists; repository private and the audited commit not yet pushed; Hugging Face returns 401; `predictions.csv` valid. | `docs/submission_checklist.md` lists each remaining step with its status; the branch is pushed; publishing is the author's |
| A12 | PASS | `evaluation.py:57-68,107-110`: unweighted mean, AUROC undefined targets dropped, AP 0, prior cells set to 0 for every model; scikit-learn recomputation of all seven val files matches `val_summary.json` to five decimals; masked 0.767 against at-risk 0.739 quantified. | none needed |

### B. Leakage control

| ID | Verdict | Evidence (short) | Disposition |
|---|---|---|---|
| B1 | PASS | allowlists `schema.py:36-115`; DEATHDATE (1,354 train/val, all after the anchor) and DISPENSES (post-anchor refills) excluded; durations never inputs (`contracts.py:111-112`). | nit: REASONCODE and ENCOUNTER are read and dropped; left |
| B2 | PASS | age and recency from the anchor or birth (`cohort.py:103`, `sequences.py:430-452`, `features.py:93-96`); `last_encounter` read nowhere outside cohort building; date-shift test. | none needed |
| B3 | PASS | DALY/QALY/QOLS removed (D36): era AUROC 0.999 with them, 0.876 without (fold 0); cost quantified; era subgroups reported. | the scan output is now committed (`reports/dataset/clock_scan.txt`) and the script fits the with-codes baseline too |
| B4 | PASS | `folds.py:24-36,51-93`: train-only KFold with a seed and a re-verified manifest; 2,791 unique patients; every member's held predictions equal its fold's held-out set. | none needed |
| B5 | PASS | tokenizer `tokenizer.py:200-267`, bag of codes `features.py:63-70`, value statistics `lightgbm_cv.py:141-164`, nested Platt `calibration.py:88-103`; recomputations equal. | none needed |
| B6 | PASS | `pretrain.py:161-165`: windows from fold fit rows, eval windows from held-out training rows; `contracts.py:184-194` loads the training split only and requires `dates < anchors`; the rejected full-timeline branch is absent from every submitted parent. | none needed |
| B7 | PASS | `finetune.py:313-335`: best epoch on fold held-out patients; the only val-label reader on the model path is `val_score.py:253-260`; val curves scored after training. | none needed |
| B8 | PARTIAL | budget commits precede their transformer runs; two bundle attempts recorded. Gaps: the step-2 baselines also scored val (28 and 29 October) and the report said "two val runs"; round 3, which added LightGBM, opened after attempt 1 had shown LightGBM ahead on val mAP. | fixed in the report, `model_results.md`, README, D37: three reads of the val labels; attempt 2 described as a second look at the same patients, the hidden test as the independent check |
| B9 | PASS | transformer OOF is each fold's held predictions; nested stacker `stacking.py:95-103`; adopted `logit_mean` has no parameter; submitted Platt weights fitted on train OOF only. Caveat: Platt fitted on the 3-model OOF blend with 80%-data LightGBM, applied to the 15-member mean with 100%-data LightGBM. | the mismatch is now documented in D38 with a check: test-time pre-Platt logits match the OOF distribution (mean shift +0.035, no target beyond 0.25, sd ratio 0.83 to 1.06; `checks/platt_shift.json`) |
| B10 | PARTIAL | the test path (`predict.py`, `submission.py`, `hf_package.py`) opens no labels; rebuild difference 0.0. Gaps: `train_baseline.py:102` reads `labels.parquet` unfiltered and scores val (the step-2 script); `lightgbm_final fit` writes the test CSVs in the same process that reads train labels. | a test now pins the set of modules that name the label file (`tests/test_audit_boundaries.py`) and documents each; the step-2 script is not on the submission path and its outputs are frozen comparison files, so it was left as is. The LightGBM layout is accepted: test labels do not exist, and the boosters saved by `fit` reproduce the CSVs (`lightgbm_final predict`, used by the package verifier) |
| B11 | PASS | six mutations on a scratch copy each fail a test (`auditor_scripts/e_mutations.sh`); future-record tests pass. | none needed |
| B12 | PARTIAL | split dependence measured (genotype files for 14.2% of train, 0.6% of val and test); presence is an input; dependence of predictions measured (D34). Gap: the outcome correlation was asserted without a committed number. | fixed: `scripts/checks/modality_presence.py` writes `reports/dataset/modality_presence.json` (2.11 against 1.83 positives per patient for genotype files, 2.35 for imaging; 9 of 40 targets at p < 0.01); cited in the report and results |
| B13 | PASS | `parity.py:27-48` and `reports/dataset/dataset_report.json`; the auditor's recomputation agrees. | none needed |
| B14 | PASS | prior from pre-anchor conditions only, equality enforced (`contracts.py:198-205`); `PredictionInputs` has no label field. | none needed |

### C. Evaluation rigour

| ID | Verdict | Evidence (short) | Disposition |
|---|---|---|---|
| C1 | PARTIAL | `KFold(5, shuffle, seed 20261028)` (`folds.py:19-36`); hash-verified file used by every model; pooled and mean-fold both reported. Gap: stratification (none) and the seed were not stated in the report. | fixed: the protocol paragraph states plain K-fold, the seed, no stratification, and the rarest target's 4 to 6 positives per fold |
| C2 | PASS | patient-indexed BCa (`evaluation.py:135-159`), paired resamples shared; an independent scipy BCa reproduces the committed intervals exactly. | none needed |
| C3 | PARTIAL | incidence, logistic regression (8-point grid), LightGBM (18-point grid) on the same folds. Gap: the LightGBM choice sits on the grid edge on all three axes in both runs and the grid was not extended. | accepted: extending the grid is a model change (new budget, refreeze, third val read); stated in the report's limits and in `model_results.md` |
| C4 | PARTIAL | budgets committed before their transformer runs; rules recomputed from `run.json`: T1, C1, S, C pass, the 4x192 screen fails; final choice follows the rules. Gaps: `budget_v3.toml` was committed while the stacking job was running and after part of its output existed; the round-2 component rule was written after two scores (disclosed in D36); the optimism estimate had no source; "24 trials" against 28 configurations on disk. | fixed in D38 and the report: the true sequence (stacking output from 21:04, commit 21:18:44, summary 21:19:52) is stated and rules S and C are described as written with point estimates possibly in view; the count is 28 configurations; the optimism estimate is now sourced (CV 0.780 mean fold / 0.774 pooled against val 0.767; epoch effect +0.008 from `val_curves.csv`) |
| C5 | PARTIAL | two bundle attempts with complete artifacts; transformer predictions byte-identical across attempts; no test score anywhere. Gaps: the step-2 val scoring was not in the report's tally; comparison models carried no single-model intervals. | fixed: tally corrected; single-model BCa intervals for the four comparison models computed after the bundle from its saved files (`reports/val/post_hoc_single_model_intervals.json`) and shown in the report, README, and results |
| C6 | PASS | per-target file (six models), era and modality subgroups recomputed exactly; near-chance targets explained. | counts corrected under C10 |
| C7 | PASS | val Brier per model; nested Platt: ECE 0.0128 to 0.0103, Brier lower on 4 of 5 folds, per-fold AUROC unchanged; reliability table and figure. | none needed |
| C8 | PASS | AUROC and AP tested against scikit-learn with ties; unweighted mean; all-negative targets handled; mask applied inside `evaluate_predictions` for every model. | none needed |
| C9 | PARTIAL | per-step and per-epoch curves in `metrics.jsonl` and synced wandb runs; run ids in the manifest; project link in the report. Gaps: the project is not publicly readable; no per-member run names in the report. | the report now names the 15 member runs; making the project public is the author's step (checklist item 7) |
| C10 | PARTIAL | about 60 numbers agree with their source files. Disagreements: "nine targets below 0.6 for every model" (9 for the blend, 7 for the three main models, 5 for all six), "seven above 0.95 for every model" (7 for the blend, 6 for the three main models), "nine targets with at most eight positives" (11), "193 unit tests" (195 collected), "24 trials" (28), and stacking intervals quoted as 1,000 resamples but computed with 200. | fixed: every count corrected in the report, results, model card, and AI-workflow note; the stacking report was regenerated with 1,000 resamples (the `make stacking` default; the three out-of-fold files were byte-identical to the frozen ones) and the quoted intervals refreshed: at three decimals the AUROC intervals are unchanged (blend minus transformer +0.014 [+0.009, +0.019]; minus LightGBM counts +0.011 [+0.007, +0.014]) and the mAP intervals move by at most 0.002 |
| C11 | PASS | the nested logistic stacker reproduced to 1e-13; logit-mean equals the mean of base logits; a non-nested fit would read 0.788 against the reported 0.769; Platt nested. | none needed |

### D. Representation, model, and training

| ID | Verdict | Evidence (short) | Disposition |
|---|---|---|---|
| D1 | PASS | `tokenizer.py:199-207` fit on fit patients; UNK per source; vocabularies 1,909 to 1,935 with unknown rates reported. | none needed |
| D2 | PASS | quantile bins (at most 10) on fit values (`numeric.py:35-40`); missing-value fallbacks; LightGBM fit-median fills. | none needed |
| D3 | PASS | per-token age features plus rotary positions and a recency feature; keep-most-recent-days policy with history summary tokens; retention measured. | nit fixed: the held-out token-loss fraction (24.6%, fold 0) now appears beside the fit-set 15 to 18% |
| D4 | PASS | 737,984 parameters instantiated on CPU (the checkpoint's 739,915 includes the output-mask buffer); 4x192 screen (2,255,040 parameters) scored 0.0144 lower on fold 0. | none needed |
| D5 | PASS | masked BCE with mask = 1 - prior (`training.py:327-332`); next-token cross-entropy with input-only masking and 5x target weight. | none needed |
| D6 | PASS | every value in `v2_c1.toml:22-54` and the report; code defaults: fine-tune betas (0.9, 0.999), constant fine-tune lr, stopping metric. | the code defaults are now stated in the report's hyperparameter paragraph |
| D7 | PASS | seeds keyed by (seed, epoch) (`training.py:445-450`); resume tests pass; seed spread reported. | the report now says members trained on MPS and the resume test runs on CPU |
| D8 | PASS | per-epoch train and held-out curves in every `metrics.jsonl` (fold 0: 0.880 train against 0.772 held at the best epoch); dropout 0.3 (+0.0071, 5 of 5 folds), early stopping, extra anchors. | none needed |
| D9 | PASS | 3 seeds x 5 folds averaged; blend parameter-free; Platt from train OOF; val only applies the frozen file. | none needed |
| D10 | PASS | S/H/F on the same folds and budget; F minus S +0.0203 [+0.0126, +0.0284]; negative pretraining results reported. | the report now says the comparison was made on `small-v1` and `small-v1-mm` and S was not re-run for the final recipe |
| D11 | PASS | genotype vocabulary from fit carriers; per-volume standardisation; CNN trained with the body on fit rows; with/without comparison +0.0005 [-0.0060, +0.0065]. | the report now frames the modalities as a test with a null result, kept |
| D12 | PARTIAL | tests cover the causal mask, padding, and the readout; none covered position encoding or the output shape directly (the auditor's probe confirmed both). | fixed: three tests added (`tests/test_model.py`: rotary identity at position 0 and relative-shift dependence; event order changes the state at equal ages; output shape) |

### E. Reproducibility and engineering

| ID | Verdict | Evidence (short) | Disposition |
|---|---|---|---|
| E1 | PASS | lock file pinned; Python 3.11+ stated, CI on 3.12; `make setup` proven by CI. | nit: unrelated packages in the lock; left (CI installs from it) |
| E2 | PARTIAL | all documented targets exist. Gap: `reports/lightgbm_clean/`, which `make stacking` and `make lightgbm-final` need, came from an undocumented `lightgbm_cv` run with `--exclude-observations DALY QALY QOLS --output-name lightgbm_clean`. | fixed: `make lightgbm-clean` added and cited in the README, the results, and the report's reproduction paragraph |
| E3 | PARTIAL | inference reproduced to 0.0 in two records and the auditor's own run. Gap: the fresh-environment record carried no environment provenance. | fixed: `hf_package verify` now records the interpreter version, platform, prefix, and lock-file hash; the earlier records predate the field |
| E4 | PARTIAL | 390 of 390 manifest hashes match; freeze and score commits exist. Gap: no record of the hash of `predictions.csv` or the PDF. | fixed: `make submission-record` writes `reports/final/submission_record.json` (hashes, sizes, commit) |
| E5 | PASS | CI green on the pushed branch; 195 tests locally (199 after this branch's additions); coverage of boundaries, labels, metrics, tokenizer, model, resume listed by the auditor. | none needed |
| E6 | PARTIAL | ruff and mypy pass under the repository configuration; `mypy --strict` reports 33 errors. | accepted and stated in the checklist: the configuration is not `--strict` |
| E7 | PASS | no secrets; one machine path in a handoff note; ignore rules cover data, models, logs, wandb; repository 16.6 MiB. | the machine path is removed; nit: 58 raw synthetic medication rows in `reports/dataset/omitted_stops.csv`, left |
| E8 | PASS | `select_device` auto (MPS else CPU); CPU used by scoring, verification, and CI. | nit: personal `WANDB_ENTITY` default, left (checklist) |
| E9 | PASS | `metrics.jsonl` and `run.json` per run; wandb names `{config}-fold{k}-{F|pretrain}`; all 202 offline sessions synced. | none needed |
| E10 | PASS | 125 files, 51 MB; card states data, use, metrics by split, how to run; the loader is the one the verifier exercises. Gaps: `license: mit` without a LICENSE file; not uploaded. | LICENSE added; upload is the author's (checklist item 6) |
| E11 | FAIL | no LICENSE file; no acknowledgement of CC BY 4.0 or MITRE anywhere. | fixed: `LICENSE` (MIT, with the dataset notice) and a data-and-licence section in the README and the model card |
| E12 | PASS | module table matches the package; a missing dataset raises a clear error. Nits: three modules absent from the README table; `full_timeline` imported unconditionally. | the three modules are in the table; the import is left |

### F. Documentation and honesty

| ID | Verdict | Evidence (short) | Disposition |
|---|---|---|---|
| F1 | PASS | four sections answer the four questions; wandb link; "test scores are the organisers'" stated; numbers labelled by split. | none needed |
| F2 | PARTIAL | 25+ numbers traced exactly. Gaps: the counts listed under C10; the clock-scan figures (0.999, 0.865, 1,155 of 1,219, 2.37 against 1.48) had no committed output. | fixed: counts corrected; `reports/dataset/clock_scan.txt` committed (0.999 with the codes, 0.876 without on fold 0; D36's 0.865 was the earlier run) |
| F3 | PASS | failed trials of all rounds, the null modality result, the clock-code cost, near-chance targets, a limits paragraph. | none needed |
| F4 | PARTIAL | D1 to D8 and D31 to D38 with evidence. Gap: ten entries undated. | fixed: a dated index (first commit of each entry) at the top of `docs/decisions.md` |
| F5 | FAIL | `docs/report.md:4` author placeholder; `report.md:228` Codex note; bracketed notes in `docs/ai_workflow.md`. | open, author (checklist items 1 and 2) |
| F6 | FAIL | README called `small-v1-mm` the submitted model and "the final model is a transformer", contradicting `configs/final.toml`; `data/README.md` said the modalities are unused. | fixed: README status rewritten; `data/README.md` corrected |
| F7 | FAIL | alias on GitHub and Hugging Face; wandb entity carries the real name; "Derek" in budget comments and decision log; a machine path in a note; placeholder author on the PDF. | partly fixed: machine paths removed (note, audit copies); the wandb entity and first name are the author's decision, recorded in the checklist with the trade-off |
| F8 | PASS | leak claims cite tests and audits; differences carry intervals; no test score. Nit: the CV blend row is the pre-Platt value. | the pre-Platt note is added in the report, results, and model card |

### G. Final gates

| ID | Verdict | Evidence (short) | Disposition |
|---|---|---|---|
| G1 | PASS | validator (`auditor_scripts/validate_predictions.py`): `patient_id` first; 358 rows = test split = `test_anchors.csv`; 40 code columns in file order; numeric; no NaN; [0.0, 0.912]; no duplicates. | none needed |
| G2 | PASS | verifier at `812393e`: `agrees: true`, difference 0.0, 358x40, CPU; `predictions.csv` unchanged since commit `49a1e33`. | none needed |
| G3 | FAIL | PDF has the four sections but the two placeholders and the name `docs/report.pdf`. | open, author (checklist items 1 to 4); the PDF was rebuilt from the corrected text |
| G4 | PARTIAL | repository private, Hugging Face 401, wandb not anonymously readable; recorded only as "publishing waits". | fixed: per-item status and steps in `docs/submission_checklist.md` |
| G5 | FAIL | no single list of what remains. | fixed: `docs/submission_checklist.md` |

## Appendix: outside the criteria (not scored)

Items the auditors raised that no criterion covers, kept here so a reader has them:

* The released test tables keep the clock codes the model drops (1,081 DALY rows), so removal can
  lower the hidden score; disclosed in the report.
* The organisers' imaging folder is truncated at the anchor for test patients (28 pre-anchor
  scans, 0 after) but not for train and val (34 and 11 post-anchor scans); the pipeline's date
  rule covers it.
* Anchors span 1916 to 2016 in every split, so "anchor before 2015" mostly means the patient
  died; the residual era signal (AUROC 0.876 on fold 0 without the clock codes) is disclosed.
* `full_timeline.py` (post-anchor pretraining, rejected in D36) exists; no submitted config
  uses it.
* A `[DNA]` presence token reaches two test patients.
* Pretraining never stopped early (best epoch 29 or 30 of 30 in every parent).
* Within-fold epoch selection makes every out-of-fold comparison slightly optimistic in the same
  direction for every model (about 0.003, D36).
* The package verifier ran in 29 seconds on CPU, not the "7 minutes" the model card said
  (corrected).
* Stored BCa warnings in the stacking report concern auxiliary eligibility counts whose intervals
  are unused.
