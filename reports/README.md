# Reports

Every file here was written by a pipeline command and committed as the record of that stage.
Folders are listed in the order the pipeline produces them. A few are inputs to later stages:
the step-2 val prediction files (the val bundle scores two of them as comparison models), the
selected LightGBM settings in `lightgbm_clean/cv_summary.json`, the out-of-fold blend in
`stacking/clean/`, and the Platt weights in `calibration/stack/`. The frozen manifest in
`final/` hashes all of them.

| Folder | Written by | What is in it |
|---|---|---|
| `dataset/` | `make dataset`, `make modalities`, `scripts/checks/` | step-1 checks: train and val against the released test files (`dataset_report.json`), label prevalence, the 58 omitted medication stops (D31), modality coverage (D34), modality-file presence against outcomes, the list of genomics and imaging files present in the copy used (`modality_files.json`), the clock-code scan (D36), and the death-window table (`death_window_oof.csv`: per-target share of positives among pre-2015 anchors and out-of-fold AUROC within each era group) |
| `baseline/` | `make baseline` | step 2: the logistic-regression and incidence baselines, their training CV, val metrics with a per-condition table, and their val prediction files |
| `p3a/` | `make preflight` | the training-only audit before any transformer work: raw-data checks, unit rules, target support per fold, reproduction of one step-2 setting, and the input manifest |
| `representation/` | `make tokenize` | per-fold tokenizer and sequence reports (vocabulary, truncation, retention) for `small-v1`, `small-v1-mm`, and `small-v2-base`, the representation the submission uses |
| `transformer/` | `make cv-summary`, `make experiments-summary`, `make pipeline-smoke` | five-fold results per configuration: `small-v1/` (the S/H/F comparison), `small-v1-mm/` (the multimodal test, D34), `small-v1-mm-do03/` (round 1's pick, D35), `small-v2-c1/` (the submitted recipe, one seed, D36); the three tuning-round registers (`experiments.*`, `round2/`, `round3/`); the per-epoch training records of the 15 members (`member_curves.csv`, drawn as the report's Figure 1); the generation-deferral record; the smoke-test reports |
| `lightgbm/` | `make lightgbm` | the LightGBM comparison baselines on the training folds with the clock codes still in the inputs; superseded by `lightgbm_clean/` |
| `lightgbm_clean/` | `make lightgbm-clean` | the same without DALY, QALY, and QOLS (D36): the grid, the settings the submission uses, the comparison with the transformer ensemble, and the race check (D15) |
| `stacking/` | `make stacking` | round 3: nested-fold stacking of the transformer ensemble with the two LightGBM models. `clean/` is what the submission uses; `with_clock_codes/` is the diagnostic with the codes kept (D38) |
| `calibration/` | `make calibration` | per-target Platt scaling on out-of-fold predictions. `stack/` is the submission's map; `transformer/` is the same procedure on the transformer ensemble alone, not used by the submission |
| `lightgbm_final/` | `make lightgbm-final` | the two final LightGBM models fitted on all training patients: settings, and their val and test probability files |
| `final/` | `make freeze`, `make predict`, `make package`, `make submission-record` | the frozen manifest (every artifact, data, code, and lock-file hash), the transformer's test probabilities, the package verification records (development machine, fresh environment), and the submission record |
| `val/` | `make val-score` | the official val run of the submitted bundle (attempt 2): summary, per-target table, prediction files, and the val curve of every saved epoch checkpoint. `attempts/` keeps attempt 1 (the transformer ensemble alone). `post_hoc_single_model_intervals.json` holds the comparison models' intervals, computed after the bundle from its saved files |
| `audit/` | the final audit, 4 November 2026 | the verdicts with evidence (`final_audit.md`), the four auditors' reports as delivered (machine paths removed; file paths are as they were at commit `812393e`, before `docs/` was reorganised), their scripts, and two follow-up checks |

The eight `.parquet` files, about 1 MB each, hold out-of-fold probabilities for every training
patient. They are the inputs to the stacking and calibration checks.
