# Plan

Goal: predict, for every test patient, the probability that each of 40 target conditions is
diagnosed for the first time in the five years after the patient's anchor date. The pipeline
must be leak-free, use the provided splits, and train only on the training split.

## Steps

| Step | Content | Status |
|------|---------|--------|
| 1 | Anchors, leak-free history, labels (`ehrforecast.build_dataset`) | done |
| 2 | Baselines: age-sex incidence tables and logistic regression on code counts plus age and sex (`ehrforecast.train_baseline`) | done |
| 3 | Representation: folds, mask, extra baselines, tokenizer, sequences, leak tests (`model_plan.md` P3a–P3c) | planned |
| 4 | Decoder transformer: pretraining, fine-tuning, zero-shot, clinical scores, one val run (`model_plan.md` P4a–P4f) | planned |
| 5 | Test predictions, report, GitHub, Hugging Face | planned |

## Where each known risk is handled

### Must fix

| Risk | Handling | Where |
|------|----------|-------|
| Train history must look like the test files | Truncation copies the test rules: keep rows dated before the anchor day; blank any `STOP` on or after it. Hard check: zero values on or after the anchor in every table, both ours and the released test files. Parity report compares train/val/test history statistics. | `truncation.py`, `build_dataset.py`, `parity.py` |
| Death and cost columns hold future data | Every table is read through a column allowlist. Dropped: `DEATHDATE`, `HEALTHCARE_EXPENSES`, `HEALTHCARE_COVERAGE`, `MARITAL`, address fields, all cost and payer columns, `DISPENSES`, `TOTALCOST`, `devices.UDI`. | `schema.py` |
| Few positives for some targets | Bootstrap 95% intervals on every val metric. Per-target table with positive counts. `C` is chosen by cross-validation on train, not on val, so val stays an unbiased estimate. | `metrics.py`, `train_baseline.py` |
| Test labels are hidden | Val is the only held-out evaluation. The report labels val numbers as val and says why test numbers come from the organisers. | report (step 5) |
| Mixed timestamp formats | One parser (`parse_timestamps`) for every time column; all comparisons at day level against a midnight anchor. Unit tests cover both formats. | `io.py`, `tests/test_io.py` |

### Lower risk

| Risk | Handling | Where |
|------|----------|-------|
| Long timelines (median 213 pre-anchor events, p90 635, max several thousand) | Collapse same-day duplicates and medication refills; condition summary plus the most recent tokens up to 1,024 (left truncation). See `model_plan.md` 8.3–8.4. | step 3 |
| Genomics download incomplete, almost no test coverage (2 of 358) | Not used. Stated in the report. | report |
| Imaging DICOM files: 9.2 GB, 28 test patients | DICOM files not used. Imaging study events from `imaging_studies.csv` are used. | `schema.py` |
| Little data for transformer pretraining | Small model (a few million parameters), dropout, weight decay, early stopping on held-out CV folds (val is not used for choices, `model_plan.md` D9). Optional extra anchors per training patient, kept only if CV improves (E3). The baseline stays in the report as the bar to beat, with paired bootstrap comparison. | step 4 |

## Step 3 to 5 design

The full plan is in `model_plan.md` (decisions D9–D22, evaluation protocol, phases, experiment,
test, and risk registers). It replaces the earlier design notes here, which chose checkpoints on
val; that is no longer the protocol (D9).

## Open decision

* Name on the PDF report and the GitHub / Hugging Face account to publish under.
* Step 3 to 5 open items (wandb entity): `model_plan.md` section 18.
