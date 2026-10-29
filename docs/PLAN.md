# Plan

Goal: predict, for every test patient, the probability that each of 40 target conditions is
diagnosed for the first time in the five years after the patient's anchor date. The pipeline
must be leak-free, use the provided splits, and train only on the training split.

## Steps

| Step | Content | Status |
|------|---------|--------|
| 1 | Anchors, leak-free history, labels (`ehrforecast.build_dataset`) | done |
| 2 | Logistic-regression baseline on code counts plus age and sex (`ehrforecast.train_baseline`) | done |
| 3 | Tokenizer: events to token sequences, time-gap tokens, lab-value bins | planned |
| 4 | Small decoder transformer: next-event pretraining, then multi-label fine-tuning, wandb | planned |
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
| Long timelines (median 213 pre-anchor events, p90 635, max several thousand) | Step 3: group events by encounter, collapse repeated identical observations and medication refills, keep demographic prefix tokens plus the most recent tokens up to the context length (left truncation). Measure the token-length distribution before choosing the context length. | step 3 |
| Genomics download incomplete, almost no test coverage (2 of 358) | Not used. Stated in the report. | report |
| Imaging DICOM files: 9.2 GB, 28 test patients | DICOM files not used. Imaging study events from `imaging_studies.csv` are used. | `schema.py` |
| Little data for transformer pretraining | Small model (a few million parameters), dropout, weight decay, early stopping on val loss. Optional extra anchors per training patient, kept only if val improves. The baseline stays in the report as the bar to beat, with paired bootstrap comparison. | step 4 |

## Step 3 to 5 design notes

* One input unit is one event token, `source:code`. Numeric observations add a value-bin token
  (per-code deciles computed on training data only). Text observations add a value token.
* Time: a gap token between consecutive events, from a fixed set of buckets (same day, under a
  week, under a month, under 3 months, under 6 months, under a year, under 2 years, longer). Age
  at anchor and sex are prefix tokens.
* The vocabulary is fitted on training patients only; rare tokens map to `[UNK]`.
* Pretraining uses training patients only. Whether to pretrain on full training timelines or on
  pre-anchor history only is decided on val loss; neither option touches val or test data.
* Fine-tuning: a 40-way sigmoid head on the final position of the pre-anchor sequence, binary
  cross-entropy. Checkpoint selection on val macro AUROC.
* Submission check script: 358 rows, IDs equal `test_anchors.csv`, 40 columns named and ordered
  as `target_conditions.csv`, probabilities in [0, 1], no missing values.

## Open decision

* Name on the PDF report and the GitHub / Hugging Face account to publish under.
