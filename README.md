# ehr-forecast

Predicts which of 40 conditions a patient will be diagnosed with for the first time in the five
years after a landmark ("anchor") date, using only the synthetic EHR events recorded before that
date. Data: the Synthea Coherent dataset as released for the research intern coding test.

Status: step 1 (anchors, leak-free history, labels) and step 2 (logistic-regression baseline) are
done. The sequence model comes next. See `docs/PLAN.md`.

## Setup

```bash
make setup                      # creates .venv and installs the package with dev tools
ln -s /path/to/research_intern_coding_test data   # folder with patient_splits.csv etc.
```

## Run

```bash
make check      # ruff lint + format check, mypy, pytest
make dataset    # step 1 -> artifacts/dataset/
make baseline   # step 2 -> artifacts/baseline/
```

Step 1 takes about 20 seconds and step 2 about 3 minutes on a laptop.

## How leakage is prevented

* Train and val timelines are cut at the anchor with the same rules the released test files
  follow: rows dated before the anchor day are kept, and a `STOP` on or after the anchor is
  blanked. `docs/DECISIONS.md` shows how these rules were checked against the test files.
* The pipeline stops with an error if any table still holds a date on or after the anchor.
* Columns that summarise the whole record (death date, lifetime costs, medication dispense
  totals, marital status, device UDI) are never read.
* Labels come from the full timeline, but only after the history has been cut, and only for
  train and val patients.
* Model choices use cross-validation on train. Val is scored once, at the end.

## Results so far (val, 365 patients)

| Model | Macro AUROC | mAP |
|---|---|---|
| Age and sex only | 0.698 [0.675, 0.722] | 0.113 [0.102, 0.118] |
| Code counts + age and sex | 0.756 [0.735, 0.775] | 0.217 [0.185, 0.225] |

95% BCa bootstrap intervals. Details and a leakage review: `docs/BASELINE.md`.

## Layout

```
src/ehrforecast/
  config.py          dataset layout and constants
  schema.py          column allowlists per table
  io.py              CSV reading and the single timestamp parser
  cohort.py          splits, anchors, per-patient cohort table
  truncation.py      cut timelines at the anchor
  events.py          long event table
  labels.py          labels and prior-condition flags
  parity.py          train/val versus test comparison statistics
  build_dataset.py   step 1 command
  features.py        bag-of-codes features
  baseline.py        multi-label logistic regression
  metrics.py         AUROC, AP, BCa bootstrap
  train_baseline.py  step 2 command
tests/               unit tests on small hand-made tables
docs/                plan, decisions with evidence, baseline results
```
