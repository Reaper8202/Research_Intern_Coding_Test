# ehr-forecast

Predicts which of 40 conditions a patient will be diagnosed with for the first time in the five
years after a landmark ("anchor") date, using only the synthetic EHR events recorded before that
date. Data: the Synthea Coherent dataset as released for the research intern coding test.

Status: step 1 (anchors, leak-free history, labels) and step 2 (logistic-regression baseline) are
done. The sequence model comes next. See `docs/plan.md`.

## Project structure

```
ehr-forecast/
├── data/
│   ├── README.md            how to provide the dataset
│   ├── raw/                 task dataset, unchanged (not committed)
│   └── processed/           step 1 output (not committed)
├── docs/
│   ├── plan.md              steps and where each known risk is handled
│   ├── decisions.md         every data decision with the evidence for it
│   └── baseline_results.md  step 2 results and leakage review
├── models/                  trained models (not committed)
├── reports/
│   ├── dataset/             step 1 checks: train/val/test parity, label prevalence
│   └── baseline/            step 2 metrics, per-condition table, CV results, val predictions
├── logs/                    run logs (not committed)
├── src/ehrforecast/         Python package
├── tests/                   unit tests on small hand-made tables
├── Makefile                 setup, checks, and pipeline commands
├── pyproject.toml           package metadata and tool settings
└── requirements-lock.txt    exact dependency versions
```

## Setup

Python 3.11 or newer.

```bash
make setup                                        # .venv with locked dependency versions
ln -s /path/to/research_intern_coding_test data/raw
```

## Run

```bash
make check      # ruff lint + format check, mypy, pytest
make dataset    # step 1: data/raw -> data/processed, reports/dataset
make baseline   # step 2: models/baseline, reports/baseline
```

Step 1 takes about 20 seconds and step 2 about 2 minutes on a laptop. `make clean` removes
everything that these commands generate, except the committed reports.

## How leakage is prevented

* Train and val timelines are cut at the anchor with the same rules the released test files
  follow: rows dated before the anchor day are kept, and a `STOP` on or after the anchor is
  blanked. `docs/decisions.md` shows how these rules were checked against the test files.
* The pipeline stops with an error if any table still holds a date on or after the anchor.
* Columns that summarise the whole record (death date, lifetime costs, medication dispense
  totals, marital status, device UDI) are never read.
* Labels come from the full timeline, but only after the history has been cut, and only for
  train and val patients.
* Model choices use cross-validation on train. Val is scored once, at the end.

## Results so far (val, 365 patients)

| Model | Macro AUROC | mAP |
|---|---|---|
| Age-sex incidence table | 0.692 [0.667, 0.716] | 0.111 [0.097, 0.116] |
| Age-sex incidence table, at-risk patients only | 0.727 [0.703, 0.749] | 0.141 [0.120, 0.148] |
| Logistic regression, age and sex | 0.698 [0.675, 0.722] | 0.113 [0.102, 0.118] |
| Logistic regression, code counts + age and sex | 0.756 [0.735, 0.775] | 0.217 [0.185, 0.225] |

95% BCa bootstrap intervals. The code-count model beats the at-risk incidence table by +0.029
macro AUROC [+0.003, +0.054] (paired bootstrap). Details, per-condition analysis, and a leakage
review: `docs/baseline_results.md`.

## Package modules

| Module | Purpose |
|---|---|
| `config.py` | dataset layout and constants |
| `schema.py` | column allowlists per table |
| `io.py` | CSV reading and the single timestamp parser |
| `cohort.py` | splits, anchors, per-patient cohort table |
| `truncation.py` | cut timelines at the anchor |
| `events.py` | long event table |
| `labels.py` | labels and prior-condition flags |
| `parity.py` | train/val versus test comparison statistics |
| `build_dataset.py` | step 1 command |
| `features.py` | bag-of-codes features |
| `incidence.py` | sex- and age-stratified incidence baseline |
| `baseline.py` | multi-label logistic regression |
| `metrics.py` | AUROC, AP, BCa bootstrap intervals, paired model comparison |
| `train_baseline.py` | step 2 command |
| `logging_utils.py` | console and file logging for the commands |
