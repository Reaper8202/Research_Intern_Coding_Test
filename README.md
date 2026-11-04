# ehr-forecast

For each of 40 conditions, the probability that a patient is diagnosed with it for the first
time in the five years after an anchor date, predicted from the synthetic EHR events recorded
before that date. Built for the research intern coding test on the Synthea Coherent dataset.

The submission is a fixed blend of three models, each trained from random weights on the 2,791
training patients: fifteen small decoder-only transformers (2 layers, width 128, about 738k
parameters each; three seeds times five cross-validation folds, self-pretrained on pre-anchor
event sequences and then fine-tuned on the 40 labels) and two LightGBM models on code counts,
one of them with the latest value of each numeric observation. Their logits are averaged, and a
per-target Platt map fitted on training out-of-fold predictions scales the result (decision
D38). On the official validation split the blend scores macro AUROC 0.767 [0.746, 0.788] and
mAP 0.258 [0.234, 0.268]; the transformer ensemble alone scores 0.751 / 0.209 and LightGBM on
counts alone 0.762 / 0.239. Test labels are hidden; the test score is the organisers'.

The transformer is the part this project built and studied. It went through a comparison of
three training methods (supervised from scratch, a fixed self-pretrained body, full fine-tuning;
the last won on every fold), a multimodal test (genotype tokens and one brain-MRI token: no
measurable gain, kept by decision D34), and three tuning rounds whose trial lists and adoption
rules were committed before the runs (`configs/budget*.toml`; D35, D36, D38). Round 2 removed
three simulator scores (DALY, QALY, QOLS) that encode the anchor year. No external data or
pretrained weights were used. Official val was read three times in all, each read recorded: the
step-2 baselines once, then the frozen bundle twice.

## The four deliverables

| Deliverable | Where |
|---|---|
| Technical report | [`docs/report.pdf`](docs/report.pdf), source [`docs/report.md`](docs/report.md) |
| Test predictions | [`predictions.csv`](predictions.csv): 358 test patients, one column per target code |
| Trained models | the package `models/hf`, built by `make package` and published at <https://huggingface.co/Reaper8202/ehr-forecast> |
| Code | this repository. `make check` runs lint, the type check, and 199 tests; CI runs the same on every push |

Training, tuning, and val-curve runs are logged in the wandb project linked from the report.

## Where to look

| To see | Open |
|---|---|
| what was decided, why, and the check behind each decision, in order | [`docs/decisions.md`](docs/decisions.md) |
| every result with its interval: training CV, both val runs, per-target tables | [`docs/model_results.md`](docs/model_results.md) |
| how leakage is prevented | the section below, then `src/ehrforecast/truncation.py` and `tests/test_history_boundary.py` |
| the final audit: 74 criteria fixed before the examination, verdicts with evidence | [`docs/audit_criteria.md`](docs/audit_criteria.md), [`reports/audit/final_audit.md`](reports/audit/final_audit.md) |
| how AI coding tools were used | section 3 of the report |
| the plans as written before the code | [`docs/planning/`](docs/planning/) |
| what each committed result file and config is | [`reports/README.md`](reports/README.md), [`configs/README.md`](configs/README.md) |

## Results on official val

365 patients, 685 positive labels, scored by the frozen bundle. Two attempts, both recorded:
attempt 1 (`reports/val/attempts/168b78f5b5d8/`) scored the transformer ensemble alone;
attempt 2 (`reports/val/`) scored the round-3 blend after it was adopted on training CV. The
step-2 baselines had scored val once before the bundle existed (28 and 29 October), so the val
labels were read three times in all. All models use the common prior-condition mask; intervals
are 95% patient BCa bootstrap intervals (the comparison models' own intervals were computed after
the bundle from its saved files, `reports/val/post_hoc_single_model_intervals.json`).

| Model | Macro AUROC | mAP | Brier |
|---|---|---|---|
| **Submitted blend**: transformer ensemble + LightGBM counts + LightGBM counts and values, mean of logits, Platt-scaled | **0.767** [0.746, 0.788] | **0.258** [0.234, 0.268] | 0.0365 |
| Transformer ensemble (`small-v2-c1`, 3 seeds x 5 folds) | 0.751 [0.730, 0.773] | 0.209 [0.192, 0.211] | 0.0381 |
| LightGBM counts (CV-selected setting, fitted on all of train) | 0.762 [0.740, 0.781] | 0.239 [0.209, 0.249] | 0.0370 |
| LightGBM counts + values | 0.758 [0.740, 0.780] | 0.257 [0.228, 0.267] | 0.0367 |
| Logistic regression, code counts + age and sex (step 2) | 0.764 [0.742, 0.782] | 0.222 [0.195, 0.229] | 0.0384 |
| Age-sex incidence table, at-risk patients only (step 2) | 0.727 [0.705, 0.751] | 0.141 [0.125, 0.146] | 0.0390 |

Paired differences, blend minus: transformer ensemble +0.016 AUROC [+0.002, +0.030] and +0.049 mAP
[+0.033, +0.072]; LightGBM counts +0.005 AUROC [-0.005, +0.016] and +0.019 mAP [+0.002, +0.040].
Training CV said the same (`docs/decisions.md`, D38). Per-target and subgroup results:
`docs/model_results.md`.

## Repository layout

```
ehr-forecast/
├── README.md
├── predictions.csv          the submission: 358 test patients x 40 target codes
├── Makefile                 setup, checks, and every pipeline command
├── pyproject.toml           package metadata and tool settings
├── requirements-lock.txt    exact dependency versions
├── LICENSE                  MIT, with the dataset notice
├── configs/                 training configs, the three tuning-round budgets, the frozen final bundle
├── data/                    how to provide the dataset; raw/ and processed/ are not committed
├── docs/                    report, results, decision log, model card, audit criteria, plans
├── reports/                 every committed result file, by pipeline stage
├── scripts/                 one-off data checks behind D15, D31, and D36; the submission record
├── src/ehrforecast/         the Python package
├── tests/                   199 tests on small hand-made and synthetic tables
├── .github/workflows/       CI: lint, format check, type check, tests
├── models/                  trained models and the public package (not committed)
└── logs/, wandb/            run logs (not committed)
```

Each of `docs/`, `reports/`, `configs/`, and `scripts/` has its own README listing its files.

## Setup

Python 3.11 or newer.

```bash
make setup                                        # .venv with locked dependency versions
ln -s /path/to/research_intern_coding_test data/raw
```

On macOS, LightGBM needs the OpenMP runtime: `brew install libomp`.

## Reproducing the submission

The full sequence, in order. One transformer fold (pretraining, then fine-tuning) takes about
20 minutes on a 16 GB M-series Mac; the 15 members took about 4.5 hours in all.

```bash
make check                                        # ruff lint + format check, mypy, pytest
make dataset && make modalities && make folds && make tokenize CONFIG=configs/v2_base.toml
# the 15 transformer members: seed 1 fine-tunes from the v2_ptw_pre pretraining; seeds 2 and 3
# pretrain under their own configs
for f in 0 1 2 3 4; do
  make pretrain CONFIG=configs/experiments/v2_ptw_pre.toml FOLD=$f
  make finetune CONFIG=configs/experiments/v2_c1.toml FOLD=$f METHOD=F
  for s in seed2 seed3; do
    make pretrain CONFIG=configs/experiments/v2_c1_$s.toml FOLD=$f
    make finetune CONFIG=configs/experiments/v2_c1_$s.toml FOLD=$f METHOD=F
  done
done
make lightgbm && make lightgbm-clean              # LightGBM comparison models on the training folds
make stacking && make calibration                 # round 3: nested-fold blend and Platt scaling (CV only)
make lightgbm-final                               # the two LightGBM members on all training patients
make freeze && make val-score                     # hash every artifact; one recorded val run
make predict && make submission                   # transformer test probabilities, then predictions.csv
make package                                      # public package in models/hf, then the clean-reload check
```

The package carries the 15 transformer checkpoints, 5 tokenizers, 80 LightGBM boosters with
their feature specs, and the Platt weights. `python -m ehrforecast.hf_package verify` rebuilds
`predictions.csv` from it on CPU and reproduced it exactly, in the development environment and
in a fresh virtual environment built from `requirements-lock.txt`
(`reports/final/package_verify_*.json`).

## The pipeline, step by step

Steps 1 and 2 (data and reference baselines):

```bash
make dataset    # step 1: data/raw -> data/processed, reports/dataset (about 20 s)
make baseline   # step 2: logistic regression and incidence tables, reports/baseline (about 2 min)
make folds      # save and verify the five training folds
make preflight  # training-only audits and baseline reproduction; does not score val (about 20 s)
make modalities # genotype files and DICOM volumes -> data/processed (about 20 s)
make tokenize   # fold tokenizers, patient histories, reports/representation (about 45 s)
make pipeline-smoke   # tiny end-to-end run on fold 0, temporary outputs (about 20 s)
```

Transformer training, one fold at a time (settings in `configs/transformer.toml` unless `CONFIG`
is given):

```bash
make pretrain FOLD=0              # self-pretraining on fit patients' pre-anchor histories
make finetune FOLD=0 METHOD=S     # S: random init; H: fixed pretrained body; F: full fine-tuning
make pretrain FOLD=0 RESUME=1     # continue an interrupted run from its last completed epoch
make cv-summary                   # after all five folds: fold table, pooled metrics, method selection
make wandb-sync                   # upload finished offline runs to wandb
```

The multimodal model (D34) uses `CONFIG=configs/multimodal.toml` for `tokenize`, `pretrain`,
`finetune`, and `cv-summary` (with `REFERENCE=configs/transformer.toml` to compare it with
`small-v1`), and `SMOKE_CONFIG=configs/smoke_mm.toml` for the smoke test. Genotypes enter as
`[DNA] dna:<site>:<variant alleles>` tokens. A scan enters as one `[IMG]` token only when its
study date is before the anchor day. Coverage is small: genomics for 396 train and 2 test
patients, and pre-anchor scans for 200 train and 28 test patients
(`reports/dataset/modalities.json`).

The tuning rounds (training CV only; official val is never read):

```bash
make experiments                  # round 1, configs/budget.toml in order; resumable
make experiments-summary          # reports/transformer/experiments.csv with the adoption rule applied
PYTHONPATH=src python -m ehrforecast.experiments run --budget configs/budget_v2.toml \
    --dataset data/processed --models models/transformer --logs logs/round2        # round 2
PYTHONPATH=src python -m ehrforecast.experiments summarize --budget configs/budget_v2.toml \
    --models models/transformer --reports reports/transformer/round2
```

Round 3 (`configs/budget_v3.toml`, D38) is `make stacking`, `make calibration`, and the
architecture screen `make pretrain` and `make finetune` with `CONFIG=configs/experiments/v3_l4w192.toml`.

Step 5. `make freeze` records every artifact, data, code, and lock-file hash in
`reports/final/manifest.json`. `make val-score` verifies them, refuses a changed tracked tree,
scores the submission, the transformer ensemble, the step-2 baselines, and the frozen LightGBM
files on val, then scores every saved epoch checkpoint of the members as val curves logged to
wandb. A second run needs `--replacement` with a documented reason; each earlier attempt is kept
under `reports/val/attempts/`. The transformer prediction runs 15 models and takes about 6
minutes on CPU.

Each training run writes `models/transformer/<config>/fold<k>/<run>/` with `best.pt`, `last.pt`,
`metrics.jsonl`, and `run.json`. wandb logs offline; `make wandb-sync` uploads finished runs to
the team `derek-wu820-university-of-toronto`, project `ehr-forecast` (set `WANDB_ENTITY` to use
another space). `make cv-summary` writes `reports/transformer/<config>/`; its bootstrap intervals
take about 12 minutes. `python -m ehrforecast.predict` scores held-out training patients or the
test split from saved fold models; it reads no labels and refuses official val. `make clean`
removes everything these commands generate, except the committed reports.

## How leakage is prevented

* Train and val timelines are cut at the anchor with the same rules the released test files
  follow: rows dated before the anchor day are kept, and a `STOP` on or after the anchor is
  blanked. `docs/decisions.md` (D1, D2) shows how these rules were checked against the test files.
* The pipeline stops with an error if any table still holds a date on or after the anchor.
* A `STOP` dated before its own `START` (58 medication rows) is omitted after the cut; the start
  is kept (D31). The prediction loader rejects event files built without this rule.
* Columns that summarise the whole record (death date, lifetime costs, medication dispense
  totals, marital status, device UDI) are never read (D3).
* Labels come from full condition records for train and val patients. Prediction inputs come
  from truncated histories, and the prediction interface never sees labels.
* Every fitted object (vocabulary, value bins, folds, pretraining windows, LightGBM vocabularies
  and medians, stacking weights, Platt maps) is fitted inside its fold on fit patients only.
* Model choices were made by cross-validation on train. Official val was read by a bundle that
  records each attempt and refuses to run on a changed tree.

## Data and licence

The copy of the dataset this submission was built from had an incomplete `genomics/` folder: the
organisers' download log marks it as not fully downloaded, and 400 genotype files were present
(396 train, 2 val, 2 test). `reports/dataset/modality_files.json` lists the genomics and imaging
files that were present. The genotype tokens, the modality coverage numbers, and the
modality-presence check in the report describe that copy. To reproduce `predictions.csv` exactly
from a complete download, restrict `genomics/` to the listed files before `make modalities`.

The dataset is not committed. It is derived from the Coherent Synthetic Data Set generated by
Synthea (Copyright 2022 The MITRE Corporation, MITRE Public Release Case Number 21-1917) and
released under the Creative Commons Attribution 4.0 International License. The code in this
repository and the published model weights are under the MIT licence (`LICENSE`).

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
| `contracts.py` | explicit model inputs; training-only label loading |
| `folds.py` | saved five-fold assignments of the training split |
| `masking.py` | prior-condition mask aligned by patient and target |
| `evaluation.py` | version-2 AUROC, AP, Brier score, and patient BCa intervals |
| `numeric.py` | unit audit and quantile bins for numeric observations |
| `provenance.py` | hashes, file checks, and atomic JSON output |
| `preflight.py` | training-only data audit and baseline reproduction |
| `modalities.py` | genotype and DICOM cache for the multimodal model (D34) |
| `tokenizer.py` | fold tokenizer fitted on fit patients only |
| `sequences.py` | patient histories, prediction sequences, pretraining windows |
| `representation.py` | `make tokenize` command and retention reports |
| `model.py` | decoder-only transformer and the 3-D CNN image encoder |
| `training.py` | configs, fold loading, batching, losses, checkpoints, logging |
| `pretrain.py` | self-pretraining (next-event prediction) |
| `finetune.py` | supervised methods S, H, and F |
| `predict.py` | label-free prediction from saved fold models |
| `smoke.py` | end-to-end execution check on tiny settings |
| `cv_summary.py` | five-fold results, method selection, config comparison |
| `experiments.py` | run and summarise a pre-registered tuning budget (`configs/budget*.toml`) |
| `lightgbm_cv.py` | LightGBM comparison baselines on the training folds (own process, no PyTorch) |
| `full_timeline.py` | the rejected full-timeline pretraining variant (D36); not used by the submission |
| `stacking.py` | round 3: nested-fold stacking of the transformer ensemble with LightGBM (CV only) |
| `calibration.py` | round 3: per-target Platt scaling, reliability tables, nested-fold check |
| `lightgbm_final.py` | final LightGBM models on all training patients; boosters, feature spec, val/test files |
| `submission.py` | the submitted probabilities: blend of the bases, then Platt scaling |
| `val_score.py` | artifact freeze, hash verification, the official val run, val curves |
| `hf_package.py` | build and verify the public model package (Hugging Face Hub) |
| `wandb_sync.py` | upload finished offline wandb runs |
