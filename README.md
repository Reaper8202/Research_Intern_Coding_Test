# ehr-forecast

Predicts which of 40 conditions a patient will be diagnosed with for the first time in the five
years after a landmark ("anchor") date, using only the synthetic EHR events recorded before that
date. Data: the Synthea Coherent dataset as released for the research intern coding test.

Status: steps 1 to 3 are done. The CSV-only transformer (`small-v1`) is trained on all five
training folds; self-pretraining then full fine-tuning (F) was selected. The submitted model is
multimodal (`small-v1-mm`, decision D34): the same transformer plus genotype tokens and a brain
MRI token from a 3-D CNN. See [the project plan](docs/plan.md) and
[model plan version 2.3](docs/model_plan.md).

The final model is a transformer trained from random weights on the supplied training data. Any
self-pretraining uses only training histories before their anchors. External datasets and
pretrained weights are excluded. Logistic regression and LightGBM are comparison methods.

The assessment demonstrates a correct, leak-free, reproducible training pipeline. The plan starts
with a small transformer and a complete run through training, checkpoint recovery, reload, and
prediction. Broader experiments follow only after these checks pass. Clinical usefulness is not
a project objective.

## Project structure

```
ehr-forecast/
├── data/
│   ├── README.md            how to provide the dataset
│   ├── raw/                 task dataset, unchanged (not committed)
│   └── processed/           step 1 output (not committed)
├── docs/
│   ├── plan.md              steps and where each known risk is handled
│   ├── model_plan.md        revised transformer design and evaluation protocol
│   ├── decisions.md         every data decision with the evidence for it
│   └── baseline_results.md  step 2 results and leakage review
├── configs/                 P3a, transformer, and smoke-test settings
├── models/                  trained models and fold tokenizers (not committed)
├── reports/
│   ├── dataset/             step 1 checks: train/val/test parity, label prevalence, omitted stops
│   ├── baseline/            step 2 metrics, per-condition table, CV results, val predictions
│   ├── p3a/                 P3a audits: baseline reproduction, units, fold support, manifest
│   ├── representation/      per-fold vocabulary, truncation, and retention reports
│   └── transformer/         generation deferral record and the G2 smoke-test report
├── logs/                    run logs (not committed)
├── src/ehrforecast/         Python package
├── tests/                   unit tests on small hand-made and synthetic tables
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

On macOS, LightGBM needs the OpenMP runtime: `brew install libomp`.

## Run

```bash
make check      # ruff lint + format check, mypy, pytest
make dataset    # step 1: data/raw -> data/processed, reports/dataset
make baseline   # step 2: models/baseline, reports/baseline
make folds      # P3a: save and verify the five training folds
make preflight  # P3a: training-only audits and baseline reproduction; does not score val
make tokenize   # P3b-P3c: fold tokenizers, patient histories, reports/representation
make pipeline-smoke   # G2: tiny end-to-end run on fold 0, temporary outputs
```

Transformer training, one fold at a time (settings in `configs/transformer.toml`):

```bash
make pretrain FOLD=0              # self-pretraining on fit patients' pre-anchor histories
make finetune FOLD=0 METHOD=S     # S: random init; H: fixed pretrained body; F: full fine-tuning
make finetune FOLD=0 METHOD=H
make finetune FOLD=0 METHOD=F
make pretrain FOLD=0 RESUME=1     # continue an interrupted run from its last completed epoch
make cv-summary                   # after all five folds: fold table, pooled metrics, selection
make wandb-sync                   # upload finished offline runs to wandb
```

Multimodal model (D34). Build the modality cache once, then use the multimodal config:

```bash
make modalities                   # genotype files and DICOM volumes -> data/processed (20 s)
make tokenize CONFIG=configs/multimodal.toml
make pretrain CONFIG=configs/multimodal.toml FOLD=0
make finetune CONFIG=configs/multimodal.toml FOLD=0 METHOD=F
make cv-summary CONFIG=configs/multimodal.toml REFERENCE=configs/transformer.toml
make pipeline-smoke SMOKE_CONFIG=configs/smoke_mm.toml
```

Genotypes enter as `[DNA] dna:<site>:<variant alleles>` tokens. A scan enters as one `[IMG]`
token only when its study date is before the anchor day. Coverage is small: genomics for 396
train and 2 test patients, and pre-anchor scans for 200 train and 28 test patients
(`reports/dataset/modalities.json`).

Each run writes `models/transformer/<config>/fold<k>/<run>/` with `best.pt`, `last.pt`,
`metrics.jsonl`, and `run.json`. wandb logs offline. `make wandb-sync` uploads finished runs to
the team `derek-wu820-university-of-toronto`, project `ehr-forecast`; set `WANDB_ENTITY` to use
another space. `make cv-summary` writes `reports/transformer/<config>/` and applies the method
selection rule of model plan section 5.3. Its bootstrap intervals take about 12 minutes.
`python -m ehrforecast.predict` scores held-out training patients or the test split from saved
fold models. It reads no labels and refuses official val.

On a laptop: step 1 takes about 20 seconds, step 2 about 2 minutes, `make preflight` and
`make pipeline-smoke` about 20 seconds each, and `make tokenize` about 45 seconds. On an M-series
Mac (MPS), one transformer fold (pretraining, then H, F, and S) takes about 20 minutes. `make clean`
removes everything that these commands generate, except the committed reports.

## How leakage is prevented

* Train and val timelines are cut at the anchor with the same rules the released test files
  follow: rows dated before the anchor day are kept, and a `STOP` on or after the anchor is
  blanked. `docs/decisions.md` shows how these rules were checked against the test files.
* The pipeline stops with an error if any table still holds a date on or after the anchor.
* A `STOP` dated before its own `START` (58 medication rows) is omitted after the cut; the start
  is kept (`docs/decisions.md`, D31). The prediction loader rejects event files built without
  this rule.
* Columns that summarise the whole record (death date, lifetime costs, medication dispense
  totals, marital status, device UDI) are never read.
* Labels come from full condition records for train and val patients. Prediction inputs come
  from truncated histories. The transformer plan keeps labels outside prediction interfaces.
* New model choices use cross-validation on train. The baseline has already been evaluated on
  val. The transformer plan freezes all choices before one recorded val evaluation bundle.

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
| `contracts.py` | explicit model inputs; training-only label loading |
| `folds.py` | saved five-fold assignments of the training split |
| `masking.py` | prior-condition mask aligned by patient and target |
| `evaluation.py` | version-2 AUROC, AP, Brier score, and patient BCa intervals |
| `numeric.py` | unit audit and quantile bins for numeric observations |
| `provenance.py` | hashes, file checks, and atomic JSON output |
| `preflight.py` | P3a data audit and baseline reproduction |
| `modalities.py` | genotype and DICOM cache for the multimodal model (D34) |
| `tokenizer.py` | fold tokenizer fitted on fit patients only |
| `sequences.py` | patient histories, prediction sequences, pretraining windows |
| `representation.py` | `make tokenize` command and retention reports |
| `model.py` | decoder-only transformer and the 3-D CNN image encoder |
| `training.py` | configs, fold loading, batching, losses, checkpoints, logging |
| `pretrain.py` | self-pretraining (next-event prediction) |
| `finetune.py` | supervised methods S, H, and F |
| `predict.py` | label-free prediction from saved fold models |
| `smoke.py` | G2 end-to-end execution check |
| `cv_summary.py` | five-fold results, method selection, config comparison |
| `wandb_sync.py` | upload finished offline wandb runs |
