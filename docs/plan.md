# Plan

Goal: predict each test patient's probability of a first diagnosis for each of the 40 target
conditions within five years after the anchor. The final product must be a transformer.
Logistic regression and LightGBM provide comparisons only.

The main objective is to demonstrate correct data handling, model training, evaluation, and
reproducibility. Clinical usefulness is not an objective. Complete a small end-to-end pipeline
before model experiments. Strong results remain an aim; model size and novelty are not requirements.

Follow the [assessment README](../../research_intern_coding_test/ReadMe.md). Keep the supplied
splits. Use only supplied training data and weights learned in this project. Use no external
datasets, pretrained weights, external tokenizers, or published clinical risk coefficients.
The coefficient restriction is a conservative project choice; the README does not discuss
clinical equations explicitly.

The detailed design is [model plan version 2.3](model_plan.md), revised on 31 October 2026.
This revision changes the plan. It does not complete an implementation phase.

## Steps

| Step | Content | Status |
|------|---------|--------|
| 1 | Anchors, leak-free history, labels (`ehrforecast.build_dataset`) | done |
| 2 | Baselines: age-sex incidence tables and logistic regression on code counts plus age and sex (`ehrforecast.train_baseline`) | done |
| 3 | Folds, common prior mask, baseline reproduction, input contracts, tokenizer, sequences, and leakage tests (P3a–P3c) | done (not yet reviewed by Derek) |
| 4 | Complete small-model run, three training methods, full comparisons, bounded extensions, and frozen val evaluation (P4a–P4f; generation deferred) | done: D34–D36 rounds complete; frozen val bundle run once on 3 November (D37) |
| 5 | Test predictions, report, GitHub, Hugging Face | done locally: `predictions.csv`, package, report PDF; publishing waits for Derek |

## Where each known risk is handled

### Must fix

| Risk | Handling | Where |
|------|----------|-------|
| Train history must look like the test files | Truncation copies the test rules: keep rows dated before the anchor day; blank any `STOP` on or after it. Hard check: zero values on or after the anchor in every table, both ours and the released test files. Parity report compares train/val/test history statistics. | `truncation.py`, `build_dataset.py`, `parity.py` |
| Death and cost columns hold future data | Every table is read through a column allowlist. Dropped: `DEATHDATE`, `HEALTHCARE_EXPENSES`, `HEALTHCARE_COVERAGE`, `MARITAL`, address fields, all cost and payer columns, `DISPENSES`, `TOTALCOST`, `devices.UDI`. | `schema.py` |
| Few positives for some targets | Show per-target counts and patient-bootstrap intervals for final comparisons. Document unsupported metrics. Use five-fold confirmation and seed checks. CV remains development evidence affected by selection. | `model_plan.md` section 5 |
| Val already informed earlier exploration | Disclose existing baseline scores and earlier descriptive analyses. Freeze the new model and evaluation operations before one recorded val bundle. Do not call val untouched or use its new scores to choose the submission. | P4e |
| Test labels are hidden | Label every metric by split. Attribute test scores to the organisers when supplied. | report (step 5) |
| Raw interval stops before their start (57 training medication rows) | D31: keep the start, omit the invalid stop, after the cut and in every split. Raw files stay unchanged. Each omitted stop is listed. An event manifest rejects histories built without the rule. | `truncation.py`, `events.py`, `reports/dataset/omitted_stops.csv`, `decisions.md` D31 |
| Mixed timestamp formats | One parser (`parse_timestamps`) for every time column; all comparisons at day level against a midnight anchor. Unit tests cover both formats. | `io.py`, `tests/test_io.py` |
| Preprocessing can depend on future records | Cut before compression. At fixed anchors and artifacts, future changes must leave all inputs, masks, and predictions unchanged. Fit preprocessing within each fold. Keep labels outside prediction interfaces. | P3a–P4a; T19–T20, T34 |
| Outside fitted parameters enter the pipeline | Initialize transformers randomly. Verify the origin of every loaded checkpoint. Keep clinical formulas as literature context only. | D16, D19, D28; T28 |
| Training recovery or batching changes the experiment | Test complete epoch-boundary state recovery and correct loss normalization across unequal microbatches. | T35–T36 |
| The model works only in the development environment | Test label-free inference and clean-environment CPU reload through the final prediction command. | T34, T37; G2, G4 |
| Too many experiments delay a complete result | Complete G1–G2 first. Reserve core evaluation and delivery costs. Limit extra fold-0 trials to four and five-fold confirmations to two. | sections 5.3 and 6 |

### Lower risk

| Risk | Handling | Where |
|------|----------|-------|
| Long timelines | Start with 1,024 tokens and an older-condition summary. F12 shows substantial loss for dense histories. Screen 512. Give 2,048 first extension priority within memory and budget limits. Always report retention. | P3b–P3c; E5, E14 |
| Quantile bins lose numeric detail | After G2, use a separately budgeted continuous-versus-binned LightGBM comparison. Test E18 only after material loss and within the extension limit. | E11, E18 |
| Genomics download incomplete, almost no test coverage (2 of 358) | Not used. Stated in the report. | report |
| Imaging DICOM files: 9.2 GB, 28 test patients | DICOM files not used. Imaging study events from `imaging_studies.csv` are used. | `schema.py` |
| Little data for transformer training | Use a 2-layer, width-128 default and compare three training methods. Self-pretrain only on fit patients' pre-real-anchor histories. A larger model and extra anchors need a reason and a budget slot. | P4a–P4c; E1, E3, E4, E11 |
| Generation lacks examples of required target events | Pneumonia and prostate cancer metastasis have no pre-anchor training diagnoses. Defer zero-shot generation. Their supervised labels remain available to the transformer's 40-output head. | F11; P4d; section 10 |

## Step 3 to 5 design

The full plan defines decisions D9–D30, phases, experiments, tests, and risks. Existing decisions
D1–D8 remain in [decisions.md](decisions.md). Planned decisions move there when implemented.

Compare these methods using the same representation and training folds:

* Train the transformer and prediction head from random weights.
* Train our own transformer on permitted histories, hold its body fixed, and fit its prediction head.
* Use the same locally trained checkpoint and update the whole model with supervised labels.

The small default uses rotary positional encoding, continuous age information, and the last clinical
event as its readout. Age at anchor, sex, and event recency also enter the prediction head.
Every adopted configuration requires five-fold confirmation. Average the five selected fold
models for val and test. Baselines cannot enter this ensemble.

G1 checks the data boundary. G2 proves a short run through pretraining, adaptation, recovery,
reload, and prediction on mock test histories from a training fold. G3 completes the required
five-fold comparisons. G4 verifies frozen evaluation and reproducible delivery. Extra model
experiments cannot replace these conditions. Preserve pretraining evidence even if scratch training wins.

The training-only coverage audit found 133 pneumonia labels and 50 prostate cancer metastasis
labels, but no pre-anchor diagnosis for either target. Zero-shot generation therefore fails its
coverage requirement. P3a recorded that decision in `reports/transformer/generation_gates.json`.
Do not budget generation training.

P4f now precedes P4e. Define technical error analyses before freezing code, settings, tokenizers,
and checkpoints. Run official val scoring and the requested val curves in one recorded bundle.
Clinical reference documents remain optional background. They create no implementation requirement.

P4a proves the complete small run and measures speed and memory before broad searches. Full baseline
searches also wait for G2. Earlier compute estimates are withdrawn.

## Status on 1 November 2026

* P3a passed: `make preflight` reproduced the step-2 training CV and did not score official val.
  It found invalid medication stops; D31 handles them.
* P3b–P3c are implemented: fold tokenizers, value bins, refill compression (D32), sequences,
  summaries, pretraining windows, and retention reports for all five folds (`make tokenize`).
* Training code is implemented: model, self-pretraining, S/H/F, resume, and label-free
  prediction. `make check` passes 129 tests, including T2–T7, T10–T13, T18–T20, T22–T23, T29–T30,
  T34–T36, and T38.
* G1 checks pass. `make pipeline-smoke` passed G2 on fold 0 with tiny settings: pause and resume,
  H/F/S, a fresh-process prediction of 559 x 40 probabilities, and CPU/MPS agreement (3e-8).
* P4a and the first part of P4c are complete: self-pretraining and S, H, and F on all five
  training folds with the default 2 x 128 model (20 runs). All runs ended without NaN values.
  Logs are in wandb, team `derek-wu820-university-of-toronto`, project `ehr-forecast`.
* Section 5.3 selected **F**. F scored above S on 5 of 5 folds. Mean masked macro AUROC: F 0.760,
  S 0.744, H 0.742. Mean mAP: F 0.186, S 0.163, H 0.161. H scored above S on 2 of 5 folds and
  failed the consistency rule. These are training-CV results; official val was not read.
  `reports/transformer/small-v1/` holds the fold table, pooled out-of-fold metrics, and paired
  patient BCa intervals.
* Pooled out-of-fold, paired patient BCa 95% intervals (1,000 resamples): F minus S, AUROC +0.020
  [+0.013, +0.028] and mAP +0.027 [+0.017, +0.036]. F minus the step-2 code-count logistic
  regression, AUROC +0.017 [+0.007, +0.025] and mAP -0.004 [-0.014, +0.007]. F therefore ranks
  better than that baseline, but its average precision is not clearly different. Pooled values:
  F 0.750 / 0.163, S 0.730 / 0.136, H 0.734 / 0.141, logistic regression 0.734 / 0.166.
* Pretraining reached the 30-epoch limit on every fold, with the best epoch at 28 or 29. H reached
  the limit on folds 0 and 2. Held-out loss still fell slowly at the limit. The limit is the plan
  setting and stayed fixed for all folds.
* Fold-4 pretraining failed four times because swap filled the disk. D33 records the cause and
  the fix (release cached MPS memory after each step). The failed attempts were not used.
* Derek required a multimodal model (D34) and chose to submit it. `small-v1-mm` adds genotype
  tokens and a brain MRI token (3-D CNN) to `small-v1`. All 20 runs completed; section 5.3 again
  selected F. Mean fold AUROC / mAP: F 0.759 / 0.197, S 0.752 / 0.170, H 0.740 / 0.162.
  Against CSV-only F (paired BCa, pooled out of fold): AUROC +0.0005 [-0.006, +0.007], mAP +0.009
  [-0.000, +0.017]. On the 541 patients with a modality input: AUROC -0.003 [-0.020, +0.015].
  The modalities show no measurable benefit (D34). Against the step-2 logistic regression:
  AUROC +0.017 [+0.007, +0.027], mAP +0.005 [-0.007, +0.016].
* E11 LightGBM (section 7.2, training CV): counts AUROC 0.779 / mAP 0.229; counts + values
  0.774 / 0.243; counts + binned 0.777 / 0.244. Every selection is at the simple edge of the grid
  (4 leaves, 100 trees). The value-bin check shows no material loss, so E18 is not triggered.
  **The transformer trails LightGBM:** multimodal F minus LightGBM counts, pooled paired BCa,
  AUROC -0.023 [-0.032, -0.015] and mAP -0.032 [-0.044, -0.021]. Section 5.4 applies: verify
  correctness, inspect training-only errors, then at most one recovery round of two configurations.
  The transformer stays the submission; the gap is reported.
* Tuning round complete (D35; `configs/budget.toml`, `reports/transformer/experiments.csv`).
  Selected: `small-v1-mm-do03`, multimodal F with fine-tuning dropout 0.3. Mean fold AUROC 0.766
  and mAP 0.203 (incumbent 0.759 / 0.197); +0.007 AUROC on 5 of 5 folds, about twice the spread
  of three seeds. Failed: lower body learning rate, weight decay, all three together (mAP), the
  age-sex head alone, 60 pretraining epochs, and context 512 and 2,048 (fold-0 screens).
  The LightGBM gap shrank from -0.023 to -0.014 AUROC [-0.021, -0.006] and remains reported.
* Round 2 (D36) replaced the candidate with `small-v2-c1` averaged over three seeds (15 fold
  models) after removing the simulator clock codes DALY, QALY, and QOLS.

## Status on 3 November 2026 (step 5)

* P4e ran once (D37). `make freeze` recorded 294 artifact, data, code, and comparison-file hashes
  in `reports/final/manifest.json` at commit 8e8725e. `make val-score` then scored official val:
  macro AUROC 0.751 [0.730, 0.773], mAP 0.209 [0.192, 0.211]. The transformer beats the at-risk
  incidence table (+0.025 AUROC, interval above zero), is indistinguishable in AUROC from
  LightGBM counts and the step-2 logistic regression, and trails LightGBM on mAP. One attempt;
  no replacement bundle. Details: `docs/model_results.md`, `reports/val/`.
* `make predict` wrote `predictions.csv` (358 test patients x 40 codes) from the same 15 members.
* `ehrforecast.hf_package` builds the public package in the `models/transformer` layout, so the
  README's prediction command works from the download; `verify` reproduces `predictions.csv`.

## Status on 3 November 2026, evening (round 3)

* Derek asked for the strongest leak-free submission and a check of every decision. An
  independent read-only audit of every data boundary found no leak (D1 to D37 hold); it flagged
  that having a genotype file or a scan is itself correlated with outcomes (a dataset artifact,
  disclosed in the report), and the residual era signal already recorded in D36.
* Round 3 was pre-registered in `configs/budget_v3.toml` (commit e93c4d7) before any of its scores.
  Rule S adopted the mean-of-logits blend of the C1 three-seed ensemble with LightGBM counts and
  counts + values (nested five-fold CV: +0.014 AUROC over the ensemble, +0.011 over LightGBM
  counts, intervals above zero). Rule C adopted per-target Platt scaling. Rule A2 (LightGBM
  probabilities as head inputs) was skipped by its condition. D38 records the numbers.
* Rule A1, the 4 x 192 architecture screen on fold 0, failed (-0.014 AUROC against C1 on the same
  fold); the architecture stays at 2 x 128 (D38).
* Refrozen (commit 6f923e0 and the manifest commit after it). The replacement val bundle
  (attempt 2, `093a88a1f5b5`) scored the blend at 0.767 [0.746, 0.788] AUROC and 0.258 mAP;
  attempt 1 stays under `reports/val/attempts/`. `predictions.csv` is the blend. The package
  reproduces it exactly from its own files. Publishing (GitHub public, Hugging Face upload,
  wandb public) waits for Derek.

Measured time on this Mac (MPS), per fold: pretraining 20–61 minutes, H 2–6, F 12–15, and S 13–23
minutes for folds 0–3. Fold 4 ran with the D33 fix and took 10, 2.5, 3, and 3 minutes. Most of the
earlier time was memory pressure, not computation. One fold now needs about 20 minutes.

### Implementation choices the plan left open

| Choice | Setting |
|---|---|
| Supervised learning rate | Constant at the starting rates; the plan gives no schedule. AdamW default betas. |
| Memory | Micro-batches of 8 with exact loss accumulation to 32 patients. |
| Pretraining windows | Start day drawn uniformly from day 0 to the prediction-window start, so each window is full. |
| H and F heads | New random head; only body weights come from the pretraining checkpoint. |
| Encounter tokens | Eligibility counts each class-and-code pair (`enc:<class>:<code>`). |
| Missing unit | Unit key `nounit`. |
| Training metrics | Fit-patient AUROC and mAP in evaluation mode, each epoch. |

## Open decision

* Name on the PDF report and the GitHub / Hugging Face account to publish under.
* Measured local compute ceiling. P4a measured the speeds above; set the ceiling before broad
  experiments.
