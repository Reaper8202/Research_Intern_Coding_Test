---
title: "Forecasting first diagnoses from synthetic EHR timelines"
subtitle: "Research intern coding test: technical report"
author: "[AUTHOR NAME]"
date: "4 November 2026"
geometry: margin=2.3cm
fontsize: 11pt
mainfont: "Helvetica Neue"
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{htbp}
---

**Links.** Code and documentation: <https://github.com/Reaper8202/Research_Intern_Coding_Test>.
Trained model package: <https://huggingface.co/Reaper8202/ehr-forecast>.
wandb project (training, tuning, and val-curve runs): <https://wandb.ai/derek-wu820-university-of-toronto/ehr-forecast>.

## Summary

The submission is a fixed blend of three models trained from random weights on the 2,791
training patients: fifteen small decoder-only transformers (2 layers, width 128, about 738k
parameters each; three seeds, five cross-validation folds, self-pretrained on pre-anchor event
sequences and fine-tuned on the 40 labels), and two LightGBM models on code counts, one with the
latest value of each numeric observation. Their logits are averaged and a per-target Platt map
fitted on training out-of-fold predictions scales the result. On the official validation split
(365 patients) the blend scores macro AUROC **[VAL_BLEND_AUROC]** and mAP **[VAL_BLEND_MAP]**; the
transformer ensemble alone scores [VAL_T_AUROC] / [VAL_T_MAP] and LightGBM on counts alone
[VAL_L1_AUROC] / [VAL_L1_MAP]. Test scores are the organisers'.

The transformer is the part of the system this project built and studied, and it is what the
report describes. The blend was adopted at the end by a rule written before its score, because in
nested cross-validation it beat both of its parts with intervals above zero. Most of the effort
went into the pipeline rather than the models: a truncation rule checked against the released test
files, column allowlists that keep future-bearing fields out, a prior-condition mask shared by
every model, five-fold cross-validation on train for every choice, three tuning rounds with
budgets committed before their scores, an independent audit of every data boundary, and two
recorded val runs, both disclosed.

## 1. Representation

**One unit of input is one clinical event token**, `source:kind:code`, with its value fused in.
A patient's sequence is

```
[SEX:F]  hx:<older conditions> ...  [SEP]  <events, oldest first, one token each>
```

| Source | Token | Example |
|---|---|---|
| conditions, medications, care plans, allergies, devices | `src:start:code`, `src:stop:code` | `med:start:310798` |
| procedures, immunizations | `proc:code`, `imm:code` | `proc:430193006` |
| numeric observations | `obs:code:unit:q<bin>` | `obs:8480-6:mm[Hg]:q7` |
| text observations | `obs:code:<normalized value>` | `obs:72166-2:never_smoker` |
| imaging studies, encounters | `img:modality|site`, `enc:class:code` | `enc:wellness:162673000` |
| genotype (D34) | `[DNA]`, `dna:<site>:<alleles>` | one token per carried variant |
| brain MRI (D34) | `[IMG]` | embedding replaced by a 3-D CNN encoding |

**Fields and modalities.** Every CSV table is read through a column allowlist. Dropped on
purpose: death date, lifetime healthcare expenses and coverage, marital status, addresses, all
cost and payer columns, medication dispense counts and total cost (in the released test files
873 of 1,049 medications that continue past the anchor have more dispenses than months elapsed,
so they leak future refills), and device UDIs (they encode manufacture and expiry dates). Race
and ethnicity are not used (D15; a LightGBM check showed they cost no measurable score). Three
observation codes, DALY, QALY, and QOLS, are removed (D36): they are simulator scores recorded
once a year from 2011, their counts predict an anchor before 2015 with AUROC 0.999, and they tell
the model how the anchor was constructed rather than anything clinical. Removing them cost about
0.014 CV AUROC in the transformer and 0.005 in the final blend, and probably costs test score,
since the test files contain them.

Genomics and the brain-MRI DICOM volumes are included at my request after an earlier decision to
leave them out. Coverage is thin: genotype files exist for 396 training and 2 test patients, and a
pre-anchor scan for 200 training and 28 test patients. The scan enters as one token only when its
study date is before the anchor day. Measured in CV they add nothing (+0.0005 AUROC; on the 541
training patients whose inputs actually change, -0.003 [-0.020, +0.015]). The volumes are
near-identical simulated brains (block-mean correlation 0.995 to 0.999 between patients). The
audit (section 3) found that having a genotype file or a scan is itself correlated with outcomes
in this dataset, so the presence tokens could act as a proxy; the CV result says the model does
not use them, and only 2 test patients have genotypes.

**Continuous values.** Each numeric observation is binned into quantiles fitted per code and
unit on the fold's fit patients only (up to 10 bins, merged where boundaries coincide), and the
bin index is fused into the token. Units stay separate; no unit conversions or clinical
thresholds come from outside. A value token needs at least five fit patients, otherwise it maps
to `obs:code:other_value`. A LightGBM check compared continuous latest values against their
bins and found no material loss (0.243 against 0.244 mAP), so the binned representation stayed.
The LightGBM part of the blend uses the continuous latest value directly, with a missing flag and
the log of days since the measurement.

**Time between events.** Every token carries the patient's age at that event, to the day, as
`age/100` plus sine and cosine terms over eight periods from 0.5 to 200 years, mapped by a learned
linear layer and added to the token embedding. The model can therefore read the gap between any
two tokens. Rotary position encoding gives token order. The prediction head also receives the
exact time from the last event to the anchor, age at anchor, sex, and a missing-history flag.
Explicit gap tokens were planned as an experiment and not reached within the budget.

**Context and summary.** Sequences are capped at 1,024 tokens. The most recent complete event
days are kept; conditions first diagnosed before the kept window become `hx:` summary tokens
(up to 128) at the front, so an old diagnosis is never lost. Under the final tokenizer about 4.6%
of training patients are truncated, but they are the densest histories: 15 to 18% of all event
tokens fall outside the window, mostly medication refills and encounters. Exact refills (a stop
and a restart of the same drug on the same day; 160,150 of them in fold 0) are merged first.
Contexts of 512 and 2,048 were screened on fold 0 and did not help.

**Leakage controls.** Train and val timelines are cut with the same rules the released test files
follow: keep rows dated before the anchor day, blank any `STOP` on or after it. The rule was
chosen among four candidates by matching the distribution of the gap from last event to anchor in
the test files (D1). The pipeline stops if any table holds a date on or after an anchor. A test
adds a future record and checks that every prediction is unchanged. Labels come from the full
record for train and val patients; the prediction interface never sees them. The vocabulary,
bins, folds, self-pretraining windows, LightGBM vocabularies and value medians, stacking
weights, and Platt maps are all fitted inside each fold on fit patients only.

## 2. Model and training

**Architecture.** Decoder-only transformer with pre-LayerNorm blocks, causal multi-head attention,
RoPE, GELU MLP (4x width), token embedding plus the age projection: 2 layers, width 128, 4
heads, vocabulary about 1,930 tokens, 738k parameters in total. The pretraining head predicts the
next token with weights tied to the embedding; input-only symbols are masked out of the output
distribution. The prediction head concatenates the last clinical-event state with the mean state
over all clinical events (the `last_mean` readout adopted in round 2) and the four head features,
applies dropout, and maps to 40 logits with independent sigmoids. The MRI encoder is a four-layer
3-D CNN (81k parameters) from random weights on a 64-cubed volume.

**Objectives.** Self-pretraining: next-token cross-entropy over clinical event positions, on
each fit patient's history strictly before the real anchor, one random full window per patient
per epoch; target-diagnosis tokens weigh 5 in the loss (round 2). Fine-tuning: unweighted binary
cross-entropy over the unmasked patient-target cells (a target the patient already has is
masked out of the loss and set to probability 0 at prediction), normalised by the cell count across
gradient-accumulation micro-batches; one extra supervised example per fit patient per epoch
from an earlier anchor five years back (round 2).

**Hyperparameters** (`configs/experiments/v2_c1.toml`): AdamW. Pretraining: lr 0.001, betas
(0.9, 0.95), weight decay 0.1, 10% warm-up then cosine to 0.1 of peak, batch 32 patients, 30
epochs, patience 3 on fixed held-out windows; every run reached epoch 30 with held-out loss still
falling slowly (fold 0: 1.449). Fine-tuning: head lr 0.001, body lr 0.0001, weight decay 0.01,
gradient clip 1.0, batch 32, dropout 0.3 in the body and head, up to 30 epochs, patience 5 on
held-out masked macro AUROC; best epochs ranged from 3 to 13. One fold (pretraining plus
fine-tuning) takes about 20 minutes on a 16 GB M-series Mac; the 15 submitted members took
about 4.5 hours in all. Everything is float32.

**Three training methods were compared** on the same folds before any tuning: S (random
initialisation, supervised only), H (pretrained body held fixed, head only), and F (pretrained
body, full fine-tuning). F won on every fold: mean fold AUROC 0.760 against 0.744 (S) and 0.742
(H); pooled paired difference F minus S +0.020 [+0.013, +0.028]. Self-pretraining helps; a fixed
body does not carry enough.

**Tuning, pre-registered.** Three rounds, each written to a budget file and committed before any
of its scores existed, with a fixed adoption rule: mean AUROC gain at least 0.005, higher on at
least 4 of 5 folds, mAP loss at most 0.005. Round 1 (11 trials) adopted dropout 0.3; lower body
learning rate, weight decay 0.1, an age-sex head, 60 pretraining epochs, and contexts 512 and
2,048 failed. Round 2 (12 trials, after removing the clock codes) adopted the combination of a
pooled readout, extra anchors, and target-weighted pretraining; a count branch, per-target loss
weights, and dropout 0.5 failed. Full-timeline pretraining (reading fit patients' post-anchor
events) passed the rule and was **not** adopted, because the task says to truncate at the anchor.
Three seeds of the winner were trained and averaged. Round 3 tested the one architecture
dimension never varied, depth and width: a 4-layer, width-192 model (2.26M parameters) reached a
slightly lower pretraining loss but scored 0.758 against 0.772 on the fold-0 screen, so the small
model stayed. The full register with every score is in
`docs/decisions.md` (D35, D36, D38) and `reports/transformer/`.

**Comparison models and the blend.** Alongside the transformer: an age-sex incidence table (the
epidemiological floor, with and without the "already has it" flag), L2 logistic regression on
log code counts, and LightGBM on counts and on counts plus latest values, each with settings
chosen by the same folds (LightGBM's grid picked its simplest corner, 4 leaves and 100 trees). In
training CV the transformer ensemble and LightGBM counts were within noise on AUROC (0.761 against
0.764 pooled out of fold) and LightGBM was ahead on mAP. Round 3 then tested output stacking with
nested folds: for each fold, the stacker is fitted on the other four folds' out-of-fold
predictions. The parameter-free mean of logits beat the transformer ensemble by +0.014 AUROC
[+0.009, +0.019] and LightGBM counts by +0.011 [+0.007, +0.014] (mean fold AUROC 0.780, mAP
0.236), a learned per-target logistic stacker gained only 0.001 more, and the pre-registered rule
adopted the fixed blend. Per-target Platt scaling of the blend, also nested, lowered Brier on 4
of 5 folds and macro expected calibration error from 0.013 to 0.010, with per-target AUROC
unchanged by construction (a non-positive fitted slope falls back to the identity).

**Curves.** Training loss, learning rate, and gradient norm are logged per step, and held-out
fold metrics per epoch, for all transformer runs in the wandb project above. Figure 1 shows the
requested validation curves: every saved epoch checkpoint of the 15 members scored on the official
val split after training, inside the frozen bundle. The epochs selected on the held-out training
folds (dots) sit near but mostly before the val optimum; val would have picked a later epoch in 13
of 15 members and gained 0.008 AUROC on average. That choice was not made.

![Official val macro AUROC of every saved fine-tuning checkpoint. Thin lines: the 15 ensemble
members; dots: the epoch each member's held-out training fold selected; the orange line is the
mean over members at that epoch.](figures/val_curves.png){width=92%}

**Evaluation protocol.** Five folds of the training split (patients never shared across folds),
one out-of-fold prediction per patient from the models that did not fit on them, macro AUROC and
mAP over 40 targets with the common prior mask, plus Brier score, with 95% patient-level BCa
bootstrap intervals (1,000 resamples) and paired bootstraps for differences. Official val was
scored by `make val-score`, which records every artifact, data, code, and lock-file hash at
freeze, refuses to run on a changed tree or artifact, writes a start record before any metric
and a completion record after, and refuses a second run without a documented reason. It ran
twice: attempt 1 scored the transformer ensemble (0.751 / 0.209) before the blend existed;
attempt 2 scored the blend after round 3 was decided on training CV. Both records are kept
(`reports/val/`, `reports/val/attempts/`). The first attempt's numbers were known when round 3
was written; its rules used only training CV, and the blend's val score could not be used to
change it.

## 3. AI workflow

Claude Code wrote most of the code, the tests, and the planning documents; I set the objective
and constraints, approved or rejected each decision that touched the data or the protocol, and read
every result before the next step. The division is recorded in `docs/ai_workflow.md` and the
reasoning for each decision in `docs/decisions.md`.

What kept the AI's work honest: the plan was written before the code and each revision says what
was known when it was made; 193 unit tests include mutation-style checks (changing `<` to `<=` in
the anchor rule fails a test; a future record must leave predictions unchanged; a resumed run must
match an uninterrupted one; adding held-out rows must not change the fitted vocabulary); the
adoption rule was fixed before the first transformer result; every experiment block was committed
before its scores; and an independent read-only audit by a second AI session checked every data
boundary (truncation, allowlists, labels, tokenizer, extra anchors, pretraining windows, early
stopping, modalities, LightGBM, evaluation, test split) with file and line references and found
no leak.

Problems the review caught: fold-4 pretraining filled the disk with swap four times because the
MPS allocator fragmented across variable-length batches (fixed by releasing cached memory each
step, verified to change no computation, and 3 to 6 times faster); LightGBM and PyTorch each
loading an OpenMP runtime and crashing in one process (LightGBM now runs in its own process, and
the test suite pins the import order); the first bootstrap intervals using the percentile method,
which for rare targets barely contained the point estimate (BCa replaced them); a preflight audit
flagging its own report files as a dirty tree; and a first Platt-scaling fit that let five
near-chance targets take a negative slope, which would have reversed their ranking (caught by a
monotonicity test; the fit now falls back to the identity). [Derek: add anything else here,
including any use of Codex CLI.]

## 4. Interpretation

**Where the model does well.** Seven targets are above 0.95 AUROC for every model: prostate
neoplasm, carcinoma in situ and metastasis of the prostate, normal pregnancy, pneumonia, obesity,
and the pregnancy-complication code. Each follows a visible Synthea pathway (PSA tests and rectal
exams, oral contraceptives, dementia care plans, BMI observations). A leakage review of these
scores found pre-anchor causes, not leaks. The transformer's clearest wins over LightGBM are the
paired brain targets, silent micro-haemorrhage and abnormal gait (0.905 against 0.832 on val; the
two labels are identical for every patient because one module emits both), and stroke. Figure 2
shows every target for the final blend and its two parts.

![Official val AUROC per target for the blend, the transformer ensemble, and LightGBM on code
counts, sorted by the blend. Numbers in parentheses are val positives; most targets have 5 to 20,
so single-target differences are noisy.](figures/val_per_target.png){width=74%}

**Where it does poorly, and why.** Nine targets are below 0.6 for every model: whiplash,
lacerations, wrist and ankle injuries, clavicle fracture, sinusitis, and strep throat. Synthea
generates these acute events with little dependence on the prior record, so no history model can
rank them; the age-sex table is at chance too. The transformer's largest losses to LightGBM are
the count-driven conditions: osteoporosis, recurrent rectal polyp, myocardial infarction, the
three lung-cancer codes, and pathological fracture, on val and in CV alike. In Synthea their risk
grows with how many of a specific event has happened (screening visits, bone-density tests,
nodule follow-ups), which a count feature states directly; the transformer has to infer it from a
window that drops 15 to 18% of event tokens for exactly the dense histories where counts are high.
The count branch tested in round 2 did not close the gap, a longer context did not help on fold 0,
and that is why the blend helps: the two models fail on different targets, and in CV the better
of the two per target would reach 0.778 against 0.774 for the blend.

**Anchor era.** Anchors are five years before the last encounter, so an anchor before 2015
almost always means the patient died by 2020 (1,155 of 1,219 training patients), and those
patients have more new diagnoses (2.37 against 1.48). Scores split by era are similar for every
model, and an era signal remains in the clinical codes and age even without the clock codes
(AUROC 0.865 for predicting the era). It cannot be removed without removing history, so it is
reported.

**Calibration.** Prior-condition cells are exact zeros. The raw blend under-predicts slightly in
the middle of its range (Figure 3); the Platt map corrects it. On val the blend's Brier score is
[VAL_BLEND_BRIER] against [VAL_T_BRIER] for the transformer alone.

![Reliability of the blend on training out-of-fold predictions, before and after Platt scaling,
10 equal-mass bins of all at-risk patient-target cells.](figures/reliability.png){width=58%}

**Limits.** The data are synthetic and the anchor is retrospective, so nothing here is evidence
about clinical use. Val is small: the macro AUROC interval is about 0.04 wide and nine targets
have at most eight positives. Every CV number for the chosen configuration is the best of 24
transformer trials plus three stackers on the same five folds, so it is optimistic; the val
scores are the first check and the hidden test the second. Two seeds of CV variation (about 0.002
AUROC in the mean, up to 0.014 per fold) are comparable to several of the differences in the
tuning tables. A macro AUROC of 0.80 is out of reach on this data without the clock codes:
the nine near-chance targets cap the average, and even an oracle choice between the transformer
and LightGBM per target reaches 0.778 in CV.

## 5. Results summary and reproduction

| Split | Model | Macro AUROC | mAP |
|---|---|---|---|
| train CV, pooled out of fold | blend (submitted) | 0.774 | 0.212 |
| train CV, pooled out of fold | transformer ensemble | 0.761 | 0.181 |
| train CV, pooled out of fold | LightGBM counts | 0.764 | 0.188 |
| **val** (365), attempt 2 | **blend (submitted)** | **[VAL_BLEND_AUROC]** | **[VAL_BLEND_MAP]** |
| val, attempt 2 | transformer ensemble | [VAL_T_AUROC] | [VAL_T_MAP] |
| val, attempt 2 | LightGBM counts | [VAL_L1_AUROC] | [VAL_L1_MAP] |
| val, attempt 2 | LightGBM counts + values | [VAL_L2_AUROC] | [VAL_L2_MAP] |
| val, attempt 2 | logistic regression, codes | [VAL_LR_AUROC] | [VAL_LR_MAP] |
| val, attempt 2 | age-sex incidence, at risk | 0.727 | 0.141 |
| val, attempt 1 | transformer ensemble | 0.751 [0.730, 0.773] | 0.209 [0.192, 0.211] |
| test (358) | blend (submitted) | organisers' score | organisers' score |

Paired on val, blend minus transformer ensemble: AUROC [VAL_BLEND_MINUS_T]; blend minus LightGBM
counts: AUROC [VAL_BLEND_MINUS_L1]. `predictions.csv` holds 358 rows by 40 code columns.

Reproduction: `make setup`, link the dataset to `data/raw`, then `make dataset`, `make modalities`,
`make folds`, `make tokenize`, `make pretrain` and `make finetune` per fold and seed, `make
lightgbm`, `make stacking`, `make calibration`, `make lightgbm-final`, `make freeze`, `make
val-score`, `make predict`, `make submission`, `make package`. The Hugging Face package carries the
15 transformer checkpoints, 5 tokenizers, 80 LightGBM boosters with their feature specs, and the
Platt weights; `python -m ehrforecast.hf_package verify` rebuilds `predictions.csv` from it on CPU
and reproduced it exactly, in the development environment and in a fresh virtual environment
built from `requirements-lock.txt`.
