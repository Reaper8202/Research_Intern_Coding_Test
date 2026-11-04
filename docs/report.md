---
title: "Forecasting first diagnoses from synthetic EHR timelines"
subtitle: "Research intern coding test: technical report"
author: "[AUTHOR NAME]"
date: "3 November 2026"
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

The submitted model is an ensemble of fifteen small decoder-only transformers (2 layers, width
128, about 738k parameters each) trained from random weights on the 2,791 training patients:
three random seeds, five cross-validation folds each, self-pretrained on pre-anchor event
sequences and then fully fine-tuned on the 40 first-diagnosis labels. On the official validation
split (365 patients) it scores macro AUROC **0.751** [0.730, 0.773] and mAP **0.209**
[0.192, 0.211], scored once by a frozen bundle. It beats the age-sex incidence reference by 0.025
AUROC (interval above zero), ranks patients about as well as logistic regression and LightGBM on
code counts, and trails LightGBM on average precision by 0.03. Test scores are the organisers'.

Most of the effort went into the pipeline rather than the model: a truncation rule checked
against the released test files, column allowlists that keep future-bearing fields out, a prior-
condition mask shared by every model, five-fold cross-validation on train for every choice, two
tuning rounds whose budgets were committed before their scores, and one recorded val run. The
transformer did not beat a tuned gradient-boosted baseline on this data, and the report says so.

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
0.014 CV AUROC and probably costs test score too, since the test files contain them.

Genomics and the brain-MRI DICOM volumes are included at my request after an earlier decision to
leave them out. Coverage is thin: genotype files exist for 396 training and 2 test patients, and a
pre-anchor scan for 200 training and 28 test patients. The scan enters as one token only when its
study date is before the anchor day. Measured in CV they add nothing (+0.0005 AUROC; on the 541
training patients whose inputs actually change, -0.003 [-0.020, +0.015]). The volumes are
near-identical simulated brains (block-mean correlation 0.995 to 0.999 between patients).

**Continuous values.** Each numeric observation is binned into quantiles fitted per code and
unit on the fold's fit patients only (up to 10 bins, merged where boundaries coincide), and the
bin index is fused into the token. Units stay separate; no unit conversions or clinical
thresholds come from outside. A value token needs at least five fit patients, otherwise it maps
to `obs:code:other_value`. A LightGBM check compared continuous latest values against their
bins and found no material loss (0.243 against 0.244 mAP), so the binned representation stayed.

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
bins, folds, and self-pretraining windows are fitted inside each fold on fit patients only.

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

**Tuning, pre-registered.** Two rounds, each written to a budget file and committed before any
of its scores existed, with a fixed adoption rule: mean AUROC gain at least 0.005, higher on at
least 4 of 5 folds, mAP loss at most 0.005. Round 1 (11 trials) adopted dropout 0.3; lower body
learning rate, weight decay 0.1, an age-sex head, 60 pretraining epochs, and contexts 512 and
2,048 failed. Round 2 (12 trials, after removing the clock codes) adopted the combination of a
pooled readout, extra anchors, and target-weighted pretraining; a count branch, per-target loss
weights, and dropout 0.5 failed. Full-timeline pretraining (reading fit patients' post-anchor
events) passed the rule and was **not** adopted, because the task says to truncate at the anchor.
Three seeds of the winner were trained and averaged. The full register with every score is in
`docs/decisions.md` (D35, D36) and `reports/transformer/`.

**Curves.** Training loss, learning rate, and gradient norm are logged per step, and held-out
fold metrics per epoch, for all 199 runs in the wandb project above. Figure 1 shows the requested
validation curves: every saved epoch checkpoint of the 15 members scored on the official val split
after training, inside the frozen bundle. The epochs selected on the held-out training folds (dots)
sit near but mostly before the val optimum; val would have picked a later epoch in 13 of 15
members and gained 0.008 AUROC on average. That choice was not made.

![Official val macro AUROC of every saved fine-tuning checkpoint. Thin lines: the 15 ensemble
members; dots: the epoch each member's held-out training fold selected; the orange line is the
mean over members at that epoch.](figures/val_curves.png){width=92%}

**Evaluation protocol.** Five folds of the training split (patients never shared across folds),
one out-of-fold prediction per patient from the model that did not fit on them, macro AUROC and
mAP over 40 targets with the common prior mask, plus Brier score, with 95% patient-level BCa
bootstrap intervals (1,000 resamples) and paired bootstraps for differences. Official val was
scored once, at the end, by `make val-score`: it recorded 294 artifact, data, code, and lock-file
hashes at freeze, refused to run on a changed tree or artifact, wrote a start record before any
metric and a completion record after, and refuses to run again without a documented reason.
Comparison models: an age-sex incidence table (the epidemiological floor, with and without the
"already has it" flag), L2 logistic regression on log code counts, and LightGBM on counts and on
counts plus latest values, each refitted on training patients with settings chosen by the same
folds.

## 3. AI workflow

Claude Code wrote most of the code, the tests, and the planning documents; I set the objective
and constraints, approved or rejected each decision that touched the data or the protocol, and read
every result before the next step. The division is recorded in `docs/ai_workflow.md` and the
reasoning for each decision in `docs/decisions.md`.

What kept the AI's work honest: the plan was written before the code and each revision says what
was known when it was made; 175 unit tests include mutation-style checks (changing `<` to `<=` in
the anchor rule fails a test; a future record must leave predictions unchanged; a resumed run must
match an uninterrupted one); the adoption rule was fixed before the first transformer result; and
every experiment block was committed before its scores.

Problems the review caught: fold-4 pretraining filled the disk with swap four times because the
MPS allocator fragmented across variable-length batches (fixed by releasing cached memory each
step, verified to change no computation, and 3 to 6 times faster); LightGBM and PyTorch each
loading an OpenMP runtime and crashing in one process (LightGBM now runs in its own process); the
first bootstrap intervals using the percentile method, which for rare targets barely contained the
point estimate (BCa replaced them); and a preflight audit flagging its own report files as a dirty
tree. [Derek: add anything else here, including any use of Codex CLI.]

## 4. Interpretation

**Where the model does well.** Seven targets are above 0.95 AUROC for every model: prostate
neoplasm, carcinoma in situ and metastasis of the prostate, normal pregnancy, pneumonia, obesity,
and the pregnancy-complication code. Each follows a visible Synthea pathway (PSA tests and rectal
exams, oral contraceptives, dementia care plans, BMI observations). A leakage review of these
scores found pre-anchor causes, not leaks. The transformer's clearest win is the paired brain
targets, silent micro-haemorrhage and abnormal gait (0.905 against 0.832 for LightGBM; the two
labels are identical for every patient because one module emits both), and stroke (0.801 against
0.783). Figure 2 shows every target.

![Official val AUROC per target, transformer ensemble against LightGBM on code counts, sorted by
the transformer. Numbers in parentheses are val positives; most targets have 5 to 20, so single-
target differences are noisy.](figures/val_per_target.png){width=74%}

**Where it does poorly, and why.** Nine targets are below 0.6 for every model: whiplash,
lacerations, wrist and ankle injuries, clavicle fracture, sinusitis, and strep throat. Synthea
generates these acute events with little dependence on the prior record, so no history model can
rank them; the age-sex table is at chance too. The transformer's largest losses to LightGBM are
the count-driven conditions: osteoporosis (0.804 against 0.895), recurrent rectal polyp (0.860
against 0.946), myocardial infarction (0.827 against 0.904), the three lung-cancer codes (0.76 to
0.79 against 0.83 to 0.85), and pathological fracture (0.847 against 0.909). These were also the
largest CV losses. In Synthea their risk grows with how many of a specific event has happened
(screening visits, bone-density tests, nodule follow-ups), which a count feature states directly;
the transformer has to infer it from a window that drops 15 to 18% of event tokens for exactly the
dense histories where counts are high. The count branch tested in round 2 did not close the gap,
and a longer context did not help on fold 0.

**Anchor era.** Anchors are five years before the last encounter, so an anchor before 2015
almost always means the patient died by 2020 (1,155 of 1,219 training patients), and those
patients have more new diagnoses (2.37 against 1.48). Scores split by era are similar for both
models (transformer 0.722 before 2015, 0.730 from 2015), and an era signal remains in the clinical
codes and age even without the clock codes (AUROC 0.865). It cannot be removed without removing
history, so it is reported.

**Calibration.** Prior-condition cells are exact zeros. Elsewhere the transformer's mean predicted
probability is slightly below prevalence on most targets and its Brier score is 0.0011 worse
than LightGBM's; the ensemble improved Brier over a single member by 0.0007.

**Limits.** The data are synthetic and the anchor is retrospective, so nothing here is evidence
about clinical use. Val is small: the macro AUROC interval is 0.04 wide and nine targets have at
most eight positives. The CV scores of the chosen configuration are the best of 23 trials on the
same folds, so they are optimistic; the val score is the first unbiased one, and the hidden test
score is the second. Two seeds of CV variation (about 0.002 AUROC in the mean, up to 0.014 per
fold) are comparable to several of the differences in the tuning table.

## 5. Results summary and reproduction

| Split | Model | Macro AUROC | mAP |
|---|---|---|---|
| train CV, pooled out of fold | transformer ensemble | 0.761 | 0.181 |
| train CV, pooled out of fold | LightGBM counts | 0.764 | 0.188 |
| **val** (365) | **transformer ensemble** | **0.751** [0.730, 0.773] | **0.209** [0.192, 0.211] |
| val | logistic regression, codes | 0.764 | 0.222 |
| val | LightGBM counts | 0.762 | 0.239 |
| val | LightGBM counts + values | 0.758 | 0.257 |
| val | age-sex incidence, at risk | 0.727 | 0.141 |
| test (358) | transformer ensemble | organisers' score | organisers' score |

Paired on val, transformer minus LightGBM counts: AUROC -0.010 [-0.031, +0.010], mAP -0.030
[-0.060, -0.011]. `predictions.csv` holds 358 rows by 40 code columns from the same 15 members.

Reproduction: `make setup`, link the dataset to `data/raw`, then `make dataset`, `make modalities`,
`make folds`, `make tokenize`, `make pretrain` and `make finetune` per fold and seed, `make freeze`,
`make val-score`, `make predict`. The Hugging Face package carries the 15 checkpoints and 5
tokenizers in the layout the prediction command reads; `python -m ehrforecast.hf_package verify`
reloads it on CPU and reproduced `predictions.csv` exactly, both in the development environment
and in a fresh virtual environment built from `requirements-lock.txt`.
