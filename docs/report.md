---
title: "Forecasting first diagnoses from synthetic EHR timelines"
subtitle: "Research intern coding test: technical report"
author: "Derek Wu"
date: "4 November 2026"
geometry: margin=1.7cm
fontsize: 10pt
mainfont: "Helvetica Neue"
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{htbp}
---

**Links.** Code: <https://github.com/Reaper8202/Research_Intern_Coding_Test>. Model package: <https://huggingface.co/Reaper8202/ehr-forecast>. wandb project: <https://wandb.ai/derek-wu820-university-of-toronto/ehr-forecast> (member run, seed 1 fold 0: <https://wandb.ai/derek-wu820-university-of-toronto/ehr-forecast/runs/cfcff203c2910f92>).

## Summary

| Split | Model | Macro AUROC | mAP | At-risk AUROC |
|---|---|---|---|---|
| official val (365) | **submitted blend** | **0.767 [0.746, 0.788]** | **0.258 [0.234, 0.268]** | 0.739 |
| official val | transformer ensemble alone | 0.751 [0.730, 0.773] | 0.209 [0.192, 0.211] | 0.722 |
| official val | LightGBM counts alone | 0.762 [0.740, 0.781] | 0.239 [0.209, 0.249] | 0.733 |
| official val | logistic regression on codes | 0.764 [0.742, 0.782] | 0.222 [0.195, 0.229] | 0.733 |
| official val | age-sex incidence table, at risk | 0.727 [0.705, 0.751] | 0.141 [0.125, 0.146] | 0.696 |
| train CV, pooled out of fold | submitted blend | 0.774 (0.780 mean fold) | 0.212 | |
| test (358) | submitted blend | organisers' score | organisers' score | |

The submission is a blend of three models trained from scratch on the 2,791 training patients: fifteen small decoder-only transformers (2 layers, width 128, about 738k parameters; three seeds times five folds, self-pretrained on pre-anchor events, then fine-tuned on the 40 labels) and two LightGBM models on code counts, one with the latest value of each numeric observation. Logits are averaged and a per-target Platt map fitted on training out-of-fold predictions scales the result. Intervals are 95% patient BCa bootstraps (1,000 resamples). Test labels are hidden; from val and CV I expect a test macro AUROC of 0.75 to 0.78 and mAP of 0.21 to 0.26. Most of the effort went into the pipeline: a truncation rule checked against the released test files, every choice made by five-fold CV on train with the adoption rules committed before the runs, two independent audits, and official val read three times in all, each read recorded.

## 1. Representation

**One clinical event is one token.** Every pre-anchor row becomes `source:kind:code` with its value folded in: `med:start:310798`, `proc:430193006`, `obs:8480-6:mm[Hg]:q7` (code, unit, seventh of up to ten quantile bins), `obs:72166-2:never_smoker`. A patient is a chronological sequence of these tokens (median about 170) behind a short prefix: sex, genotype tokens, the MRI token, and a summary of older diagnoses.

**Fields.** Every table is read through a column allowlist. I dropped what summarises the whole record rather than a moment in it: death date, lifetime costs and coverage, marital status, addresses, payer columns, and device UDIs (which encode manufacture and expiry dates). The tricky one was medication dispense counts: even in the truncated test files, 873 of the 1,049 medications that extend past the anchor show more dispenses than months elapsed, so the column counts future refills. Race and ethnicity are out; a LightGBM check showed no measurable cost (-0.002 AUROC, interval straddling zero). I also removed DALY, QALY, and QOLS: Synthea logs them yearly from 2011, so their count is a clock for the anchor year, and an early anchor almost always means the patient died. Not a temporal leak, but the model would be reading how the dataset was built. It cost about 0.014 CV AUROC for the transformer and 0.005 for the blend, which I'd rather state than hide.

**Modalities.** Genotypes enter as one token per carried variant; a pre-anchor brain MRI enters as one token from a small 3-D CNN trained from scratch. The copy I worked from had an incomplete genomics download (the organisers' log says so; 400 files, 396 of them training patients, listed in `reports/dataset/modality_files.json`), scans for 200 training and 28 test patients, and near-identical simulated volumes. I had decided before the result to keep the modalities and submit either way: +0.0005 AUROC overall in CV, -0.003 on the 541 patients whose inputs changed. In this copy, having a genotype file correlates with outcomes (older patients, more diagnoses), most likely an artifact of the partial download; CV says the network doesn't exploit it, and two test patients have one. Whether real modalities would help at this size is a question this dataset can't answer; on real data they have (Soenksen et al., 2022; Acosta et al., 2022).

**Continuous values.** Up to ten quantile bins per code and unit, fitted within each fold's training patients; units never converted (only five codes have more than one); no external thresholds. A LightGBM check found raw values no better than bins (0.243 vs 0.244 mAP). Recent benchmarks also favour fused bins (Lee et al., 2026; McCann et al., 2026).

**Time.** Every token carries the patient's exact age at the event as age/100 plus sine and cosine terms at eight periods from six months to 200 years, added to the token embedding, so attention can read the gap between any two events; RoPE gives order. The head also gets the time from the last event to the anchor, age at anchor, sex, and an empty-history flag.

**Context.** Sequences are capped at 1,024 tokens, keeping the most recent complete days. Diagnoses that fall off become up to 128 summary tokens at the front with their first-diagnosis age. Exact refills (stop and restart of the same drug the same day; 160,150 in one fold) are merged first. Only 4.6% of patients are truncated, but they are the densest histories: 15 to 18% of all event tokens fall outside the window. Contexts of 512 and 2,048 didn't help on fold 0.

**Leakage.** The brief's anchor sentence can be read four ways; I simulated all four and kept the one whose last-event-to-anchor gaps match the released test files. The pipeline halts if any post-anchor date survives; a test adds a future record and checks no prediction changes; labels never reach the prediction interface; every fitted object, from the vocabulary to the Platt maps, is fitted inside its fold, which the tests and two read-only audits checked.

## 2. Model and training

**Architecture.** A decoder-only transformer, structurally a miniature GPT: two blocks, width 128, four causal heads, feed-forward 512 with GELU, pre-LayerNorm, RoPE, dropout 0.3. Token embeddings plus a learned projection of the 17 age features. The pretraining head uses tied weights over the vocabulary with input-only symbols masked. The prediction head concatenates the last-event state, the mean state over events, and the four head features (260 values) and maps to 40 logits with sigmoids. The MRI encoder is a four-layer 3-D CNN (81k parameters). Total 737,984 parameters, most in the embedding (about 1,930 tokens). With 2,791 patients, overfitting is the main risk, so 2 x 128 was the default; the one larger model tried, 4 x 192 (2.26M parameters), scored worse on the fold-0 screen (0.758 vs 0.772).

**Objectives.** Pretraining: next-token cross-entropy over each patient's pre-anchor history, one random window per patient per epoch, target-condition start tokens weighted 5x, stopping on fixed held-out windows from the fold's held-out training patients (never val); every run hit the 30-epoch cap with the loss still falling slowly, to about 1.5 nats per token. Pretraining on full timelines (post-anchor events included) passed my adoption rule and I rejected it, because the brief says to truncate at the anchor. Fine-tuning: binary cross-entropy over the 40 outputs, with cells where the patient already has the condition excluded from the loss and set to zero at prediction. Each patient also contributes one extra example per epoch from a random earlier cut day, with labels rebuilt relative to it.

**Hyperparameters.** AdamW, float32. Pretraining: lr 0.001, betas (0.9, 0.95), weight decay 0.1, 10% warm-up then cosine, clip 1.0, batch 32, up to 30 epochs, patience 3. Fine-tuning: head lr 0.001, body lr 0.0001, weight decay 0.01, clip 1.0, batch 32, dropout 0.3, patience 5 on held-out masked macro AUROC; best epochs 3 to 13. One fold takes about 20 minutes on a 16 GB M-series Mac.

**Three methods, compared before tuning:** S (from scratch), H (pretrained body frozen, head only), F (pretrained body and head fine-tuned). F won on all five folds, mean AUROC 0.760 vs 0.744 (S) and 0.742 (H), paired F minus S +0.020 [+0.013, +0.028]. Pretraining works if the body is allowed to adapt.

**Tuning.** Every round was a budget file committed before its runs, with one rule: mean AUROC gain of at least 0.005, higher on at least four of five folds, mAP loss at most 0.005. Round 1 found fine-tuning overfitting (train AUROC 0.84 to 0.90 vs held-out 0.75 to 0.77) and adopted dropout 0.3; five other variants failed, including longer context and 60 pretraining epochs. Round 2, after the clock codes were removed, adopted the last-plus-mean readout, the extra anchors, and target-weighted pretraining together (five of five folds); a count branch, loss weights, and dropout 0.5 failed. Three seeds of the winner give the 15 members. Round 3 tried the larger architecture and the blend. One transparency note from the final audit: round 3's budget file was committed at 21:18 on 3 November while its stacking job was already running, so its stacking and calibration rules can't be shown to have been fixed before any number was visible; the architecture screen started after the commit. The rules weren't changed afterwards, and the blend passes them by margins well outside the intervals.

**Comparison models and the blend.** An age-sex incidence table, L2 logistic regression on code counts, and LightGBM on counts and on counts plus latest values, all tuned on the same folds; LightGBM's grid picked its simplest corner (4 leaves, 100 trees), on the grid edge on all three axes, and I didn't extend it. In CV the transformer ensemble and LightGBM counts were tied on AUROC (0.761 vs 0.764 pooled) and LightGBM led on mAP. Nested stacking (the combiner fitted on the other four folds' out-of-fold predictions) showed a parameter-free mean of logits beating the transformer by +0.014 AUROC [+0.009, +0.019] and LightGBM by +0.011 [+0.007, +0.014]; a learned logistic stacker gained only 0.001 more, under the rule's 0.003 margin, so the fixed blend stayed. Nested per-target Platt scaling lowered Brier on four of five folds and calibration error from 0.013 to 0.010 without changing any ranking.

![Training curves of the 15 members: pretraining held-out next-token loss; fine-tuning training loss and held-out fold macro AUROC per epoch (thin lines: members; orange: mean while at least five are training). The held-out AUROC flattens after about epoch 7 while the training loss keeps falling, which dropout and early stopping hold in check.](figures/training_curves.png){width=76%}

**Validation curves.** Figure 2 scores every saved epoch checkpoint of the 15 members on official val, inside the frozen bundle. The epoch each held-out fold chose sits mostly before the val optimum; choosing on val would have gained 0.008 AUROC in 13 of 15 members, and I didn't.

![Official val macro AUROC of every saved fine-tuning checkpoint. Thin lines: members; dots: the epoch chosen on the held-out training fold; orange: the mean at that epoch.](figures/val_curves.png){width=48%}

**Evaluation protocol.** Five-fold CV on the training split (shuffled, seed 20261028, patients never shared; the rarest target has 4 to 6 positives per fold); macro AUROC and mAP after the common prior mask, Brier, at-risk variants, paired bootstraps for differences. Official val was scored only through a frozen bundle that hashes every artifact, refuses to run on a modified tree, logs a start record before any metric, and refuses a second run without a written reason. It ran twice: the transformer ensemble alone (0.751 / 0.209), then the blend. The step-2 baselines had scored val once before the bundle existed, so val was read three times in all. Round 3 used training CV only, but it was opened after the first bundle run had shown LightGBM ahead on val mAP, so the second run is a second look at the same 365 patients, not an independent check. The hidden test set is the independent check.

## 3. AI workflow

I started without any AI: I read BEHRT, Med-BERT, ETHOS, Context Clues, and two 2026 tokenisation benchmarks by hand, then fed them to the models as context, so the AI reasoned from sources I had already read. Planning was a conversation, not a prompt: an interrogation-style session (the "grill me" workflow) pressed on every part of the design, from what counts as a leak to when to read val, until I had answers I could defend. The result was a written plan with dated revisions, finished before the code, and the adoption rule was fixed in it before the first transformer result.

Implementation used Codex (GPT Astra, via the Codex CLI) and Claude (Opus 5.5 and Fable 5.1 sessions, via Claude Code). I broke the plan into subunits small enough to test alone (from the anchor rules to the val bundle and the package); each was implemented by one AI and audited and tested by the other. I approved or rejected each decision that touched the data or the protocol and read the results before the next step.

The rule underneath it: no AI output was trusted until a different process had attacked it. The writer of a subunit was never its judge; the 199 tests were written to break the code (changing `<` to `<=` in the anchor rule fails a test; a future record must leave every prediction unchanged); every tuning round was committed before its scores; a table in the plan says what each operation may read; and at the end I had 74 audit criteria written and committed first, then applied by fresh sessions that could read the repository but not edit it. Their verdicts came from the files, and that is how the three things I'd least have wanted a reviewer to find first were found: the genotype-file correlation, the era signal surviving the clock-code removal, and the mis-dated round-3 budget. Each is disclosed where it belongs. The AI wrote more lines than I did; the data boundary, the selection rules, and every call that changed the data or the protocol were mine, and the decision log is where that can be checked.

## 4. Interpretation

**Strong targets.** Seven score above 0.95 on val (six of them for the transformer and LightGBM too): neoplasm of prostate (0.998), normal pregnancy (0.993), prostate metastasis (0.993), prostate carcinoma in situ (0.992), pneumonia (0.980), obesity (0.979), fetus with unknown complication (0.964). Each follows a visible Synthea pathway (PSA tests before the prostate codes, contraceptives before pregnancy, dementia care plans before pneumonia, BMI before obesity); a leakage review of every score above 0.98 found a pre-anchor cause.

**Where the blend repairs the transformer.** The transformer loses to LightGBM on count-driven conditions (rectal polyp, myocardial infarction, pathological fracture, osteoporosis, colon polyp). In Synthea their risk grows with the number of specific earlier events; a count feature states that number directly, while the transformer must infer it from a window that drops 15 to 18% of event tokens for exactly the dense histories where counts are high. The blend closes most of the gap (myocardial infarction 0.827 to 0.927, osteoporosis 0.804 to 0.883, pathological fracture 0.847 to 0.911); the lung-cancer codes stay below LightGBM.

**Where the transformer leads.** The paired brain targets, silent micro-haemorrhage and abnormal gait (0.905 vs 0.832; identical labels, one module emits both), heart failure, stroke, and concussion: conditions that depend on order and timing, which a bag of counts cannot see. The blend keeps most of it (0.892 on the brain pair) and is above the transformer on 29 of 40 targets and above LightGBM on 17. The two models fail on different targets; that is why the blend works (Figure 3).

![Official val AUROC per target for the blend, the transformer ensemble, and LightGBM on code counts, sorted by the blend. Numbers in parentheses are val positives; most targets have 5 to 20, so single-target differences are noisy.](figures/val_per_target.png){width=44%}

**Where no model works.** Nine targets are below 0.6 for the blend (sprains, fractures, lacerations, sinusitis, strep throat, whiplash), five of them for every model including the age-sex table. Synthea generates these acute events with almost no dependence on the prior record. They cap the macro average: even an oracle choosing the better of the transformer and LightGBM per target reaches only 0.778 in CV, so 0.80 is out of reach on this data without the clock codes.

**The death window.** The anchor is five years before the last encounter, so an anchor before 2015 almost always means the patient died by record end (1,155 of 1,219 training patients), and those patients have more new diagnoses (2.37 vs 1.48). The era signal survives the clock-code removal (era AUROC 0.876 on fold 0 from age and clinical codes; it cannot be removed without removing history). It is concentrated: 43.7% of training patients have a pre-2015 anchor, and nine targets have at least 90% of their positives there (the prostate and lung-cancer codes, heart failure, pneumonia, myocardial infarction). On training out-of-fold predictions those nine score 0.925 overall but 0.879 ranking within the pre-2015 group, and the 40-target mean falls from 0.745 to 0.718 within it. So part of every model's score is telling patients in their last five years from the rest; the within-group numbers are the fairer measure of risk ranking among patients with the same horizon.

**Calibration and limits.** Prior-condition cells are exact zeros; the Platt map corrects a slight under-prediction in the middle of the range; val Brier is 0.0365 for the blend vs 0.0381 for the transformer alone. The data are synthetic and the anchor is retrospective, so nothing here is evidence about clinical use. Val is small (interval about 0.04 wide; eleven targets with at most eight positives). The CV number is optimistic: the chosen configuration is the best of 28 transformer configurations and three stackers on the same folds (mean-fold CV 0.780 vs 0.767 on val). The LightGBM setting sits on the edge of its grid, and the grid was not extended.

## 5. Reproduction

`make setup`, link the dataset to `data/raw` (restricting `genomics/` to the listed files), then the `make` targets in the order the README gives, from `make dataset` to `make package`. The Hugging Face package holds the 15 checkpoints, 5 tokenizers, 80 LightGBM boosters, and the Platt weights; `python -m ehrforecast.hf_package verify` rebuilt `predictions.csv` (358 rows by 40 code columns) exactly on CPU, in the development environment and in a fresh virtual environment from `requirements-lock.txt`. Paired on val, blend minus transformer: AUROC +0.016 [+0.002, +0.030], mAP +0.049 [+0.033, +0.072]; blend minus LightGBM counts: +0.005 [-0.005, +0.016], +0.019 [+0.002, +0.040].

## References

Acosta JN et al. Multimodal biomedical AI. *Nat Med* 28, 1773–1784 (2022). <https://doi.org/10.1038/s41591-022-01981-2> · Lee I et al. Representation Before Training: A Practical Benchmark for Generative Medical Event Model Tokenization. arXiv:2604.16775 (2026) · Li Y et al. BEHRT. *Sci Rep* 10, 7155 (2020) · McCann KA et al. Continuous Value Tokenization Improves Medical Event Foundation Models. medRxiv (2026) · Rasmy L et al. Med-BERT. *npj Digit Med* 4, 86 (2021) · Renc P et al. Zero shot health trajectory prediction using transformer (ETHOS). *npj Digit Med* (2024) · Soenksen LR et al. Integrated multimodal artificial intelligence framework for healthcare applications. *npj Digit Med* 5, 149 (2022). <https://doi.org/10.1038/s41746-022-00689-4> · Wornow M et al. Context Clues. ICLR 2025
