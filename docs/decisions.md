# Decisions and evidence

Each entry states the decision, the reason, and the check that supports it. Numbers come from the
released data (train+val 3,156 patients, test 358).

## D1. Anchor date and history cut-off

**Decision.** `anchor_date` = calendar day of (last encounter end − 5 years), where the last
encounter end is the latest `STOP` (or `START` when `STOP` is blank). History = events dated
strictly before `anchor_date`.

**Evidence.** Four conventions were simulated on train/val and compared with the test files by
the share of patients whose last history event is 0–4 days before the anchor:

| Convention | 0 d | 1 d | 2 d | 3 d |
|---|---|---|---|---|
| history < floor(T), anchor = floor(T) (chosen) | 0.0% | 4.7% | 0.9% | 0.7% |
| history < T, anchor = floor(T) | 1.9% | 4.4% | 0.8% | 0.6% |
| history < T, anchor = floor(T)+1 | 0.0% | 1.9% | 4.4% | 0.8% |
| history < floor(T)+1, anchor = floor(T)+1 | 0.0% | 2.2% | 4.4% | 0.7% |
| **Released test files** | **0.0%** | **3.4%** | **0.3%** | **1.1%** |

(T = last encounter − 5 years.) Only the chosen convention has no events on the anchor day and no
peak at 2 days. Subtracting calendar years (not 1,825 days) keeps the 1-day peak where the test
files have it.

## D2. Rows that span the anchor

**Decision.** Keep a row whose start is before the anchor; blank its `STOP` when the `STOP` is on
or after the anchor day.

**Evidence.** Train encounters have no blank `STOP`, but test encounters have 5. The rate fits:
51 of 3,156 train/val patients (1.6%) have an encounter that spans the anchor, which predicts
about 6 test patients. Blank-`STOP` share after our truncation versus the test files:

| Table | train/val (ours) | test (released) | if spanning rows were dropped |
|---|---|---|---|
| encounters | 0.0003 | 0.0003 | 0.0000 |
| conditions | 0.731 | 0.730 | 0.729 |
| medications | 0.036 | 0.044 | 0.012 |
| careplans | 0.657 | 0.667 | 0.648 |

The medication gap is sampling noise: resampling 358 train patients 2,000 times gives a 95% range
of 0.030–0.045.

## D3. Columns left out

**Decision.** Read every table through an allowlist (`schema.py`).

**Evidence.** 1,354 of 3,156 train/val patients have a `DEATHDATE`; no test patient has one.
`HEALTHCARE_EXPENSES` and `HEALTHCARE_COVERAGE` are lifetime totals. In the test files, 873 of
1,049 medications that continue past the anchor have more `DISPENSES` than months elapsed before
the anchor, so dispense counts and total costs include future refills even in test. `MARITAL` and
the address are end-of-record values. `devices.UDI` encodes manufacture and expiry dates.

## D4. Label window

**Decision.** Positive when the first-ever diagnosis date is in `[anchor_date, anchor_date + 5
years]`, both ends included. First-ever dates come from the full timeline.

**Evidence.** 119 of 5,899 positive labels (2%) are first diagnosed on the day of the last
encounter, which is the window end. A half-open window would drop them, but they are inside "the
five years after the anchor" by the task definition. No condition is dated after the last
encounter, so the closed window equals "anchor to end of record". No patient has a 29 February
last encounter, so the leap-day edge case does not occur in this data.

## D5. Optional modalities

**Decision.** Do not use genomics or DICOM imaging. Use `imaging_studies.csv` events.

**Superseded on 1 November 2026 by D34.** Derek required a multimodal model.

**Evidence.** `___All_Errors.txt` reports that the genomics folder did not download completely.
Of the 400 genomics files present, 396 are train, 2 val, 2 test. DICOM covers 28 test patients
and is 9.2 GB. The README says the CSV tables are enough.

## D6. Baseline protocol

**Decision.** One L2 logistic regression per target on log-count features (whole history and
last 2 years), age, age bands, and sex. `C` is chosen by 5-fold cross-validation on train, with the
vocabulary refitted inside each fold. The final model is fitted on train only and scored once on
val. Primary metric: macro AUROC; also mAP, BCa bootstrap 95% intervals (1,000 patient
resamples), and AUROC among patients who did not already have the condition ("at risk").

**Reason.** Choosing `C` on val would make the val score optimistic, and val is the only labelled
held-out set. A demographics-only model is fitted the same way to show what the history adds.

## D7. BCa instead of percentile bootstrap intervals

**Decision.** Use BCa (bias-corrected and accelerated) intervals from `scipy.stats.bootstrap`.

**Evidence.** The first run used percentile intervals and gave val mAP 0.217 with interval
[0.210, 0.268]: the point value was almost at the lower end. Average precision on resampled data
is biased upwards for rare targets, because duplicated positives raise precision at the top of
the ranking. BCa corrects this bias and gives [0.185, 0.225].

## D8. Sex- and age-stratified incidence as the epidemiological reference

**Decision.** Add two incidence-table baselines: the observed five-year rate of first diagnosis
per sex and age band, and the same rate among patients at risk (prior cases excluded from the
denominator and given probability 0). Small strata are smoothed towards the sex-specific rate
with a pseudo-count. Band width (2.5, 5, 10, 20 years) and pseudo-count (1, 10, 100) are chosen
by the same cross-validation on train as `C`; the result is 5-year bands and pseudo-count 10,
inside both grids.

**Reason.** A learned model is worth something only if it beats the population rates. The
at-risk version also isolates the "already has it" effect seen in D6, so the gain from the
clinical history can be measured on its own. The codes model is compared with every baseline by
a paired BCa bootstrap of the difference on the same val patients, because overlapping marginal
intervals do not show whether one model is better.

**Implementation note.** The bootstraps call AUROC and AP about one million times per run. The
sklearn functions spend most of that time on input validation, so `metrics.py` has exact
rank-based versions. Unit tests check that they equal the sklearn values, ties included.

## D31. Interval stops recorded before their own start

D9–D30 are reserved for the planned decisions in `docs/model_plan.md`. This implementation
decision therefore uses the next free number. Derek approved it on 31 October 2026.

**Decision.** When an interval row has a `STOP` earlier than its own `START`, keep the recorded
start and omit the stop. Apply the rule after the anchor cut and before event construction, to
every split, including the released test files. The raw files stay unchanged. Encounters are
excluded: their ends define the anchors, so an invalid encounter stop stops the pipeline.

**Evidence.** The P3a raw-data audit found 57 training medication rows (27 patients) with a stop
2 to 6 days before the start. These are not same-day rounding errors. The anchor cut removes 8 of
these rows. After the cut, 49 training, 3 val, and 6 test rows (22, 1, and 3 patients) remain.
No other interval table has such a row in any split. In the typical case, the invalid row is a
second prescription of the same drug, from the same encounter, inside a valid prescription.

**Which field is wrong (`scripts/checks/medication_stops.py`, training patients).** The start
falls on an encounter day of that patient in 57 of 57 invalid rows and in 97.8% of valid rows. The
stop does so in 8 of 57 invalid rows and in 95.8% of valid rows. So the stop is the wrong field.

**Reason.** The start is a real prescription event; only its stop date is impossible. Each stop
is linked to its start only through its own row. Other prescriptions of the same patient and drug
cannot identify the correct stop: 1,579 train/val rows share a patient and code with an invalid
row. Removing all stops for that patient and code would delete valid stops. The rule reads only
the row's own pre-anchor dates, so later records cannot change a prediction history.

**Effect.** The rebuilt `events.parquet` has 58 fewer medication `stop` rows (1,431,561 to
1,431,503). All other step-1 outputs are byte-identical. The baseline features count only
`start` and `point` events, so the step-2 baseline is unchanged. P3a reproduces its training-CV
scores within 1e-12. `reports/dataset/omitted_stops.csv` lists every omitted stop.

**Implementation note.** `truncation.omit_invalid_stops` applies the rule. `build_events` rejects
a remaining invalid stop, so no caller can skip the repair. `events.manifest.json` records the
rule version (`omit-stop-before-start-v1`) and the event-file hash. The prediction loader rejects
an event table without a matching manifest, for example one built before this rule.

## D32. Refill compression links each stop to its own start

**Decision.** Merge a medication stop with a later start of the same drug when the start is at
most one day after the stop and is not the stop's own start. Remove both events, so the treatment
span continues. Each start merges at most once. The event table stores `interval_start`, the
`START` of the source row, on both start and stop events. It is the link to the own start.

**Evidence.** M1 on fold-0 fit patients: 160,150 stops are followed by a restart at exactly the
same timestamp. No restart falls more than 0 and at most 2 days after a stop. 14 fall 2 to 7 days
after, and 585 later. The one-day threshold therefore merges only exact refills. The raw training
medications contain 39,274 zero-length prescriptions (stop equal to start). Without the link, the
stop of a repeated zero-length prescription would merge with its own start and delete it.

**Effect.** Event schema `events-v2` and input schema `prediction-inputs-v3`. Row count and
history rule are unchanged. Compression runs inside the fold tokenizer, after the anchor cut, so
a restart after the anchor cannot remove a visible stop. E14 can still compare compression on
and off, because `events.parquet` itself stays uncompressed.

## D33. Release cached MPS memory after each training step

**Decision.** On MPS, call `torch.mps.empty_cache()` after each optimizer update and after each
held-out evaluation (`training.release_cached_memory`). The call returns only memory blocks that
no tensor uses. It changes no computation and no random draw.

**Evidence.** Sequence lengths change from batch to batch, so the MPS cache fragments and grows.
Folds 0–3 completed without the call; they ran while the disk had free space for swap. Fold-4
pretraining then failed four times on 31 October 2026. On this 16 GB Mac one process passed 20 GB,
and swap filled the disk (`No space left on device`). Hard allocator limits failed because of the
fragmentation (out of memory at 4.7 GB and 7.7 GB limits, with 0.03 GB of live tensors between
steps). A 40-step probe on fold-4 data: driver memory reached 7.8 GB without the call and stayed at
0.1–0.5 GB with it. After 20 identical steps, the maximum weight difference with the call was
2.1e-7. Two runs without the call already differed by 2.6e-7, the normal MPS run-to-run variation.
The call added under 3% to the probe time.

**Effect.** Fold 4 used the call; folds 0–3 did not. Their computations are the same. The
restarted fold-4 run reproduced the first epochs of the failed attempts to four decimals. The
failed attempts were moved out of `models/` and were not used. Later runs use the call.
The call also made training 3–6 times faster: fold-4 pretraining took 10 minutes, against
20–61 minutes for folds 0–3, because memory pressure had slowed every step.

## D34. Multimodal inputs: genomics and pre-anchor brain MRI

**Decision.** Derek required a multimodal model and chose to submit it whatever the CV result.
The configuration `small-v1-mm` adds two inputs to `small-v1` and changes nothing else:

* Genomics: `[DNA]`, then one `dna:<site>:<variant alleles>` token per carried genotype, after
  `[SEX]`. A genotype token needs at least 5 fit carriers in the fold. Age is 0 (germline).
* Brain MRI: one `[IMG]` token after the genotype tokens. A 3-D CNN from random weights (four
  stride-2 convolutions with 8 to 64 channels, mean pooling, one linear layer; 81,232 parameters)
  maps the volume to the token embedding. Its age is the age at the scan.

**Date rules.** A scan enters only when its study date is before the anchor day, the rule of every
event table (`truncation`). A pretraining window also needs the scan before its first day, as for
summary conditions, so no next-token target follows a scan shown earlier in the window. Genotypes
carry no dates. Both modality tokens are input-only and are never next-token targets.

**Preprocessing.** `make modalities` reads every file once. Genotype: the number of variant
alleles over the two strands. Volume: area averaging from 256 cubed to 64 cubed, then the volume's
own mean and standard deviation, clipped to [-5, 5]. No statistic is shared between patients, and
no external weights or atlases are used. One file has truncated pixel data; it is recorded in
`reports/dataset/modalities.json` and skipped.

**Coverage.** Genomics: 396 train, 2 val, 2 test patients (the download log reports an incomplete
folder). Readable scans before the anchor day: 200 train, 19 val, 28 test. Scans on or after the
anchor day (34 train, 11 val) are excluded. The volumes are near-identical simulated brains:
the correlation of 32-cubed block means between patients is 0.995 to 0.999.

**Expected effect.** Small. Genomics reaches 2 of 358 test patients, and the images vary little.
The comparison with `small-v1` is reported for all patients and for patients with a modality.

**Result (training CV, 1 November 2026).** Section 5.3 again selected F within `small-v1-mm`
(AUROC gain over S +0.008, 4 of 5 folds). Multimodal F minus CSV-only F, pooled out of fold,
paired patient BCa 95%: AUROC +0.0005 [-0.006, +0.007], mAP +0.009 [-0.000, +0.017]. On the 541
training patients with a genotype file or a pre-anchor scan, the only patients whose inputs
changed: AUROC -0.003 [-0.020, +0.015], mAP -0.013 [-0.053, +0.020]. The modalities therefore
show no measurable benefit. The small all-patient mAP difference does not appear where the inputs
differ, so it is most likely training variation (a different vocabulary and random
initialization); seed variation was not measured. The multimodal model is submitted by Derek's
decision. `reports/transformer/small-v1-mm/` holds the full comparison.

**Verification.** `tests/test_modalities.py`: genotype counting, DICOM resampling, a truncated
file, cache hashes, fit-only genotype vocabulary, the anchor-day and window rules (a mutation of
`<` to `<=` fails a test), the context budget, and image-to-row routing in the model. A CSV-only
tokenizer keeps its hash, and the trained `small-v1` fold-0 model reproduces its saved held-out
predictions (maximum difference 1.2e-6, CPU against MPS). `make pipeline-smoke
SMOKE_CONFIG=configs/smoke_mm.toml` passed with images and genotypes in the inputs.

## D35. Tuning round outcome: fine-tuning dropout 0.3 is selected

D36 replaces this selection as the candidate. This configuration stays as the fallback.

**Decision.** The selected configuration is `small-v1-mm-do03`: the multimodal model (D34) with
method F and fine-tuning dropout 0.3 (`configs/experiments/mm_dropout03.toml`). It fine-tunes
from the incumbent's self-pretraining; only the fine-tuning dropout and head dropout change.

**Round.** `configs/budget.toml`, fixed before any of its scores (commit 39e6b1c), with two dated
revisions: the section 5.4 recovery round (874c962) and the time-saving scope change (3c9ab12).
Rule (section 5.3): mean macro AUROC gain at least 0.005, higher AUROC on at least 4 of 5 folds,
mean mAP loss at most 0.005, against the incumbent's fold scores. Training CV only.

| Experiment | Folds | Mean AUROC | Mean mAP | AUROC gain | Folds higher | mAP loss | Result |
|---|---|---|---|---|---|---|---|
| Incumbent `small-v1-mm` | 5 | 0.7591 | 0.1971 | | | | |
| T1 dropout 0.3 | 5 | 0.7662 | 0.2033 | +0.0071 | 5 | -0.0062 | passes |
| T2 body learning rate 3e-5 | 5 | 0.7612 | 0.1930 | +0.0021 | 4 | +0.0041 | fails |
| T3 weight decay 0.1 | 5 | 0.7592 | 0.1968 | +0.0001 | 4 | +0.0003 | fails |
| T4 T1 + T2 + T3 | 5 | 0.7659 | 0.1897 | +0.0068 | 5 | +0.0074 | fails (mAP) |
| R1 age-sex head | 5 | 0.7627 | 0.1991 | +0.0036 | 4 | -0.0020 | fails |
| R2 age-sex head + dropout 0.3 | 5 | 0.7664 | 0.2012 | +0.0073 | 4 | -0.0040 | passes |
| P1 60 pretraining epochs | 5 | 0.7554 | 0.1885 | -0.0036 | 3 | +0.0086 | fails |
| E5 context 512 (screen) | fold 0 | | | +0.0019 | | +0.0157 | fails screen |
| E5 context 2,048 (screen) | fold 0 | | | +0.0018 | | +0.0228 | fails screen |

**Seed stability (E0).** Three seeds of the incumbent: mean AUROC 0.7591, 0.7579, and 0.7608;
mean mAP 0.197, 0.193, and 0.205. Per-fold AUROC ranges 0.0015 to 0.010. Against the per-fold
mean of the three seeds, T1 still passes: +0.0069 AUROC, 4 of 5 folds, mAP +0.0051. The AUROC
gain is about twice the spread of the seed means; the mAP gain is within the mAP seed spread.

**Selection.** T1 and R2 pass and lie within 0.005 of each other on both metrics (R2 AUROC
+0.0002, mAP -0.0021). The section 5.3 order prefers fewer supervised parameters: the T1 head has
5,320 parameters and the R2 head 5,800. T1 is selected. The combination slot was not used,
because no passing change came from a different family.

**Pooled out of fold, paired patient BCa 95% (`reports/transformer/small-v1-mm-do03/`).** T1 minus
the incumbent: AUROC +0.009 [+0.004, +0.014], mAP +0.007 [+0.001, +0.014]. T1 minus the step-2
logistic regression: AUROC +0.026 [+0.017, +0.036], mAP +0.012 [+0.002, +0.023].

**Interpretation.** Fine-tuning overfitting was the main limitation that this budget could fix.
The age-sex head helps on its own but adds nothing beside dropout. A lower pretraining loss (P1:
1.389 against 1.405 on fold 0) did not give better predictions, and neither context length helped.

**Comparison with LightGBM (pooled out of fold, paired patient BCa 95%,
`reports/lightgbm/transformer_comparison.json`).** T1 minus LightGBM counts: AUROC -0.014
[-0.021, -0.006], mAP -0.025 [-0.036, -0.013]. T1 minus LightGBM counts + values: AUROC -0.010
[-0.017, -0.001]. The incumbent's AUROC gap was -0.023, so the round closed about 40% of it.
Section 5.4 applies: the transformer remains the submission and the gap is reported.

**Caveats.** The selected configuration is the best of 11 trials on the same folds, so its CV
score is somewhat optimistic; the single official val check (P4e) is the unbiased estimate.
E0-seed3 fold 3 was resumed from its epoch-6 checkpoint after the 18:10 restart (the G2-tested
resume path). Its first wandb session ends with an incomplete record, so wandb shows that
run's steps out of order; its `metrics.jsonl` is the authoritative record. Pooled out-of-fold AUROC (0.760 for T1) is lower than the mean fold AUROC (0.766)
because pooling mixes fold-specific probability scales.

## D36. Round 2: simulator clock codes removed; C1 with three seeds is the candidate

**Decision.** The candidate is `small-v2-c1` with three seeds (`configs/experiments/v2_c1.toml`,
`v2_c1_seed2.toml`, `v2_c1_seed3.toml`). It averages 15 fold models. Against D35 it has four
changes: DALY, QALY, and QOLS leave the inputs; the head reads the last event and the mean event;
each fit patient gives one extra supervised example per epoch from an earlier cut day; and target
diagnoses weigh 5 in the pretraining loss. D16 stays in force: pretraining reads pre-anchor
histories only. The author reviews this choice before P4e.

**Why the three codes leave.** DALY, QALY, and QOLS are simulator scores recorded yearly from 2011
to 2021, so their counts work as a calendar clock. They predict an anchor before 2015 with AUROC
0.999 (`scripts/checks/clock_scan.py`). Of those 1,219 training patients, 1,155 died, and they have
more new diagnoses (2.37 against 1.48 per patient). All these rows are before the anchor, so this
is not a future leak. The problem is validity: the model reads how the anchor was built, not a
clinical state. Removal costs score: A1 trails the D35 model by 0.0143 AUROC and 0.0153 mAP, and
LightGBM counts falls from 0.773 to 0.764 pooled AUROC. The test files contain the codes too (1,081
DALY rows), so removal can lower the hidden score. An era signal remains in clinical codes and age
(AUROC 0.865). It is kept, because it cannot be removed without removing clinical history, and it
is reported below.

**Round.** `configs/budget_v2.toml`. Each block was committed before its scores (15d569d, 6791ab2,
ed19115, bce43e9, 02e8074, dedc88e). Rule: section 5.3, against A1 unless stated. Training CV only;
official val was not read.

| Run | Change | AUROC | mAP | AUROC gain | Folds higher | mAP gain | Result |
|---|---|---|---|---|---|---|---|
| A0 | clean inputs (against `small-v1-mm`) | 0.7461 | 0.1831 | -0.0130 | 1 | -0.0140 | measures the cost |
| A1 | A0 with dropout 0.3 (against D35) | 0.7518 | 0.1880 | -0.0143 | 0 | -0.0153 | reference |
| A2 | dropout 0.5 | 0.7456 | 0.1682 | -0.0062 | 2 | -0.0198 | fails |
| A3 | last plus mean readout | 0.7549 | 0.1927 | +0.0030 | 4 | +0.0046 | fails, enters C1 |
| X1 | extra anchors | 0.7560 | 0.1887 | +0.0042 | 4 | +0.0007 | fails, enters C1 |
| P2a | full-timeline parent (against A0) | 0.7489 | 0.1832 | +0.0028 | 3 | +0.0000 | fails |
| P2b | fine-tune from P2a | 0.7569 | 0.1955 | +0.0050 | 5 | +0.0075 | passes, not adopted |
| F1 | count branch | 0.7513 | 0.1888 | -0.0005 | 2 | +0.0007 | fails |
| F2 | per-target loss weights | 0.7506 | 0.1796 | -0.0013 | 1 | -0.0084 | fails |
| P3a | target-weighted parent (against A0) | 0.7473 | 0.1806 | +0.0012 | 3 | -0.0026 | fails |
| P3b | fine-tune from P3a | 0.7539 | 0.1864 | +0.0020 | 4 | -0.0016 | fails, enters C1 |
| C1 | A3 + X1 + P3b | 0.7601 | 0.1982 | +0.0083 | 5 | +0.0101 | passes |

**Combination.** The component test was: mean AUROC gain above 0, higher on at least 4 of 5 folds,
mean mAP loss at most 0.005. C1 joins the components that pass, then needs the section 5.3 rule by
itself. Pooled out of fold, paired patient BCa 95%: C1 minus A1 is AUROC +0.010 [+0.003, +0.018]
and mAP +0.006 [-0.002, +0.014].

**Full-timeline pretraining is not adopted.** P2b passes the rule, but it reads post-anchor events
of fit patients, and the task says "Truncate the timeline at the anchor". Held-out, val, and test
patients stay out and inputs stay cut, so no held-out score is contaminated. The gain does not
justify departing from the written instruction, so D16 stands. No combination with this parent ran.

**Seeds.** Each seed runs its own pretraining. Mean fold AUROC is 0.7601, 0.7579, and 0.7599; mean
mAP is 0.1982, 0.1920, and 0.1892. Per-fold AUROC range is 0.0035 to 0.0144. The seed mean against
A1 is +0.0074 AUROC, 5 of 5 folds, mAP +0.0051 (A1 has one seed).

**Three-seed ensemble (`reports/lightgbm_clean/ensemble_comparison.json`).** Pooled out of fold:
AUROC 0.7607, mAP 0.1809. Mean fold: AUROC 0.7653, mAP 0.2036. All intervals are paired patient BCa
95%, with AUROC first and mAP second:

| Ensemble minus | AUROC | mAP | Brier |
|---|---|---|---|
| its first member | +0.006 [+0.001, +0.011] | +0.010 [+0.005, +0.016] | -0.0007 [-0.0009, -0.0006] |
| clean LightGBM counts | -0.003 [-0.010, +0.005] | -0.007 [-0.017, +0.002] | +0.0005 [+0.0002, +0.0007] |
| clean LightGBM counts + values | -0.000 [-0.009, +0.007] | -0.021 [-0.031, -0.011] | +0.0006 [+0.0003, +0.0009] |

The AUROC gap to LightGBM is no longer distinguishable from zero. LightGBM stays ahead on mAP
against the value features, and it is better calibrated.

**By anchor era.**

| Anchor | Patients | Ensemble AUROC, mAP | LightGBM counts AUROC, mAP |
|---|---|---|---|
| before 2015 | 1,219 | 0.740, 0.200 | 0.736, 0.210 |
| from 2015 | 1,572 | 0.771, 0.142 | 0.781, 0.139 |

**By condition (`reports/lightgbm_clean/ensemble_per_target.csv`).** Ten targets stay near chance
(mean AUROC 0.544): sprains, lacerations, concussion, whiplash, wrist fracture, strep throat, and
sinusitis. They are acute events, and the history before the anchor holds little about them. The
ensemble is higher on 19 of 40 targets. Its largest win is stroke (0.854 against 0.758). Its largest
losses are myocardial infarction (0.808 against 0.885), pathological fracture (0.874 against 0.928),
colon polyp (0.807 against 0.859), and rectal polyp (0.823 against 0.874). These losses are on the
count-driven conditions, and the count branch (F1) did not close them.

**The 0.80 question.** A macro AUROC of 0.80 needs the other 30 targets to average 0.885. They
average 0.833. Best-per-target over all models reaches 0.777 on clean inputs. CV 0.80 is not a goal
of this work, and the round stopped at its written limit without reaching it.

**Related check on D15 (`scripts/checks/race_check.py`).** LightGBM counts with race and ethnicity
minus without: AUROC -0.002 [-0.004, +0.001], mAP +0.001 [-0.001, +0.005]. So D15 costs no
measurable score.

**Caveats.** C1 is the best of 12 configurations after the 11 in D35, on the same folds, so its CV
score is optimistic. A3 and X1 scores were known when the component test was written; F1, F2, and
P3b scores were not. All three components failed the rule alone, so the component test is more
lenient than the rule. Best epochs are chosen on the held-out fold, which adds about 0.003. The
ensemble is three fits per patient against one LightGBM fit, and a bagged LightGBM was not tested.
Only the single P4e val check and the hidden test correct these effects.
