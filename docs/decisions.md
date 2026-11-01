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
