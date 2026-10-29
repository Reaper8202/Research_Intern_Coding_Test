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
