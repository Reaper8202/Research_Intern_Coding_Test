# Implementation handoff

Updated: 31 October 2026. Work stopped at Derek's request because usage was nearly exhausted.

**Resolved later on 31 October 2026.** Claude Code completed next steps 1–6. D31 is implemented
and recorded in `decisions.md`. `make check` passed with 93 tests, and `make preflight` passed.
Changes are not yet committed; Derek reviews the P3a boundary first. The text below is the
original handoff.

## Goal and rules

Repository: `/Users/derek/Downloads/research_intern_coding_test/ehr-forecast`.

Read `docs/model_plan.md` version 2.2 and `docs/plan.md`. The assessment authority is
`../research_intern_coding_test/ReadMe.md`, relative to the repository root.

The objective is a rigorous training and evaluation pipeline. The final system must be a
transformer. Use only supplied training data and locally learned weights. No external datasets,
pretrained weights, external tokenizers, or clinical coefficients. Clinical usefulness is not
an objective. Consult Derek before unexpected decisions. Use short, complete sentences and
ASD-STE100 writing guidelines. Do not claim formal certification.

## Current implementation

P3a is in progress. It is **not complete**. No transformer training or new official validation
scoring has occurred. No changes have been committed.

New files:

* `contracts.py`: explicit prediction inputs; training-only loading; identifier, date, label,
  and prior-mask checks. Removes encounter duration from numeric model inputs.
* `folds.py`: saved five-fold assignments and provenance verification.
* `masking.py`: prior-condition masking aligned by patient ID and target code.
* `evaluation.py`: version-2 AUROC, AP, Brier score, and patient BCa intervals. Includes support
  counts and degenerate-result reporting. The old `metrics.py` behavior remains for reproduction.
* `numeric.py`: unit audit and shared quantile-bin functions. Units remain separate.
* `provenance.py`: hashes, file verification, and atomic JSON output.
* `preflight.py`: raw training-data checks and reproduction of one historical baseline setting.

These modules are under `src/ehrforecast/`. New tests cover their behavior, future-record
perturbations, and label boundaries. `configs/p3a.toml` fixes the reproduction setting at
`C=0.003`, selected by the existing training CV table. The Makefile adds `folds` and `preflight`.

The loader successfully returned 2,791 training patients, 1,154,581 events, and 40 target columns.
Fold sizes are 559, 558, 558, 558, and 558. The last complete `make check` passed 86 tests, lint,
formatting, and mypy. A later addition checks raw demographics; rerun the full checks after work resumes.

## Blocking data issue and approved decision

`make preflight` stopped at a newly detected raw-data issue. In the training split, 57 medication
rows across 27 patients have STOP before START. The difference is about 2–6 days. Of these,
49 rows have both dates before the anchor. These are not same-day timestamp rounding errors.
Other interval tables had no such rows in this audit.

Derek explicitly approved: **Keep the recorded start; omit the invalid stop.**

This rule is **not implemented yet**. Keep the raw files unchanged. Record each affected row.
Apply the rule after the sample cutoff, before compression and event construction. Later records
must not influence a prediction history.

Important integration issue: existing `data/processed/events.parquet` already includes the invalid
stop events. Its flattened rows do not retain the original interval linkage. Do not remove all
stops for a patient and code. Preserve historical baseline reproduction and define a consistent
versioned input path for the corrected histories. Consult Derek if this requires another
unplanned design decision.

The failed audit is recorded in `reports/p3a/run.json` and `logs/preflight.log`. Fold files were
written to `data/processed/`. Baseline reproduction and the remaining preflight reports did not run.

## Next steps

1. Implement the approved medication rule at the appropriate raw-history boundary.
2. Add regression tests for invalid intervals, overlapping records, and cutoff invariance.
3. Resolve corrected-history provenance while preserving the original baseline comparison.
4. Run `make check`.
5. Run `make preflight` without invoking `make baseline`, which would score official val.
6. Review all audit outputs before marking P3a complete.

Update phase status, implementation decisions, README instructions, and the AI workflow record
after verification. Follow the plan's phase review and commit requirements. G1 also needs later
tokenizer/sequence checks; P3a alone does not complete that gate.

Preserve the existing untracked research files `docs/clinical_reference_feasibility.md` and
`docs/endpoint_risk_scores.md`. Earlier approved plan edits are also uncommitted. No process remains
running from this work.
