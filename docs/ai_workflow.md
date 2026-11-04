# AI workflow

How AI coding tools were used on this project, what I reviewed myself, and what the review
caught. Written 3 November 2026. [Items in brackets are for Derek to confirm or edit.]

## Tools

* **Claude Code** (Anthropic's agentic CLI, Claude Fable 5.1 and Opus 4.x sessions) did most of
  the implementation: the data pipeline, the transformer, the training loop, the evaluation code,
  the tests, and the run scripts. It also drafted the planning documents in `docs/` and kept
  the decision log current.
* [**Codex CLI** (OpenAI): add here if it was used for any module, or delete this line.]
* **wandb** tracked every training run. The AI tools wrote the logging code; I did not edit
  runs by hand.
* No AI tool had access to the hidden test labels, and none was allowed to score official val
  until the frozen bundle (`make val-score`) ran once at the end.

## Division of work

I set the objective, the constraints, and the order of work: a leak-free pipeline and a small
transformer with a complete run before any model experiments, comparison baselines that are
hard to beat, cross-validation on the training split only, and one recorded val run at the end.
The AI tools planned inside those constraints, wrote the code, and ran the experiments. I
reviewed the plan documents, approved or rejected every decision that changed the data or the
protocol, and read the results before the next step started.

Decisions I made myself, with the AI presenting the evidence:

* D31: medication rows with a `STOP` before their own `START`. I chose to keep the start and
  omit the stop rather than drop the row or move the date.
* D34: I required a multimodal model (genotype tokens and a brain MRI token) and chose to submit
  it whatever the CV result showed. The CV result showed no measurable gain, and that is
  reported as such.
* D35 and D36: I approved each tuning budget before its scores existed. Both budgets were
  committed before any run in them finished, so the selection rule could not be bent to fit a
  result.
* D36: removing the three simulator quality-of-life scores (DALY, QALY, QOLS) that work as a
  calendar clock, although removal costs CV score.

## How the work was kept honest

* **Plan before code.** `docs/model_plan.md` was written and revised before the matching code.
  Each revision is dated and states whether any result was known when it was made.
* **Tests as the review.** 199 tests (184 test functions), including mutation-style checks: changing `<` to
  `<=` in the anchor-day rule fails a test; a future record added after the anchor must leave
  every prediction unchanged; a resumed run must match an uninterrupted one.
* **Pre-registered selection.** The adoption rule (mean AUROC gain of at least 0.005, higher on at
  least 4 of 5 folds, mAP loss at most 0.005) was fixed before the first transformer result.
* **One val read.** `ehrforecast.val_score` refuses to run on a changed working tree, a changed
  artifact, or a completed bundle, and records a start and an end.
* **Commit log as the record.** Every experiment block was committed before its scores.
* **A second pair of eyes.** A separate, read-only AI session audited every data boundary with
  file and line references (truncation, allowlists, labels, tokenizer, extra anchors, pretraining
  windows, early stopping, modalities, LightGBM, evaluation, test split) and found no leak; its
  two real findings (modality-file presence correlates with outcomes; a residual era signal)
  are disclosed in the report. Another session wrote the regression tests it recommended.

## What the review found

Problems the AI tools introduced or missed, found by my review or by the checks above:

* Fold-4 pretraining filled the disk with swap four times (D33). The cause was MPS cache
  fragmentation across variable-length batches, not the model. The fix (release cached memory
  after each step) was verified to change no computation and made training 3 to 6 times faster.
* The preflight audit flagged a dirty working tree because of its own report files; the flag was
  limited to tracked files, and the audit now reads git state before writing.
* LightGBM and PyTorch each load an OpenMP runtime on macOS, and the two in one process crashed.
  The LightGBM module now runs in its own process and never imports PyTorch.
* A resumed run (E0 seed 3, fold 3) left wandb with steps out of order; `metrics.jsonl` is the
  record of truth and the caveat is in D35.
* The first bootstrap intervals used the percentile method and gave an interval that barely
  contained the point estimate (D7). BCa intervals replaced them.
* The first Platt-scaling fit let five near-chance targets take a negative slope, which would
  have reversed their ranking. A monotonicity test caught it; the fit now keeps the model's own
  order when the slope is not positive.
* A session ran out of usage mid-way through P3a. The handoff note in `docs/handoff_claude.md`
  let the next session finish the work without redoing it.

## What I would not claim

The AI tools wrote more lines than I did. The design choices that matter for the assessment
(what counts as a leak, which comparisons are fair, when to stop tuning) were mine, and the
evidence for each is in `docs/decisions.md` so a reader can check that claim rather than take it.
