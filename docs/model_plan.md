# Model plan: steps 3 and 4

| Item | Value |
|---|---|
| Version | 2.3, 31 October 2026 |
| Status | Reviewed against the assessment's pipeline-first requirement. Version 2.2 makes a complete small-model pipeline the first milestone and limits further experiments (section 18). Version 2.3 adds the approved interval-stop rule D31 (P3a) and the refill link D32 (P3b). P3a to P3c and the training code are implemented; no full training run has started. |
| Primary deliverable | A transformer that estimates 40 five-year first-diagnosis probabilities. All weights originate in this project. |
| Scope | Representation, training, model selection, and final validation. Step 5 covers test prediction and publishing. |
| Assessment authority | `../../research_intern_coding_test/ReadMe.md` |
| Related documents | `plan.md`, `decisions.md`, `baseline_results.md`, `clinical_reference_feasibility.md`, `endpoint_risk_scores.md` |

## 0. Purpose and terms

The final product must be a transformer. Logistic regression and LightGBM provide comparisons.
They cannot contribute predictions to the submitted model or its ensemble.

The main aim is to demonstrate a correct, leak-free, reproducible training and evaluation pipeline.
Competitive transformer results remain an aim. Model size and novelty are not success criteria.
A paper's score is not a target score for this dataset. Its evaluation conditions differ.

Apply these priorities in order:

1. Verify the data boundary, labels, patient separation, and fitted preprocessing.
2. Complete a small training-to-prediction run, including checkpoint recovery and independent reload.
3. Complete the required comparisons and evaluate the selected system under the frozen protocol.
4. Test additional model choices only within the remaining, recorded budget.

A failed correctness check blocks further experiments and final scoring, even when metrics look
strong. A modest score does not make a valid pipeline incorrect. Report both facts separately.

The project demonstrates a foundation-model training workflow: learn a representation from
unlabelled permitted histories, reuse that checkpoint, evaluate supervised adaptation, and package
reproducible inference. S/H/F comparisons remain required even if scratch training wins. This small
synthetic cohort and one downstream task do not establish a broadly useful clinical foundation model.
Clinical usefulness is not a project objective. The report states that limit briefly. It does not
require clinical deployment studies, clinical risk calculators, or extra experiments about clinical utility.

A **target** is one of the 40 supplied condition codes. A **label** records whether its first
diagnosis occurs in the prediction period. A **token** is one model input symbol. A **tokenizer**
converts permitted clinical events into token identifiers and associated ages.

**Self-pretraining** means training a randomly initialized transformer on our permitted training
histories. It does not mean downloading or adapting an externally pretrained transformer.
**Fine-tuning** means updating that locally trained transformer with the 40 supervised labels.
The **body** is the transformer that processes the sequence. The **head** is its output layer.

**Fit patients** are four training folds used to fit one model. The **held-out fold** is the
remaining training fold. **Val** always means the official validation split. **CV** means
cross-validation within the official training split. An **epoch** is one training pass through
the samples selected for that pass.

| Term | Meaning |
|---|---|
| AUROC | A ranking metric. It measures how often a positive case ranks above a negative case, with half credit for ties. |
| mAP | Mean average precision across targets. Average precision measures how well a ranked list retrieves positive cases. It depends on outcome prevalence. |
| Brier score | Mean squared difference between a predicted probability and its observed label. Lower is better. |
| Calibration | Agreement between predicted probabilities and observed outcome rates. |
| Prior-condition mask | A rule that sets risk to zero for a target already diagnosed before the anchor. |
| Checkpoint | Saved model parameters and the information needed to identify their training run. |
| Causal attention | A sequence operation that lets each position use its own and earlier inputs only. |
| RoPE | Rotary positional encoding. It adds information about token order to attention. |
| Quantile bins | Value groups whose boundaries come from ranks in the training values. |
| BCa bootstrap | Bias-corrected and accelerated bootstrap intervals. Patient resampling estimates uncertainty from the evaluation sample. |

IDs retain the version-1 meanings where possible: F = fact, D = decision, P = phase,
M = measurement, E = experiment, T = test, and R = risk. Retired experiments remain listed.
Planned decisions move to `decisions.md` only when implemented with their evidence.

## 1. Changes from version 1

| Area | Revised rule |
|---|---|
| Weight origin | Initialize every model with random weights. Load only checkpoints produced by this project's permitted training runs. |
| Self-pretraining data | Use fit patients' histories strictly before their real anchors. Full-timeline self-pretraining is removed. |
| Main prediction method | Compare supervised training from scratch, a self-pretrained fixed body, and full fine-tuning. |
| Readout | Use the last retained clinical-event state and explicit demographic and recency inputs. Do not require a new, untrained anchor token. |
| Position and time | Start with rotary positional encoding and continuous age information. Test whether gap tokens help direct prediction. |
| Search | Start with the 2-layer, width-128 model. Complete the pipeline before searches. Bound additional trials and confirm adopted candidates across five folds. |
| History length | Start at 1,024 tokens. Screen 512. Give 2,048 first priority among extensions when memory and the reserved core-work budget allow it (F12). Measure lost clinical content. |
| Numeric values | Keep fused quantile-bin tokens as the default. Complete the value-bin diagnostic after the first complete pipeline run. Test E18 only after material loss and within the extension budget. |
| Leakage tests | With anchors and fitted artifacts fixed, future changes must leave all prediction inputs, masks, and probabilities unchanged. |
| Generation | Defer it. Two required targets have no pre-anchor training events. Keep the temporal and sampling requirements as future design notes. |
| Validation | Fix code, configuration, checkpoints, and evaluation operations before one recorded validation bundle. |
| Clinical formulas | Retain as literature context only. Do not execute published coefficients in the assessment pipeline. |
| Estimates | Replace the old 13–23 hour compute estimate with a measured run budget before broad experiments. |

These changes follow Derek's request to improve the plan while retaining a transformer final
product. They also apply a conservative interpretation of the assessment's data restrictions.

## 2. Assessment rules and compliance

The supplied assessment README was reread for this revision. It recommends a transformer
pretrained on timelines and then fine-tuned. It also states: "Do not use external datasets or
pretrained weights. Train only on the provided training split."

Those statements permit learning our own weights from the supplied training data. They do not
permit importing an existing EHR model, language model, tokenizer, or embedding table.

| Assessment rule | Implementation requirement |
|---|---|
| Keep the supplied train, val, and test splits. | Preserve every assignment in `patient_splits.csv`. Internal CV uses training patients only. |
| Predict from events before the anchor. | Build each prediction history from dates strictly before its anchor day. Preserve D1–D4. |
| Do not train on val or test. | Exclude them from weight updates, vocabulary fitting, value bins, imputation, scaling, and all learned preprocessing. |
| Do not use external datasets or pretrained weights. | Use random initialization and supplied CSV data. Record the origin of every loaded training checkpoint. |
| Submit probabilities for every test patient. | Produce 358 rows, `patient_id`, and the 40 condition-code columns in the supplied order. |
| Track training with wandb. | Record training and held-out curves. Produce official val curves only after the final freeze. Sync offline logs before delivery. |
| Explain representation, training, interpretation, and AI use. | Maintain the report evidence and `docs/ai_workflow.md` during implementation. |
| Publish code and the trained model. | Package all five fold models, their tokenizers, and one prediction entrypoint for GitHub and Hugging Face. |
| Report test AUROC and mAP. | Test labels are hidden. Report CV and val results accurately. Attribute test scores to the organisers when supplied. |

### 2.1 Allowed and excluded resources

Allowed resources are the supplied data, general software libraries, architecture descriptions,
and mathematical methods from papers. PyTorch modules start with random weights. Literature
can inform the design without supplying trained parameters or patient data.

The following are excluded from model construction and scoring:

* External transformer weights, including Delphi, ETHOS, CEHR-GPT, Llama, and EHRSHOT checkpoints.
* External tokenizers, code embeddings, language-model features, and teacher-model predictions.
* External patient records, synthetic records generated by outside models, and additional Synthea simulations.
* External ontologies, SNOMED hierarchies, code-group mappings, and generator disease rules.
* Published clinical risk coefficients fitted on external cohorts.

The last restriction is a conservative project choice. The README does not explicitly discuss
clinical equations. This revision avoids treating outside fitted coefficients as an exception
to its weight restriction. The clinical feasibility documents remain background research.
Their numerical comparison work is not part of the assessment pipeline.

Descriptions supplied with the dataset may explain results in the report. They do not supply
semantic embeddings or expanded target definitions. Genomics and DICOM remain excluded under D5.
Imaging-study CSV events remain permitted.

### 2.2 Data boundary for each operation

| Operation | Permitted records |
|---|---|
| Fit a fold tokenizer or preprocessing rule | Fit patients' pre-real-anchor histories only. |
| Self-pretrain the fold transformer | Those same histories, with random initialization. |
| Stop self-pretraining | Fixed pre-anchor windows from that fold's held-out training patients. No weight updates. |
| Train a supervised head or fine-tune | Fit-patient input histories and their supervised labels. |
| Build labels | First diagnosis dates from full condition records for the relevant labelled training patients. |
| Select or stop supervised training | Held-out training-fold predictions and labels. |
| Evaluate final val models | Val histories and labels, after the final freeze. |
| Predict test | Released test histories and supplied anchors. No label construction or fitting. |

Full future records are not an unsupervised training resource in version 2. The README's wording
about truncating timelines motivates this conservative choice. It is not a claim that the README
explicitly discusses every possible self-pretraining design.

Historical exploration already used val and test summary statistics. The baseline report also
contains val outcomes. Record that history. Do not describe val as a dataset that nobody has seen.
No new val result may change this revision's model configuration or submission choice.

### 2.3 Data interfaces and early failure

Store labels separately from model inputs. The label builder can read full condition records.
Anchor construction can read the full encounter table as the assessment requires. Prediction
builders receive fixed anchors and permitted histories. They do not receive future records or labels.

The current `cohort.parquet` also contains `last_encounter` and `window_end`. The current event
table includes encounter duration in `value_num`. These are storage fields, not model features.
Use explicit input-column and source allowlists. Only observations supply numeric value features.
Patient identifiers select records and folds; they never become learned inputs.

Training loaders must select the official training split before any fitting operation. Prediction
must run when label files and future tables are unavailable. Evaluation joins predictions to labels
by patient ID and target code, never by an assumed row order. Reuse these interfaces for CV and test.

P3a records raw-file hashes, schema versions, split counts, target order, and processing code version.
Reject duplicate patient or target IDs, unknown patient references, missing required dates,
non-finite model tensors, and invalid label/mask combinations. A documented missing-value rule is distinct
from silent parsing failure. Reuse existing step-1 checks and add only missing checks.
Cache keys include fold patients, data, tokenizer, configuration, and code versions. Reject stale
or incompatible caches. Checks of the anchor calculation remain separate from fixed-anchor leakage tests.

## 3. Evidence and limits

### 3.1 Dataset facts

| ID | Evidence already recorded | Consequence |
|---|---|---|
| F1 | Of train + val patients with anchors before 2015, 1,303 were dead and 71 alive at record end. For later anchors, the counts were 51 and 1,731. | Exclude calendar time. The retrospective anchor also limits clinical interpretation. |
| F2 | Before refill compression, train sequence length had median 353 and maximum 15,449. About 33% exceeded 512; 8% exceeded 1,024. | Measure truncation effects. Recompute lengths under the revised tokenizer. |
| F3 | Earlier raw-row counts included train + val and full timelines. | They are not the available self-pretraining budget. Count permitted fit-patient tokens directly. |
| F4 | Gaps between event days range from days to many years. About 24% of previously counted gaps were 365–729 days. | Retain age and elapsed-time information. Coarse gaps cannot define an accurate generation clock. |
| F5 | Medication refill starts per patient and drug had p90 55 and maximum 588. | Test compression without assuming that repetition has no predictive value. |
| F6 | There are 113 numeric observation codes and 37 text codes. Five numeric codes have multiple units. | Audit units and sparse value groups before fitting bins. |
| F7 | About 9.5% of patient–target combinations have a prior condition. | Apply a common prior-condition mask. |
| F8 | Training positives range from 29 to 741 per target. Two targets have identical labels. | Keep all required targets, but report scarcity and duplication. |
| F9 | In the existing five folds, the minimum held-out positive counts are 5, 4, 4, 6, and 6. No target has zero positives in an original fold. | Single-fold estimates can be unstable. Subgroups and bootstrap samples can still lack positives. |
| F10 | A preliminary fold-0 clock replay on 559 held-out training patients had median absolute error 120 days. About 60.8% exceeded 90 days. | Do not use bucket medians for final generation timing. |
| F11 | Pneumonia and prostate cancer metastasis have no pre-anchor condition rows in the training split. Their supervised labels contain 133 and 50 positives. | A supervised head can learn these outcomes. Zero-shot generation lacks training examples of their target-start symbols. |
| F12 | With one token per event, 5.8% of training patients exceed 1,024 event tokens, but 27.4% of all training event tokens lie beyond that limit. The 161 truncated patients keep a median 3.6 of 64.6 history years. At 2,048, 3.6% of patients exceed the limit; with upper-bound refill compression, 0.6% do. | Truncation removes most of the record for a few dense histories. Prioritize 2,048 within the memory and extension budget. Always report who is truncated (M7, R19). |

F10 used raw event days before refill compression. Medians came from fit patients' full timelines
under the version-1 proposal. The replay included the anchor and end boundaries. It was not a
model result or a validation result. It does not justify fitting a clock in the current plan.
F1–F8 are historical measurements. New fitted settings must come from training data only.

F11 was checked on 30 October 2026 using training patients and their fixed stored anchors.
The raw `train_val/conditions.csv`, processed event table, and prior-condition flags agree.
Codes `233604007` and `314994000` each have zero pre-anchor occurrences across the full training
split and thus in every CV fit set. Their post-anchor labels remain valid supervised training
targets. The other 38 targets each meet the five-patient threshold in all five fit sets.
This check fitted no model and calculated no new validation score.

F12 was measured on 31 October 2026 on the 2,791 training patients and fitted nothing. It counts
one token per pre-anchor event after same-day deduplication, with numeric and text values fused
into the event token. It excludes structure, summary, and gap tokens, so real event budgets are
slightly smaller. F2 is larger because it also counted one gap token per event day.

| Context | Patients over the limit | Event tokens beyond the limit | Patients over, upper-bound refill compression |
|---|---:|---:|---:|
| 512 | 15.5% | 37.9% | 6.6% |
| 1,024 | 5.8% | 27.4% | 4.6% |
| 2,048 | 3.6% | 16.0% | 0.6% |

The upper-bound compression keeps only the first start and the last stop for each patient and
drug, so it also merges true interruptions. The planned one-day rule (section 8.5) lies between
the two columns. M3 and M7 replace these figures with exact counts under the final tokenizer.

### 3.2 Literature transfer

| Source | Observed finding | Design implication and limit |
|---|---|---|
| Shmatko et al., Delphi | Age encoding, event timing, and cohort effects matter. The model trained on about 403,000 patients. | Retain age and audit biases. Its two-million-parameter scale is not evidence of the optimum here. |
| Pang et al., CEHR-GPT v2 | A generated record can preserve code statistics while distorting time-dependent outcomes. | Validate chronology and five-year target inclusion. Do not copy its calendar-year inputs. |
| Wornow et al., Context Clues v2 | Longer context helped some architectures. RoPE reduced spikes in GPT perplexity, a next-token prediction metric. Time tokens did not consistently help. | Test context and time representation on our data. These are hypotheses, not promised gains. |
| Context Clues, section 3.2 and Appendix H | Main evaluations fitted supervised heads on fixed representations. Generated-future predictions were weaker in three tested tasks. | Include a fixed-body candidate. Check generation feasibility separately; F11 fails that check here. |

Two later sources address numeric-value encoding (added in version 2.1):

| Source | Observed finding | Design implication and limit |
|---|---|---|
| Lee et al., tokenization benchmark (MIMIC-IV, 2026 preprint) | Fused code–decile tokens were a practical starting configuration. 20, 30, or 100 bins gave small, non-monotonic differences. Reference-range bins did not consistently help. xVal encodings scored about 0.05 AUROC below deciles. | Keep fused deciles as the default. Single site. The authors state that the xVal result does not show a general disadvantage of continuous encoding. |
| McCann et al., continuous value tokenization (MIMIC-IV, 2026 medRxiv preprint) | A continuous value embedding fused to the measurement token lowered next-token loss against 10 equal-width bins. ICD-code AU-PRC rose from 0.446 to 0.457; other tasks were equivalent. Calibration error was higher (ECE 0.076 versus 0.038). | Diagnosis-prediction gains were small, and calibration worsened. The baseline used equal-width bins, not quantile bins. This motivates E18 only if the value-bin check finds material loss. |

No source establishes a winning configuration for 2,791 synthetic training patients and these
40 five-year targets. Numerical scores from the papers are not directly comparable to ours.

## 4. Decision register

D1–D8 in `decisions.md` remain implemented decisions. Entries below are planned rules for version 2.
D31 and D32 are implementation decisions from P3a and P3b; `decisions.md` records their evidence.

| ID | Decision |
|---|---|
| D9 | Select models using training CV only. Treat CV as development evidence affected by selection. |
| D10 | Exclude calendar year, birth year, absolute dates, death date, and prohibited record-summary fields from prediction inputs. E8 remains an isolated, non-submitted calendar diagnostic. |
| D11 | Defer zero-shot generation because F11 fails its target-coverage requirement. Do not spend the current training budget on it. |
| D12 | Prioritize a verified complete pipeline and the small supervised transformer. LightGBM is a required comparison. Model extensions have a fixed conditional budget. |
| D13 | Set prior-condition probabilities to zero in every method. Exclude those cells from newly trained supervised losses. |
| D14 | Fit each tokenizer and preprocessing object using that fold's fit patients only. Store their identifiers and hashes. |
| D15 | Learned demographic inputs are age and sex. Race and ethnicity do not enter model inputs. |
| D16 | Self-pretraining uses histories before fit patients' real anchors. Post-anchor conditions supply supervised labels only. |
| D17 | Official val metrics and curves are produced in one frozen evaluation bundle. They do not select settings, epochs, or models. |
| D18 | Submit the selected five-model supervised transformer ensemble. Baselines never enter its predictions. No zero-shot generation model is eligible in this revision. |
| D19 | Use no external fitted weights, tokenizers, ontologies, semantic features, clinical coefficients, or additional datasets. |
| D20 | Direct-classification default: event tokens, RoPE, and continuous age embeddings. Compare adding coarse gap tokens. |
| D21 | Read the last retained clinical-event state. Join age at anchor, sex, and time since the last event before the 40-output linear head. |
| D22 | Average probabilities from the five fold models. Use the primary seed for final artifacts. Do not select the best seed. |
| D23 | Compare scratch supervised training, self-pretraining with a fixed body, and self-pretraining with full fine-tuning. |
| D24 | Select preprocessing rules from training data. Val and test parity reports are descriptive after the freeze. |
| D25 | At fixed anchors and with fixed artifacts, later-record changes must not change sequences, head inputs, prior masks, baseline features, or predictions. |
| D26 | Preserve the inclusive calendar-five-year label boundary. Any future generation design must use exact integer days. |
| D27 | Extra-anchor training samples patients uniformly. More eligible anchors must not give a patient greater sampling weight. |
| D28 | Freeze checkpoint, tokenizer, configuration, code, and data identifiers together before official val evaluation. |
| D29 | Complete the small end-to-end run before broad searches. Preserve the budget for correctness, core evaluation, and delivery before model extensions. |
| D30 | Keep labels and future records outside prediction interfaces. Test training recovery and label-free inference before full experiments. |
| D31 | Keep the recorded start and omit a `STOP` earlier than its own `START`. Apply this after the anchor cut, before event construction and compression, in every split. Encounters are excluded. Implemented in P3a. |
| D32 | Refill compression merges a medication stop with a restart of the same drug at most one day later, never with the stop's own start. Events carry `interval_start` to link them. Implemented in P3b. |

## 5. Evaluation and selection

### 5.1 Folds and patient separation

Reuse `KFold(n_splits=5, shuffle=True, random_state=20261028)` on the training-row order returned
by `train_baseline.load_data`. Save `patient_id` and `fold` in `data/processed/folds.parquet`.
Verify this assignment and reproduce one saved baseline configuration's fold scores during P3a.
Run any full baseline-grid reproduction after G2. Preserve the step-2 metric definition for that check.

All samples from a patient belong to the same fold. A held-out fold contains real-anchor
supervised samples only. Pretraining evaluation uses fixed, pre-real-anchor windows from those
same held-out training patients. It does not use their post-anchor event sequences.

An out-of-fold prediction comes from the one model that did not fit on that patient. Never
average the other four models for that patient's CV prediction. Those models fitted on the patient.
Val and test predictions can average all five models because none fitted on those patients.

Held-out fold labels still select supervised epochs and configurations. These are development
predictions, not an independent performance estimate for the selected procedure. Five-fold
confirmation does not remove that selection effect. State it in the results and confidence intervals.
Do not add a nested model search merely to make the project larger. Use the frozen val bundle for
the final local assessment and disclose its historical exposure.

### 5.2 Metrics

| Role | Metric and scope |
|---|---|
| Primary | Macro AUROC across the 40 targets, after the prior-condition mask. |
| Secondary | mAP across the 40 targets, with the same mask. |
| Secondary | At-risk macro AUROC, mAP, and Brier score. Exclude prior cases separately for each target. |
| Probability quality | Macro Brier score, pooled reliability plot, and per-target observed versus mean predicted rates. |
| Descriptive | Per-target AUROC and AP, with patient and positive counts. |
| Robustness | Scores by retained-history length, truncation, and event density. Boundaries come from training data. |

Report both the mean of the five fold macro AUROCs and pooled out-of-fold metrics. Their values
can differ. The mean of fold scores controls model selection and preserves the step-2 definition.

Require 40 valid target AUROCs for the original full-fold point estimates. For smaller subgroups,
report unsupported targets and denominators explicitly. AUROC requires both classes. For a nonempty
sample, AP is zero with no positives and one with only positives, under the reference-library
convention. An empty sample has no metric. Report each metric's own denominator and class counts.
Do not silently report an unsupported subgroup as a complete 40-target AUROC result.

The existing metric wrapper excludes both AUROC and AP when either class is absent. Preserve that
behavior only for step-2 reproduction. Version the revised metrics and test their edge cases before
new comparisons. Keep historical and revised tables distinct.

Use the existing 95% BCa bootstrap with 1,000 patient resamples for final comparisons. Resample
patients jointly across all targets. Use identical resample indices and target eligibility for
paired model differences. Log degenerate resamples and any non-finite interval. Apply the revised
metric-specific eligibility rules equally to all methods. Report how many AUROC targets each
resample excludes; do not relabel a varying-target interval as a fixed-40-target interval.
These intervals describe evaluation-sample variation. They do not include the whole model-search process.

Calibration is diagnostic in this version. Do not fit a calibrator on val or test. Report at-risk
calibration separately so deterministic masked zeros do not hide probability errors. Use coarse
bins for rare targets. No claim should rest on one target with fewer than ten positives.

### 5.3 Staged selection

Primary seed: 20261028. Stability seeds: 20261029 and 20261030. The seeds are fixed before results.
The initial reference is supervised training from scratch with the 2-layer, width-128 architecture.
All searches start after the complete pipeline check in section 6.

1. Run S, H, and F on fold 0 with the small default architecture.
2. Complete all three methods across five folds.
3. Select the reference complete configuration using the rules below.
4. Screen the 512-token alternative on fold 0.
5. Screen only the additional configurations listed in the frozen run budget.
6. Confirm each proposed replacement across five folds before adopting it.

Allow at most four additional complete configurations on fold 0, including the 512-token run.
Allow at most two of these configurations to proceed to five-fold confirmation. Prioritize the
2,048-token context check when feasible, then E18 if the value-bin check triggers it. Refill, gap,
position, larger-model, and extra-anchor variants compete for the remaining slots. Record the
ordered choices before their scores are available. An unrun variant is reported as untested.
The final combined configuration also consumes a slot; combinations are not free additional trials.

A replacement of equal or greater complexity needs a mean macro AUROC gain of at least 0.005,
an AUROC improvement in at least four folds, and a mean mAP loss no greater than 0.005.
This is a practical consistency rule, not a significance test.
A smaller model can replace a larger one within 0.005 if mean mAP does not fall by more than 0.005.
For training methods within 0.005 on both metrics, prefer fewer supervised parameters, then
fewer total parameters, then less training work. Record the exact comparison and tie resolution.
Apply these simplification rules against the best confirmed configuration by mean macro AUROC,
with mean mAP as its tie-break. Do not accumulate small permitted losses through successive changes.

E0 repeats the two leading complete configurations with the stability seeds on fold 0. If their
ranking changes, or a claimed 0.005 gain lacks four-fold consistency, repeat both configurations
on all five folds with those seeds before choosing a replacement, if the reserved budget permits.
Otherwise retain the incumbent and report the unresolved ranking. Compare the mean across seeds
within each fold. Retain the incumbent if the improvement still fails the consistency rule.
Never use the maximum-minus-minimum spread of three seeds as a statistical significance threshold.

The final shortlist consists of complete configurations. It must include the actual combination
of tokenizer, context, position encoding, training method, and optional extra anchors used at
inference. Do not combine individually successful changes without evaluating their combination.
Save every trial in the experiment register. Screening results are exploratory.

Generation does not enter this selection. F11 fails its target-coverage requirement before
training begins. A later change to that scope requires a dated protocol revision and a valid
training-data justification. It cannot relax the assessment restrictions or use new val outcomes.

### 5.4 Interpretation of strength

The target is a transformer competitive with the strongest permitted baseline, with stable
at-risk performance and useful probabilities. Prefer consistent gains in AUROC and mAP.
Do not promise a numerical score before training. Do not claim sequence order caused an improvement
when the competing methods also differ in laboratory values, timing, or retained information.

If the transformer trails the baselines, first verify data and training correctness. Then inspect
training-only errors before expanding model size.
Check age and recency inputs, target coverage, lost history, training convergence, and rare-target
behavior. If the remaining budget permits, allow one documented recovery round of at most two
additional complete configurations. Reserve final evaluation and delivery costs first.
Confirm any proposed replacement across five folds. Report the best valid transformer honestly
if the gap remains. A weaker result does not authorize a baseline submission or validation tuning.

### 5.5 Predeclared official val comparisons

* Selected direct transformer versus masked code-count logistic regression.
* Selected direct transformer versus at-risk age–sex incidence.
* Selected direct transformer versus LightGBM counts and LightGBM counts + values. The binned
  diagnostic set stays training-CV only.
* If E8 was included in the budget, anchor-year logistic regression versus ordinary code-count
  logistic regression, as an artifact diagnostic.

All probability differences use matched patients and the common prior mask. Source perturbations,
subgroups, and condition-level differences are descriptive. They cannot change the submitted model.

## 6. Phases, priority, and compute budget

| Phase | Required output | Dependency |
|---|---|---|
| P3a | Fold file, common mask, unit audit, versioned metric checks, baseline reproduction, input contracts and provenance | Existing steps 1 and 2 |
| P4d | Confirm the F11 coverage audit and record generation deferral. No generation training. | P3a; complete this check early |
| P3b | Fold tokenizers, value bins, deterministic token and age rules | P3a |
| P3c | Prediction and pretraining samples, retention report, future-perturbation tests | P3b |
| P4a | Complete small-model run: short S/H/F training, recovery, reload, prediction, metrics, and speed report | P3c |
| P4b | Self-pretraining runs and shared checkpoints for fixed-body and fine-tuned candidates | P4a |
| P4c | Three-way small-model comparison, baseline CV, budgeted value-bin diagnostic and extensions, final classifier CV | P4a, P4b |
| P4f | Training-only error analysis and the final evaluation specification | P4c |
| P4e | Artifact freeze, one val bundle, results document | P4c, P4f, and P4d decision |
| Step 5 | Test predictions, report PDF, public repository, model package | P4e |

P4f now precedes the final scoring bundle. It does not implement external clinical formulas.
P4d retains its identifier but ends at the early coverage check. Generation fails that requirement
under the current data boundary. Extra anchors and broader interpretation work remain conditional.

These completion conditions control the work. A strong metric cannot replace them.

| Gate | Evidence required before continuing |
|---|---|
| G1: data boundary | Existing anchor and label tests pass. Fit scope, input allowlists, target alignment, and input-level fixed-anchor future perturbations pass. |
| G2: complete small run | One fold completes short random-init pretraining, H/F adaptation, and S training. A saved run resumes. A fresh process reloads the artifacts and produces 40 probabilities through the final prediction interface. |
| G3: evaluated system | Required small-model S/H/F and baseline comparisons have five-fold results. Metrics, support counts, uncertainty, failure records, and selection evidence are saved. |
| G4: reproducible delivery | Frozen artifacts pass clean-environment CPU inference and submission checks. The val bundle, report, and public package agree on the selected model. |

For G2, treat held-out training patients as mock test patients. Supply their anchors and cut
histories without labels or future tables. Score their saved predictions in a separate evaluator.
Exercise the freeze and evaluation runner on these training-fold artifacts. Do not consume official
val for this check. The small run tests execution; it is not a performance claim or a grid search.
Repeat the future-perturbation check through the saved model's probabilities as part of G2.

Baseline grid searches and the value-bin diagnostic run after G2. P3a supplies their reusable
unit rules, bin-boundary function, fold file, and simple reproduction check. They must not delay
the first complete transformer pipeline. New representation variants return through G1 and G2.

The old 27-hour implementation and 13–23-hour compute totals are withdrawn. P4a must measure:
training tokens/second, epoch duration, peak memory, and ensemble inference duration.
Compare a tiny CPU run with MPS. Benchmark actual vocabulary and length distributions.

Write a run-count budget before P4b. Count shared self-pretraining checkpoints only once.
Include five-fold confirmation, seed repeats, baseline searches, and val curves. Record a local
compute ceiling in `configs/budget.toml` before large searches. Reserve core evaluation, recovery
checks, CPU packaging, and report work before allocating extension runs. If the small default
exceeds 15 minutes per epoch or available memory, reduce the physical batch size with correctly
normalized gradient accumulation before removing useful history. Record any further budget-driven
protocol change before its results. Do not drop correctness checks to fund a larger model.

P3a must time one baseline configuration before its grid search and set a baseline budget.
P4a extends that budget with measured transformer costs. No broad search precedes G2 or its cost check.
Stop extensions when their fixed run or time limit is reached. Report skipped work explicitly.

Do not include generation training or inference in the current run budget. Its failed coverage
check is enough to stop that branch. Section 10 retains the other concerns from the literature review.

Each implementation phase requires its outputs, a passing `make check`, an entry in `plan.md`,
and a commit. Derek reviews each phase boundary. This document revision is planning work and
is not completion of P3a.

## 7. P3a: comparison methods and data controls

### 7.1 Prior-condition mask

Build the mask from `prior_conditions.parquet`, in `target_conditions.csv` order. This operation
uses the complete permitted history, before token truncation or unknown-token replacement.
Apply the mask to every model's predictions. Exclude masked cells from new supervised losses.
A masked probability is zero because the task defines first diagnosis, not recurrence.

Write training-CV masked and unmasked tables for the existing baselines. Defer every new official
val calculation to P4e. Verify that the mask changes no other cells. Masked ranking metrics must
not decrease for these known-negative cells.

Keep the step-2 reproduction separate from the revised comparisons. For the new code-count
logistic regression, fit each target on at-risk fit patients and select C using the same original
grid with masked CV scores. Refit its vocabulary within each fold. Retain the existing historical
val results as historical results. They are not scores for this revised baseline.
Select the at-risk incidence configuration with its original grid and the common masked metrics.

### 7.2 LightGBM

P3a defines the data and feature interfaces. Full comparisons run after G2. The first two feature
sets are required. The binned diagnostic uses a separate, declared budget; report if it is not run.

Fit one binary classifier per target using at-risk fit patients only. Use the same fold file.
Define three feature sets:

| Feature set | Content |
|---|---|
| Counts | Existing `BagOfCodes`, age, age bands, and sex. Refit the code vocabulary in each fold. |
| Counts + values | The same inputs plus the most recent valid numeric observations, missingness flags, and observation recency. |
| Counts + binned values | Counts + values, with each latest numeric value replaced by its quantile-bin index from section 8.3, fitted on the fold's fit patients. Missingness flags and recency stay unchanged. Training-CV diagnostic only. |

For the second set, use observation codes present in at least five fit patients. Complete the M2
unit audit in P3a before this comparison. Share its documented unit rules with P3b.
Record one latest value per code and unit, before the sample anchor. Impute with
fit-patient medians. Add a missingness flag and log-scaled elapsed days since measurement.
If distinct values share the latest recorded timestamp, use their median within the code/unit group.
Do not use a clinical normal range or outside code mapping. Keep numeric features continuous.

**Value-bin information check (introduced in version 2.1).** Run this after G2, using P3a's bin rules.
The binned set differs from counts + values only in
numeric resolution. Trees can then split only at the fitted bin boundaries, which is the
information that quantile tokens discard. Compare the two by mean CV macro AUROC and mean CV mAP,
each with its own grid and the same selection rule. The loss is material when the binned set is
lower by at least 0.005 on either metric. Record all five fold differences. A material loss gives
E18 priority within the extension budget; it is not proof that E18 will help a transformer.
Otherwise E18 is not run. Record the decision before fitting an E18 tokenizer or training that variant.

Implement the bin-boundary function once in P3a and reuse it in P3b, so both use identical rules.
The check sees only the latest value per code. The transformer sees every retained value over
time, and rare value tokens can collapse into a fallback. Thus, the check is a partial proxy for
the transformer's information loss. The binned set does not enter the
official val bundle, and it is never submitted.

For each feature set, choose one configuration shared across all 40 targets. Grid:
`num_leaves` = {4, 8, 16}; `n_estimators` = {100, 300, 600};
`min_child_samples` = {20, 50}. This gives 18 settings per feature set.

Fixed settings: learning rate 0.05, feature fraction 0.5, bagging fraction 0.8, bagging frequency 1,
L2 penalty 1.0, deterministic mode, fixed thread count, and seed 20261028. Select by mean CV macro
AUROC. Report grid-edge selections. Fit final baseline models on the training split only after
selection. Record the macOS `libomp` requirement in setup instructions during implementation.

These are system comparisons. They do not isolate event order. The baselines have access to
complete permitted histories, while the transformer can have truncated histories.

### 7.3 Calendar and era diagnostics

E8 and E9 are optional after G2. Their absence does not relax the exclusion of calendar features.

E8 adds `anchor_before_2015` and standardized anchor year to code-count logistic regression.
Use the step-2 regularization grid and training folds. Save it in a separate artifact directory.
It is never an eligible submission model. Its new val results wait until P4e.

E9 predicts the early-anchor indicator from code features. A second diagnostic predicts anchor
year within the earlier-anchor group. Both use training CV. Report AUROC or Spearman correlation
as appropriate. These probes detect associations. They do not separate calendar era from age,
illness, record duration, or the anchor-selection mechanism.

Combined outputs of P3a and the later comparisons: `folds.parquet`,
`reports/baseline/masked_vs_unmasked_cv.csv`, `reports/lightgbm/`,
`reports/artifacts/`, `reports/transformer/generation_gates.json`, and an updated baseline document
with clearly labelled CV results. The generation record documents the coverage failure, not a model run.

## 8. P3b–P3c: representation

### 8.1 Measurements before settings are fixed

| ID | Training-only measurement | Use |
|---|---|---|
| M1 | Stop-to-restart gaps for each patient and drug | Check the one-day refill rule. |
| M2 | Units, counts, and value ranges for numeric codes | Define explicit unit handling. |
| M3 | Sequence lengths, summary sizes, and retained clinical years | Check context and summary budgets. |
| M4 | Unknown-token rates and distinct patients per code/value group | Check vocabulary coverage and rare-value fallback. |
| M5 | Last-event-to-anchor gap and empty-history counts | Verify recency inputs and prediction coverage. |
| M6 | Extra-anchor counts, age distributions, and target rates | Check eligibility and patient sampling. |
| M7 | Post-truncation source mix, lost events, and positive counts by history group | Identify selective information loss. |
| M8 | Permitted pretraining patients, tokens, windows, and optimizer steps per fold | Set the measured training budget. |

Compute development reports on fit and held-out training patients. Hold settings fixed when
later creating val/test parity reports. Test-distribution warnings do not authorize retuning.
M2 starts in P3a because the numeric-value baseline needs the same unit rules as the tokenizer.

### 8.2 Token types and vocabulary

| Source | Token form |
|---|---|
| Conditions | `cond:start:<code>`, `cond:stop:<code>` |
| Medications | `med:start:<code>`, `med:stop:<code>` |
| Care plans, allergies, devices | `<source>:start:<code>`, `<source>:stop:<code>` |
| Procedures | `proc:<code>` |
| Immunizations | `imm:<code>` |
| Numeric observations | `obs:<code>:<unit-key>:q<bin>` |
| Text observations | `obs:<code>:<normalized-value>` |
| Unusable observation value | `obs:<code>:missing_value` |
| Rare value of a known observation code | `obs:<code>:other_value` |
| Imaging study | `img:<modality>\|<body-site>` |
| Encounter | `enc:<class>:<code>` |
| Structure | `[PAD]`, `[SEP]`, `[SEX:F]`, `[SEX:M]`, `[UNK:<source>]`, `hx:<condition-code>` |
| Optional classification gaps | `[GAP:<bucket>]`, used only in the E6 gap variant |

Encounter duration, lifetime costs, death date, marital status, addresses, device identifiers,
and future medication dispense totals remain excluded. Use the existing source allowlists.

A base clinical code is eligible when it occurs in at least five fit patients' permitted histories.
Count patients, not rows. Unknown codes map to a source-specific unknown token. Known observation
codes retain their identity when a rare value maps to `other_value`. Reserve missing/other-value
forms for eligible observation codes. Derive summary forms only from eligible condition codes.
Structural tokens have a fixed, documented vocabulary and do not need a five-patient count.

Normalize text by fixed lowercase and whitespace rules. Sort identifiers deterministically.
Never fit a vocabulary using descriptions, held-out codes, or post-real-anchor occurrences.
The 40 supervised output columns always exist, even if a target code never appears in fit inputs.

### 8.3 Numeric values and units

Fit quantile boundaries separately per code and compatible unit. Start with
`max(1, min(10, floor(n_values / 20)))` bins. Merge duplicate boundaries. Record the number of
fit patients in each resulting bin. Value tokens with fewer than five fit patients use the
known-code fallback. Apply the same five-patient rule to normalized text-value tokens.
Clip values outside fitted boundaries to the edge bins. Preserve missing-value indicators.

Use only explicit arithmetic conversions between equivalent units identified in supplied fields.
Document each conversion. Keep ambiguous or incompatible units separate. Do not infer clinical
thresholds from outside sources. The tokenizer stores all unit rules and bin boundaries.

Fused quantile-bin tokens remain the default value encoding. E18 runs only after a material loss
in the value-bin check and a slot in the extension budget. E18 keeps the bin token and adds a
continuous value channel:

* Standardize each value within its code and unit, using the mean and standard deviation of fit
  patients' values after clipping at their 1st and 99th percentiles.
* Map the standardized value through a learned linear layer to the model width. Add it to the
  token and age embeddings.
* Use zero for tokens without a numeric value. Add a learned value-present embedding for tokens
  that have one.
* Store the clipping limits and standardization statistics in the fold tokenizer.

For a zero-variance fit group, use standardized value zero. Missing, non-finite, and unseen-unit
values follow explicit stored fallback rules. Never fit statistics on a prediction batch.
E18 is a representation alternative under section 5.3. It needs five-fold confirmation. Report its
calibration, because continuous value encoding worsened calibration in one source (section 3.2).

### 8.4 Sequence construction and summary

Default direct-classification sequence:

```text
[SEX:*]  hx:<older-condition> ... [SEP]  <clinical events, oldest first>
```

The last retained clinical event is the readout position. There is no appended `[ANCHOR]` token.
Age at anchor and elapsed time since the last event enter the head separately.

For every prediction sample, apply this procedure:

1. Fix the sample anchor.
2. Apply the existing strict pre-anchor truncation rules to raw records.
3. Apply value normalization and any permitted refill compression.
4. Convert the retained records into event tokens.
5. Deduplicate identical tokens within each day.
6. Select the most recent complete event days that fit the context budget.
7. Construct the condition summary from diagnoses before the selected window.
8. Attach the age and recency inputs.

Within each day, sort by a fixed source/kind/code order, then normalized unit and value token.
Values mapped to one bin share a token in the default representation. For E18, the continuous
value is also part of the ordering and deduplication key. Do not discard distinct values merely
because their bin tokens match. Source-file row order must not change the sequence.

The prefix and summary share a budget of at most 128 summary tokens plus required structure.
Prioritize older target conditions, then the most recently first-diagnosed other conditions.
Order selected summary tokens by their first diagnosis date, with a code tie-break. Assign each
summary token its actual first-diagnosis age. A summary token does not imply an active condition.
A condition first diagnosed inside the retained window must remain an event, not a summary token.

Start with total context length L = 1,024. E5 screens 512 and prioritizes 2,048 within the budget
(F12). All structure and summary tokens count against L.
Drop the oldest event days until the sequence fits. Recompute the summary for the final boundary.
If one day alone exceeds the event budget, retain its final tokens in the fixed source/kind/code
order and flag the partial day. Never drop the patient. Test and report this exceptional path.

If no clinical event remains, use a zero history vector and the permitted head inputs. Set the
recency value to zero with a separate missing-history flag. Do not invent a diagnosis or discard
an empty-history patient. The mask still uses the full permitted history.

### 8.5 Refill compression

Default candidate: merge a same-drug stop/restart pair separated by at most one day. M1 must
check its distribution before implementation. Perform compression after the sample cutoff.
Future restarts must never remove a pre-anchor stop.

Preserve every patient–drug pair, its first start, the final known stop, and true interruptions
longer than the threshold. Test overlapping intervals and changes at the anchor. After D31,
58 medication starts have no stop event, although the prescription ended. Do not assume that
every start has a matching stop. Report counts
removed and retained treatment spans. Optional E14 compares compression on and off. Its result must pass
five-fold confirmation before a replacement is adopted. Do not collapse repeated lab measurements
across days or treat every repeated diagnosis as documentation noise.

### 8.6 Age, position, and optional gaps

Age is `(event_day - birth_date).days / 365.25`. Use age/100 plus sine and cosine terms for eight
periods spaced geometrically from 0.5 to 200 years. A learned linear layer maps these features
to the model width. Add this result to the token embedding.

RoPE encodes token position within attention. It does not encode calendar time or replace age.
Optional E13 compares RoPE against learned absolute token positions with other settings fixed.

Clinical events use their event ages. Summary tokens use first-diagnosis ages. Sex and separator
tokens use the first retained event age. No prefix uses age at the end of a pretraining window.

For E6, compare the default against added gap tokens with buckets 1–6, 7–29, 30–90, 91–181,
182–364, 365–729, 730–1,825, and at least 1,826 days. Insert gaps only between distinct event
days. Do not insert an anchor-ending gap for the direct classifier. The head already receives
that exact recency. These coarse buckets are never a generation clock.

### 8.7 Pretraining samples

Use each fit patient's history strictly before the real anchor. For a long history, choose one
window per patient per epoch, starting at a day boundary. Recompute the prefix and summary from
information before that window. Keep the same patient sampling weights across model candidates.

For held-out pretraining evaluation, use fixed windows chosen by a logged seed. Reuse them at
every epoch. Do not make early stopping depend on newly sampled evaluation windows.

The data pipeline may cache deterministic tokenized histories. Training-window sampling is
stochastic by design. Repeating a run with the same seed must repeat its sampling schedule.

### 8.8 Extra anchors

Extra anchors remain optional. Set `a_k = real_anchor - 5*k calendar years`, for k = 1 to K.
Start with K = 3. Require at least one prior event and complete five-year follow-up. In this
revision, an extra label window must end no later than that patient's real anchor.

Reuse the existing truncation, first-diagnosis, and prior-condition functions. Both endpoints
are included under D4. Adjacent windows therefore share one boundary day. Do not describe them
as strictly disjoint. Report any diagnoses that fall on shared boundaries.

Sample patients uniformly. At each epoch, choose one supervised sample for each patient.
If extra anchors exist, choose the real anchor with probability 0.5. Otherwise choose one extra
anchor uniformly. Patients without extra anchors always contribute their real anchor.
Compare against the real-anchor-only method with matched optimizer-step budgets.

E3 tests no extras, extras without an indicator, and extras with a binary early-anchor head input.
The indicator is zero for every real-anchor evaluation patient. It does not isolate the shared
transformer parameters from extra-anchor updates. Its benefit must be measured.

### 8.9 Representation reports and outputs

Store each fold tokenizer, unit rules, value boundaries, fit-patient hash, and input-data hash.
Write separate pretraining, supervised-fit, and held-out sample files. Build val/test samples
only with the frozen fold tokenizers. Never refit them for those splits.

Report sequence lengths, unknown rates, clinical events retained, years retained, truncation
rates, source-specific losses, summary coverage, and last-event recency. Separate training
patients by truncation and density. Include target-positive counts for these development groups.
After the final freeze, apply the same reporting rules to val and test. Warnings are descriptive.

## 9. P4a–P4c: transformer and training

### 9.1 Common architecture

| Component | Small default | Optional larger control |
|---|---|---|
| Type | Decoder-only transformer with causal attention and pre-LayerNorm | Same |
| Layers | 2 | 4 |
| Width | 128 | 192 |
| Attention heads | 4 | 4 |
| Feed-forward width | Four times model width, GELU activation | Same ratio |
| Position | RoPE | RoPE |
| Age | Continuous age embedding | Same |
| Dropout | 0.1 | 0.1 |
| Context | 1,024 | 1,024 |
| Pretraining output | Token prediction with tied input/output token weights | Same |
| Supervised output | Linear layer with 40 logits, then independent sigmoids | Same |

Build the small default first. The larger control requires a budget slot and a specific training-only
reason, such as persistent underfitting after data and optimization checks. Record the actual
parameter count after vocabulary fitting. Do not label either model as
optimal or enforce an arbitrary two-million-parameter target. No downloaded model configuration
may silently load weights. General architecture code and random initialization are sufficient.

The head input is the last clinical-event state plus:

* Age at anchor divided by 100.
* Sex in the existing dataset encoding.
* `log1p(days_since_last_event) / log1p(36525)`.
* A missing-history flag.
* An early-anchor indicator only in that E3 variant.

Use the same readout in all three training methods. Apply head dropout 0.1. The fixed-body
method runs the body in evaluation mode and trains only the head. Tests must verify that body
weights and tokenizer files remain unchanged.

### 9.2 Self-pretraining

Initialize the body and token-prediction head randomly. Use next-token cross-entropy on clinical
events and, in the gap variant, gap tokens. Prefix, summary, separator, and padding positions
are context, not prediction targets. Define the shifted input/target mask explicitly.
Mask input-only symbols out of the output distribution. Do not concatenate different patients
without separate attention masks.

Use AdamW with betas (0.9, 0.95), weight decay 0.1 on weight matrices, and gradient clipping 1.0.
Start with peak learning rate 0.001. Warm up for 10% of the configured maximum optimizer steps,
then use cosine decay to 10% of the peak. Use float32. Start with 32 patients per batch and
length bucketing. Log token counts and effective batch size if accumulation is required.

Maximum 30 epochs. Evaluate fixed held-out windows each epoch. Stop after three evaluations
without improvement and retain the best checkpoint. Report overall next-token loss, per-source
loss, and loss for target-condition start tokens that have support. Log support counts.

Checkpoint reuse is permitted only across candidates with identical fit patients, tokenizer,
representation, architecture, context, and pretraining configuration. Fixed-body and full
fine-tuning candidates should share the same initial self-pretrained checkpoint.

### 9.3 Supervised methods

| Method | Initial body | Parameters updated |
|---|---|---|
| S: supervised scratch | Random | Body and head |
| H: fixed body | Our self-pretrained checkpoint | Head only |
| F: full fine-tuning | The same self-pretrained checkpoint as H | Body and head |

Use unweighted binary cross-entropy over unmasked cells. Normalize by their count. Skip a batch
only if it contains no unmasked cell, and log that event. Start without class weights or focal
loss because probability quality is part of the task. Do not claim that weighting can never
help ranking; it is outside the initial search.

Use AdamW with weight decay 0.01 and clipping 1.0. Head learning rate starts at 0.001. Body learning
rate starts at 0.0003 for S and 0.0001 for F. H has no body optimizer. A bounded recovery search
may test a factor-three lower or higher body rate for a documented convergence problem.
Do not give one training method a large hidden search advantage.

Use batches of 32 patients. Maximum 30 epochs, patience five on held-out masked macro AUROC.
Use mAP to break exactly tied checkpoint scores, then prefer the earlier epoch. Store training
and held-out losses, AUROC, mAP, gradient norms, and effective learning rates. Keep the epoch
checkpoints needed for the final validation curves. Official val never controls early stopping.

### 9.4 Checks before broad training

T10 must verify causal attention in evaluation mode. T11 must demonstrate that the trainable
model can fit eight distinct supervised samples with dropout disabled. Test the language-model
loop on an unambiguous synthetic sequence batch. Do not require loss below 0.05 on an impossible
batch with identical inputs and conflicting next-token targets.

Check padding invariance, readout indices, loss shifting, fixed-body immutability, and finite
outputs. Compare a tiny CPU/MPS run within documented numerical tolerances. Record observed
nondeterminism rather than promising byte-identical device computations.

The selected model must use its own five-fold tokenizer and weight artifacts. No external
checkpoint is permitted at any stage, including debugging and feature extraction.

### 9.5 Loss accounting and recovery

For each optimizer update, divide summed next-token loss by the number of eligible target tokens.
Divide summed supervised loss by the number of unmasked patient–target cells. Gradient accumulation
must use these totals across all contributing microbatches. A microbatch is one part of a larger
optimizer batch. Do not average microbatch means when their eligible counts differ. Handle the
last partial batch explicitly. Use the same denominators for reported epoch losses.
Clip gradients once after accumulation. If an optimizer batch has no eligible targets, record it
and skip both the optimizer and schedule update. Do not divide by zero or invent a loss value.

Uniform patient sampling does not imply equal patient weight in a token-averaged pretraining loss.
Report tokens per patient and the objective used. Compare next-token losses only when vocabulary,
target masks, and evaluation windows match. A changed tokenizer can change the prediction problem.
Stop on non-finite loss, gradients, or probabilities. Record the failed run; do not silently skip it.

Support recovery at completed epoch boundaries. Save model, optimizer, learning-rate schedule,
epoch, global step, early-stopping state, and random-generator states. Save the sampler seed and
epoch, configuration, tokenizer hash, and fold/data identifiers. Restore them before creating
the next epoch's batches. Save to a temporary file, then replace the completed checkpoint atomically.
Reject incompatible or incomplete checkpoints. Use a new run ID for an intentional changed run.

T35 compares uninterrupted and resumed training on a deterministic CPU fixture, including the next
sample order, learning rate, and parameter update. Document numerical tolerance and device limits.
Do not promise identical results across hardware or software releases. Checkpoint and reproducibility
details follow the [PyTorch checkpoint guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
and [reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## 10. P4d: generation feasibility and deferral

### 10.1 Decision from the permitted data

Zero-shot generation estimates a target's risk by counting its appearance in sampled futures.
Here, zero-shot means that the event generator has not been fitted to the supervised outcome labels.
It still requires training examples of the events it must generate.

The training-only coverage audit found:

| Required target | Code | Patients with a pre-anchor diagnosis | Positive supervised training labels |
|---|---|---:|---:|
| Pneumonia | `233604007` | 0 | 133 |
| Prostate cancer metastasis | `314994000` | 0 | 50 |

The generator would receive no positive next-event training examples for these two targets.
Reserving output symbols would not teach their probability or timing. Generating an arbitrary
nonzero probability for them would not resolve the missing evidence.

All three supervised methods can use their permitted outcome labels. The 40-output head remains
complete even when an input vocabulary lacks a target code. This is a concrete reason to prioritize
direct supervised prediction, independent of the papers' architecture comparisons.

P3a must reproduce and save the coverage counts. P4d then records deferral. Do not train an additional
model for generated-future risk scoring or run a future-sampling grid. Generated probabilities
cannot enter this submission. Next-token self-pretraining for H and F remains required.
Do not add post-anchor self-pretraining or outside data to repair this coverage failure.

### 10.2 Findings retained for a future design

The literature review identified other requirements. They remain useful research notes, but do
not require generation implementation in this version.

| Requirement | Needed evidence |
|---|---|
| Target coverage | Each required target-start symbol has observed support in every permitted fit set. |
| Exact time | An integer-day clock reproduces long gaps and calendar-five-year boundaries without bucket-median drift. |
| Arbitrary anchors | Trained query conditioning represents both quiet days and event days. The prefix cannot reveal the next event's time. |
| Observation cutoff | The final gap is right-censored: its later event time is unknown. Known silence contributes a defined survival likelihood. The cutoff cannot become an invented clinical event. |
| Valid trajectories | A declared token grammar, age updates, and context-rebuild rule pass hand-built tests. Invalid-token masks have reported effects. |
| Sampling stability | Doubled future counts and independent sampling seeds give sufficiently stable probabilities and CV metrics. |
| Completion | Capped trajectories remain unresolved. Lower and upper probability bounds account for them. |
| Cost | Measured generation speed and memory support the full CV and final ensemble workload. |

One possible exact clock uses repeated `[WAIT:365]` tokens and a final remainder of 1–364 days.
Its full grammar and censoring loss would need a separate specification. The direct classifier's
coarse gap tokens must never serve as that clock.

The prediction window includes both offset zero and the calendar-five-year end day. A future
generator must process all events on that final day before stopping. Count target starts only.
Apply the observed-history prior mask independently of any generated history.

At N = 64 futures and a true model probability of 0.01, the probability of zero sampled hits is
about 0.526. Thus, 64 futures is not an established sufficient sample count. Any later proposal
must choose N from training-only convergence checks and keep it fixed for CV, val, and test.
A capped trajectory cannot automatically count as a negative outcome.

At 300 tokens per future and N = 64, training CV would require about 53.6 million generated tokens.
Five-model val and test inference would add about 69.4 million. These are workload estimates,
not measured runtimes. None belongs in the current implementation budget.

Reopening this branch requires a dated protocol revision that resolves its coverage problem
within the assessment rules. The current plan does not assume such a resolution.

## 11. P4e: final artifacts and official validation

The five-fold comparison runs already produce the candidate model artifacts. Do not retrain them
with new settings after selection. The final ensemble uses the primary-seed artifacts from the
chosen complete configuration. Stability-seed runs describe variation; they are not selected by score.

Use this procedure:

1. Select the direct classifier using training CV.
2. Verify that the generation coverage decision is recorded.
3. Fix the five-fold ensemble under D18.
4. Fix all baseline configurations and descriptive analyses.
5. Write `configs/final.toml` and an artifact manifest.
6. Commit the code, configuration, manifest, and evaluation specification.
7. Verify every model, tokenizer, input, and configuration hash.
8. Run `make val-score` once.
9. Write `docs/model_results.md`.

The manifest includes checkpoint paths and hashes, fold fit-patient hashes, tokenizer hashes,
training seeds, chosen epochs, data fingerprints, mask and target order, ensemble settings, and
the exact checkpoint list for validation curves. Generated binaries remain outside Git if needed.

The val bundle evaluates final probabilities and the predefined epoch checkpoints for learning
curves. This supplies the requested official validation curves without using them for selection.
The bundle logs these curves to wandb, in the same project as the training curves, with one run
per fold model that records the ID of that model's training run.
Do not choose a different epoch after viewing these curves. Predefine any source perturbations
and subgroup analyses in the same bundle. There is no separate clinical-score scoring run.

The script refuses a changed tracked working tree, hash mismatch, or an already completed bundle.
Its code fingerprint covers every executed project module and the dependency lock file. An
untracked imported module cannot bypass the freeze. Run these rejection paths during G2.
It writes a start record before computing metrics and a completion or failure record afterwards.
Store run ID, timestamp, code commit, config and manifest hashes, and models evaluated.

A documented implementation bug can justify a replacement bundle. Record the bug and new manifest
before rerunning. Retain prior results and disclose all attempts. A disappointing score is not a bug.
There is no independent untouched local test after this process; test labels remain with organisers.

## 12. P4f: interpretation and literature context

Complete the analysis specification before P4e. Use training out-of-fold results to develop it.

Required analyses are per-target performance, at-risk comparisons, history-retention groups,
source coverage, and training behavior. Calendar probes are optional diagnostics. Use broad groups
and show positive counts. Avoid
unreliable AUROC claims when a group lacks both classes. Brier scores can still be descriptive,
but differing prevalence also affects their interpretation.

Optional E10 replaces one source's event tokens with source-specific unknown tokens at inference.
Retain the fixed observed-history prior mask. State whether summary tokens are also perturbed.
This is a sensitivity check, not a causal explanation. A source-removal test cannot establish
how changing real care would change outcomes.

Clinical reference documents remain optional background. They create no implementation or
evaluation requirement. Do not load their coefficients or evaluate their calculators. The report
explains technical errors, such as sparse labels, lost history, poor convergence, or miscalibration.
It does not try to establish clinical benefit or superiority over clinical care.

Keep the supplied three papers in the literature section. Distinguish their observed results
from our hypotheses. No imported weights, code ontologies, or source-generator rules enter the model.

## 13. Experiment register

| ID | Experiment | Priority and decision rule |
|---|---|---|
| E0 | Matched-seed stability of leading complete configurations | Required; section 5.3. |
| E1 | S versus H versus F using the same representation and fold patients | Required five-fold comparison on the small default. Preserve this evidence even if S wins. |
| E2 | Full-timeline versus history-only self-pretraining | Retired. Only pre-real-anchor self-pretraining is permitted in version 2. |
| E3 | No extra anchors versus patient-balanced extras, with or without early indicator | Conditional after a stable direct classifier; five-fold confirmation. |
| E4 | Small default 2 x 128 versus larger 4 x 192 | Conditional on a documented capacity problem and an extension slot. Five-fold confirmation before adoption. |
| E5 | Context 1,024 versus 512 and versus 2,048 | Screen 512. Give 2,048 first extension priority when memory and the core-work budget permit (F12). Record skipped comparisons. Confirm every adopted alternative across five folds. |
| E6 | Age-only event representation versus age plus coarse gap tokens | Optional extension. Record retained clinical events and years. |
| E7 | Zero-shot transformer versus selected direct transformer | Deferred. F11 fails coverage before generation training. No current submission comparison. |
| E8 | Code-count LR with and without anchor year | Optional diagnostic within a fixed budget; never submitted. Declare its inclusion before val. |
| E9 | Era information in code features | Optional training-only diagnostic. Calendar exclusions remain required without it. |
| E10 | Source perturbation of the final transformer | Optional descriptive analysis, fixed before val. |
| E11 | LightGBM counts, counts + values, and counts + binned values | Counts and counts + values are required comparisons after G2. The binned diagnostic has a separate budget and cannot delay core evaluation. None is submitted. |
| E12 | Published clinical-score numerical comparison | Retired from assessment execution. Literature context only. |
| E13 | RoPE versus learned absolute positions | Optional extension; five-fold confirmation before adoption. |
| E14 | Refill compression on versus off | Optional extension; five-fold confirmation before adoption. Compression correctness and retention checks remain required. |
| E15 | Sensitivity to ordering of events on the same day | Optional training-only robustness check. Do not interpret sorted code order as clinical causality. |
| E16 | Transformer without numeric value groups | Optional information-content comparison against code-count methods. |
| E17 | Training-patient learning curve for the selected transformer | Conditional if data limitation or model capacity remains unclear. |
| E18 | Fused bin tokens versus bin tokens plus a continuous value channel (section 8.3) | Conditional on a material value-bin loss and an extension slot. Five-fold confirmation and calibration reporting required if adopted. |

Sequential screening is deliberate. This is not a Cartesian search across all experiment choices.
Use identical fit patients and matched seeds for paired runs. Record exposure in patients, windows,
tokens, and optimizer steps. Confirm the final combination rather than adding isolated gains.

Each run record contains experiment ID, complete configuration hash, fold, seed, parent checkpoint,
training data scope, tokenizer hash, compute cost, metrics, wandb link, and code commit.

## 14. Test register

| ID | Required test |
|---|---|
| T1 | Real and extra prediction histories contain no event on or after their fixed anchor. |
| T2 | Shifting birth, event, and anchor dates together by +/-3,652 days preserves token IDs, ages, and recency inputs. |
| T3 | A code exclusive to a held-out patient maps to the appropriate unknown token. |
| T4 | Value boundaries and unit rules use fit histories only. Boundaries increase and out-of-range values clip correctly. |
| T5 | Summaries use only selected conditions diagnosed before the retained window. Their ages match first diagnoses. |
| T6 | Cached histories and fixed evaluation samples reproduce exactly. Seeded training-window schedules also reproduce. |
| T7 | Refill compression preserves patient–drug pairs, first starts, known final stops, and true interruptions. |
| T8 | Extra-anchor labels and masks equal the existing functions. Shared boundary-day behavior is explicit. |
| T9 | No patient crosses fit and held-out folds. Val/test never enter a fit set. |
| T10 | Later tokens cannot change earlier causal outputs in evaluation mode. |
| T11 | The trainable model fits a suitable tiny batch. The test does not demand impossible predictions. |
| T12 | Masked supervised cells contribute zero gradient. An all-masked batch is handled explicitly. |
| T13 | Prediction masking sets exactly the prior-condition cells to zero. |
| T14 | Generation counts only target starts and respects the exact inclusive five-year horizon. |
| T15 | AUROC, AP, Brier score, ties, and declared degenerate cases match reference calculations. |
| T16 | The val runner checks the manifest, records start/completion, and rejects accidental repeat scoring. |
| T17 | Submission has 358 unique test IDs and 40 correctly ordered target columns, finite values in [0,1], and no missing predictions. |
| T18 | Reloaded fold artifacts and the ensemble reproduce in-memory predictions within declared device tolerance. |
| T19 | With anchors and fitted artifacts fixed, changing future events, stops, values, or excluded fields leaves sequences, head inputs, masks, baseline features, and predictions unchanged. Labels may change. |
| T20 | No vocabulary, bins, scaler, imputer, or self-pretraining sample uses a fit patient's post-real-anchor records. |
| T21 | Extra-anchor sampling gives equal patient selection probability, independent of eligible-anchor count. |
| T22 | Fixed-body training leaves all body parameters unchanged and uses a defined readout. |
| T23 | Padding length and batch composition do not change a patient's evaluation prediction. Readout excludes padding and structure. |
| T24 | Generation encodes and decodes exact day offsets, long gaps, and leap-year horizons without accumulated timing error. |
| T25 | Quiet-query and event-day-query prefixes contain no future-event timing. Grammar transitions and age updates are valid. |
| T26 | A capped generation trajectory is unresolved. Its probability bounds and deterministic continuation are correct. |
| T27 | Out-of-fold predictions use only the patient's held-out model. Final val/test ensemble uses all five distinct fold artifacts. |
| T28 | Every loaded training checkpoint has a permitted local parent and verified provenance. No external model or tokenizer is fetched. |
| T29 | Empty histories, all-unknown histories, oversized summaries, and oversized single days still produce valid predictions. |
| T30 | Pretraining targets and input-only symbols are masked at the correct shifted positions. Patients cannot attend across sequence boundaries. |
| T31 | The training-only target-coverage audit identifies unsupported generation targets and records deferral. |
| T32 | Target-code headers and prior masks remain correct across fold-specific vocabularies. |
| T33 | Generation handles a right-censored final gap without inventing its next event. Known silence has the specified likelihood. |
| T34 | Prediction runs without labels, future tables, or prohibited cohort fields. Label changes cannot change fixed-model predictions. |
| T35 | Epoch-boundary recovery restores sampling, optimizer, schedule, early stopping, and random states. Corrupt or mismatched checkpoints fail clearly. |
| T36 | Unequal microbatches match their combined batch on a dropout-free CPU fixture for both objectives. Zero-target batches make no optimizer or schedule update. |
| T37 | A clean-environment CPU command scores mock test histories from saved artifacts. No training-only path, label file, or network model download is required. |
| T38 | Duplicate/missing IDs, invalid dates, stale caches, and target mismatches fail clearly. Independent row permutations do not alter correctly joined metrics. |
| T39 | If E18 runs, constant values, missing values, unseen units, and distinct same-bin values follow the stored rules without non-finite tensors. |
| T40 | Hand-calculated cases verify real anchors, first-ever labels, prior cases, repeats, both horizon endpoints, and leap-year boundaries. |

Generation implementation tests T14, T24, T25, T26, and T33 are deferred with that branch. Their
absence must not be presented as completed generation work. T31 remains a required data check.
T8 and T21 apply only if extra anchors are implemented; T39 applies only to E18. Do not build an
optional feature merely to pass its test. All core and activated-feature tests precede final scoring.

## 15. Risk register

| ID | Risk | Control |
|---|---|---|
| R1 | Calendar information or end-of-record fields enter predictions. | Allowlists, D10, T2, T19, and separate era diagnostics. |
| R2 | Train and test preprocessing differs. | One prediction builder and frozen descriptive parity checks. |
| R3 | Small patient count causes overfitting. | Small default, fixed-body method, dropout, early stopping, seed checks, and honest CV interpretation. |
| R4 | Self-pretraining memorizes individual histories. | Patient-disjoint held-out loss, training/held-out gap, and S/H/F comparison. |
| R5 | Fold-0 search selects noise. | Five-fold confirmation and bounded searches. |
| R6 | MPS variation or device errors affect reproducibility. | Tiny CPU comparison, logged versions, seeds, and numerical tolerances. |
| R7 | Summary or preprocessing uses future information. | Cut before compression; T5, T19, and T20. |
| R8 | Generated time is inaccurate. | Generation deferred. Any future branch requires exact days and T24. |
| R9 | Extra anchors overweight older patients or change the task distribution. | Uniform patient sampling, real-anchor evaluation, and E3 confirmation. |
| R10 | Codes encode era or observation patterns. | Retention/source reports, optional era probes, and a stated limit on generalization. |
| R11 | Validation is reused for choices. | Artifact freeze, one logged bundle, fixed curve schedule, and disclosed historical val use. |
| R12 | Computation exceeds the local budget. | Measured run plan, shared checkpoints, and optional-branch gates. |
| R13 | Identical target labels duplicate part of the macro score. | Keep required targets and report duplication. |
| R14 | Refill compression removes meaningful information. | E14, interval tests, and retention reports. |
| R15 | Dependencies fail on reviewer hardware. | Lock file, CPU inference path, setup notes, and reload smoke test. |
| R16 | Baseline and transformer inputs differ. | Counts + values baseline, optional E16, and careful claims about sequence information. |
| R17 | Generation cannot learn absent targets or estimate rare events reliably. | F11 and T31 stop the current branch. Any future branch also needs sampling and completion checks. |
| R18 | Pretraining loss improves without better condition prediction. | Per-source losses and supervised representation comparisons. |
| R19 | Truncation harms high-density histories selectively. F12: truncated patients keep a median 3.6 of 64.6 history years. | Required retained-time/source reports, subgroup evaluation, condition summary, and priority for the E5 2,048 comparison. |
| R20 | An external coefficient or checkpoint enters the pipeline. | Resource exclusions, provenance manifest, and T28. |
| R21 | Quantile bins hide numeric differences that matter for a target. | Budgeted value-bin check after G2; conditional E18; no outside thresholds. |
| R22 | Unequal token or mask counts distort accumulated gradients. | Explicit objective denominators and T36. |
| R23 | A resumed run changes the experiment or a package depends on training files. | Full recovery state, provenance checks, and T34–T38. |
| R24 | Experiment scope displaces correctness and delivery. | G1–G4, a small default, fixed trial limits, and reserved core-work budget. |

## 16. Logging, reproducibility, and proposed files

Use wandb project `ehr-forecast`. Use offline mode until the account or team is selected. Sync logs
before submission. Record the full TOML configuration, code commit, model parent, fit-patient hash,
input hashes, tokenizer hash, software versions, seed, device, and training resource use.

Log training loss, learning rate, and gradient norm by step. Log held-out metrics and training
metrics by epoch. Save per-target results and support counts. Official val metrics appear only
in the P4e bundle. Do not label internal held-out-fold curves as official val curves.

The README asks for a wandb link in the report. Before delivery, group the final configuration's
training runs and val-bundle runs in one wandb report or project view, make it public, and check
the link while logged out. Reviewers cannot open a private project.

Keep all settings in `configs/*.toml`. Record exact dependency versions in `requirements-lock.txt`.
New packages are expected to include `torch`, `wandb`, and `lightgbm`. Install packages without
fetching external models or sample datasets. Store generated data and binaries outside tracked source.

Proposed layout:

```text
configs/                       default.toml, budget.toml, experiments/, final.toml
src/ehrforecast/
  folds.py                     saved patient assignments and fold loaders
  masking.py                   common prior-condition mask
  lgbm.py                      comparison methods
  artifacts.py                 calendar and era diagnostics
  tokenizer.py                 fit-only vocabulary, units, values, and age features
  sequences.py                 cutoffs, summaries, windows, and retention reports
  anchors.py                   optional patient-balanced extra anchors
  model.py                     random-init transformer and common prediction head
  pretrain.py                  permitted history self-pretraining
  finetune.py                  S, H, and F supervised methods
  zeroshot.py                  deferred; do not implement in version 2
  evaluate.py                  metrics and frozen evaluation bundle
models/transformer/<config>/fold<k>/
                               tokenizer.json, checkpoints/, selected.pt, provenance.json
reports/                       representation/, transformer/, lightgbm/, artifacts/, val_runs.log
reports/transformer/            experiments.csv, budget.json, generation_gates.json
models/final/                  manifest.json and five selected fold artifacts
docs/                          model_plan.md, model_results.md, ai_workflow.md
```

File names can be consolidated where clearer. Do not create an external clinical-score loader.
Proposed commands: `folds`, `baselines-extra`, `tokenize`, `pipeline-smoke`, `pretrain`,
`finetune`, `freeze`, `val-score`, and `predict`. Training commands support epoch-boundary resume.
Every command writes a log. `pipeline-smoke` exercises G2 with training-fold data and temporary outputs.

## 17. Completion and submission

A successful implementation produces a valid transformer system and an accurate account of its
performance. Strong results are the aim; changing labels, using external weights, or concealing a
stronger baseline is never an acceptable way to obtain them.

Before step 5, require:

* G1–G3 and all core tests pass. Every test patient can be scored.
* The chosen complete configuration has five-fold results and recorded selection evidence.
* Every artifact has verified training-only provenance.
* The official validation bundle is complete and its attempts are disclosed.
* Per-target, at-risk, probability-quality, and retention results are available.
* The report explains the synthetic data and retrospective anchor limitations.

Step 5 produces `predictions.csv`, the technical PDF, a reproducible public GitHub repository,
and the trained transformer package on Hugging Face. The public package must include tokenizer
artifacts, all ensemble members, target order, prior-mask logic, input schema, and a CPU prediction
entrypoint. Publish only the project's own trained weights. Verify a clean reload before upload.
G4 also requires a recorded run in a fresh environment from the lock file. Its prediction command
uses only supplied truncated histories, anchors, and the model package. It must not need a local
training cache or label file. Retain the self-pretraining and adaptation evidence if S is submitted.

The report covers representation, model and training, AI workflow, and interpretation. Record
which AI tools performed work, what was reviewed, and which bugs the review found. Label every
metric by split. Do not fabricate hidden test metrics or present val scores as test scores.

The main evidence of engineering ability is concrete:

| Capability | Reviewable evidence |
|---|---|
| Data engineering | Saved splits, input contracts, label-boundary tests, and future-perturbation tests. |
| Representation learning | A locally trained next-token checkpoint, its data lineage, and held-out learning curves. |
| Model training | Correct losses, finite gradients, recovery checks, resource measurements, and S/H/F comparisons. |
| Evaluation | Versioned metrics, strong comparison methods, fold and per-target results, and a frozen val record. |
| Reproducibility | Locked dependencies, artifact hashes, CPU reload, one prediction command, and a valid CSV. |

Keep the report focused on these artifacts and the observed results. Larger parameter counts,
more ablations, and claims of clinical utility do not replace this evidence.

Each question in the README's report section has a planned answer:

| README question | Planned answer | Plan section |
|---|---|---|
| What is one unit of input? | One clinical event token, `source:kind:code`, with numeric or text values fused into it. | 8.2 |
| Which fields and modalities, and why? | Structured CSV tables through allowlists, including imaging-study events. No genomics or DICOM (D5). No race or ethnicity (D15). No future-bearing or calendar fields (D3, D10). | 2.1, 8.2 |
| How were continuous values such as labs handled? | Quantile bins per code and unit, fitted on fit patients. A budgeted value-bin diagnostic can motivate E18. Report untested alternatives explicitly. | 7.2, 8.3 |
| How was the time between events handled? | Every token carries its age to the day, so the model can infer the time between any two events. RoPE gives token order. The head receives the exact time from the last event to the anchor. E6 tests explicit gap tokens. | 8.6, 9.1 |
| Architecture, objective, hyperparameters | Decoder-only transformer, S/H/F training methods, and the frozen settings in `configs/final.toml`. | 9 |
| wandb training and validation curves, with a link | Training and held-out curves during training; official val curves from the frozen bundle; one public link. | 11, 16 |
| How AI coding tools were used | `docs/ai_workflow.md`, kept during implementation. | 17 |
| Why the model does well or poorly across conditions | Per-target, at-risk, retention-group, and source analyses. | 12 |
| Test macro AUROC and mAP | Test labels are hidden. Report CV and val results, labelled by split, and attribute test scores to the organisers. | 5, 17 |

## 18. Open administrative items and revision log

| Item | Needed by | Default handling |
|---|---|---|
| wandb account or team | Before final report delivery | Log offline and sync later. Make the final runs public and test the link while logged out. |
| Compute ceiling and available device | P4a budget freeze | Measure the local machine first. Do not assume access to a cloud GPU. |
| Report name and GitHub/Hugging Face accounts | Step 5 | Preserve existing open items. |

Scientific choices in the experiment register are resolved by training-only evidence. They are
not permission questions that block P3a. Record any protocol change after the first transformer
result with its date, reason, affected runs, and whether results were already known.

| Version | Change |
|---|---|
| 1 | Initial model plan, 30 October 2026. |
| 2 | Applied the three-paper review and Derek's transformer-only requirement. Restricted self-pretraining to pre-anchor training histories. Removed external clinical coefficients. Added S/H/F, RoPE, retention tests, future-perturbation checks, and a consistent artifact/val freeze. Deferred generation after the F11 coverage audit. Retired E2 and numerical E12. |
| 2.1 | 31 October 2026. Added F12 (measured context loss) and made the 2,048 comparison in E5 required when memory allows. Added the P3a value-bin information check (third LightGBM feature set), conditional E18 (continuous value channel), R21, and two numeric-encoding sources. No implementation or val result informed these changes. After a README audit, specified wandb logging of the val-bundle curves and a public link, and added a README-question map to section 17. |
| 2.2 | 31 October 2026. Made pipeline correctness and reproducibility the primary objective. Clinical usefulness is not an objective. Added G1–G4, input interfaces, full prediction-invariance checks, loss accounting, recovery, and clean-environment inference. Made 2 x 128 the default and bounded extensions. Deferred full baseline searches until after the first complete run. Kept S/H/F, strong baselines, five-fold evaluation, and the frozen val bundle. Clarified selection bias and metric edge cases. Preserved F12 and the value-bin diagnostic as priorities within the budget. No new model or val result informed this review. |
| 2.3 | 31 October 2026. Added D31 after the P3a raw-data audit found 57 training medication rows with a stop before the start. Derek approved the rule: keep the start, omit the invalid stop. Event histories in every split use it; the historical baseline is unchanged. Added D32 (refill compression links each stop to its own start) after M1 found 39,274 zero-length prescriptions. No model or val result informed these changes. |

## 19. Sources

Assessment rules: [supplied README](../../research_intern_coding_test/ReadMe.md).
Existing data decisions: [D1–D8](decisions.md).
Existing reference results: [baseline results](baseline_results.md).

* Shmatko A, et al. Learning the natural history of human disease with generative transformers.
  [Nature (2025)](https://doi.org/10.1038/s41586-025-09529-3).
  Relevant: architecture, age and time representation, model scale, and cohort bias.
* Pang C, et al. CEHR-GPT: Generating Electronic Health Records with Chronological Patient Timelines.
  [arXiv:2402.04400v2 (2024)](https://arxiv.org/abs/2402.04400v2).
  Relevant: sections 3, 5.1, and 5.2 on representation and temporal distortion.
* Wornow M, et al. Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHRs.
  [ICLR 2025; arXiv:2412.16178v2](https://arxiv.org/abs/2412.16178v2).
  Relevant: section 3.2, section 4.4, Appendices D and H, and Figures 11–12.
* Renc P, et al. Zero shot health trajectory prediction using transformer.
  [npj Digital Medicine (2024)](https://doi.org/10.1038/s41746-024-01235-0).
  Relevant: the generation-based prediction method. No released weights or tokenizers are used.
* Li Y, et al. BEHRT: Transformer for Electronic Health Records.
  [Scientific Reports (2020)](https://doi.org/10.1038/s41598-020-62922-y).
* Rasmy L, et al. Med-BERT. npj Digital Medicine (2021).
  [Paper](https://doi.org/10.1038/s41746-021-00455-y).
* Vaswani A, et al. Attention Is All You Need. NeurIPS (2017).
  [Paper](https://arxiv.org/abs/1706.03762).
* Lee I, et al. Representation Before Training: A Practical Benchmark for Generative Medical
  Event Model Tokenization. [arXiv:2604.16775v3 (2026)](https://arxiv.org/abs/2604.16775v3).
  Relevant: fused versus unfused value tokens, bin granularity, soft discretization, and xVal.
* McCann KA, et al. Continuous Value Tokenization Improves Medical Event Foundation Models.
  [medRxiv preprint (2026)](https://www.medrxiv.org/content/10.64898/2026.08.04.26359713v1).
  Relevant: continuous value embeddings, their calibration, and the equal-width-bin baseline.
* Karpathy A. [nanoGPT](https://github.com/karpathy/nanoGPT).
  Architecture reference only. No pretrained weights are imported.
