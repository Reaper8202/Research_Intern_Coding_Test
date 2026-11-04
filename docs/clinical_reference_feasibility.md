# Clinical reference baseline feasibility

Reviewed on 29 October 2026. This review applies the clinical-score inventory to the proposed comparison against LightGBM and the transformer. It does not implement scores or report model performance.

**The practical shortlist is four endpoint candidates, with heart failure as a conditional fifth.** These are four or five distinct endpoint codes, not four or five independent score formulas. The wider inventory found seven long-term event candidates and 14 broader clinical-tool associations. Diagnosis and recovery tools do not answer this project's forecasting question.

| Target | SNOMED code | Candidate reference | Fit and limits |
|---|---|---|---|
| Coronary heart disease | 53741008 | Framingham coronary heart disease model | Select the CHD-specific version. The general CVD model includes other diseases. Record its exact age range and exclude patients with existing disease as its intended use requires. [Framingham risk functions](https://www.framinghamheartstudy.org/fhs-for-researchers/fhs-risk-functions/). |
| Myocardial infarction | 22298006 | Framingham hard CHD or ASCVD pooled cohort equations | These predict combined outcomes. Hard CHD includes coronary death. ASCVD includes stroke. Comparing their ranks against MI alone is a component-outcome comparison, not an exact endpoint validation. [Framingham hard CHD](https://www.framinghamheartstudy.org/fhs-for-researchers/fhs-risk-functions/hard-coronary-heart-disease-10-year-risk/), [ACC ASCVD estimator](https://tools.acc.org/ascvd-risk-estimator/default.aspx). |
| Stroke | 230690007 | Revised Framingham Stroke Risk Profile | The revised model omits ECG left ventricular hypertrophy, which was not identified as a structured input in this data review. Its age range, previous stroke exclusion, and other inputs still need enforcement. Use CHA2DS2-VASc only for a separate subgroup with AF before the anchor. [Revised stroke model](https://pmc.ncbi.nlm.nih.gov/articles/PMC5504355/), [NICE AF guidance](https://www.nice.org.uk/guidance/ng196/chapter/recommendations). |
| Atrial fibrillation | 49436004 | CHARGE-AF | Predicts five-year incident AF. The guideline recognizes the model, but widespread routine use and the benefit of score-directed screening should not be assumed. It is a published reference model. [2023 AF guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC11095842/). |
| Chronic congestive heart failure — conditional | 88805009 | PREVENT-HF | Predicts future HF, with a broader outcome than the exact chronic congestive HF code. Kidney function is required and often missing. [AHA PREVENT calculator](https://professional.heart.org/en/guidelines-and-statements/prevent-calculator). |

Type 2 diabetes is not one of the 40 target conditions. Prediabetes is a target, but it is a different outcome. FINDRISC therefore has no direct target match. [Original diabetes risk score study](https://diabetesjournals.org/care/article/26/3/725/29197/The-Diabetes-Risk-ScoreA-practical-tool-to-predict).

FRAX is an optional extension for pathological fracture due to osteoporosis (`443165006`). It predicts fractures at specified sites. Parental hip fracture history, alcohol intake, and other required clinical factors need an explicit availability review. The absence of bone density alone does not exclude FRAX, since the model can operate without it. [FRAX tool description](https://www.fraxplus.org/).

CAPRA is not suitable for the main validation comparison here. It requires an existing prostate cancer diagnosis and biopsy findings. The validation split has 27 patients with prostate neoplasm recorded before the anchor. None developed the metastasis target in the follow-up window. AUROC and AP cannot provide a useful positive-event comparison in that subgroup. Six validation patients have the metastasis target overall, but none has the prostate-neoplasm target recorded before the anchor. Gleason grade and positive biopsy-core fraction were not found in the structured observation descriptions. [CAPRA inputs and intended population](https://urology.ucsf.edu/research/cancer/prostate-cancer-risk-assessment-and-the-ucsf-capra-score).

The following local counts come from the existing processed cohort, events, labels, and prior-condition tables. The validation split has 365 patients. All observation rows used in this check had a timestamp strictly before the patient's anchor. The check selected the most recent observation per patient and code. These counts establish field presence only. They do not validate units, values, observation age, or all clinical input definitions.

| Input | Observation code | Validation patients with a value |
|---|---|---:|
| Systolic blood pressure | 8480-6 | 350 |
| Diastolic blood pressure | 8462-4 | 350 |
| Total cholesterol | 2093-3 | 294 |
| HDL cholesterol | 2085-9 | 294 |
| Smoking status | 72166-2 | 350 |
| Height | 8302-2 | 350 |
| Weight | 29463-7 | 350 |
| Body mass index | 39156-5 | 350 |
| Estimated glomerular filtration rate, a kidney function measure | 33914-3 | 115 |
| Creatinine | 38483-4 | 188 |
| Serum or plasma creatinine | 2160-0 | 5 |
| Prostate-specific antigen | 2857-1 | 22 |
| DXA T-score | 38265-5 | 40 |

Systolic blood pressure, total cholesterol, HDL, and smoking status are jointly present for 292 of 365 validation patients. Of these, 182 are aged 40 through 79. This is an input-coverage count, not the final eligible ASCVD cohort. Existing cardiovascular disease and the other score requirements still need assessment. Age at anchor ranges from about 3 to 103 years, so scoring every patient with an adult calculator would be inappropriate.

Only 69 validation patients aged 30 through 79 have that common input bundle plus recorded kidney function and body mass index. This count does not apply PREVENT's existing-CVD exclusion or check all other inputs. Kidney-function observations contain both `mL/min/{1.73_m2}` and `mL/min` units. These must not be treated as interchangeable without checking the export and measurement definition. Deriving kidney function from suitable creatinine measurements could increase coverage. That would require a documented equation and a unit check.

| Target | Validation patients without that target before the anchor | New diagnoses |
|---|---:|---:|
| Coronary heart disease | 324 | 15 |
| Myocardial infarction | 363 | 19 |
| Stroke | 324 | 35 |
| Atrial fibrillation | 323 | 14 |
| Chronic congestive heart failure | 360 | 29 |
| Pathological fracture due to osteoporosis | 355 | 14 |
| Prostate cancer metastasis | 365 | 6 |

These are endpoint-level counts before score-specific eligibility restrictions. For CHA2DS2-VASc, only 27 validation patients have AF before the anchor and no previous stroke. Seven develop a first stroke. This comparison can be exploratory, but its uncertainty will be large. Do not apply the score to the whole cohort or use it to predict AF onset.

The Synthea check found a material version difference:

| Source revision inspected | Evidence |
|---|---|
| `50d8ce2d2b7a031751518d03a881b00f1b186929`, 27 May 2021 | The cardiovascular module calculates Framingham-based coronary risk and uses it to trigger CHD onset. It also uses Framingham-based point rules for AF and stroke. MI can arise through a later CHD progression pathway. [Pinned source](https://github.com/synthetichealth/synthea/blob/50d8ce2d2b7a031751518d03a881b00f1b186929/src/main/java/org/mitre/synthea/modules/CardiovascularDiseaseModule.java). |
| `bd8a52ba0a54ad8c6a1004b74787a14d8c25dbcc`, 21 October 2026 | The cardiovascular module defaults to ASCVD instead of Framingham for coronary risk. It still calls Framingham routines for AF and stroke. [Pinned source](https://github.com/synthetichealth/synthea/blob/bd8a52ba0a54ad8c6a1004b74787a14d8c25dbcc/src/main/java/org/mitre/synthea/modules/CardiovascularDiseaseModule.java). |

The supplied README identifies the Coherent dataset but does not pin a Synthea revision or configuration. The 2021 source is evidence of historical generator behavior, not proof that it generated these particular files. The current source alone cannot establish how a dataset published in 2022 was generated. [Coherent dataset publication](https://doi.org/10.3390/electronics11081199).

A published clinical equation and Synthea's implementation must also be distinguished. The 2021 AF implementation uses a reduced set of predictors. Its stroke implementation uses age-based probabilities below age 55. Generator rules can use internal state, whereas the reference baseline must use recorded pre-anchor data. Shared formulas therefore make strong performance plausible, but do not guarantee it.

For the comparison design, use these steps:

1. Fix each score version and its outcome mapping before computing validation metrics.
2. Define each score's eligible patient group from pre-anchor information.
3. Map required diagnoses and active medications to explicit code lists.
4. Select the most recent valid pre-anchor observation for each required measurement.
5. Record the elapsed time since each measurement.
6. Fit any missing-value imputation on training patients only.
7. Evaluate the clinical score, LightGBM, and transformer on the same patients within each comparison row.
8. Report the patient count and positive count beside AUROC and average precision (AP).
9. Report uncertainty using the same patient resamples for all methods in a row.
10. Keep a generator-rule comparison separate from the published clinical-score comparison.

Do not use absence of a diagnosis code as evidence of an absent clinical factor without documenting that assumption. Do not replace blood pressure treatment status with a hypertension diagnosis. If missing inputs lead to patient exclusion, also evaluate the learned models on that same subset. A check restricted to patients with all required inputs can reveal sensitivity to imputation.

The horizon statement needs qualification. AUROC and AP depend on rankings. A strictly increasing conversion of one score leaves those metrics unchanged, provided it introduces no ties. However, a model developed for 10-year outcomes may rank five-year events differently from a model developed for five-year outcomes. Evaluating a 10-year score against five-year labels is an informative comparison with a different prediction period. It is not a matched five-year risk validation. No conversion is needed for ranking metrics, and a 10-year probability must not be reported as a calibrated five-year probability.

ASCVD also has an outcome mismatch. Its combined outcome includes coronary death, nonfatal MI, and fatal or nonfatal stroke. The listed target codes do not identify that complete outcome. Separate MI and stroke rows should be labeled component-outcome comparisons. A union of available MI and stroke labels would still be a proxy, not the complete published ASCVD outcome. [ACC outcome definition](https://tools.acc.org/ascvd-risk-estimator/default.aspx).

The report should describe any result as performance against selected published reference scores on this synthetic dataset. It would not establish superiority over clinical care. The clinical reference table should identify the score version, target code, prediction period, patient eligibility, sample size, event count, input coverage, AUROC, AP, and uncertainty for each model.
