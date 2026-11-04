# Documentation

Start with the report. The decision log is the evidence behind it.

| File | What it is |
|---|---|
| `report.md`, `report.pdf` | the technical report: representation, model and training, AI workflow, interpretation |
| `model_results.md` | every result with its interval: training CV, both official val runs, per-target and subgroup tables |
| `decisions.md` | the decision log (D1 to D8, D31 to D38): each decision, the reason, and the check behind it |
| `model_card.md` | the model card shipped with the Hugging Face package (`make package` copies it there) |
| `baseline_results.md` | the step-2 reference models (logistic regression, age-sex incidence tables) and their leakage review |
| `audit_criteria.md` | the 74 audit criteria written before the final audit; the verdicts are in `../reports/audit/final_audit.md` |
| `submission_checklist.md` | what remained before submission, with the status of each item |
| `figures/` | the report figures and the script that draws them from the committed reports |
| `planning/` | the plans as written before the code: `plan.md` (steps and risks), `model_plan.md` (the transformer design and evaluation protocol, revisions 1 to 2.5) |

Suggested order for a reviewer: `report.pdf`, then `decisions.md`, then `model_results.md`.
The planning documents are kept as they were written. Where they and the decision log differ,
the decision log is what was done.
