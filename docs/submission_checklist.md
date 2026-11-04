# Submission checklist

The one list of what remains before the four submission items leave this machine, with the
status observed by the final audit on 4 November 2026 (`reports/audit/final_audit.md`). Items
marked **author** need a decision or an account that only the author has.

| # | Item | Status on 4 November | What remains |
|---|---|---|---|
| 1 | Report author line (`docs/report.md`, `author:` in the front matter) | placeholder `[AUTHOR NAME]` | **author**: set the name the organisers expect |
| 2 | AI-workflow note on Codex CLI (`docs/report.md` section 3; `docs/ai_workflow.md` tools list) | bracketed note | **author**: state whether Codex CLI was used, or delete the line |
| 3 | Rebuild the PDF after 1 and 2: `cd docs && pandoc report.md -o report.pdf --pdf-engine=xelatex` | PDF matches the current `report.md` except the author line | run after 1 and 2 |
| 4 | Save the PDF as `<your_name>_results.pdf` (the name the brief asks for) | `docs/report.pdf` | **author**: copy under the final name; then `make submission-record` |
| 5 | GitHub repository public, with `final-deliverable` merged to `main` | private; branch pushed | **author**: merge, then set the repository to public and open the link logged out |
| 6 | Model on the Hugging Face Hub (`Reaper8202/ehr-forecast`) | not uploaded (`models/hf` built and verified locally, 125 files, 51 MB) | **author**: `hf auth login`, then `hf upload Reaper8202/ehr-forecast models/hf . --repo-type model`; open the page logged out |
| 7 | wandb project public (`derek-wu820-university-of-toronto/ehr-forecast`) | private (anonymous read returns no project) | **author**: set the project to public; open the link in the report logged out. The entity name carries the author's real name; keeping it is the author's call (the brief requires the link) |
| 8 | `predictions.csv` | 358 x 40, validated, regenerated from the package to 0.0 difference | nothing; send the committed file |
| 9 | `reports/final/submission_record.json` (hashes of the files sent, with the commit) | written for the current files | re-run `make submission-record` after 3 and 4 |

Not changed by the audit, on purpose: the LightGBM grid, which stopped at its simplest corner on
every axis (any extension is a model change that needs a new budget, a refreeze, and a disclosed
val attempt 3); the personal `WANDB_ENTITY` default in the Makefile; the `mypy` configuration,
which is not `--strict` (33 errors under `--strict`, none under the repository configuration).
