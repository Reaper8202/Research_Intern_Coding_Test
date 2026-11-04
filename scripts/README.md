# Scripts

One-off data checks that produced evidence in the decision log, and the submission record. The
pipeline itself is the `make` targets over `src/ehrforecast/`.

| Script | Decision | What it does |
|---|---|---|
| `checks/invalid_stops.py` | D31 | finds interval rows whose `STOP` is before their own `START` |
| `checks/medication_stops.py` | D31 | shows that the stop, not the start, is the wrong field in those rows |
| `checks/race_check.py` | D15 | LightGBM with and without race and ethnicity (`reports/lightgbm_clean/race_check.json`) |
| `checks/clock_scan.py` | D36 | how well observation counts predict the anchor era, with and without the three clock codes (`reports/dataset/clock_scan.txt`) |
| `checks/modality_presence.py` | final audit | whether having a genotype or imaging file correlates with outcomes (`reports/dataset/modality_presence.json`) |
| `submission_record.py` | final audit | hashes of the deliverables at a commit (`make submission-record`, `reports/final/submission_record.json`) |
