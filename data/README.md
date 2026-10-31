# data

Nothing in this folder is committed except this file.

| Path | Content | Created by |
|------|---------|------------|
| `raw/` | The task dataset, unchanged: `patient_splits.csv`, `target_conditions.csv`, `test_anchors.csv`, `train_val/`, `test/` (plus the optional `genomics/` and `imaging/`, which the pipeline does not use) | you, see below |
| `processed/` | Step 1 output: cohort, pre-anchor events with their manifest, labels, prior-condition flags (Parquet) | `make dataset` |
| `processed/folds.*` | Saved five-fold training assignments and their manifest | `make folds` |

Link or copy the dataset to `data/raw`, for example:

```bash
ln -s /path/to/research_intern_coding_test data/raw
```
