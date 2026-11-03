"""Count STOP < START rows in every interval table, every split (read-only audit)."""
import sys
import pandas as pd
from ehrforecast.io import read_event_table
from ehrforecast.schema import EVENT_TABLES
from ehrforecast.cohort import load_splits, load_test_anchors
from pathlib import Path

raw = Path("data/raw")
splits = load_splits(raw / "patient_splits.csv")
cohort = pd.read_parquet("data/processed/cohort.parquet", columns=["split", "anchor_date"])
for d in ("train_val", "test"):
    for name, spec in EVENT_TABLES.items():
        if spec.stop_col is None:
            continue
        t = read_event_table(raw / d, spec)
        bad = t[t[spec.stop_col].notna() & (t[spec.stop_col] < t[spec.time_col])].copy()
        if bad.empty:
            print(d, name, 0)
            continue
        bad["split"] = bad["PATIENT"].map(cohort["split"])
        bad["anchor"] = bad["PATIENT"].map(cohort["anchor_date"])
        bad["start_pre"] = bad[spec.time_col] < bad["anchor"]
        bad["stop_pre"] = bad[spec.stop_col] < bad["anchor"]
        bad["gap_days"] = (bad[spec.time_col] - bad[spec.stop_col]).dt.total_seconds() / 86400
        print(d, name, len(bad), "patients", bad["PATIENT"].nunique())
        print(bad.groupby(["split", "start_pre", "stop_pre"]).size().to_string())
        print("gap days", bad["gap_days"].describe().round(2).to_dict())
        if name == "medications":
            # Same patient+code rows: do valid stops exist for the same drug?
            same = t.merge(bad[["PATIENT", "CODE"]].drop_duplicates(), on=["PATIENT", "CODE"])
            print("rows sharing patient+code with an invalid row:", len(same))
            print(bad[["split", "START", "STOP", "CODE", "DESCRIPTION"]].head(8).to_string())
