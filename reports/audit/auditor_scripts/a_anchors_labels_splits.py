"""Audit script A: splits (A3), anchors (A1), processed events vs anchor (A6), labels (A4),
prior mask (A5/B14), predictions.csv structure (A8), DEATHDATE presence (B1 context).
Reads raw CSVs and processed parquet directly; uses no ehrforecast code. pandas only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("<repo>")
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
OUT = Path(__file__).parent / "a_results.json"
res: dict = {}


def ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="ISO8601", utc=True).dt.tz_localize(None)


# ---------------------------------------------------------------- A3 splits
splits = pd.read_csv(RAW / "patient_splits.csv", dtype=str)
cohort = pd.read_parquet(PROC / "cohort.parquet")
tv_pat = pd.read_csv(RAW / "train_val" / "patients.csv", dtype=str)
te_pat = pd.read_csv(RAW / "test" / "patients.csv", dtype=str)
ta = pd.read_csv(RAW / "test_anchors.csv", dtype=str)
res["A3"] = {
    "patient_splits_counts": splits["split"].value_counts().to_dict(),
    "patient_splits_dup_ids": int(splits["Id"].duplicated().sum()),
    "cohort_counts": cohort["split"].value_counts().to_dict(),
    "cohort_ids_equal_split_ids": set(cohort.index) == set(splits["Id"]),
    "cohort_split_equal": bool(
        (cohort["split"].reindex(splits["Id"]).to_numpy() == splits["split"].to_numpy()).all()
    ),
    "train_val_patients_csv_eq_trainval_ids": set(tv_pat["Id"])
    == set(splits.loc[splits["split"].isin(["train", "val"]), "Id"]),
    "test_patients_csv_eq_test_ids": set(te_pat["Id"]) == set(splits.loc[splits["split"] == "test", "Id"]),
    "test_anchors_ids_eq_test_ids": set(ta["Id"]) == set(splits.loc[splits["split"] == "test", "Id"]),
    "cohort_columns": list(cohort.columns),
}
print("A3", json.dumps(res["A3"], indent=1, default=str))

# ---------------------------------------------------------------- A1 anchors
enc = pd.read_csv(RAW / "train_val" / "encounters.csv", dtype=str, keep_default_na=False, na_values=[""])
enc["START"] = ts(enc["START"])
enc["STOP"] = ts(enc["STOP"])
end = enc["STOP"].fillna(enc["START"])
last = end.groupby(enc["PATIENT"]).max()
anchor_cal = (last - pd.DateOffset(years=5)).dt.normalize()
anchor_1826 = (last - pd.Timedelta(days=1826)).dt.normalize()
anchor_1825 = (last - pd.Timedelta(days=1825)).dt.normalize()
anchor_36525 = (last - pd.Timedelta(days=365.25 * 5)).dt.normalize()
proc_anchor = cohort.loc[cohort["split"].isin(["train", "val"]), "anchor_date"]
common = proc_anchor.index.intersection(anchor_cal.index)
rng = np.random.default_rng(20261103)
sample = rng.choice(np.asarray(common), size=200, replace=False)


def mism(alt: pd.Series, idx) -> int:
    return int((alt.loc[idx] != proc_anchor.loc[idx]).sum())


res["A1"] = {
    "n_trainval_patients_with_encounters": int(len(last)),
    "n_trainval_in_cohort": int(len(proc_anchor)),
    "all_trainval_have_encounters": bool(set(proc_anchor.index) <= set(last.index)),
    "blank_STOP_encounters_trainval": int(enc["STOP"].isna().sum()),
    "last_is_blank_stop_START_for_n_patients": int(
        (last.loc[common] == enc.loc[enc["STOP"].isna()].groupby("PATIENT")["START"].max().reindex(common)).sum()
    ),
    "sample200_mismatch_calendar_years": mism(anchor_cal, sample),
    "all_mismatch_calendar_years": mism(anchor_cal, common),
    "all_mismatch_1826_days": mism(anchor_1826, common),
    "all_mismatch_1825_days": mism(anchor_1825, common),
    "all_mismatch_365.25x5_days": mism(anchor_36525, common),
    "last_encounter_on_feb29": int((last.dt.month.eq(2) & last.dt.day.eq(29)).sum()),
    "last_encounter_has_time_of_day": int((last != last.dt.normalize()).sum()),
    "processed_anchor_is_midnight": bool((proc_anchor == proc_anchor.dt.normalize()).all()),
    "window_end_eq_anchor_plus_5cal_years": bool(
        (cohort["window_end"] == cohort["anchor_date"] + pd.DateOffset(years=5)).all()
    ),
    "cohort_last_encounter_col_eq_recomputed": bool((cohort.loc[common, "last_encounter"] == last.loc[common]).all()),
}
# test anchors are taken from the file, not recomputed
test_anch = ts(ta.set_index("Id")["anchor_date"])
res["A2"] = {
    "test_anchor_in_cohort_eq_file": bool((cohort.loc[test_anch.index, "anchor_date"] == test_anch).all()),
    "test_last_encounter_is_null_in_cohort": bool(cohort.loc[test_anch.index, "last_encounter"].isna().all()),
}
print("A1", json.dumps(res["A1"], indent=1, default=str))
print("A2", json.dumps(res["A2"], indent=1, default=str))

# ---------------------------------------------------------------- A6: processed events before anchor
ev = pd.read_parquet(PROC / "events.parquet")
anchor_of = ev["patient_id"].map(cohort["anchor_date"])
birth_of = ev["patient_id"].map(cohort["birthdate"])
res["A6_processed_events"] = {
    "n_events": int(len(ev)),
    "events_on_or_after_anchor": int((ev["time"] >= anchor_of).sum()),
    "events_before_birth": int((ev["time"] < birth_of).sum()),
    "interval_start_on_or_after_anchor": int((ev["interval_start"] >= anchor_of).sum()),
    "max_days_before_anchor_min": float(((anchor_of - ev["time"]).dt.total_seconds() / 86400).min()),
    "patients_with_events_by_split": ev.groupby(ev["patient_id"].map(cohort["split"]), observed=True)["patient_id"].nunique().to_dict(),
    "encounter_stop_rows": int((ev["source"].astype(str).eq("encounters") & ev["kind"].astype(str).eq("stop")).sum()),
    "events_per_patient_median_by_split": {
        s: float(ev[ev["patient_id"].isin(cohort.index[cohort["split"] == s])].groupby("patient_id").size().median())
        for s in ("train", "val", "test")
    },
}
# any event exactly ON the anchor day (midnight) -> must be 0
res["A6_processed_events"]["events_on_anchor_day"] = int((ev["time"].dt.normalize() == anchor_of).sum())
print("A6", json.dumps(res["A6_processed_events"], indent=1, default=str))

# ---------------------------------------------------------------- A4 labels
cond = pd.read_csv(RAW / "train_val" / "conditions.csv", dtype=str, keep_default_na=False, na_values=[""])
cond["START"] = ts(cond["START"])
targets = pd.read_csv(RAW / "target_conditions.csv", dtype=str)
codes = targets["CODE"].tolist()
labels = pd.read_parquet(PROC / "labels.parquet")
prior = pd.read_parquet(PROC / "prior_conditions.parquet")
tv = cohort[cohort["split"].isin(["train", "val"])]
first = cond.groupby(["PATIENT", "CODE"])["START"].min().reset_index()
first = first[first["CODE"].isin(codes) & first["PATIENT"].isin(tv.index)]
a = first["PATIENT"].map(tv["anchor_date"])
w = first["PATIENT"].map(tv["window_end"])


def wide(mask: pd.Series) -> pd.DataFrame:
    hit = first.loc[mask]
    out = pd.DataFrame(0, index=tv.index, columns=codes, dtype="int8")
    for p, c in zip(hit["PATIENT"], hit["CODE"]):
        out.at[p, c] = 1
    return out


mine = wide((first["START"] >= a) & (first["START"] <= w))
lab = labels.loc[tv.index, codes]
alt_excl_start = wide((first["START"] > a) & (first["START"] <= w))
alt_excl_end = wide((first["START"] >= a) & (first["START"] < w))
# "any diagnosis in window" (not first-ever): a wrong-label construction
anyw = cond[cond["CODE"].isin(codes) & cond["PATIENT"].isin(tv.index)].copy()
anyw_a = anyw["PATIENT"].map(tv["anchor_date"])
anyw_w = anyw["PATIENT"].map(tv["window_end"])
anyw = anyw[(anyw["START"] >= anyw_a) & (anyw["START"] <= anyw_w)][["PATIENT", "CODE"]].drop_duplicates()
any_in_window = pd.DataFrame(0, index=tv.index, columns=codes, dtype="int8")
for p, c in zip(anyw["PATIENT"], anyw["CODE"]):
    any_in_window.at[p, c] = 1
# prior from raw: first START strictly before anchor
prior_mine = wide(first["START"] < a)
res["A4_A5"] = {
    "labels_shape": list(labels.shape),
    "labels_index_eq_trainval": set(labels.index) == set(tv.index),
    "labels_columns_eq_targets_order": list(labels.columns) == codes,
    "label_cells_differ_from_recompute_inclusive_both_ends": int((mine.to_numpy() != lab.to_numpy()).sum()),
    "label_positives_processed": int(lab.to_numpy().sum()),
    "label_positives_recomputed": int(mine.to_numpy().sum()),
    "cells_differ_if_start_exclusive": int((alt_excl_start.to_numpy() != lab.to_numpy()).sum()),
    "cells_differ_if_end_exclusive": int((alt_excl_end.to_numpy() != lab.to_numpy()).sum()),
    "cells_differ_if_any_diagnosis_in_window_not_first_ever": int((any_in_window.to_numpy() != lab.to_numpy()).sum()),
    "first_dx_exactly_on_anchor_day": int(((first["START"].dt.normalize() == a)).sum()),
    "first_dx_exactly_on_window_end_day": int(((first["START"].dt.normalize() == w)).sum()),
    "condition_START_has_time_of_day": int((cond["START"] != cond["START"].dt.normalize()).sum()),
    "prior_cells_differ_from_recompute": int((prior_mine.to_numpy() != prior.loc[tv.index, codes].to_numpy()).sum()),
    "prior_rows_all_splits": int(len(prior)),
    "label_and_prior_overlap": int((lab.to_numpy() & prior.loc[tv.index, codes].to_numpy()).sum()),
    "prior_rate_by_split": {s: float(prior.loc[cohort.index[cohort["split"] == s]].to_numpy().mean()) for s in ("train", "val", "test")},
    "targets_without_positives_train": [c for c in codes if lab.loc[cohort.index[cohort["split"] == "train"], c].sum() == 0],
    "targets_without_positives_val": [c for c in codes if lab.loc[cohort.index[cohort["split"] == "val"], c].sum() == 0],
}
print("A4/A5", json.dumps(res["A4_A5"], indent=1, default=str))

# ---------------------------------------------------------------- A8 predictions.csv
pred = pd.read_csv(REPO / "predictions.csv", dtype={"patient_id": str})
test_ids = splits.loc[splits["split"] == "test", "Id"]
vals = pred.drop(columns="patient_id")
numeric = vals.apply(pd.to_numeric, errors="coerce")
res["A8"] = {
    "n_rows": int(len(pred)),
    "n_test_patients": int(len(test_ids)),
    "first_col_is_patient_id": pred.columns[0] == "patient_id",
    "columns_eq_target_codes_in_order": list(pred.columns[1:]) == codes,
    "ids_eq_test_split": set(pred["patient_id"]) == set(test_ids),
    "duplicate_ids": int(pred["patient_id"].duplicated().sum()),
    "non_numeric_cells": int(numeric.isna().sum().sum() - vals.isna().sum().sum()),
    "missing_cells": int(vals.isna().sum().sum()),
    "min": float(numeric.min().min()),
    "max": float(numeric.max().max()),
    "cells_exactly_zero": int((numeric == 0).sum().sum()),
    "prior_true_cells_test": int(prior.loc[test_ids, codes].to_numpy().sum()),
    "zero_cells_eq_prior_cells": bool(
        np.array_equal((numeric.set_index(pred["patient_id"]).loc[test_ids, codes].to_numpy() == 0), prior.loc[test_ids, codes].to_numpy() == 1)
    ),
}
print("A8", json.dumps(res["A8"], indent=1, default=str))

# ---------------------------------------------------------------- B1 context: DEATHDATE in raw
pat_tv = pd.read_csv(RAW / "train_val" / "patients.csv", dtype=str, keep_default_na=False, na_values=[""])
pat_te = pd.read_csv(RAW / "test" / "patients.csv", dtype=str, keep_default_na=False, na_values=[""])
d_tv = ts(pat_tv.set_index("Id")["DEATHDATE"])
d_te = ts(pat_te.set_index("Id")["DEATHDATE"])
res["B1_context"] = {
    "raw_patient_columns": list(pat_tv.columns),
    "trainval_deathdate_nonblank": int(d_tv.notna().sum()),
    "trainval_death_after_anchor": int((d_tv > cohort.loc[d_tv.index, "anchor_date"]).sum()),
    "trainval_death_eq_last_encounter_day": int((d_tv.dt.normalize() == cohort.loc[d_tv.index, "last_encounter"].dt.normalize()).sum()),
    "test_deathdate_nonblank": int(d_te.notna().sum()),
    "test_death_after_anchor": int((d_te > cohort.loc[d_te.index, "anchor_date"]).sum()),
    "raw_medication_columns": pd.read_csv(RAW / "train_val" / "medications.csv", nrows=0).columns.tolist(),
    "raw_encounter_columns": pd.read_csv(RAW / "train_val" / "encounters.csv", nrows=0).columns.tolist(),
}
print("B1ctx", json.dumps(res["B1_context"], indent=1, default=str))
OUT.write_text(json.dumps(res, indent=2, default=str))
print("wrote", OUT)
