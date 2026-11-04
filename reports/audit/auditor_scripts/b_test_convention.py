"""Audit script B (A6/A7/B13): the released test tables against test_anchors.csv, and the
train/val truncation conventions against what the test files show. pandas only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path("<repo>")
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
OUT = Path(__file__).parent / "b_results.json"
TABLES = {
    "encounters": ("START", "STOP"),
    "conditions": ("START", "STOP"),
    "medications": ("START", "STOP"),
    "careplans": ("START", "STOP"),
    "allergies": ("START", "STOP"),
    "devices": ("START", "STOP"),
    "procedures": ("DATE", None),
    "immunizations": ("DATE", None),
    "observations": ("DATE", None),
    "imaging_studies": ("DATE", None),
}


def ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="ISO8601", utc=True).dt.tz_localize(None)


def read(split: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / split / f"{name}.csv", dtype=str, keep_default_na=False, na_values=[""])
    t, s = TABLES[name]
    df[t] = ts(df[t])
    if s:
        df[s] = ts(df[s])
    return df


cohort = pd.read_parquet(PROC / "cohort.parquet")
ta = pd.read_csv(RAW / "test_anchors.csv", dtype=str).set_index("Id")["anchor_date"]
ta = ts(ta)
res: dict = {"test_anchor_file": {"n": int(len(ta)), "all_midnight": bool((ta == ta.dt.normalize()).all()),
                                   "min": str(ta.min()), "max": str(ta.max())}}

# -------------------------------------------------------------- test tables vs anchors
test_rows = {}
test_any_on_anchor_day = set()
test_any_on_day_before = set()
for name, (t, s) in TABLES.items():
    df = read("test", name)
    a = df["PATIENT"].map(ta)
    rec = {
        "rows": int(len(df)),
        "patients_without_anchor": int(a.isna().sum()),
        f"{t}_on_or_after_anchor": int((df[t] >= a).sum()),
        f"{t}_on_anchor_day": int((df[t].dt.normalize() == a).sum()),
        f"{t}_on_day_before_anchor": int((df[t].dt.normalize() == a - pd.Timedelta(days=1)).sum()),
        f"max_{t}_minus_anchor_days": float(((df[t] - a).dt.total_seconds() / 86400).max()),
    }
    test_any_on_anchor_day |= set(df.loc[df[t].dt.normalize() == a, "PATIENT"])
    test_any_on_day_before |= set(df.loc[df[t].dt.normalize() == a - pd.Timedelta(days=1), "PATIENT"])
    if s:
        rec.update({
            "STOP_blank": int(df[s].isna().sum()),
            "STOP_blank_frac": round(float(df[s].isna().mean()), 4),
            "STOP_on_or_after_anchor": int((df[s] >= a).sum()),
            "STOP_eq_anchor_midnight": int((df[s] == a).sum()),
            "STOP_before_START": int((df[s] < df[t]).sum()),
            "max_STOP_minus_anchor_days": float(((df[s] - a).dt.total_seconds() / 86400).max()),
        })
    test_rows[name] = rec
res["test_tables"] = test_rows
res["test_patients_with_event_on_anchor_day"] = len(test_any_on_anchor_day)
res["test_patients_with_event_on_day_before_anchor"] = len(test_any_on_day_before)
print("TEST", json.dumps(test_rows, indent=1))

# -------------------------------------------------------------- train/val: what the raw data has at the anchor
tv = cohort[cohort["split"].isin(["train", "val"])]
anchor = tv["anchor_date"]
conv = {}
on_anchor_day = set()
day_before = set()
blank_strict = {}
blank_drop_spanning = {}
blank_le = {}
for name, (t, s) in TABLES.items():
    df = read("train_val", name)
    a = df["PATIENT"].map(anchor)
    on_anchor_day |= set(df.loc[df[t].dt.normalize() == a, "PATIENT"])
    day_before |= set(df.loc[df[t].dt.normalize() == a - pd.Timedelta(days=1), "PATIENT"])
    conv[name] = {
        "rows_raw": int(len(df)),
        f"rows_{t}_before_anchor": int((df[t] < a).sum()),
        f"rows_{t}_on_anchor_day": int((df[t].dt.normalize() == a).sum()),
    }
    if s:
        keep = df[t] < a
        kept = df.loc[keep]
        ak = a[keep]
        spanning = kept[s] >= ak
        # pipeline convention: keep row, blank STOP >= anchor
        blank_strict[name] = round(float((kept[s].isna() | spanning).mean()), 4)
        # alternative: drop spanning rows entirely
        nonspan = kept.loc[~spanning]
        blank_drop_spanning[name] = round(float(nonspan[s].isna().mean()), 4) if len(nonspan) else None
        # alternative: blank only STOP > anchor (keep STOP == anchor midnight)
        blank_le[name] = round(float((kept[s].isna() | (kept[s] > ak)).mean()), 4)
        conv[name].update({
            "rows_spanning_anchor(START<anchor<=STOP)": int(spanning.sum()),
            "rows_STOP_eq_anchor_midnight": int((kept[s] == ak).sum()),
            "rows_STOP_blank_raw": int(df[s].isna().sum()),
            "rows_STOP_before_START_raw": int((df[s] < df[t]).sum()),
        })
res["trainval_raw_at_anchor"] = conv
res["trainval_patients_with_event_on_anchor_day"] = {"n": len(on_anchor_day), "frac": round(len(on_anchor_day) / len(tv), 4)}
res["trainval_patients_with_event_on_day_before_anchor"] = {"n": len(day_before), "frac": round(len(day_before) / len(tv), 4)}
res["test_patients_frac_event_on_anchor_day"] = round(res["test_patients_with_event_on_anchor_day"] / len(ta), 4)
res["test_patients_frac_event_on_day_before"] = round(res["test_patients_with_event_on_day_before_anchor"] / len(ta), 4)
res["blank_stop_fraction"] = {
    "trainval_pipeline_rule(keep START<anchor, blank STOP>=anchor)": blank_strict,
    "trainval_if_spanning_rows_dropped": blank_drop_spanning,
    "trainval_if_only_STOP>anchor_blanked": blank_le,
    "test_released": {n: r["STOP_blank_frac"] for n, r in test_rows.items() if "STOP_blank_frac" in r},
}
# spanning encounters: expected number in test from the train/val rate
span_enc = conv["encounters"]["rows_spanning_anchor(START<anchor<=STOP)"]
res["spanning_encounters"] = {
    "trainval_rows": span_enc,
    "trainval_rate": round(span_enc / len(tv), 4),
    "expected_in_test": round(span_enc / len(tv) * len(ta), 1),
    "test_blank_STOP_encounters": test_rows["encounters"]["STOP_blank"],
}
# test medications: DISPENSES on spanning (blank-STOP) rows (are totals untruncated?)
med = pd.read_csv(RAW / "test" / "medications.csv", dtype=str, keep_default_na=False, na_values=[""])
med["START"] = ts(med["START"]); med["STOP"] = ts(med["STOP"])
a = med["PATIENT"].map(ta)
blank = med["STOP"].isna()
months_to_anchor = ((a - med["START"]).dt.days / 30.44)
disp = pd.to_numeric(med["DISPENSES"], errors="coerce")
res["test_medications_blank_stop_dispenses"] = {
    "rows_blank_STOP": int(blank.sum()),
    "rows_blank_STOP_with_DISPENSES_gt_months_to_anchor_plus1": int((blank & (disp > months_to_anchor + 1)).sum()),
    "example": med.loc[blank, ["START", "DISPENSES"]].assign(months_to_anchor=months_to_anchor[blank].round(1)).head(5).astype(str).to_dict("records"),
}
print(json.dumps({k: v for k, v in res.items() if k != "test_tables"}, indent=1, default=str))
OUT.write_text(json.dumps(res, indent=2, default=str))
print("wrote", OUT)
