"""Audit script D (B10, B12): rebuild predictions.csv from the frozen base files through
submission.build (label-free path) and compare; modality-file presence by split and vs labels.
No torch in this process."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("<repo>")
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
OUT = Path(__file__).parent / "d_results.json"
sys.path.insert(0, str(REPO / "src"))
import tomllib  # noqa: E402
from ehrforecast.contracts import load_prediction_inputs, load_splits  # noqa: E402
from ehrforecast.submission import build, read_predictions  # noqa: E402

assert "torch" not in sys.modules
res: dict = {}
final = tomllib.loads((REPO / "configs" / "final.toml").read_text())
splits = load_splits(PROC)
test = pd.Index(splits.index[splits.eq("test")], name="patient_id")
inputs = load_prediction_inputs(PROC, test)
bases = {"transformer": read_predictions(REPO / "reports/final/test_predictions_transformer.csv", inputs.cohort.index, inputs.codes)}
for name in final["stack"]["bases"]:
    if str(name).startswith("lightgbm_"):
        bases[str(name)] = read_predictions(REPO / "reports/lightgbm_final" / f"test_predictions_{name}.csv", inputs.cohort.index, inputs.codes)
rebuilt = build(final["stack"], bases, inputs.prior, inputs.codes, REPO)
submitted = read_predictions(REPO / "predictions.csv", inputs.cohort.index, inputs.codes)
res["B10_rebuild"] = {
    "max_abs_diff_rebuilt_vs_predictions_csv": float(np.abs(rebuilt.to_numpy() - submitted.to_numpy()).max()),
    "modules_loaded_contain_torch": "torch" in sys.modules,
    "modules_loaded_contain_lightgbm": "lightgbm" in sys.modules,
    "prior_cells_test": int(inputs.prior.to_numpy().sum()),
}
print("B10", json.dumps(res["B10_rebuild"], indent=1))

# ------------------------------------------------------------------ B12 presence
cohort = pd.read_parquet(PROC / "cohort.parquet")
labels = pd.read_parquet(PROC / "labels.parquet")
codes = inputs.codes
gen_ids = {p.name.removesuffix("_dna.csv") for p in (RAW / "genomics").glob("*_dna.csv")}
img_ids = {p.stem for p in (RAW / "imaging").glob("*.dcm")}
cohort["has_genomics"] = cohort.index.isin(gen_ids)
cohort["has_imaging"] = cohort.index.isin(img_ids)
pres = {}
for mod in ("has_genomics", "has_imaging"):
    tab = pd.crosstab(cohort["split"], cohort[mod])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    pres[mod] = {"by_split": {s: {"n": int(tab.loc[s].sum()), "present": int(tab.loc[s].get(True, 0)),
                                   "frac": round(float(tab.loc[s].get(True, 0) / tab.loc[s].sum()), 4)} for s in tab.index},
                 "chi2_split_vs_presence_p": float(p)}
    # outcome association on train+val (labelled patients)
    tv = cohort[cohort["split"].isin(["train", "val"])]
    y = labels.loc[tv.index, codes]
    npos = y.sum(axis=1)
    with_, without = npos[tv[mod]], npos[~tv[mod]]
    pres[mod]["trainval_positive_labels_per_patient_mean"] = {"present": float(with_.mean()), "absent": float(without.mean()),
                                                              "mannwhitney_p": float(stats.mannwhitneyu(with_, without).pvalue) if len(with_) else None}
    pres[mod]["trainval_age_at_anchor_mean"] = {"present": float(tv.loc[tv[mod], "age_at_anchor"].mean()), "absent": float(tv.loc[~tv[mod], "age_at_anchor"].mean())}
    pres[mod]["trainval_female_frac"] = {"present": float((tv.loc[tv[mod], "gender"] == "F").mean()), "absent": float((tv.loc[~tv[mod], "gender"] == "F").mean())}
    # per-target: targets whose prevalence differs most (Fisher) between present/absent, train only
    tr = cohort[cohort["split"] == "train"]
    ytr = labels.loc[tr.index, codes]
    rows = []
    for c in codes:
        a = int(ytr.loc[tr[mod], c].sum()); b = int(tr[mod].sum()) - a
        cc = int(ytr.loc[~tr[mod], c].sum()); d = int((~tr[mod]).sum()) - cc
        pv = stats.fisher_exact([[a, b], [cc, d]])[1]
        rows.append({"code": c, "prev_present": round(a / max(1, a + b), 4), "prev_absent": round(cc / max(1, cc + d), 4), "fisher_p": pv})
    rows = sorted(rows, key=lambda r: r["fisher_p"])
    pres[mod]["train_per_target_most_associated"] = rows[:6]
    pres[mod]["train_targets_fisher_p_lt_0.01"] = int(sum(r["fisher_p"] < 0.01 for r in rows))
    pres[mod]["train_targets_fisher_p_lt_0.05"] = int(sum(r["fisher_p"] < 0.05 for r in rows))
# imaging: scan date vs anchor (only pre-anchor scans enter) - use processed cache
with np.load(PROC / "imaging.npz") as d:
    scans = pd.Series(pd.to_datetime(d["study_dates"]), index=pd.Index(d["patients"].astype(str)))
scan_before = scans.lt(cohort.loc[scans.index, "anchor_date"])
pres["imaging_prescan_by_split"] = {s: int(scan_before[cohort.loc[scans.index, "split"].eq(s).to_numpy()].sum()) for s in ("train", "val", "test")}
pres["imaging_scan_year_range"] = [str(scans.min().date()), str(scans.max().date())]
# genomics presence among died patients (DEATHDATE) - is genomics tied to death/era?
pat = pd.read_csv(RAW / "train_val" / "patients.csv", dtype=str, keep_default_na=False, na_values=[""]).set_index("Id")
died = pat["DEATHDATE"].notna().reindex(cohort.index).fillna(False)
tv = cohort[cohort["split"].isin(["train", "val"])]
pres["trainval_died_frac"] = {"genomics_present": float(died[tv.index][tv["has_genomics"]].mean()), "genomics_absent": float(died[tv.index][~tv["has_genomics"]].mean()),
                              "imaging_present": float(died[tv.index][tv["has_imaging"]].mean()), "imaging_absent": float(died[tv.index][~tv["has_imaging"]].mean())}
pres["trainval_anchor_year_mean"] = {"genomics_present": float(tv.loc[tv["has_genomics"], "anchor_date"].dt.year.mean()), "genomics_absent": float(tv.loc[~tv["has_genomics"], "anchor_date"].dt.year.mean())}
res["B12"] = pres
print("B12", json.dumps(pres, indent=1, default=str))
# val summary subgroup for modality input
vs = json.loads((REPO / "reports/val/val_summary.json").read_text())
res["val_subgroups"] = {k: {"n": v.get("n_patients"), "final_auroc": (v.get("final") or {}).get("masked", {}).get("macro_auroc") if isinstance(v.get("final"), dict) else None} for k, v in vs["subgroups"].items()}
res["val_headline"] = {"auroc": vs["final"]["masked"]["macro_auroc"], "mAP": vs["final"]["masked"]["mAP"], "n": vs["n_patients"], "submission": vs["submission"]}
print(json.dumps({"val_subgroups": res["val_subgroups"], "val_headline": res["val_headline"]}, indent=1))
OUT.write_text(json.dumps(res, indent=2, default=str))
print("wrote", OUT)
