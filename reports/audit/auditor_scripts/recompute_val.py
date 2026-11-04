"""Audit script: recompute official-val metrics with scikit-learn and reproduce the bootstrap.

Read-only. Uses only data/processed, reports/val, reports/lightgbm_final, reports/baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("<repo>")
D = ROOT / "data/processed"

cohort = pd.read_parquet(D / "cohort.parquet")
val_ids = cohort.index[cohort["split"] == "val"]
codes = pd.read_csv(D / "targets.csv", dtype=str, keep_default_na=False)["CODE"].tolist()
labels = pd.read_parquet(D / "labels.parquet").loc[val_ids, codes]
prior = pd.read_parquet(D / "prior_conditions.parquet").loc[val_ids, codes]
print(f"val patients {len(val_ids)}, targets {len(codes)}, positive labels {int(labels.to_numpy().sum())}")
print(f"targets with zero val positives: {int((labels.sum(axis=0) == 0).sum())}")
print(f"min val positives per target: {int(labels.sum(axis=0).min())}; targets with <=8 positives: {int((labels.sum(axis=0) <= 8).sum())}")
print(f"label/prior overlap cells: {int(np.logical_and(labels.to_numpy(), prior.to_numpy()).sum())}")


def read_pred(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id")
    f.columns = [str(c) for c in f.columns]
    return f.loc[val_ids, codes]


files = {
    "final": ROOT / "reports/val/val_predictions_final.csv",
    "transformer": ROOT / "reports/val/val_predictions_transformer.csv",
    "lightgbm_counts": ROOT / "reports/lightgbm_final/val_predictions_lightgbm_counts.csv",
    "lightgbm_counts_values": ROOT / "reports/lightgbm_final/val_predictions_lightgbm_counts_values.csv",
    "logreg_codes": ROOT / "reports/baseline/val_predictions_logreg_codes.csv",
    "incidence_at_risk": ROOT / "reports/baseline/val_predictions_incidence_age_sex_at_risk.csv",
    "attempt1_final": ROOT / "reports/val/attempts/168b78f5b5d8/val_predictions_final.csv",
}
preds = {k: read_pred(v) for k, v in files.items()}

Y = labels.to_numpy(dtype=int)
PR = prior.to_numpy(dtype=bool)


def macro(y: np.ndarray, p: np.ndarray, eligible: np.ndarray | None = None):
    aucs, aps = [], []
    for j in range(y.shape[1]):
        keep = slice(None) if eligible is None else eligible[:, j]
        yj, pj = y[keep, j], p[keep, j]
        if 0 < yj.sum() < len(yj):
            aucs.append(roc_auc_score(yj, pj))
        aps.append(average_precision_score(yj, pj) if yj.sum() else 0.0)
    return float(np.mean(aucs)), float(np.mean(aps)), len(aucs)


summary = json.load(open(ROOT / "reports/val/val_summary.json"))
print("\n== masked (prior cells set to 0), all patients ==")
print(f"{'model':24s} {'sk_auroc':>9s} {'sk_mAP':>8s} | {'file_auroc':>10s} {'file_mAP':>9s} | prior cells nonzero")
for name, f in preds.items():
    p = f.to_numpy(dtype=float)
    nonzero_prior = int((p[PR] != 0).sum())
    pm = np.where(PR, 0.0, p)
    a, m, n = macro(Y, pm)
    if name == "final":
        ref = summary["final"]["masked"]
    elif name == "attempt1_final":
        ref = json.load(open(ROOT / "reports/val/attempts/168b78f5b5d8/val_summary.json"))["final"]["masked"]
    else:
        ref = summary["comparison_models"][name]["masked"]
    print(f"{name:24s} {a:9.5f} {m:8.5f} | {ref['macro_auroc']:10.5f} {ref['mAP']:9.5f} | {nonzero_prior} (n_auc={n})")

print("\n== at-risk (prior cells excluded) ==")
for name, f in preds.items():
    p = np.where(PR, 0.0, f.to_numpy(dtype=float))
    a, m, n = macro(Y, p, eligible=~PR)
    ref = summary["final"]["at_risk"] if name == "final" else summary["comparison_models"].get(name, {}).get("at_risk")
    print(f"{name:24s} {a:9.5f} {m:8.5f} | file {ref['macro_auroc']:.5f} {ref['mAP']:.5f}" if ref else f"{name:24s} {a:9.5f} {m:8.5f}")

print("\n== effect of the prior-cell convention on the blend (val) ==")
p = preds["final"].to_numpy(dtype=float)
a_masked, m_masked, _ = macro(Y, np.where(PR, 0.0, p))
a_atrisk, m_atrisk, _ = macro(Y, p, eligible=~PR)
# 'unmasked' alternative: fill prior cells with the model's mean at-risk probability per target (a neutral value)
fill = np.array([p[~PR[:, j], j].mean() for j in range(len(codes))])
p_fill = np.where(PR, fill[None, :], p)
a_fill, m_fill, _ = macro(Y, p_fill)
print(f"masked zeros: {a_masked:.4f}/{m_masked:.4f}; excluded (at-risk): {a_atrisk:.4f}/{m_atrisk:.4f}; prior cells filled with target mean: {a_fill:.4f}/{m_fill:.4f}")
print(f"prior cells in val: {int(PR.sum())} of {PR.size} ({PR.mean():.3%})")

print("\n== attempt-1 final == attempt-2 transformer? ==")
print("max abs diff:", float(np.abs(preds["attempt1_final"].to_numpy() - preds["transformer"].to_numpy()).max()))

# ---------------------------------------------------------------- bootstrap
print("\n== my own patient-level bootstrap (percentile and BCa, n_boot=1000, seed=0) ==")
pf = np.where(PR, 0.0, preds["final"].to_numpy(dtype=float))
pt = np.where(PR, 0.0, preds["transformer"].to_numpy(dtype=float))
pl = np.where(PR, 0.0, preds["lightgbm_counts"].to_numpy(dtype=float))


def stat_single(idx: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, m, _ = macro(Y[idx], p[idx])
    return np.array([a, m])


def stat_pair(idx: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return stat_single(idx, p) - stat_single(idx, q)


n = len(Y)
rng = np.random.default_rng(0)
t0 = time.time()
boots_single, boots_pair_t, boots_pair_l = [], [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    boots_single.append(stat_single(idx, pf))
    boots_pair_t.append(stat_pair(idx, pf, pt))
    boots_pair_l.append(stat_pair(idx, pf, pl))
bs = np.array(boots_single)
bpt = np.array(boots_pair_t)
bpl = np.array(boots_pair_l)
print(f"({time.time()-t0:.0f}s)")
point = stat_single(np.arange(n), pf)
for i, name in enumerate(("macro_auroc", "mAP")):
    lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
    print(f"blend {name}: point {point[i]:.4f}, percentile CI [{lo:.4f}, {hi:.4f}], bootstrap mean {bs[:, i].mean():.4f} (bias {bs[:, i].mean()-point[i]:+.4f}); committed BCa [{summary['final_bca'][name]['ci_low']:.4f}, {summary['final_bca'][name]['ci_high']:.4f}]")
pd_t = stat_pair(np.arange(n), pf, pt)
pd_l = stat_pair(np.arange(n), pf, pl)
for i, name in enumerate(("macro_auroc", "mAP")):
    lo, hi = np.percentile(bpt[:, i], [2.5, 97.5])
    ref = summary["paired_bca"]["final_minus_transformer"][name]
    print(f"blend-transformer {name}: point {pd_t[i]:+.4f}, percentile [{lo:+.4f}, {hi:+.4f}]; committed BCa {ref['point']:+.4f} [{ref['ci_low']:+.4f}, {ref['ci_high']:+.4f}]")
    lo, hi = np.percentile(bpl[:, i], [2.5, 97.5])
    ref = summary["paired_bca"]["final_minus_lightgbm_counts"][name]
    print(f"blend-lightgbm_counts {name}: point {pd_l[i]:+.4f}, percentile [{lo:+.4f}, {hi:+.4f}]; committed BCa {ref['point']:+.4f} [{ref['ci_low']:+.4f}, {ref['ci_high']:+.4f}]")

# scipy BCa with my own statistic (macro AUROC and mAP only), seed 0
print("\n== scipy BCa with my own statistic (auroc, mAP), 1000 resamples, seed 0 ==")
t0 = time.time()
res = stats.bootstrap((np.arange(n),), lambda idx: stat_single(np.asarray(idx), pf), vectorized=False, n_resamples=1000, method="BCa", rng=np.random.default_rng(0))
print(f"({time.time()-t0:.0f}s) blend BCa auroc [{res.confidence_interval.low[0]:.4f}, {res.confidence_interval.high[0]:.4f}], mAP [{res.confidence_interval.low[1]:.4f}, {res.confidence_interval.high[1]:.4f}]")
res = stats.bootstrap((np.arange(n),), lambda idx: stat_pair(np.asarray(idx), pf, pt), vectorized=False, n_resamples=1000, method="BCa", rng=np.random.default_rng(0))
print(f"blend-transformer BCa auroc [{res.confidence_interval.low[0]:+.4f}, {res.confidence_interval.high[0]:+.4f}], mAP [{res.confidence_interval.low[1]:+.4f}, {res.confidence_interval.high[1]:+.4f}]")

# ---------------------------------------------------------------- subgroups
print("\n== subgroups (val) ==")
anchor_year = cohort.loc[val_ids, "anchor_date"].dt.year.to_numpy()
for name, mask in (("anchor_before_2015", anchor_year < 2015), ("anchor_from_2015", anchor_year >= 2015)):
    out = []
    for model in ("final", "transformer", "lightgbm_counts"):
        p = np.where(PR, 0.0, preds[model].to_numpy(dtype=float))[mask]
        a, m, k = macro(Y[mask], p)
        ref = summary["subgroups"][name][model]
        out.append(f"{model} {a:.3f}/{m:.3f} (file {ref['macro_auroc']:.3f}/{ref['mAP']:.3f}, n_auc {ref['n_targets_auroc']})")
    print(f"{name} n={int(mask.sum())} (file {summary['subgroups'][name]['n_patients']}): " + "; ".join(out))
mod = summary["subgroups"].get("modality_input", {})
print("modality_input n (file):", mod.get("n_patients"), {k: (round(v["macro_auroc"], 3), round(v["mAP"], 3)) for k, v in mod.items() if isinstance(v, dict)})

# ---------------------------------------------------------------- per target
print("\n== per-target checks (val_per_target.csv) ==")
pt_tab = pd.read_csv(ROOT / "reports/val/val_per_target.csv", dtype={"code": str})
piv = pt_tab.pivot(index="code", columns="model", values="auroc")
desc = pd.read_csv(D / "targets.csv", dtype=str, keep_default_na=False).set_index("CODE")["DESCRIPTION"]
piv["desc"] = desc.reindex(piv.index)
npos = pt_tab[pt_tab["model"] == "final"].set_index("code")["n_pos"]
piv["n_pos"] = npos.reindex(piv.index)
print("blend > transformer on", int((piv["final"] > piv["transformer"]).sum()), "of 40; blend > lightgbm_counts on", int((piv["final"] > piv["lightgbm_counts"]).sum()))
models = ["final", "transformer", "lightgbm_counts", "lightgbm_counts_values", "logreg_codes", "incidence_at_risk"]
below = piv[(piv[models] < 0.6).all(axis=1)]
print("below 0.6 for every model:", len(below), list(below["desc"]))
above = piv[(piv[models] > 0.95).all(axis=1)]
print("above 0.95 for every model:", len(above), list(above["desc"]))
print("targets with n_pos <= 8:", int((piv["n_pos"] <= 8).sum()), "; 5..20:", int(((piv["n_pos"] >= 5) & (piv["n_pos"] <= 20)).sum()))
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
show = piv[["desc", "n_pos", "final", "transformer", "lightgbm_counts"]].round(3)
keys = ["myocardial", "osteopor", "rectal polyp", "pathological fracture", "polyp of colon", "micro", "gait", "heart failure", "stroke", "lung", "bronch", "concussion"]
print(show[show["desc"].str.lower().str.contains("|".join(keys))].sort_values("desc").to_string())

# recompute my own per-target AUROC for the blend and compare with the file
mine = {}
pf_full = np.where(PR, 0.0, preds["final"].to_numpy(dtype=float))
for j, c in enumerate(codes):
    mine[c] = roc_auc_score(Y[:, j], pf_full[:, j])
diff = (pd.Series(mine) - piv["final"].reindex(codes)).abs().max()
print("max |my per-target AUROC - file| for blend:", float(diff))
