"""Audit script: recompute pooled and mean-fold OOF metrics of the blend; verify nesting.

Read-only. No torch, no lightgbm imports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("<repo>")
D = ROOT / "data/processed"
cohort = pd.read_parquet(D / "cohort.parquet")
train_ids = cohort.index[cohort["split"] == "train"]
codes = pd.read_csv(D / "targets.csv", dtype=str, keep_default_na=False)["CODE"].tolist()
labels = pd.read_parquet(D / "labels.parquet").loc[train_ids, codes]
prior = pd.read_parquet(D / "prior_conditions.parquet").loc[train_ids, codes]
folds = pd.read_parquet(D / "folds.parquet").set_index("patient_id")["fold"].reindex(train_ids)
Y = labels.to_numpy(int)
PR = prior.to_numpy(bool)
F = folds.to_numpy()
print("train", len(train_ids), "folds", np.bincount(F).tolist(), "positives", int(Y.sum()))


def macro(y, p, eligible=None):
    aucs, aps = [], []
    for j in range(y.shape[1]):
        keep = slice(None) if eligible is None else eligible[:, j]
        yj, pj = y[keep, j], p[keep, j]
        if 0 < yj.sum() < len(yj):
            aucs.append(roc_auc_score(yj, pj))
        aps.append(average_precision_score(yj, pj) if yj.sum() else 0.0)
    return float(np.mean(aucs)), float(np.mean(aps)), len(aucs)


def mean_fold(y, p):
    a, m = [], []
    for k in range(5):
        s = F == k
        aa, mm, _ = macro(y[s], p[s])
        a.append(aa)
        m.append(mm)
    return float(np.mean(a)), float(np.mean(m)), [round(x, 4) for x in a]


stack = json.load(open(ROOT / "reports/stacking/clean/stacking.json"))
print("\n== committed OOF files vs my sklearn recomputation (prior cells forced to 0) ==")
oof = {}
for name in ("logit_mean", "prob_mean", "logistic"):
    f = pd.read_parquet(ROOT / f"reports/stacking/clean/oof_stack_{name}.parquet").loc[train_ids, codes]
    p = f.to_numpy(float)
    print(f"{name}: prior cells nonzero {int((p[PR]!=0).sum())}; NaN {int(np.isnan(p).sum())}")
    p = np.where(PR, 0.0, p)
    oof[name] = p
    a, m, n = macro(Y, p)
    mf = mean_fold(Y, p)
    ref = stack["stacked_pooled"][name]
    fs = stack["fold_scores"][name]
    print(f"  pooled sk {a:.4f}/{m:.4f} (file {ref['macro_auroc']:.4f}/{ref['mAP']:.4f}); mean fold sk {mf[0]:.4f}/{mf[1]:.4f} (file {np.mean([x['macro_auroc'] for x in fs]):.4f}/{np.mean([x['mAP'] for x in fs]):.4f}); folds {mf[2]}")

# base OOF inputs
print("\n== base OOF inputs ==")
bases = {}
parts = []
for cfg in ("small-v2-c1", "small-v2-c1-seed2", "small-v2-c1-seed3"):
    fr = pd.concat([pd.read_parquet(ROOT / f"models/transformer/{cfg}/fold{k}/F/held_predictions_best.parquet") for k in range(5)])
    parts.append(fr.loc[train_ids, codes].to_numpy(float))
bases["transformer"] = np.mean(parts, axis=0)
for name in ("counts", "counts_values"):
    bases[f"lightgbm_{name}"] = pd.read_parquet(ROOT / f"models/lightgbm_clean/oof_{name}.parquet").loc[train_ids, codes].to_numpy(float)
for name, p in bases.items():
    pm = np.where(PR, 0.0, p)
    a, m, _ = macro(Y, pm)
    ref = stack["base_pooled"][name]
    mf = mean_fold(Y, pm)
    print(f"{name}: pooled sk {a:.4f}/{m:.4f} (file {ref['macro_auroc']:.4f}/{ref['mAP']:.4f}); mean fold {mf[0]:.4f}/{mf[1]:.4f}; prior cells nonzero in raw file {int((p[PR]!=0).sum())}")

# verify logit_mean OOF == sigmoid(mean logits of the three bases) on at-risk cells
EPS = 1e-6
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
mine = sigmoid(np.mean([logit(b) for b in bases.values()], axis=0))
mine = np.where(PR, 0.0, mine)
print("\nmax |my logit-mean blend - committed oof_stack_logit_mean| :", float(np.abs(mine - oof["logit_mean"]).max()))

# verify the logistic stacker is nested: refit per fold on the other folds and compare
print("\n== nested logistic stacker check (refit on other folds' rows, at-risk only) ==")
Z = np.stack([logit(b) for b in bases.values()], axis=-1)  # n x 40 x 3
re = np.zeros_like(oof["logistic"])
for k in range(5):
    fit_rows = F != k
    app_rows = F == k
    for j in range(40):
        fr = fit_rows & ~PR[:, j]
        ar = app_rows & ~PR[:, j]
        y = Y[fr, j]
        if len(np.unique(y)) < 2:
            re[ar, j] = sigmoid(Z[ar, j, :].mean(axis=1))
            continue
        m = LogisticRegression(C=1.0, max_iter=1000).fit(Z[fr, j, :], y)
        re[ar, j] = m.predict_proba(Z[ar, j, :])[:, 1]
print("max |my nested logistic - committed oof_stack_logistic| :", float(np.abs(re - oof["logistic"]).max()))
# contrast: a NON-nested stacker fitted on all rows would score higher pooled
nn = np.zeros_like(re)
for j in range(40):
    rows = ~PR[:, j]
    m = LogisticRegression(C=1.0, max_iter=1000).fit(Z[rows, j, :], Y[rows, j])
    nn[rows, j] = m.predict_proba(Z[rows, j, :])[:, 1]
print("non-nested logistic (fit on all rows) pooled:", [round(x, 4) for x in macro(Y, nn)[:2]], "vs nested", [round(x, 4) for x in macro(Y, re)[:2]])

# rule S recompute
print("\n== rule S recompute from committed numbers ==")
sm = stack["paired_bca"]["stack_logit_mean_minus_transformer"]
print("S-T AUROC interval above 0:", sm["macro_auroc"]["ci_low"] > 0, (round(sm["macro_auroc"]["ci_low"], 4), round(sm["macro_auroc"]["ci_high"], 4)))
print("S-T mAP point >= 0:", sm["mAP"]["point"] >= 0, round(sm["mAP"]["point"], 4))
print("S pooled AUROC >= L1:", stack["stacked_pooled"]["logit_mean"]["macro_auroc"] >= stack["base_pooled"]["lightgbm_counts"]["macro_auroc"])
lm = np.mean([x["macro_auroc"] for x in stack["fold_scores"]["logit_mean"]])
lg = np.mean([x["macro_auroc"] for x in stack["fold_scores"]["logistic"]])
print(f"logistic - logit_mean mean fold AUROC = {lg-lm:+.4f} (>= 0.003? {lg-lm >= 0.003})")

# Platt: calibrated OOF
cal = json.load(open(ROOT / "reports/calibration/stack/calibration.json"))
calp = pd.read_parquet(ROOT / "reports/calibration/stack/oof_calibrated.parquet").loc[train_ids, codes].to_numpy(float)
calp = np.where(PR, 0.0, calp)
a, m, _ = macro(Y, calp)
print(f"\ncalibrated OOF pooled sk {a:.4f}/{m:.4f} (file {cal['pooled']['calibrated']['macro_auroc']:.4f}/{cal['pooled']['calibrated']['mAP']:.4f})")
# Brier per fold raw vs calibrated with my own computation (macro over targets of mean squared error, all patients)
def macro_brier(y, p):
    return float(np.mean([np.mean((p[:, j] - y[:, j]) ** 2) for j in range(y.shape[1])]))
raw = oof["logit_mean"]
rows = []
for k in range(5):
    s = F == k
    rows.append((k, macro_brier(Y[s], raw[s]), macro_brier(Y[s], calp[s]), macro(Y[s], raw[s])[0], macro(Y[s], calp[s])[0]))
for r in rows:
    print(f"fold {r[0]}: brier raw {r[1]:.5f} cal {r[2]:.5f} lower={r[2]<r[1]}; auroc raw {r[3]:.4f} cal {r[4]:.4f}")
print("folds with lower Brier:", sum(r[2] < r[1] for r in rows), "(file:", cal["folds_with_lower_brier"], ")")

# ECE recompute (equal-mass 10 bins, per target on at-risk cells, macro over targets)
def ece(y, p, n_bins=10):
    order = np.argsort(p, kind="stable")
    tot = 0.0
    for chunk in np.array_split(order, min(n_bins, len(order))):
        if len(chunk):
            tot += len(chunk) * abs(y[chunk].mean() - p[chunk].mean())
    return tot / len(y)
for name, p in (("raw", raw), ("calibrated", calp)):
    e = np.mean([ece(Y[~PR[:, j], j], p[~PR[:, j], j]) for j in range(40)])
    print(f"macro ECE {name}: {e:.4f} (file {cal['pooled'][name]['macro_ece']:.4f})")

# platt weights sanity: how many targets fitted vs identity, any slope <= 0
w = json.load(open(ROOT / "reports/calibration/stack/platt_weights.json"))
slopes = {c: v["a"] for c, v in w.items()}
print("platt: fitted", sum(v["fitted"] == 1.0 for v in w.values()), "identity", sum(v["fitted"] == 0.0 for v in w.values()), "min slope", round(min(slopes.values()), 3), "max slope", round(max(slopes.values()), 3))

# oracle
pt = {}
for name, p in bases.items():
    pm = np.where(PR, 0.0, p)
    pt[name] = [roc_auc_score(Y[:, j], pm[:, j]) for j in range(40)]
oracle = np.mean(np.max(np.array(list(pt.values())), axis=0))
print("oracle best base per target:", round(float(oracle), 4), "(file", round(stack["oracle_best_base_per_target_macro_auroc"], 4), ")")
