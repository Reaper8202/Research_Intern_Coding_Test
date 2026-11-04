"""Audit script: recompute the tuning-round adoption rules from the per-fold run records.

Reads models/transformer/<config>/fold<k>/F/run.json (local, not committed) and compares with
the committed experiments CSVs and docs/decisions.md tables. Read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("<repo>")
M = ROOT / "models/transformer"
MARGIN = 0.005


def scores(cfg: str) -> pd.DataFrame:
    rows = []
    for k in range(5):
        p = M / cfg / f"fold{k}" / "F" / "run.json"
        if p.is_file():
            d = json.loads(p.read_text())
            if d.get("status") == "completed":
                rows.append({"fold": k, "auroc": d["best_held_macro_auroc"], "mAP": d["best_held_mAP"], "best_epoch": d["best_epoch"]})
    return pd.DataFrame(rows).set_index("fold")


def rule(cand: pd.DataFrame, ref: pd.DataFrame, label: str, component: bool = False) -> None:
    f = cand.index.intersection(ref.index)
    gain = cand.loc[f, "auroc"] - ref.loc[f, "auroc"]
    map_loss = ref.loc[f, "mAP"] - cand.loc[f, "mAP"]
    higher = int((gain > 0).sum())
    if len(f) == 5:
        if component:
            ok = gain.mean() > 0 and higher >= 4 and map_loss.mean() <= MARGIN
            kind = "component"
        else:
            ok = gain.mean() >= MARGIN and higher >= 4 and map_loss.mean() <= MARGIN
            kind = "5.3 rule"
    else:
        ok = gain.mean() >= MARGIN and map_loss.mean() <= MARGIN
        kind = f"screen(folds {list(f)})"
    print(f"{label:38s} mean AUROC {cand.loc[f,'auroc'].mean():.4f} mAP {cand.loc[f,'mAP'].mean():.4f} | gain {gain.mean():+.4f} folds_higher {higher} mAP_loss {map_loss.mean():+.4f} -> {kind} {'PASS' if ok else 'fail'}")


print("== Round 1 (budget.toml): incumbent small-v1-mm ==")
inc = scores("small-v1-mm")
print("incumbent mean", round(inc.auroc.mean(), 4), round(inc.mAP.mean(), 4))
for cfg, label in [("small-v1-mm-do03", "T1 dropout 0.3"), ("small-v1-mm-lr3e5", "T2 body lr"), ("small-v1-mm-wd01", "T3 wd 0.1"), ("small-v1-mm-reg", "T4 regularized"), ("small-v1-mm-agesex", "R1 age-sex head"), ("small-v1-mm-agesex-do03", "R2 age-sex + do03"), ("small-v1-mm-pt60", "P1 pretrain 60"), ("small-v1-mm-ctx512", "E5 ctx512"), ("small-v1-mm-ctx2048", "E5 ctx2048")]:
    s = scores(cfg)
    if len(s):
        rule(s, inc, label)
seeds = [inc, scores("small-v1-mm-seed2"), scores("small-v1-mm-seed3")]
seed_mean = pd.concat(seeds).groupby(level=0).mean()
print("seed means:", [round(s.auroc.mean(), 4) for s in seeds], "per-fold range", [round(x, 4) for x in (pd.concat([s.auroc for s in seeds], axis=1).max(axis=1) - pd.concat([s.auroc for s in seeds], axis=1).min(axis=1))])
rule(scores("small-v1-mm-do03"), seed_mean, "T1 vs seed mean")
rule(scores("small-v1-mm-agesex-do03"), seed_mean, "R2 vs seed mean")
t1, r2 = scores("small-v1-mm-do03"), scores("small-v1-mm-agesex-do03")
print(f"T1 vs R2 within 0.005 on both? AUROC diff {r2.auroc.mean()-t1.auroc.mean():+.4f}, mAP diff {r2.mAP.mean()-t1.mAP.mean():+.4f}")
for cfg in ("small-v1-mm-do03", "small-v1-mm-agesex-do03"):
    d = json.loads((M / cfg / "fold0/F/run.json").read_text())
    print(cfg, "trainable params", d["trainable_parameters"], "total", d["parameters"])

print("\n== Round 2 (budget_v2.toml): reference A1 = small-v2-do03 ==")
a1 = scores("small-v2-do03")
d35 = scores("small-v1-mm-do03")
rule(scores("small-v2-base"), inc, "A0 clean base vs small-v1-mm")
rule(a1, d35, "A1 clean do03 vs D35 model")
for cfg, label in [("small-v2-do05", "A2 dropout 0.5"), ("small-v2-pool", "A3 pooled readout"), ("small-v2-xa", "X1 extra anchors"), ("small-v2-ftl", "P2b full-timeline"), ("small-v2-count", "F1 count branch"), ("small-v2-tw", "F2 target weights"), ("small-v2-ptw", "P3b weighted parent"), ("small-v2-c1", "C1 combination")]:
    rule(scores(cfg), a1, label)
rule(scores("small-v2-ftl-pre"), scores("small-v2-base"), "P2a vs A0")
rule(scores("small-v2-ptw-pre"), scores("small-v2-base"), "P3a vs A0")
print("-- component test (gain > 0, 4 of 5, mAP loss <= 0.005) against A1:")
for cfg, label in [("small-v2-pool", "A3"), ("small-v2-xa", "X1"), ("small-v2-count", "F1"), ("small-v2-tw", "F2"), ("small-v2-ptw", "P3b"), ("small-v2-ftl", "P2b")]:
    rule(scores(cfg), a1, label, component=True)
c1 = [scores("small-v2-c1"), scores("small-v2-c1-seed2"), scores("small-v2-c1-seed3")]
print("C1 seed means AUROC", [round(s.auroc.mean(), 4) for s in c1], "mAP", [round(s.mAP.mean(), 4) for s in c1])
rng = pd.concat([s.auroc for s in c1], axis=1)
print("C1 per-fold AUROC range", [round(x, 4) for x in (rng.max(axis=1) - rng.min(axis=1))])
rule(pd.concat(c1).groupby(level=0).mean(), a1, "C1 seed mean vs A1")
print("C1 best epochs over seeds:", sorted(int(x) for s in c1 for x in s.best_epoch))

print("\n== Round 3 (budget_v3.toml): A1 screen on fold 0 vs C1 fold 0 ==")
rule(scores("small-v3-l4w192"), scores("small-v2-c1"), "A1 4x192 screen")
for cfg in ("small-v3-l4w192", "small-v2-c1"):
    d = json.loads((M / cfg / "fold0/F/run.json").read_text())
    print(cfg, "params", d["parameters"], "best_epoch", d["best_epoch"], "held auroc", round(d["best_held_macro_auroc"], 4), "mAP", round(d["best_held_mAP"], 4))
for cfg in ("small-v3-l4w192", "small-v2-ptw-pre"):
    p = M / cfg / "fold0/pretrain/run.json"
    if p.is_file():
        d = json.loads(p.read_text())
        print(cfg, "pretrain best_held_loss", round(d["best_held_loss"], 4), "epochs", d["epochs_completed"], "stopped_early", d["stopped_early"])

print("\n== S/H/F on small-v1 ==")
for m in ("S", "H", "F"):
    rows = []
    for k in range(5):
        d = json.loads((M / "small-v1" / f"fold{k}" / m / "run.json").read_text())
        rows.append((d["best_held_macro_auroc"], d["best_held_mAP"]))
    r = np.array(rows)
    print(m, "mean fold", round(r[:, 0].mean(), 4), round(r[:, 1].mean(), 4))
f_ = np.array([json.loads((M / "small-v1" / f"fold{k}/F/run.json").read_text())["best_held_macro_auroc"] for k in range(5)])
s_ = np.array([json.loads((M / "small-v1" / f"fold{k}/S/run.json").read_text())["best_held_macro_auroc"] for k in range(5)])
print("F > S on folds:", int((f_ > s_).sum()), "of 5")
