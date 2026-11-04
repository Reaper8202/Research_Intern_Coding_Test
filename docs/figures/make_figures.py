"""Report figures from the committed reports (run: /opt/miniconda3/bin/python -I docs/figures/make_figures.py)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out = root / "docs" / "figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5f5e58", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

# Figure 1: val curves, one thin line per member, mean in front.
c = pd.read_csv(root / "reports/val/val_curves.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for (cfg, fold), d in c.groupby(["config", "fold"]):
    ax.plot(d.epoch, d.val_macro_auroc, color=BLUE, alpha=0.25, lw=1)
    b = d[d.is_best]
    ax.plot(b.epoch, b.val_macro_auroc, "o", color=BLUE, ms=4, alpha=0.9, mec="white", mew=0.8)
m = c.groupby("epoch").val_macro_auroc.agg(["mean", "count"])
m = m[m["count"] >= 5]
ax.plot(m.index, m["mean"], color=ORANGE, lw=2, label="mean over members")
ax.set_xlabel("fine-tuning epoch")
ax.set_ylabel("val macro AUROC")
ax.set_ylim(0.68, 0.78)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.text(0.99, 0.04, "thin lines: 15 members; dots: epoch chosen on the held-out training fold",
        transform=ax.transAxes, ha="right", color=MUTED, fontsize=7.5)
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.12), fontsize=8)
fig.tight_layout()
fig.savefig(out / "val_curves.png")

# Figure 2: per-target val AUROC, transformer vs LightGBM counts, sorted by transformer.
t = pd.read_csv(root / "reports/val/val_per_target.csv")
desc = pd.read_csv(root / "data/raw/target_conditions.csv", dtype=str).set_index("CODE")["DESCRIPTION"]
w = t.pivot(index="code", columns="model", values="auroc")
npos = t[t.model == "final"].set_index("code")["n_pos"]
w = w.sort_values("final")
labels = [f"{desc[str(k)].replace(' (disorder)', '').replace(' (finding)', '').replace(' (situation)', '')[:40]} ({npos[k]})" for k in w.index]
fig, ax = plt.subplots(figsize=(6.4, 7.2))
y = range(len(w))
for yi, (a, b) in enumerate(zip(w["final"], w["lightgbm_counts"])):
    ax.plot([a, b], [yi, yi], color=GRID, lw=1.2, zorder=1)
ax.scatter(w["lightgbm_counts"], list(y), color=ORANGE, s=22, zorder=2, label="LightGBM counts", edgecolor="white", lw=0.6)
ax.scatter(w["final"], list(y), color=BLUE, s=22, zorder=3, label="transformer ensemble", edgecolor="white", lw=0.6)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=7)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_xlim(0.38, 1.0)
ax.set_xlabel("val AUROC (positives in parentheses)")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(out / "val_per_target.png")
print("wrote", out / "val_curves.png", out / "val_per_target.png")

# Figure 3: reliability of the blend before and after Platt scaling (training CV, nested folds).
# Log axes: with prevalence of 1 to 27 percent, equal-mass bins of all at-risk cells sit below 0.1.
rel = pd.read_csv(root / "reports/calibration/stack/reliability.csv")
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([1e-3, 1], [1e-3, 1], color=MUTED, lw=0.8, ls=":")
for name, color, label in (("raw", BLUE, "blend, raw"), ("calibrated", ORANGE, "blend, Platt-scaled")):
    d = rel[rel.model == name]
    ax.plot(d.mean_probability, d.observed_rate.clip(lower=1e-3), "-o", color=color, lw=1.5, ms=5,
            mec="white", mew=0.8, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2e-3, 0.5)
ax.set_ylim(2e-3, 0.5)
ax.set_xlabel("mean predicted probability (10 equal-mass bins, log scale)")
ax.set_ylabel("observed first-diagnosis rate (log scale)")
ax.grid(color=GRID, lw=0.6, which="both")
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(out / "reliability.png")
print("wrote", out / "reliability.png")
