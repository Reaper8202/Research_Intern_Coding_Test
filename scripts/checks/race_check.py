"""D15 check, training CV: LightGBM counts with and without race and ethnicity.

Run from the repository root: python scripts/checks/race_check.py <output.json>
"""
import json, sys, time
from pathlib import Path
import lightgbm, numpy as np, pandas as pd
from ehrforecast.contracts import load_training_data
from ehrforecast.evaluation import bootstrap_comparison, evaluate_predictions
from ehrforecast.features import BagOfCodes
from ehrforecast.folds import fold_patients, load_folds
from ehrforecast.lightgbm_cv import FIXED
assert "torch" not in sys.modules
t0 = time.time()
ds = Path("data/processed"); data = load_training_data(ds)
labels, prior, codes, events, cohort = data.labels, data.inputs.prior, data.inputs.codes, data.inputs.events, data.inputs.cohort
assign = load_folds(ds)
demo = pd.read_parquet(ds / "cohort.parquet", columns=["race", "ethnicity"]).loc[labels.index].astype(str)
saved = pd.read_parquet("models/lightgbm/oof_counts.parquet").loc[labels.index]
def fit(x, y, h):
    if len(np.unique(y)) < 2:
        return np.full(len(h), float(y.mean()) if len(y) else 0.0)
    return lightgbm.LGBMClassifier(n_estimators=100, num_leaves=4, min_child_samples=50, **FIXED).fit(x, y).predict_proba(h)[:, 1]
withr = pd.DataFrame(np.nan, index=labels.index, columns=codes); rows = []
for fold in range(5):
    fit_ids, held = fold_patients(assign, fold)
    bag = BagOfCodes().fit(events, cohort.loc[fit_ids])
    xf, xh = bag.transform(events, cohort.loc[fit_ids]), bag.transform(events, cohort.loc[held])
    cats = pd.get_dummies(demo.loc[fit_ids], dtype="float32")
    extra_f = cats
    extra_h = pd.get_dummies(demo.loc[held], dtype="float32").reindex(columns=cats.columns, fill_value=0.0)
    xf2, xh2 = pd.concat([xf, extra_f], axis=1).to_numpy("float32"), pd.concat([xh, extra_h], axis=1).to_numpy("float32")
    parity = 0.0
    for t in codes:
        ar = prior.loc[fit_ids, t].to_numpy() == 0
        y = labels.loc[fit_ids, t].to_numpy()[ar]
        withr.loc[held, t] = fit(xf2[ar], y, xh2)
        if fold == 0 and t in codes[:5]:
            parity = max(parity, float(np.abs(fit(xf.to_numpy("float32")[ar], y, xh.to_numpy("float32")) - saved.loc[held, t].to_numpy()).max()))
    a = evaluate_predictions(labels.loc[held], withr.loc[held], codes, prior=prior.loc[held]).summary
    b = evaluate_predictions(labels.loc[held], saved.loc[held], codes, prior=prior.loc[held]).summary
    rows.append({"fold": fold, "auroc_with": a["macro_auroc"], "auroc_without": b["macro_auroc"], "map_with": a["mAP"], "map_without": b["mAP"]})
    print(f"fold {fold}: AUROC {b['macro_auroc']:.4f} -> {a['macro_auroc']:.4f}  mAP {b['mAP']:.4f} -> {a['mAP']:.4f}  {time.time()-t0:.0f}s" + (f"  parity check {parity:.1e}" if fold == 0 else ""), flush=True)
r = pd.DataFrame(rows); g = r.auroc_with - r.auroc_without; m = r.map_without - r.map_with
print(f"mean AUROC gain {g.mean():+.4f}, folds higher {(g>0).sum()}/5, mean mAP loss {m.mean():+.4f}; passes rule: {bool(g.mean()>=0.005 and (g>0).sum()>=4 and m.mean()<=0.005)}", flush=True)
ci = bootstrap_comparison(labels, withr, codes, other=saved, prior=prior, n_boot=500)
f = lambda d: f"{d['point']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
print("pooled with minus without (500 resamples): AUROC", f(ci["macro_auroc"]), "mAP", f(ci["mAP"]), f"total {time.time()-t0:.0f}s")
json.dump({"folds": rows, "bca": {k: ci[k] for k in ("macro_auroc", "mAP")}}, open(sys.argv[1], "w"), default=str)
