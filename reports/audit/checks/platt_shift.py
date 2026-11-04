"""Compare the blend's pre-Platt logit distribution on training OOF with the test-time blend."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from ehrforecast.stacking import logit
root = Path(".")
oof = pd.read_parquet(root / "reports/stacking/clean/oof_stack_logit_mean.parquet")
print("oof columns sample:", oof.columns.tolist()[:5], oof.shape, "index name:", oof.index.name)
t = pd.read_csv(root / "reports/final/test_predictions_transformer.csv", index_col=0)
l1 = pd.read_csv(root / "reports/lightgbm_final/test_predictions_lightgbm_counts.csv", index_col=0)
l2 = pd.read_csv(root / "reports/lightgbm_final/test_predictions_lightgbm_counts_values.csv", index_col=0)
codes = [c for c in t.columns]
l1 = l1.loc[t.index, codes]; l2 = l2.loc[t.index, codes]
test_logit = np.mean([logit(t.to_numpy()), logit(l1.to_numpy()), logit(l2.to_numpy())], axis=0)
oof_cols = [c for c in codes if c in oof.columns]
oof_logit = logit(oof[oof_cols].to_numpy())
print("targets matched:", len(oof_cols))
prior_test = pd.read_parquet(root / "data/processed/prior_conditions.parquet").loc[t.index, codes].to_numpy().astype(bool)
prior_oof = pd.read_parquet(root / "data/processed/prior_conditions.parquet").loc[oof.index, oof_cols].to_numpy().astype(bool)
def stats(x, prior):
    x = np.where(prior, np.nan, x)
    return np.nanmean(x, axis=0), np.nanstd(x, axis=0)
mo, so = stats(oof_logit, prior_oof); mt, st = stats(test_logit, prior_test)
print("mean logit over targets: oof %.3f test %.3f ; mean sd: oof %.3f test %.3f" % (mo.mean(), mt.mean(), so.mean(), st.mean()))
d = mt - mo
print("per-target mean-logit shift (test - oof): mean %.3f, min %.3f, max %.3f, |shift|>0.25 on %d targets" % (d.mean(), d.min(), d.max(), int((np.abs(d) > 0.25).sum())))
r = st / so
print("per-target sd ratio test/oof: mean %.3f, min %.3f, max %.3f" % (r.mean(), r.min(), r.max()))
json.dump({"n_targets": len(oof_cols), "mean_logit_oof": float(mo.mean()), "mean_logit_test": float(mt.mean()), "sd_oof": float(so.mean()), "sd_test": float(st.mean()), "shift_mean": float(d.mean()), "shift_min": float(d.min()), "shift_max": float(d.max()), "n_abs_shift_gt_0.25": int((np.abs(d) > 0.25).sum()), "sd_ratio_mean": float(r.mean()), "sd_ratio_min": float(r.min()), "sd_ratio_max": float(r.max())}, open(sys.argv[1], "w"), indent=2)
