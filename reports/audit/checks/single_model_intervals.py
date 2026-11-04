"""Post-hoc single-model BCa intervals for the comparison models on official val (attempt 2 files).

Reads the saved val prediction files and the val labels once more; changes no model and no choice.
"""
import json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, "src")
from ehrforecast.contracts import load_prediction_inputs, load_splits
from ehrforecast.evaluation import bootstrap_comparison
from ehrforecast.val_score import load_val_labels, read_comparison
root = Path(".")
splits = load_splits(root / "data/processed")
val = pd.Index(splits.index[splits.eq("val")], name="patient_id")
inputs = load_prediction_inputs(root / "data/processed", val)
codes, prior = inputs.codes, inputs.prior
labels = load_val_labels(root / "data/processed", val, codes)
files = {
    "lightgbm_counts": "reports/lightgbm_final/val_predictions_lightgbm_counts.csv",
    "lightgbm_counts_values": "reports/lightgbm_final/val_predictions_lightgbm_counts_values.csv",
    "logreg_codes": "reports/baseline/val_predictions_logreg_codes.csv",
    "incidence_at_risk": "reports/baseline/val_predictions_incidence_age_sex_at_risk.csv",
}
out = {"scope": "official val, attempt-2 prediction files, computed after the bundle completed; no model or choice changed",
       "bundle_id": json.load(open("reports/val/bundle.json"))["bundle_id"], "n_boot": 1000, "seed": 0, "models": {}}
for name, path in files.items():
    t0 = time.time()
    probs = read_comparison(root / path, val, codes) if "read_comparison" in dir() else pd.read_csv(root / path, index_col=0).loc[val, codes]
    r = bootstrap_comparison(labels, probs, codes, prior=prior, n_boot=1000, seed=0)
    out["models"][name] = r
    print(name, {k: r[k] for k in r if k in ("macro_auroc", "mAP", "macro_brier")}, "%.0fs" % (time.time() - t0), flush=True)
Path("reports/val/post_hoc_single_model_intervals.json").write_text(json.dumps(out, indent=2, default=str))
print("wrote reports/val/post_hoc_single_model_intervals.json")
