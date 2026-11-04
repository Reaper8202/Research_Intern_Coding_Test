"""Audit script C (B4, B5, A10): folds file; every fold tokenizer's fit-patient hash; the final
LightGBM feature spec (vocabulary, value medians) against a train-only recompute; Platt weights
against a recompute from training OOF; stacking OOF index. No torch in this process (lightgbm ok)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("<repo>")
PROC = REPO / "data" / "processed"
MODELS = REPO / "models"
OUT = Path(__file__).parent / "c_results.json"
sys.path.insert(0, str(REPO / "src"))
from ehrforecast.provenance import object_hash  # noqa: E402
from ehrforecast.folds import load_folds, fold_patients  # noqa: E402
from ehrforecast.contracts import load_training_data, load_splits  # noqa: E402
from ehrforecast.features import BagOfCodes  # noqa: E402
from ehrforecast.tokenizer import drop_observations  # noqa: E402

assert "torch" not in sys.modules
res: dict = {}
cohort = pd.read_parquet(PROC / "cohort.parquet")
train_ids = cohort.index[cohort["split"] == "train"]

# ------------------------------------------------------------------ B4 folds
folds = pd.read_parquet(PROC / "folds.parquet")
assignments = load_folds(PROC)  # raises if the file or manifest does not match the current split
res["B4"] = {
    "n_rows": int(len(folds)),
    "unique_patients": int(folds["patient_id"].nunique()),
    "all_in_train_split": bool(folds["patient_id"].isin(train_ids).all()),
    "covers_all_train": bool(set(folds["patient_id"]) == set(train_ids)),
    "fold_sizes": folds["fold"].value_counts().sort_index().to_dict(),
    "load_folds_ok": True,
    "manifest": json.loads((PROC / "folds.manifest.json").read_text()),
}
print("B4", json.dumps(res["B4"], indent=1, default=str))

# ------------------------------------------------------------------ B5 tokenizers: fit patients per fold
tok_rows = []
for tok_path in sorted(MODELS.glob("transformer/*/fold*/tokenizer.json")):
    fold = int(tok_path.parent.name[len("fold"):])
    fit, held = fold_patients(assignments, fold)
    data = json.loads(tok_path.read_text())
    tok_rows.append({
        "config": tok_path.parent.parent.name,
        "fold": fold,
        "fit_hash_matches_fold_fit_patients": data["fit_patients_sha256"] == object_hash(sorted(fit.tolist())),
        "fit_hash_matches_all_train": data["fit_patients_sha256"] == object_hash(sorted(train_ids.tolist())),
        "vocab": len(data["vocabulary"]),
        "excluded": data.get("excluded_observations"),
        "modalities": data.get("modalities"),
    })
res["B5_tokenizers"] = tok_rows
print("B5 tokenizers", pd.DataFrame(tok_rows).to_string())

# which tokenizers do the submitted members use?
manifest = json.loads((REPO / "reports" / "final" / "manifest.json").read_text())
res["final_members"] = [(m["config"], m["fold"], m["tokenizer"]) for m in manifest["members"]]
print("final members tokenizers", res["final_members"])

# ------------------------------------------------------------------ B5 LightGBM feature spec vs train-only recompute
data = load_training_data(PROC)
splits = load_splits(PROC)
excl = frozenset({"DALY", "QALY", "QOLS"})
events_clean = drop_observations(data.inputs.events, excl)
bag = BagOfCodes().fit(events_clean, data.inputs.cohort)
lgb = {}
from ehrforecast.lightgbm_cv import latest_values  # noqa: E402  (imports lightgbm, no torch)
from ehrforecast.lightgbm_final import fit_value_spec  # noqa: E402
latest_train = latest_values(events_clean, data.inputs.cohort)
spec_train = fit_value_spec(latest_train, data.inputs.cohort.index)
for name in ("counts", "counts_values"):
    spec = json.loads((MODELS / "lightgbm_final" / name / "feature_spec.json").read_text())
    rec = {
        "excluded_observations": spec["excluded_observations"],
        "vocab_eq_train_only_recompute": list(spec["vocabulary"]) == list(bag.vocabulary_ or []),
        "vocab_size": len(spec["vocabulary"]),
        "n_targets_with_booster": sum("booster" in t for t in spec["targets"].values()),
        "n_fit_per_target_max": max(t["n_fit"] for t in spec["targets"].values()),
        "n_train": int(len(train_ids)),
        "setting": spec["setting"],
    }
    if spec.get("values") is not None:
        v = spec["values"]
        rec["value_keys_eq_train_recompute"] = list(v["keys"]) == list(spec_train["keys"])
        rec["value_median_max_abs_diff"] = float(max(abs(v["value_median"][k] - spec_train["value_median"][k]) for k in v["keys"]))
        rec["days_median_max_abs_diff"] = float(max(abs(v["days_median"][k] - spec_train["days_median"][k]) for k in v["keys"]))
    lgb[name] = rec
res["B5_lightgbm_final_spec"] = lgb
print("B5 lightgbm", json.dumps(lgb, indent=1))

# ------------------------------------------------------------------ A10 Platt weights recompute from training OOF
from ehrforecast.calibration import fit_platt  # noqa: E402
from ehrforecast.masking import apply_prior_mask  # noqa: E402
oof = pd.read_parquet(REPO / "reports" / "stacking" / "clean" / "oof_stack_logit_mean.parquet")
labels, prior, codes = data.labels, data.inputs.prior, data.inputs.codes
raw = apply_prior_mask(oof.loc[labels.index], prior, codes)
recomputed = fit_platt(raw, labels, prior, codes)
saved = json.loads((REPO / "reports" / "calibration" / "stack" / "platt_weights.json").read_text())
diff = max(abs(recomputed[c][k] - saved[c][k]) for c in codes for k in ("a", "b", "fitted"))
res["A10_platt"] = {
    "oof_index_is_exactly_train": set(oof.index) == set(train_ids),
    "oof_n": int(len(oof)),
    "saved_weights_eq_recompute_from_train_oof_max_abs_diff": diff,
    "n_fitted": int(sum(saved[c]["fitted"] for c in codes)),
}
print("A10 platt", json.dumps(res["A10_platt"], indent=1))
# stacking inputs: transformer OOF and LightGBM OOF are train-only
for p in ("models/lightgbm_clean/oof_counts.parquet", "models/lightgbm_clean/oof_counts_values.parquet"):
    f = pd.read_parquet(REPO / p)
    res[f"oof_train_only:{p}"] = set(f.index) == set(train_ids)
# held predictions of final members are exactly the fold's held-out train patients
hp = []
for m in manifest["members"]:
    f = pd.read_parquet(MODELS / m["run_dir"] / "held_predictions_best.parquet")
    _, held = fold_patients(assignments, m["fold"])
    hp.append({"run": m["run_dir"], "held_eq_fold_heldout": set(f.index) == set(held), "n": len(f)})
res["member_held_predictions"] = hp
print(pd.DataFrame(hp).to_string())
OUT.write_text(json.dumps(res, indent=2, default=str))
print("wrote", OUT)
