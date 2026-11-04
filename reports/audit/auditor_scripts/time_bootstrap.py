"""Time one paired BCa bootstrap_comparison call (n_boot=20) on the real OOF data: the BCa
jackknife (one statistic evaluation per patient) dominates, so this bounds the stacking runtime."""
import sys, time
sys.path.insert(0, "<repo>/src")
import pandas as pd
from pathlib import Path
from ehrforecast.contracts import load_training_data
from ehrforecast.evaluation import bootstrap_comparison
ROOT = Path("<repo>")
data = load_training_data(ROOT / "data/processed")
labels, prior, codes = data.labels, data.inputs.prior, data.inputs.codes
a = pd.read_parquet(ROOT / "reports/stacking/clean/oof_stack_logit_mean.parquet").loc[labels.index]
b = pd.read_parquet(ROOT / "models/lightgbm_clean/oof_counts.parquet").loc[labels.index]
t = time.time()
bootstrap_comparison(labels, a, codes, other=b, prior=prior, n_boot=20)
dt = time.time() - t
print(f"one paired bootstrap_comparison with n_boot=20 took {dt:.0f} s (jackknife over {len(labels)} patients dominates)")
print(f"stacking.run makes 9 paired calls with n_boot=1000: lower bound ~{9*dt/60:.0f} min")
