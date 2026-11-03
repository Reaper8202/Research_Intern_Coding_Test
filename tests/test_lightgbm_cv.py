"""E11 LightGBM baselines: latest values, fit-only statistics, tree truncation, selection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ehrforecast.lightgbm_cv import ensemble_oof, latest_values, select_setting, value_features


def _events(rows: list[tuple[str, str, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [r[0] for r in rows],
            "time": pd.to_datetime([r[3] for r in rows]),
            "source": "observations",
            "kind": "point",
            "code": [r[1] for r in rows],
            "value_num": [r[2] for r in rows],
            "value_text": None,
            "unit": [r[4] for r in rows],
        }
    )


COHORT = pd.DataFrame(
    {"anchor_date": pd.to_datetime(["2015-01-01"] * 7)},
    index=pd.Index([f"p{i}" for i in range(7)], name="patient_id"),
)


def test_latest_value_takes_the_median_of_ties_at_the_latest_time() -> None:
    events = _events(
        [
            ("p0", "A", 1.0, "2010-01-01", "mg"),
            ("p0", "A", 4.0, "2014-12-31", "mg"),
            ("p0", "A", 6.0, "2014-12-31", "mg"),
            ("p0", "A", 9.0, "2014-12-31", " mg "),  # same unit after stripping
            ("p0", "A", 2.0, "2014-06-01", ""),  # a different (missing) unit
        ]
    )
    out = latest_values(events, COHORT).set_index("key")
    assert out.loc["A\tmg", "value"] == 6.0
    assert np.isclose(out["log_days"].to_numpy()[out.index.get_loc("A\tmg")], np.log1p(1.0))
    assert "A\tnounit" in out.index


def test_value_features_use_fit_patients_only() -> None:
    rows = [(f"p{i}", "A", float(i), "2014-01-01", "mg") for i in range(6)]
    rows += [("p6", "A", 1000.0, "2014-01-01", "mg")]  # a held patient with an extreme value
    rows += [(f"p{i}", "B", 1.0, "2014-01-01", "mg") for i in range(4)]  # 4 fit carriers only
    events = _events(rows)
    latest = latest_values(events, COHORT)
    fit = pd.Index([f"p{i}" for i in range(6)])
    held = pd.Index(["p6"])
    x_fit = value_features(latest, events.assign(key="A\tmg"), fit, fit, binned=False)
    assert list(x_fit.columns) == ["value:A\tmg", "missing:A\tmg", "days:A\tmg"]  # B dropped
    x_held = value_features(latest, events.assign(key="A\tmg"), fit, held, binned=False)
    assert x_held.loc["p6", "value:A\tmg"] == 1000.0
    missing = value_features(latest, events.assign(key="A\tmg"), fit[:5], pd.Index(["p5"]), False)
    # p5's own value is ignored only if absent; here an absent patient takes the fit median.
    absent = value_features(
        latest.loc[latest["patient_id"].ne("p5")], events, fit[:5], pd.Index(["p5"]), False
    )
    assert absent.loc["p5", "value:A\tmg"] == 2.0 and absent.loc["p5", "missing:A\tmg"] == 1.0
    assert missing.loc["p5", "missing:A\tmg"] == 0.0


def test_binned_values_use_fit_patient_edges() -> None:
    rows = [(f"p{i}", "A", float(i), "2014-01-01", "mg") for i in range(6)]
    rows += [("p6", "A", 1000.0, "2014-01-01", "mg")]
    events = _events(rows)
    latest = latest_values(events, COHORT)
    values = events.assign(key="A\tmg")[["patient_id", "key", "value_num"]]
    fit = pd.Index([f"p{i}" for i in range(6)])
    held = value_features(latest, values, fit, pd.Index(["p6"]), binned=True)
    # Six fit values give one bin (fewer than 20 values per bin), so every value maps to bin 0.
    assert held.loc["p6", "value:A\tmg"] == 0.0


TRUNCATION_CHECK = """
import lightgbm, numpy as np
from ehrforecast.lightgbm_cv import FIXED
rng = np.random.default_rng(0)
x = rng.normal(size=(300, 6)).astype("float32")
y = (x[:, 0] + rng.normal(size=300) > 0).astype(int)
long = lightgbm.LGBMClassifier(n_estimators=30, num_leaves=8, **FIXED).fit(x, y)
short = lightgbm.LGBMClassifier(n_estimators=10, num_leaves=8, **FIXED).fit(x, y)
assert np.array_equal(long.predict_proba(x, num_iteration=10), short.predict_proba(x))
import sys
assert "torch" not in sys.modules, "lightgbm_cv must not import torch (two OpenMP runtimes)"
"""


def test_the_first_trees_of_a_longer_model_equal_a_shorter_model() -> None:
    # A clean interpreter: other tests load PyTorch, whose OpenMP runtime can crash LightGBM.
    result = subprocess.run(
        [sys.executable, "-c", TRUNCATION_CHECK], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_selection_uses_mean_auroc_and_flags_grid_edges() -> None:
    rows = []
    for leaves, child, trees in [(4, 20, 100), (8, 50, 300), (16, 20, 600)]:
        for fold in range(5):
            score = 0.70 + (0.02 if (leaves, child, trees) == (8, 50, 300) else 0.0)
            rows.append(
                {
                    "num_leaves": leaves,
                    "min_child_samples": child,
                    "n_estimators": trees,
                    "fold": fold,
                    "macro_auroc": score,
                    "mAP": 0.1,
                }
            )
    choice = select_setting(pd.DataFrame(rows))
    assert (choice["num_leaves"], choice["min_child_samples"], choice["n_estimators"]) == (
        8,
        50,
        300,
    )
    assert choice["grid_edge"] == ["min_child_samples"]


def _write_member(models: Path, name: str, values: dict[str, float]) -> Path:
    config = models / f"{name}.toml"
    config.write_text(f'name = "{name}"\n')
    for fold in range(5):
        run = models / "transformer" / name / f"fold{fold}" / "F"
        run.mkdir(parents=True)
        frame = pd.DataFrame({"1": [values[f"p{fold}"]]}, index=[f"p{fold}"])
        frame.to_parquet(run / "held_predictions_best.parquet")
    return config


def test_ensemble_is_the_patient_level_mean_of_its_members(tmp_path: Path) -> None:
    assignments = pd.DataFrame({"patient_id": [f"p{k}" for k in range(5)], "fold": list(range(5))})
    first = _write_member(tmp_path, "a", {f"p{k}": 0.2 for k in range(5)})
    second = _write_member(tmp_path, "b", {f"p{k}": 0.4 + k / 10 for k in range(5)})
    mean = ensemble_oof(tmp_path, [first, second], assignments)
    assert mean.loc["p3", "1"] == pytest.approx((0.2 + 0.7) / 2)
    assert mean.loc["p0", "1"] == pytest.approx((0.2 + 0.4) / 2)
