from __future__ import annotations

import pandas as pd

from ehrforecast.cv_summary import select_method


def _table(scores: dict[str, tuple[list[float], list[float]]]) -> pd.DataFrame:
    params = {"S": 1000, "H": 50, "F": 1000}
    rows = []
    for method, (aurocs, maps) in scores.items():
        for fold, (a, p) in enumerate(zip(aurocs, maps, strict=True)):
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "macro_auroc": a,
                    "mAP": p,
                    "trainable_parameters": params[method],
                    "parameters": 1050,
                }
            )
    return pd.DataFrame(rows)


def test_selection_needs_five_folds_for_every_method() -> None:
    table = _table(
        {"S": ([0.7] * 4, [0.2] * 4), "H": ([0.7] * 5, [0.2] * 5), "F": ([0.7] * 5, [0.2] * 5)}
    )
    assert select_method(table)["status"] == "pending"


def test_consistent_gain_replaces_the_reference() -> None:
    s = ([0.70, 0.71, 0.72, 0.70, 0.71], [0.20] * 5)
    f = ([0.72, 0.72, 0.73, 0.71, 0.70], [0.20] * 5)  # +0.008 mean, four folds better
    h = ([0.65] * 5, [0.15] * 5)
    decision = select_method(_table({"S": s, "H": h, "F": f}))
    assert decision["consistency_checks_against_S"]["F"]["passes_consistency_rule"]
    assert decision["best_confirmed"] == "F" and decision["selected_method"] == "F"


def test_small_gain_keeps_the_reference_and_close_methods_prefer_fewer_parameters() -> None:
    s = ([0.70] * 5, [0.20] * 5)
    f = ([0.704] * 5, [0.20] * 5)  # gain below 0.005
    h = ([0.698] * 5, [0.197] * 5)  # within 0.005 on both metrics, fewest supervised parameters
    decision = select_method(_table({"S": s, "H": h, "F": f}))
    assert decision["best_confirmed"] == "S"
    assert decision["selected_method"] == "H"
    h_far = ([0.69] * 5, [0.20] * 5)
    assert select_method(_table({"S": s, "H": h_far, "F": f}))["selected_method"] == "S"


def test_fold_differences_subtract_the_reference_per_fold_and_method() -> None:
    from ehrforecast.cv_summary import fold_differences

    ours = _table({"F": ([0.75, 0.76, 0.74, 0.75, 0.77], [0.20] * 5)})
    reference = _table({"F": ([0.74] * 5, [0.21] * 5), "S": ([0.70] * 5, [0.2] * 5)})
    out = fold_differences(ours, reference)
    assert out["method"].unique().tolist() == ["F"]
    assert out["macro_auroc_difference"].round(4).tolist() == [0.01, 0.02, 0.0, 0.01, 0.03]
    assert out["mAP_difference"].round(4).tolist() == [-0.01] * 5


def test_modality_subgroup_uses_genotype_files_and_prescans_only() -> None:
    import numpy as np

    from ehrforecast.cv_summary import modality_patients
    from ehrforecast.modalities import Modalities

    cohort = pd.DataFrame(
        {"anchor_date": pd.to_datetime(["2015-01-01 10:00"] * 4)},
        index=pd.Index(["a", "b", "c", "d"], name="patient_id"),
    )
    genomics = pd.DataFrame({"patient_id": ["a"], "site": ["rs1"], "count": [0]})
    scans = pd.Series(
        pd.to_datetime(["2014-12-31", "2015-01-01", "2016-01-01"]), index=["b", "c", "d"]
    )
    modalities = Modalities(genomics, scans, np.zeros((3, 2, 2, 2), "float16"), 2, "x")
    # a: genotype file; b: scan the day before the anchor; c: scan on the anchor day; d: later.
    assert modality_patients(modalities, cohort).tolist() == ["a", "b"]
