"""Audit boundaries: fit-only statistics, the anchor-midnight cut, and landmark windows."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

from ehrforecast.contracts import prediction_cohort, prediction_events
from ehrforecast.events import EVENT_COLUMNS
from ehrforecast.lightgbm_cv import MIN_VALUE_PATIENTS, all_values, latest_values, value_features
from ehrforecast.schema import EVENT_TABLES
from ehrforecast.sequences import DAYS_PER_YEAR, LANDMARK_HORIZON_DAYS, SequenceConfig
from ehrforecast.tokenizer import FoldTokenizer
from ehrforecast.truncation import find_post_anchor_values, truncate_table
from tests.helpers import ts
from tests.transformer_fixtures import ANCHOR, fold_data, permitted_events, raw_tables

MID_2014 = pd.Timestamp("2014-06-01")


def _point_rows(
    patients: Iterable[str],
    source: str,
    code: str,
    *,
    value: float = np.nan,
    text: str | None = None,
    unit: str | None = None,
    time: pd.Timestamp = MID_2014,
) -> pd.DataFrame:
    """One point event per patient, in the event-table format."""
    ids = list(patients)
    frame = pd.DataFrame(
        {
            "patient_id": ids,
            "time": pd.Series([time] * len(ids), dtype="datetime64[ns]"),
            "source": source,
            "kind": "point",
            "code": code,
            "value_num": value,
            "value_text": pd.Series([text] * len(ids), dtype="string"),
            "unit": pd.Series([unit] * len(ids), dtype="string"),
            "interval_start": pd.Series([pd.NaT] * len(ids), dtype="datetime64[ns]"),
        }
    )
    return frame[list(EVENT_COLUMNS)]


def _new_rows(patients: Iterable[str]) -> pd.DataFrame:
    """A new code, an extreme value of a known code, a new numeric code, and a new text value."""
    ids = list(patients)
    return pd.concat(
        [
            _point_rows(ids, "procedures", "P_new"),
            _point_rows(ids, "observations", "O1", value=9999.0, unit="mg/dL"),
            _point_rows(ids, "observations", "O_new", value=3.5, unit="kg"),
            _point_rows(ids, "observations", "O2", text="Current smoker"),
        ],
        ignore_index=True,
    )


def test_tokenizer_fit_ignores_held_out_rows_but_not_fit_rows() -> None:
    """Catches a tokenizer whose vocabulary, bin edges, or text values read held-out histories."""
    cohort, tables = raw_tables(12, seed=5)
    events = permitted_events(cohort, tables)
    fit, held = cohort.index[:8], cohort.index[8:]
    base = FoldTokenizer.fit(events, fit, min_patients=1)
    with_held = FoldTokenizer.fit(
        pd.concat([events, _new_rows(held)], ignore_index=True), fit, min_patients=1
    )
    assert with_held.vocabulary == base.vocabulary
    assert with_held.numeric_bins == base.numeric_bins
    assert with_held.text_values == base.text_values
    assert with_held.sha256 == base.sha256
    # The same rows for fit patients must change the hash, so the check above is not vacuous.
    with_fit = FoldTokenizer.fit(
        pd.concat([events, _new_rows(fit[:2])], ignore_index=True), fit, min_patients=1
    )
    assert with_fit.sha256 != base.sha256
    assert {"proc:P_new", "obs:O2:current smoker"} <= set(with_fit.vocabulary)
    assert with_fit.numeric_bins["O1\tmg/dL"]["edges"] != base.numeric_bins["O1\tmg/dL"]["edges"]


def test_truncation_cuts_interval_tables_at_anchor_midnight() -> None:
    """Catches an off-by-one cut that lets anchor-day starts or stops into the history."""
    anchors = pd.Series({"p": ANCHOR})
    for table in ("medications", "careplans", "allergies", "devices"):
        spec = EVENT_TABLES[table]
        df = pd.DataFrame(
            {
                "PATIENT": ["p", "p", "p"],
                "START": ts("2014-12-31T23:59:00Z", "2014-12-31T23:59:00Z", "2015-01-01T00:00:00Z"),
                "STOP": ts("2015-01-01T00:00:00Z", "2014-12-31T23:59:59Z", None),
                "CODE": ["spans", "ended", "anchor_day"],
            }
        )
        out = truncate_table(df, spec, anchors)
        assert out["CODE"].tolist() == ["spans", "ended"], table
        assert out["STOP"].isna().tolist() == [True, False], table
        assert out.loc[1, "STOP"] == pd.Timestamp("2014-12-31 23:59:59"), table
        assert find_post_anchor_values(out, spec, anchors) == 0, table


def test_value_feature_statistics_read_fit_patients_only() -> None:
    """Catches medians, bin edges, or key selection that read held-out patients' values."""
    cohort, tables = raw_tables(24, seed=0)
    prepared = prediction_cohort(cohort)
    events = permitted_events(cohort, tables)
    fit, held = prepared.index[:16], prepared.index[16:]
    perturbed = events.copy()
    held_numeric = perturbed["patient_id"].isin(held) & perturbed["value_num"].notna()
    perturbed.loc[held_numeric, "value_num"] = perturbed.loc[held_numeric, "value_num"] * 50 + 7
    assert len(held) >= MIN_VALUE_PATIENTS  # a held-only code would be selected if counted
    perturbed = pd.concat(
        [perturbed, _point_rows(held, "observations", "O_held", value=1.0, unit="kg")],
        ignore_index=True,
    )
    for binned in (False, True):
        base = value_features(latest_values(events, prepared), all_values(events), fit, fit, binned)
        changed = value_features(
            latest_values(perturbed, prepared), all_values(perturbed), fit, fit, binned
        )
        pd.testing.assert_frame_equal(base, changed)  # values and the key set alike
        assert "value:O1\tmg/dL" in base.columns
        assert not any("O_held" in column for column in base.columns)
    # latest_values raises for a value after the anchor. A value at anchor midnight itself passes
    # with zero days; the input contract (prediction_events) is what rejects that row.
    on_anchor_day = _point_rows(
        fit[:1], "observations", "O1", value=1.0, unit="mg/dL", time=ANCHOR + pd.Timedelta(hours=9)
    )
    with pytest.raises(ValueError, match="after its patient's anchor"):
        latest_values(pd.concat([events, on_anchor_day], ignore_index=True), prepared)
    at_midnight = on_anchor_day.assign(time=ANCHOR)
    assert latest_values(at_midnight, prepared)["log_days"].tolist() == [0.0]
    with pytest.raises(ValueError, match="strictly before the anchor"):
        prediction_events(at_midnight, prepared)


def test_landmark_windows_end_before_the_real_anchor_and_see_only_earlier_days() -> None:
    """Catches an extra anchor whose label window or input reaches the real anchor's history."""
    fold = fold_data()
    hist, seq = fold.histories, SequenceConfig(48, 6)
    cohort = fold.data.inputs.cohort
    anchor_days = (cohort["anchor_date"] - cohort["birthdate"]).dt.days
    rng = np.random.default_rng(0)
    checked = 0
    for i in range(len(hist)):
        anchor_day = int(anchor_days[str(hist.patients[i])])
        allowed = hist.landmark_range(i, anchor_day)
        if allowed is None:
            continue
        low, high = allowed
        assert high + LANDMARK_HORIZON_DAYS < anchor_day  # so every cut's window ends earlier
        base = int(hist.token_offsets[i])
        tokens = hist.tokens[base : base + hist.n_tokens(i)]
        token_days = hist.token_days[base : base + hist.n_tokens(i)]
        for raw_cut in (low, high, *rng.integers(low, high + 1, size=6)):
            cut = int(raw_cut)
            assert cut + LANDMARK_HORIZON_DAYS < anchor_day
            sample = hist.landmark_sample(i, cut, seq)
            days = np.round(sample.ages.astype("float64") * DAYS_PER_YEAR)
            assert sample.n_events >= 1 and (days < cut).all()  # prefix and events alike
            visible = tokens[token_days < cut]
            assert np.array_equal(sample.tokens[sample.n_prefix :], visible[-sample.n_events :])
            checked += 1
    assert checked > 0
