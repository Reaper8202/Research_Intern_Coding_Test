from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ehrforecast.tokenizer import FoldTokenizer, age_features, compress_refills
from tests.helpers import ts
from tests.transformer_fixtures import permitted_events, raw_tables


def _events(rows: list[tuple[str, str, str, str, str]], **extra: list[Any]) -> pd.DataFrame:
    """Rows of (patient, time, source, kind, code) in the event format."""
    frame = pd.DataFrame(rows, columns=["patient_id", "time", "source", "kind", "code"])
    frame["time"] = pd.to_datetime(frame["time"])
    n = len(frame)
    frame["value_num"] = extra.get("value_num", [np.nan] * n)
    frame["value_text"] = pd.Series(extra.get("value_text", [None] * n), dtype="string")
    frame["unit"] = pd.Series(extra.get("unit", [None] * n), dtype="string")
    starts = extra.get("interval_start", [None] * n)
    frame["interval_start"] = pd.to_datetime(
        [
            t if s is None and k in ("start", "stop") else s
            for t, s, k in zip(frame["time"], starts, frame["kind"], strict=True)
        ]
    )
    return frame


def _med(patient: str, *intervals: tuple[str, str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for start, stop in intervals:
        out.append({"patient_id": patient, "time": start, "kind": "start", "interval_start": start})
        out.append({"patient_id": patient, "time": stop, "kind": "stop", "interval_start": start})
    return out


def _med_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["interval_start"] = pd.to_datetime(frame["interval_start"])
    frame["source"], frame["code"] = "medications", "drug"
    return frame


def _kinds(events: pd.DataFrame) -> list[tuple[str, str]]:
    ordered = events.sort_values(["time", "kind"])
    return list(zip(ordered["time"].dt.strftime("%Y-%m-%d"), ordered["kind"], strict=True))


def test_refill_chain_merges_but_true_interruptions_and_pairs_remain() -> None:
    events = _med_events(
        _med(
            "p",
            ("2010-01-01", "2010-02-01"),
            ("2010-02-01", "2010-03-01"),
            ("2010-03-01", "2010-04-01"),
        )
        + _med("p", ("2010-06-01", "2010-07-01"))  # 61-day interruption
        + _med(
            "q", ("2011-01-01", "2011-02-01"), ("2011-02-02", "2011-03-01")
        )  # one-day gap merges
    )
    out, merged = compress_refills(events, 1.0)
    assert merged == 3
    assert _kinds(out[out["patient_id"].eq("p")]) == [
        ("2010-01-01", "start"),
        ("2010-04-01", "stop"),
        ("2010-06-01", "start"),
        ("2010-07-01", "stop"),
    ]
    assert _kinds(out[out["patient_id"].eq("q")]) == [
        ("2011-01-01", "start"),
        ("2011-03-01", "stop"),
    ]


def test_zero_length_prescriptions_never_merge_with_their_own_start() -> None:
    events = _med_events(_med("p", ("2010-01-01", "2010-01-01"), ("2012-01-01", "2012-01-01")))
    out, merged = compress_refills(events, 1.0)
    assert merged == 0
    pd.testing.assert_frame_equal(out, events)


def test_compression_after_the_cut_keeps_a_stop_whose_restart_is_later() -> None:
    cohort, tables = raw_tables(1, seed=3)
    tables["medications"] = pd.DataFrame(
        {
            "PATIENT": ["p00", "p00"],
            "START": ts("2014-11-01", "2014-12-31"),
            "STOP": ts("2014-12-31", "2015-02-01"),
            "CODE": ["drug", "drug"],
        }
    )
    events = permitted_events(cohort, tables)
    med = events[events["source"].eq("medications")]
    out, merged = compress_refills(med, 1.0)
    assert merged == 1  # the restart before the anchor merges; its own stop is not visible
    assert _kinds(out) == [("2014-11-01", "start")]
    tables["medications"].loc[1, "START"] = pd.Timestamp("2015-01-02")  # restart after the anchor
    med = permitted_events(cohort, tables).query("source == 'medications'")
    out, merged = compress_refills(med, 1.0)
    assert merged == 0
    assert _kinds(out) == [("2014-11-01", "start"), ("2014-12-31", "stop")]


def _observations() -> pd.DataFrame:
    rows: list[tuple[str, str, str, str, str]] = []
    values: list[Any] = []
    units: list[Any] = []
    texts: list[Any] = []
    for i in range(6):
        p = f"f{i}"
        for v in (1.0 + i, 50.0 + i):
            rows.append((p, "2010-01-01", "observations", "point", "lab"))
            values.append(v)
            units.append(" mg/L ")
            texts.append(None)
        rows.append((p, "2010-01-02", "observations", "point", "smoke"))
        values.append(np.nan)
        units.append(None)
        texts.append("Never  Smoker")
    rows.append(("f0", "2010-01-03", "observations", "point", "smoke"))
    values.append(np.nan)
    units.append(None)
    texts.append("rare answer")
    return _events(rows, value_num=values, unit=units, value_text=texts)


def test_value_tokens_bins_units_and_rare_values() -> None:
    events = _observations()
    fit = pd.Index([f"f{i}" for i in range(6)])
    tokenizer = FoldTokenizer.fit(events, fit, min_patients=5)
    rule = tokenizer.numeric_bins["lab\tmg/L"]
    assert np.all(np.diff(rule["edges"]) > 0)
    assert "obs:smoke:never smoker" in tokenizer.vocabulary
    assert "obs:smoke:rare answer" not in tokenizer.vocabulary
    held = _events(
        [("h", "2011-01-01", "observations", "point", "lab")] * 3
        + [("h", "2011-01-01", "observations", "point", "smoke")] * 2,
        value_num=[-1e9, 1e9, 5.0, np.nan, np.nan],
        unit=["mg/L", "mg/L", "g/L", None, None],
        value_text=[None, None, None, "RARE answer", None],
    )
    tokens = tokenizer.tokens(held).tolist()
    edge_tokens = rule["tokens"]
    assert tokens[0] == edge_tokens[0] and tokens[1] == edge_tokens[-1]  # clipped to edge bins
    assert tokens[2] == "obs:lab:other_value"  # unseen unit
    assert tokens[3] == "obs:smoke:other_value"  # rare text value
    assert tokens[4] == "obs:smoke:missing_value"


def test_fitting_ignores_held_out_patients_and_counts_patients_not_rows() -> None:
    events = _observations()
    fit = pd.Index([f"f{i}" for i in range(6)])
    held = _events(
        [("h", "2011-01-01", "observations", "point", "lab")] * 2
        + [("h", "2011-01-01", "procedures", "point", "only_held")] * 9,
        value_num=[1e6, -1e6] + [np.nan] * 9,
        unit=["mg/L", "mg/L"] + [None] * 9,
    )
    base = FoldTokenizer.fit(events, fit, min_patients=5)
    with_held = FoldTokenizer.fit(pd.concat([events, held], ignore_index=True), fit, min_patients=5)
    assert base.to_dict() == with_held.to_dict()
    # A code exclusive to a held-out patient maps to the source-specific unknown token (T3).
    assert with_held.tokens(held).tolist()[2:] == ["[UNK:proc]"] * 9
    many_rows = _events([("f0", "2010-01-01", "procedures", "point", "busy")] * 20)
    assert (
        "proc:busy" not in FoldTokenizer.fit(many_rows, pd.Index(["f0"]), min_patients=5).vocabulary
    )


def test_encoding_is_independent_of_row_order_and_deduplicates_within_day() -> None:
    cohort, tables = raw_tables(8, seed=1)
    events = permitted_events(cohort, tables)
    tokenizer = FoldTokenizer.fit(events, cohort.index[:6], min_patients=1)
    first = tokenizer.encode(events)
    shuffled = tokenizer.encode(events.sample(frac=1.0, random_state=4))
    pd.testing.assert_frame_equal(first, shuffled)
    assert not first.duplicated(["patient_id", "day", "token_id"]).any()
    doubled = tokenizer.encode(pd.concat([events, events], ignore_index=True))
    pd.testing.assert_frame_equal(first, doubled)


def test_tokenizer_round_trip_and_version_check(tmp_path: Path) -> None:
    cohort, tables = raw_tables(6, seed=2)
    events = permitted_events(cohort, tables)
    tokenizer = FoldTokenizer.fit(events, cohort.index[:4], min_patients=1)
    tokenizer.save(tmp_path / "tokenizer.json")
    loaded = FoldTokenizer.load(tmp_path / "tokenizer.json")
    assert loaded.sha256 == tokenizer.sha256
    pd.testing.assert_frame_equal(loaded.encode(events), tokenizer.encode(events))
    text = (
        (tmp_path / "tokenizer.json").read_text().replace("fold-tokenizer-v1", "fold-tokenizer-v0")
    )
    (tmp_path / "old.json").write_text(text)
    with pytest.raises(ValueError, match="incompatible"):
        FoldTokenizer.load(tmp_path / "old.json")


def test_age_features() -> None:
    features = age_features(np.array([0.0, 50.0]))
    assert features.shape == (2, 17)
    assert features[1, 0] == pytest.approx(0.5)
    np.testing.assert_allclose(features[0, 1:9], 0.0, atol=1e-7)  # sin(0)
    np.testing.assert_allclose(features[0, 9:], 1.0)  # cos(0)
