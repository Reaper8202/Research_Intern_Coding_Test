"""P3c: patient histories, prediction sequences, pretraining windows, and retention reports.

Prediction sequence (model plan section 8.4)::

    [SEX:*]  hx:<older condition> ...  [SEP]  <clinical events, oldest first>

The most recent complete event days that fit the context length are retained. The summary lists
conditions first diagnosed before the retained window: all target conditions first, oldest
first, then the most recently diagnosed other conditions, up to ``max_summary`` tokens. Selected
summary tokens are ordered by first-diagnosis day. Ages are measured from birth, so a common shift
of all dates leaves every input unchanged. The readout is the last retained clinical event.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ehrforecast.events import SOURCE_ORDER
from ehrforecast.provenance import file_hash, object_hash, write_json
from ehrforecast.tokenizer import PREFIX, SEP, SEX_TOKENS, FoldTokenizer

HISTORY_VERSION = "patient-histories-v1"
N_HEAD_FEATURES = 4  # age at anchor / 100, male, scaled recency, missing-history flag
RECENCY_SCALE = math.log1p(36525)
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class SequenceConfig:
    context_length: int = 1024
    max_summary: int = 128

    def __post_init__(self) -> None:
        if self.max_summary < 0 or self.context_length < self.max_summary + 3:
            raise ValueError("context length must hold the summary, structure, and one event")


@dataclass
class Sample:
    """One model input. ``readout`` is -1 when no clinical event exists."""

    tokens: np.ndarray
    ages: np.ndarray
    sources: np.ndarray  # source index per token (``events.SOURCE_ORDER``); -1 for the prefix
    n_prefix: int
    readout: int
    head: np.ndarray
    first_day: int = 0
    partial_day: bool = False
    n_summary: int = 0
    n_candidates: int = 0

    @property
    def n_events(self) -> int:
        return len(self.tokens) - self.n_prefix


@dataclass
class Histories:
    """Tokenized permitted histories in flat arrays with per-patient offsets."""

    patients: pd.Index
    token_offsets: np.ndarray
    tokens: np.ndarray
    sources: np.ndarray
    token_days: np.ndarray  # days since birth, per token
    day_offsets: np.ndarray
    day_starts: np.ndarray  # global token index where each day starts
    hx_offsets: np.ndarray
    hx_tokens: np.ndarray
    hx_days: np.ndarray
    hx_target: np.ndarray
    male: np.ndarray
    age_at_anchor: np.ndarray
    days_since_last: np.ndarray  # -1 for an empty history
    sex_ids: tuple[int, int] = (0, 0)  # token IDs of [SEX:F], [SEX:M]
    sep_id: int = 0
    _position: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._position = {p: i for i, p in enumerate(self.patients)}

    def __len__(self) -> int:
        return len(self.patients)

    def index_of(self, patients: pd.Index) -> np.ndarray:
        missing = [p for p in patients if p not in self._position]
        if missing:
            raise ValueError(f"{len(missing)} requested patients have no history entry")
        return np.array([self._position[p] for p in patients], dtype="int64")

    def n_tokens(self, i: int) -> int:
        return int(self.token_offsets[i + 1] - self.token_offsets[i])

    # ------------------------------------------------------------------ pieces
    def _days(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Relative day start indices and their days since birth."""
        start = self.token_offsets[i]
        starts = self.day_starts[self.day_offsets[i] : self.day_offsets[i + 1]]
        return starts - start, self.token_days[starts]

    def _hx(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = self.hx_offsets[i], self.hx_offsets[i + 1]
        return self.hx_tokens[a:b], self.hx_days[a:b], self.hx_target[a:b]

    def _summary(self, i: int, before_day: int, max_summary: int) -> tuple[np.ndarray, np.ndarray]:
        """Summary tokens and their first-diagnosis days for a window starting ``before_day``."""
        tokens, days, target = self._hx(i)
        candidate = days < before_day
        if not candidate.any() or max_summary == 0:
            return np.zeros(0, dtype="int64"), np.zeros(0, dtype="int64")
        idx = np.flatnonzero(candidate)
        targets = idx[target[idx]]
        targets = targets[np.lexsort((tokens[targets], days[targets]))]  # oldest first
        others = idx[~target[idx]]
        others = others[np.lexsort((tokens[others], -days[others]))]  # most recent first
        chosen = np.concatenate([targets, others])[:max_summary]
        chosen = chosen[np.lexsort((tokens[chosen], days[chosen]))]
        return tokens[chosen].astype("int64"), days[chosen].astype("int64")

    def _summary_counts(self, i: int, window_days: np.ndarray, max_summary: int) -> np.ndarray:
        _, days, _ = self._hx(i)
        counts = np.searchsorted(np.sort(days), window_days, side="left")
        return np.minimum(counts, max_summary)

    def _assemble(
        self, i: int, start: int, stop: int, first_day: int, partial: bool, config: SequenceConfig
    ) -> Sample:
        base = self.token_offsets[i]
        events = self.tokens[base + start : base + stop].astype("int64")
        event_days = self.token_days[base + start : base + stop]
        event_sources = self.sources[base + start : base + stop].astype("int64")
        _, day_values = self._days(i)
        hx_tokens, hx_days = self._summary(i, int(day_values[first_day]), config.max_summary)
        first_age = float(event_days[0]) / DAYS_PER_YEAR
        sex = self.sex_ids[int(self.male[i])]
        tokens = np.concatenate([[sex], hx_tokens, [self.sep_id], events])
        ages = np.concatenate(
            [[first_age], hx_days / DAYS_PER_YEAR, [first_age], event_days / DAYS_PER_YEAR]
        ).astype("float32")
        return Sample(
            tokens=tokens,
            ages=ages,
            sources=np.concatenate([np.full(len(hx_tokens) + 2, -1), event_sources]),
            n_prefix=len(hx_tokens) + 2,
            readout=len(tokens) - 1,
            head=self.head_features(i),
            first_day=first_day,
            partial_day=partial,
            n_summary=len(hx_tokens),
            n_candidates=int((self._hx(i)[1] < day_values[first_day]).sum()),
        )

    def head_features(self, i: int) -> np.ndarray:
        missing = self.days_since_last[i] < 0
        recency = 0.0 if missing else math.log1p(int(self.days_since_last[i])) / RECENCY_SCALE
        return np.array(
            [self.age_at_anchor[i] / 100.0, float(self.male[i]), recency, float(missing)],
            dtype="float32",
        )

    def _empty(self, i: int) -> Sample:
        sex = self.sex_ids[int(self.male[i])]
        return Sample(
            tokens=np.array([sex, self.sep_id], dtype="int64"),
            ages=np.zeros(2, dtype="float32"),
            sources=np.full(2, -1, dtype="int64"),
            n_prefix=2,
            readout=-1,
            head=self.head_features(i),
        )

    # ------------------------------------------------------------------ samples
    def prediction_start(self, i: int, config: SequenceConfig) -> int:
        """First retained day: the oldest day whose window, prefix, and summary fit."""
        n = self.n_tokens(i)
        starts, days = self._days(i)
        total = 2 + self._summary_counts(i, days, config.max_summary) + (n - starts)
        fits = np.flatnonzero(total <= config.context_length)
        return int(fits[0]) if len(fits) else len(starts) - 1

    def prediction_sample(self, i: int, config: SequenceConfig) -> Sample:
        n = self.n_tokens(i)
        if n == 0:
            return self._empty(i)
        starts, days = self._days(i)
        first = self.prediction_start(i, config)
        budget = (
            config.context_length
            - 2
            - int(self._summary_counts(i, days[[first]], config.max_summary)[0])
        )
        start = int(starts[first])
        partial = n - start > budget
        if partial:  # one day alone exceeds the event budget: keep its final tokens
            start = n - budget
        return self._assemble(i, start, n, first, partial, config)

    def window_starts(self, i: int, config: SequenceConfig) -> int:
        """Number of permitted pretraining window starts (day indices 0 to the prediction start)."""
        return 0 if self.n_tokens(i) == 0 else self.prediction_start(i, config) + 1

    def window_sample(self, i: int, first: int, config: SequenceConfig) -> Sample:
        """Pretraining window from day ``first``: as many complete days as fit."""
        n = self.n_tokens(i)
        starts, days = self._days(i)
        if not 0 <= first < len(starts):
            raise ValueError("window start day is outside the history")
        budget = (
            config.context_length
            - 2
            - int(self._summary_counts(i, days[[first]], config.max_summary)[0])
        )
        ends = np.append(starts[1:], n)
        fitting = np.flatnonzero(ends[first:] - starts[first] <= budget)
        if len(fitting):
            return self._assemble(
                i, int(starts[first]), int(ends[first + fitting[-1]]), first, False, config
            )
        stop = int(ends[first])
        return self._assemble(i, stop - budget, stop, first, True, config)


def build_histories(
    encoded: pd.DataFrame,
    cohort: pd.DataFrame,
    tokenizer: FoldTokenizer,
    codes: list[str],
) -> Histories:
    """Group encoded events by patient, in ``cohort`` order. ``cohort`` is a prediction cohort."""
    patients = pd.Index(cohort.index)
    position = pd.Series(np.arange(len(patients)), index=patients)
    pos = encoded["patient_id"].map(position)
    if pos.isna().any():
        raise ValueError("encoded events contain patients outside the cohort")
    frame = encoded.assign(_pos=pos.astype("int64")).sort_values("_pos", kind="stable")
    pos_values = frame["_pos"].to_numpy()
    birth = cohort["birthdate"].to_numpy("datetime64[D]")
    anchor = cohort["anchor_date"].to_numpy("datetime64[D]")
    day = frame["day"].to_numpy("datetime64[D]")
    token_days = (day - birth[pos_values]).astype("int64")
    token_offsets = np.searchsorted(pos_values, np.arange(len(patients) + 1))

    new_day = np.ones(len(frame), dtype=bool)
    new_day[1:] = (pos_values[1:] != pos_values[:-1]) | (token_days[1:] != token_days[:-1])
    day_starts = np.flatnonzero(new_day)
    day_offsets = np.searchsorted(pos_values[day_starts], np.arange(len(patients) + 1))

    diagnoses = frame.loc[frame["first_diagnosis"].ne(""), ["_pos", "first_diagnosis"]].copy()
    diagnoses["day"] = token_days[frame["first_diagnosis"].ne("").to_numpy()]
    first = diagnoses.groupby(["_pos", "first_diagnosis"], sort=False)["day"].min().reset_index()
    first["token"] = [tokenizer.token_id(f"hx:{c}") for c in first["first_diagnosis"]]
    first = first.sort_values(["_pos", "day", "token"], kind="stable")
    hx_pos = first["_pos"].to_numpy()

    last_day = np.full(len(patients), -1, dtype="int64")
    has = np.diff(token_offsets) > 0
    last_day[has] = token_days[token_offsets[1:][has] - 1]
    anchor_day = (anchor - birth).astype("int64")
    days_since_last = np.where(has, anchor_day - last_day, -1)

    source_index = {s: i for i, s in enumerate(SOURCE_ORDER)}
    return Histories(
        patients=patients,
        token_offsets=token_offsets.astype("int64"),
        tokens=frame["token_id"].to_numpy("int32"),
        sources=frame["source"].map(source_index).to_numpy("int8"),
        token_days=token_days.astype("int32"),
        day_offsets=day_offsets.astype("int64"),
        day_starts=day_starts.astype("int64"),
        hx_offsets=np.searchsorted(hx_pos, np.arange(len(patients) + 1)).astype("int64"),
        hx_tokens=first["token"].to_numpy("int32"),
        hx_days=first["day"].to_numpy("int32"),
        hx_target=first["first_diagnosis"].isin(codes).to_numpy(),
        male=cohort["gender"].eq("M").to_numpy("int8"),
        age_at_anchor=cohort["age_at_anchor"].to_numpy("float32"),
        days_since_last=days_since_last.astype("int32"),
        sex_ids=(tokenizer.token_id(SEX_TOKENS["F"]), tokenizer.token_id(SEX_TOKENS["M"])),
        sep_id=tokenizer.token_id(SEP),
    )


_ARRAYS = (
    "token_offsets",
    "tokens",
    "sources",
    "token_days",
    "day_offsets",
    "day_starts",
    "hx_offsets",
    "hx_tokens",
    "hx_days",
    "hx_target",
    "male",
    "age_at_anchor",
    "days_since_last",
)


def save_histories(histories: Histories, path: Path, provenance: dict[str, Any]) -> None:
    """Write arrays and a manifest. Loaders compare the manifest with the expected provenance."""
    arrays = {name: getattr(histories, name) for name in _ARRAYS}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, patients=np.asarray(histories.patients, dtype=str), **arrays)
    tmp.replace(path)
    manifest = {
        "version": HISTORY_VERSION,
        **provenance,
        "sex_ids": list(histories.sex_ids),
        "sep_id": histories.sep_id,
        "file_sha256": file_hash(path),
    }
    write_json(path.with_suffix(".manifest.json"), manifest)


def load_histories(path: Path, expected: dict[str, Any]) -> Histories:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"{path} or its manifest is missing; run `make tokenize`")
    manifest = json.loads(manifest_path.read_text())
    stale = {k for k, v in expected.items() if manifest.get(k) != v}
    if manifest.get("version") != HISTORY_VERSION or manifest.get("file_sha256") != file_hash(path):
        stale.add("file")
    if stale:
        raise ValueError(f"{path} is stale ({', '.join(sorted(stale))}); run `make tokenize`")
    with np.load(path) as data:
        arrays = {name: data[name] for name in _ARRAYS}
        patients = pd.Index(data["patients"].astype(str), name="patient_id")
    sex_ids = manifest["sex_ids"]
    return Histories(
        patients=patients, sex_ids=(sex_ids[0], sex_ids[1]), sep_id=manifest["sep_id"], **arrays
    )


def retention_report(
    histories: Histories,
    rows: np.ndarray,
    tokenizer: FoldTokenizer,
    config: SequenceConfig,
    labels: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Sequence, truncation, summary, unknown-token, and recency statistics (M3 to M5, M7)."""
    records = []
    unk_ids = np.array([tokenizer.token_id(f"[UNK:{PREFIX[s]}]") for s in SOURCE_ORDER])
    lost_by_source = np.zeros(len(SOURCE_ORDER), dtype="int64")
    all_by_source = np.zeros(len(SOURCE_ORDER), dtype="int64")
    unk_by_source = np.zeros(len(SOURCE_ORDER), dtype="int64")
    for i in rows:
        sample = histories.prediction_sample(int(i), config)
        n = histories.n_tokens(int(i))
        a, b = histories.token_offsets[i], histories.token_offsets[i + 1]
        kept_from = b - sample.n_events
        np.add.at(all_by_source, histories.sources[a:b], 1)
        np.add.at(lost_by_source, histories.sources[a:kept_from], 1)
        unknown = np.isin(histories.tokens[a:b], unk_ids)
        np.add.at(unk_by_source, histories.sources[a:b][unknown], 1)
        days = histories.token_days[a:b]
        records.append(
            {
                "patient_id": histories.patients[i],
                "history_tokens": n,
                "sequence_length": len(sample.tokens),
                "retained_events": sample.n_events,
                "truncated": n > sample.n_events,
                "partial_day": sample.partial_day,
                "summary_tokens": sample.n_summary,
                "summary_candidates": sample.n_candidates,
                "history_years": float(days[-1] - days[0]) / DAYS_PER_YEAR if n else 0.0,
                "retained_years": float(days[-1] - days[kept_from - a]) / DAYS_PER_YEAR
                if n
                else 0.0,
                "days_since_last_event": int(histories.days_since_last[i]),
                "empty": n == 0,
            }
        )
    table = pd.DataFrame(records).set_index("patient_id")

    def quantiles(values: pd.Series) -> dict[str, float]:
        if values.empty:
            return {}
        q = values.quantile([0.0, 0.5, 0.9, 0.99, 1.0])
        return {
            k: round(float(v), 3)
            for k, v in zip(("min", "median", "p90", "p99", "max"), q, strict=True)
        }

    truncated = table.loc[table["truncated"]]
    nonempty = table.loc[~table["empty"]]
    density = nonempty["history_tokens"] / nonempty["history_years"].clip(lower=1.0)
    report: dict[str, Any] = {
        "n_patients": len(table),
        "empty_histories": int(table["empty"].sum()),
        "sequence_length": quantiles(table["sequence_length"]),
        "history_tokens": quantiles(table["history_tokens"]),
        "truncated_patients": int(table["truncated"].sum()),
        "truncated_fraction": round(float(table["truncated"].mean()), 4),
        "partial_day_patients": int(table["partial_day"].sum()),
        "event_tokens_lost_fraction": round(
            float(1 - table["retained_events"].sum() / max(1, table["history_tokens"].sum())), 4
        ),
        "truncated_history_years_median": round(float(truncated["history_years"].median()), 2)
        if len(truncated)
        else None,
        "truncated_retained_years_median": round(float(truncated["retained_years"].median()), 2)
        if len(truncated)
        else None,
        "summary_tokens": quantiles(table["summary_tokens"]),
        "summary_at_cap": int((table["summary_tokens"] >= config.max_summary).sum()),
        "summary_coverage": round(
            float(table["summary_tokens"].sum() / max(1, table["summary_candidates"].sum())), 4
        ),
        "days_since_last_event": quantiles(nonempty["days_since_last_event"]),
        "lost_fraction_by_source": {
            s: round(float(lost_by_source[k] / all_by_source[k]), 4)
            for k, s in enumerate(SOURCE_ORDER)
            if all_by_source[k]
        },
        "unknown_rate_by_source": {
            s: round(float(unk_by_source[k] / all_by_source[k]), 5)
            for k, s in enumerate(SOURCE_ORDER)
            if all_by_source[k]
        },
        "density_tertile_edges": [round(float(x), 2) for x in density.quantile([1 / 3, 2 / 3])]
        if len(density)
        else [],
    }
    if labels is not None:
        positives = labels.loc[table.index].sum(axis=1)
        tertile = (
            pd.qcut(density, 3, labels=["low", "middle", "high"]) if len(density) >= 3 else None
        )
        groups = {
            "truncated": table.index[table["truncated"]],
            "not_truncated": table.index[~table["truncated"]],
            "empty": table.index[table["empty"]],
        }
        if tertile is not None:
            groups.update(
                {f"density_{t}": tertile.index[tertile.eq(t)] for t in tertile.cat.categories}
            )
        report["positives_by_group"] = {
            name: {"patients": len(idx), "label_positives": int(positives.loc[idx].sum())}
            for name, idx in groups.items()
        }
    return report


def histories_provenance(
    tokenizer: FoldTokenizer, events_manifest_sha256: str, patients: pd.Index
) -> dict[str, Any]:
    return {
        "tokenizer_sha256": tokenizer.sha256,
        "events_manifest_sha256": events_manifest_sha256,
        "patients_sha256": object_hash(list(patients)),
    }
