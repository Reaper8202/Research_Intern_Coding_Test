"""P3b: a fold tokenizer fitted on the fold's fit patients only (model plan sections 8.2 to 8.6).

Token forms:

======================== ===================================================================
source                   token
======================== ===================================================================
conditions               ``cond:start:<code>``, ``cond:stop:<code>``
medications              ``med:start:<code>``, ``med:stop:<code>``
care plans, allergies,   ``careplans:start:<code>`` (and ``stop``); same for ``allergies``
devices                  and ``devices``
procedures               ``proc:<code>``
immunizations            ``imm:<code>``
numeric observations     ``obs:<code>:<unit>:q<bin>``
text observations        ``obs:<code>:<normalized value>``
observation fallbacks    ``obs:<code>:missing_value``, ``obs:<code>:other_value``
imaging studies          ``img:<modality>|<body site>``
encounters               ``enc:<class>:<code>``
structure                ``[PAD]``, ``[SEP]``, ``[SEX:F]``, ``[SEX:M]``, ``[UNK:<prefix>]``,
                         ``hx:<condition code>``
genomics (D34)           ``[DNA]`` (a genotype file exists), ``dna:<site>:<variant alleles>``
brain MRI (D34)          ``[IMG]``; the model replaces its embedding with the image encoding
======================== ===================================================================

A base code (for example ``med:<code>`` or ``obs:<code>``) is eligible when at least
``min_patients`` fit patients have it. Unknown base codes map to ``[UNK:<prefix>]``. Value tokens
also need ``min_patients`` fit patients; rarer values and unseen units of an eligible observation
code map to ``other_value``. Nothing is fitted on descriptions, labels, or held-out patients.
A genotype token also needs ``min_patients`` fit carriers; rarer genotypes are left out. A
tokenizer fitted without modalities has the same vocabulary and hash as before D34.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ehrforecast.contracts import validate_ids
from ehrforecast.events import KIND_ORDER, SOURCE_ORDER
from ehrforecast.numeric import UNIT_RULE_VERSION, assign_bins, fit_bin_edges
from ehrforecast.provenance import object_hash

TOKENIZER_VERSION = "fold-tokenizer-v1"
REFILL_RULE_VERSION = "merge-stop-restart-v1"

PREFIX = {
    "encounters": "enc",
    "conditions": "cond",
    "medications": "med",
    "careplans": "careplans",
    "allergies": "allergies",
    "devices": "devices",
    "procedures": "proc",
    "immunizations": "imm",
    "observations": "obs",
    "imaging_studies": "img",
}
INTERVAL_SOURCES = ("conditions", "medications", "careplans", "allergies", "devices")
PAD, SEP = "[PAD]", "[SEP]"
SEX_TOKENS = {"F": "[SEX:F]", "M": "[SEX:M]"}
SPECIAL_TOKENS = (PAD, SEP, *SEX_TOKENS.values(), *(f"[UNK:{p}]" for p in PREFIX.values()))
DNA, IMG = "[DNA]", "[IMG]"
NO_UNIT = "nounit"

# Age features: age/100 plus sine and cosine terms for eight periods from 0.5 to 200 years.
AGE_PERIODS = np.geomspace(0.5, 200.0, 8)
N_AGE_FEATURES = 1 + 2 * len(AGE_PERIODS)


def age_features(ages: np.ndarray) -> np.ndarray:
    ages = np.asarray(ages, dtype="float64")
    angles = 2 * np.pi * ages[..., None] / AGE_PERIODS
    return np.concatenate([ages[..., None] / 100.0, np.sin(angles), np.cos(angles)], -1).astype(
        "float32"
    )


def normalize_text(values: pd.Series) -> pd.Series:
    """Lowercase and collapse whitespace. No other rewriting."""
    return values.astype("string").str.lower().str.split().str.join(" ")


def normalize_unit(values: pd.Series) -> pd.Series:
    """Strip surrounding whitespace and preserve case (``separate-literal-units-v1``)."""
    units = values.astype("string").fillna("").str.strip()
    return units.mask(units.eq(""), NO_UNIT)


def compress_refills(events: pd.DataFrame, max_gap_days: float) -> tuple[pd.DataFrame, int]:
    """Merge a medication stop with a restart of the same drug no more than ``max_gap_days`` later.

    Both the stop and the restart are removed, so the treatment span continues. A stop never
    merges with its own start (same ``interval_start``) or with a start before it. Each start
    merges at most once. Apply this to permitted histories only: a restart after the anchor is
    not in the table, so it cannot remove a pre-anchor stop. Returns the events and the number
    of merged pairs.
    """
    med = events.loc[events["source"].astype(str).eq("medications")]
    kind = med["kind"].astype(str)
    columns = ["patient_id", "code", "time", "interval_start"]
    stops = med.loc[kind.eq("stop"), columns].sort_values(["patient_id", "code", "time"])
    starts = med.loc[kind.eq("start"), columns].sort_values(["patient_id", "code", "time"])
    gap = np.timedelta64(round(max_gap_days * 86_400), "s")
    start_groups = {key: group for key, group in starts.groupby(["patient_id", "code"], sort=False)}
    removed: list[Any] = []
    for key, group in stops.groupby(["patient_id", "code"], sort=False):
        candidates = start_groups.get(key)
        if candidates is None:
            continue
        start_times = candidates["time"].to_numpy("datetime64[s]")
        start_index = candidates.index.to_numpy()
        used = np.zeros(len(candidates), dtype=bool)
        for stop_index, stop_time, own_start in zip(
            group.index.to_numpy(),
            group["time"].to_numpy("datetime64[s]"),
            group["interval_start"].to_numpy("datetime64[s]"),
            strict=True,
        ):
            low = np.searchsorted(start_times, stop_time, side="left")
            high = np.searchsorted(start_times, stop_time + gap, side="right")
            for j in range(low, high):
                if not used[j] and start_times[j] != own_start:
                    used[j] = True
                    removed.extend((stop_index, start_index[j]))
                    break
    return events.drop(index=removed), len(removed) // 2


def _base_keys(events: pd.DataFrame) -> pd.Series:
    source = events["source"].astype(str)
    prefix = source.map(PREFIX)
    code = events["code"].astype(str)
    base = prefix + ":" + code
    encounters = source.eq("encounters")
    encounter_class = events.loc[encounters, "value_text"].astype("string").fillna("unknown")
    base[encounters] = "enc:" + encounter_class.astype(str) + ":" + code[encounters]
    return base


@dataclass(frozen=True)
class FoldTokenizer:
    """Fitted vocabulary and value rules. Use :meth:`fit`, :meth:`save`, and :meth:`load`."""

    vocabulary: tuple[str, ...]
    eligible: frozenset[str]
    numeric_bins: dict[str, dict[str, Any]]  # "<code>\\t<unit>" -> edges and per-bin tokens
    text_values: dict[str, tuple[str, ...]]  # observation code -> retained normalized values
    min_patients: int
    refill_gap_days: float
    fit_patients_sha256: str
    input_sha256: str
    genomics: bool = False
    imaging: bool = False

    def __post_init__(self) -> None:
        index = {token: i for i, token in enumerate(self.vocabulary)}
        if len(index) != len(self.vocabulary) or self.vocabulary[0] != PAD:
            raise ValueError("tokenizer vocabulary must be unique and start with [PAD]")
        object.__setattr__(self, "_index", index)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        events: pd.DataFrame,
        fit_patients: pd.Index,
        *,
        min_patients: int = 5,
        refill_gap_days: float = 1.0,
        input_sha256: str = "",
        genomics: pd.DataFrame | None = None,
        imaging: bool = False,
    ) -> FoldTokenizer:
        """Fit on fit patients' permitted histories. Other patients' rows are ignored.

        ``genomics`` (``modalities.read_genomics`` rows) adds genotype tokens; ``imaging`` adds
        ``[IMG]``. Both are fitted on fit patients only.
        """
        validate_ids(fit_patients)
        fit_events = events.loc[events["patient_id"].isin(fit_patients)]
        fit_events, _ = compress_refills(fit_events, refill_gap_days)
        base = _base_keys(fit_events)
        patients_per_base = fit_events["patient_id"].groupby(base).nunique()
        eligible = frozenset(patients_per_base.index[patients_per_base >= min_patients])

        observations = fit_events.loc[fit_events["source"].astype(str).eq("observations")]
        observations = observations.loc[("obs:" + observations["code"].astype(str)).isin(eligible)]
        numeric = observations.loc[observations["value_num"].notna()]
        numeric_bins: dict[str, dict[str, Any]] = {}
        for (code, unit), group in numeric.groupby(
            [numeric["code"].astype(str), normalize_unit(numeric["unit"])], sort=True
        ):
            fit_values = group["value_num"].to_numpy(dtype="float64")
            edges = fit_bin_edges(fit_values)
            bins = assign_bins(fit_values, edges)
            counts = pd.Series(group["patient_id"].to_numpy()).groupby(bins).nunique()
            n_bins = max(1, len(edges) - 1)
            patients = [int(counts.get(b, 0)) for b in range(n_bins)]
            bin_tokens = [
                f"obs:{code}:{unit}:q{b}"
                if patients[b] >= min_patients
                else f"obs:{code}:other_value"
                for b in range(n_bins)
            ]
            numeric_bins[f"{code}\t{unit}"] = {
                "edges": edges.tolist(),
                "fit_patients_per_bin": patients,
                "tokens": bin_tokens,
            }
        text = observations.loc[
            observations["value_num"].isna() & observations["value_text"].notna()
        ]
        text_patients = text.groupby(
            [text["code"].astype(str), normalize_text(text["value_text"])], sort=True
        )["patient_id"].nunique()
        text_values: dict[str, tuple[str, ...]] = {}
        counted = text_patients.reset_index(name="n")
        counted.columns = ["code", "value", "n"]
        for code, value in counted.loc[counted["n"] >= min_patients, ["code", "value"]].to_numpy():
            text_values[str(code)] = (*text_values.get(str(code), ()), str(value))

        tokens: set[str] = set()
        for key in eligible:
            prefix, rest = key.split(":", 1)
            if prefix in {PREFIX[s] for s in INTERVAL_SOURCES}:
                tokens.update({f"{prefix}:start:{rest}", f"{prefix}:stop:{rest}"})
                if prefix == "cond":
                    tokens.add(f"hx:{rest}")
            elif prefix == "obs":
                tokens.update({f"obs:{rest}:missing_value", f"obs:{rest}:other_value"})
            else:
                tokens.add(key)
        for rule in numeric_bins.values():
            tokens.update(rule["tokens"])
        for code, values in text_values.items():
            tokens.update(f"obs:{code}:{value}" for value in values)
        specials: list[str] = list(SPECIAL_TOKENS)
        if genomics is not None:
            carried = genomics.loc[
                genomics["patient_id"].isin(fit_patients) & genomics["count"].gt(0)
            ]
            carriers = carried["patient_id"].groupby(_genotype_tokens(carried)).nunique()
            tokens.update(carriers.index[carriers >= min_patients])
            specials.append(DNA)
        if imaging:
            specials.append(IMG)
        return cls(
            vocabulary=(*specials, *sorted(tokens)),
            eligible=eligible,
            numeric_bins=numeric_bins,
            text_values=text_values,
            min_patients=min_patients,
            refill_gap_days=refill_gap_days,
            fit_patients_sha256=object_hash(sorted(fit_patients.tolist())),
            input_sha256=input_sha256,
            genomics=genomics is not None,
            imaging=imaging,
        )

    # ------------------------------------------------------------------ lookup
    @property
    def size(self) -> int:
        return len(self.vocabulary)

    def token_id(self, token: str) -> int:
        return int(self._index[token])  # type: ignore[attr-defined]

    def ids(self, tokens: pd.Series) -> np.ndarray:
        return tokens.map(self._index).to_numpy(dtype="int64")  # type: ignore[attr-defined]

    def input_only_ids(self) -> np.ndarray:
        """Structure, summary, and modality tokens: model inputs, never next-token targets."""
        return np.array(
            [
                i
                for i, token in enumerate(self.vocabulary)
                if token in {PAD, SEP, DNA, IMG, *SEX_TOKENS.values()}
                or token.startswith(("hx:", "dna:"))
            ],
            dtype="int64",
        )

    def genomic_tokens(self, genomics: pd.DataFrame) -> pd.DataFrame:
        """``patient_id`` and ``token_id``: ``[DNA]`` for each patient, then known genotypes.

        Genotype tokens follow in token-ID order. Genotypes outside the vocabulary are left out.
        """
        if not self.genomics:
            raise ValueError("this tokenizer was fitted without genomics")
        patients = pd.unique(genomics["patient_id"].astype(str))
        marker = pd.DataFrame({"patient_id": patients, "token_id": self.token_id(DNA), "rank": 0})
        carried = genomics.loc[genomics["count"].gt(0)]
        ids = _genotype_tokens(carried).map(self._index)  # type: ignore[attr-defined]
        known = ids.notna()
        variants = pd.DataFrame(
            {
                "patient_id": carried.loc[known, "patient_id"].astype(str),
                "token_id": ids[known].astype("int64"),
                "rank": 1,
            }
        )
        out = pd.concat([marker, variants], ignore_index=True)
        out = out.sort_values(["patient_id", "rank", "token_id"], kind="stable")
        return out[["patient_id", "token_id"]].reset_index(drop=True)

    # ------------------------------------------------------------------ encoding
    def tokens(self, events: pd.DataFrame) -> pd.Series:
        """Map permitted, compressed event rows to token strings (unknowns to ``[UNK:*]``)."""
        source = events["source"].astype(str)
        prefix = source.map(PREFIX)
        code = events["code"].astype(str)
        kind = events["kind"].astype(str)
        base = _base_keys(events)
        known = base.isin(self.eligible)
        out = pd.Series(pd.NA, index=events.index, dtype="string")

        interval = source.isin(INTERVAL_SOURCES)
        out[interval] = prefix[interval] + ":" + kind[interval] + ":" + code[interval]
        simple = source.isin(["procedures", "immunizations", "imaging_studies", "encounters"])
        out[simple] = base[simple]

        obs = source.eq("observations")
        numeric = obs & events["value_num"].notna()
        text = obs & events["value_num"].isna() & events["value_text"].notna()
        missing = obs & events["value_num"].isna() & events["value_text"].isna()
        out[missing] = "obs:" + code[missing] + ":missing_value"
        if text.any():
            values = normalize_text(events.loc[text, "value_text"])
            retained = [
                value in self.text_values.get(c, ())
                for c, value in zip(code[text], values, strict=True)
            ]
            out[text] = np.where(
                retained,
                "obs:" + code[text] + ":" + values,
                "obs:" + code[text] + ":other_value",
            )
        if numeric.any():
            frame = pd.DataFrame(
                {
                    "code": code[numeric],
                    "unit": normalize_unit(events.loc[numeric, "unit"]),
                    "value": events.loc[numeric, "value_num"].astype("float64"),
                }
            )
            frame["token"] = "obs:" + frame["code"] + ":other_value"
            for (c, unit), group in frame.groupby(["code", "unit"], sort=False):
                rule = self.numeric_bins.get(f"{c}\t{unit}")
                if rule is None:
                    continue  # an unseen unit keeps the known-code fallback
                bins = assign_bins(group["value"].to_numpy(), np.asarray(rule["edges"]))
                frame.loc[group.index, "token"] = np.asarray(rule["tokens"], dtype=object)[bins]
            out[numeric] = frame["token"]

        unknown = ~known | ~out.isin(self._index.keys())  # type: ignore[attr-defined]
        out[unknown] = "[UNK:" + prefix[unknown] + "]"
        if out.isna().any():
            raise ValueError("an event row produced no token")
        return out

    def encode(self, events: pd.DataFrame) -> pd.DataFrame:
        """Compress refills, tokenize, and sort deterministically within each patient day.

        Returns ``patient_id``, ``day``, ``source``, ``token_id``, and ``first_diagnosis``
        (the condition code of a known ``cond:start`` row, else empty). Identical tokens on the
        same day are kept once. Source-file row order does not change the result.
        """
        compressed, _ = compress_refills(events, self.refill_gap_days)
        tokens = self.tokens(compressed)
        source = compressed["source"].astype(str)
        out = pd.DataFrame(
            {
                "patient_id": compressed["patient_id"].astype(str),
                "day": compressed["time"].dt.normalize(),
                "source": source,
                "source_rank": source.map({s: i for i, s in enumerate(SOURCE_ORDER)}),
                "kind_rank": compressed["kind"]
                .astype(str)
                .map({k: i for i, k in enumerate(KIND_ORDER)}),
                "token_id": self.ids(tokens),
            }
        )
        condition_start = source.eq("conditions") & tokens.str.startswith("cond:start:")
        out["first_diagnosis"] = compressed["code"].astype(str).where(condition_start, "")
        out = out.sort_values(
            ["patient_id", "day", "source_rank", "kind_rank", "token_id"], kind="stable"
        )
        out = out.drop_duplicates(["patient_id", "day", "token_id"])
        return out.drop(columns=["source_rank", "kind_rank"]).reset_index(drop=True)

    # ------------------------------------------------------------------ files
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": TOKENIZER_VERSION,
            "unit_rule_version": UNIT_RULE_VERSION,
            "refill_rule_version": REFILL_RULE_VERSION,
            "min_patients": self.min_patients,
            "refill_gap_days": self.refill_gap_days,
            "fit_patients_sha256": self.fit_patients_sha256,
            "input_sha256": self.input_sha256,
            "vocabulary": list(self.vocabulary),
            "eligible": sorted(self.eligible),
            "numeric_bins": self.numeric_bins,
            "text_values": {k: list(v) for k, v in sorted(self.text_values.items())},
            # Only multimodal tokenizers carry this key, so earlier hashes are unchanged.
            **(
                {"modalities": {"genomics": self.genomics, "imaging": self.imaging}}
                if self.genomics or self.imaging
                else {}
            ),
        }

    @property
    def sha256(self) -> str:
        return object_hash(self.to_dict())

    def save(self, path: Path) -> None:
        from ehrforecast.provenance import write_json

        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> FoldTokenizer:
        data = json.loads(path.read_text())
        if (
            data.get("version") != TOKENIZER_VERSION
            or data.get("unit_rule_version") != UNIT_RULE_VERSION
            or data.get("refill_rule_version") != REFILL_RULE_VERSION
        ):
            raise ValueError(f"{path} was written by an incompatible tokenizer version")
        return cls(
            vocabulary=tuple(data["vocabulary"]),
            eligible=frozenset(data["eligible"]),
            numeric_bins=data["numeric_bins"],
            text_values={k: tuple(v) for k, v in data["text_values"].items()},
            min_patients=int(data["min_patients"]),
            refill_gap_days=float(data["refill_gap_days"]),
            fit_patients_sha256=data["fit_patients_sha256"],
            input_sha256=data["input_sha256"],
            genomics=bool(data.get("modalities", {}).get("genomics", False)),
            imaging=bool(data.get("modalities", {}).get("imaging", False)),
        )


def _genotype_tokens(genomics: pd.DataFrame) -> pd.Series:
    return "dna:" + genomics["site"].astype(str) + ":" + genomics["count"].astype(str)
