"""B12 check: does having a genomics or imaging file correlate with the split or the outcomes?

Reads the organisers' folder listings and the processed cohort and labels; fits nothing. Writes
``reports/dataset/modality_presence.json``. Run from the repository root:
``.venv/bin/python scripts/checks/modality_presence.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

RAW, PROC, OUT = Path("data/raw"), Path("data/processed"), Path("reports/dataset")


def main() -> None:
    cohort = pd.read_parquet(PROC / "cohort.parquet")
    labels = pd.read_parquet(PROC / "labels.parquet")
    codes = list(labels.columns)
    genomics = {p.name.removesuffix("_dna.csv") for p in (RAW / "genomics").glob("*_dna.csv")}
    imaging = {p.stem for p in (RAW / "imaging").glob("*.dcm")}
    cohort["has_genomics"] = cohort.index.isin(genomics)
    cohort["has_imaging"] = cohort.index.isin(imaging)
    result: dict[str, object] = {
        "scope": "file presence only; no file content is read; labels are train and val"
    }
    labelled = cohort[cohort["split"].isin(["train", "val"])]
    train = cohort[cohort["split"] == "train"]
    for modality in ("has_genomics", "has_imaging"):
        table = pd.crosstab(cohort["split"], cohort[modality])
        present = table.get(True, pd.Series(0, index=table.index))
        entry: dict[str, object] = {
            "by_split": {
                split: {
                    "n": int(table.loc[split].sum()),
                    "present": int(present.loc[split]),
                    "fraction": round(float(present.loc[split] / table.loc[split].sum()), 4),
                }
                for split in table.index
            },
            "chi2_split_vs_presence_p": float(stats.chi2_contingency(table)[1]),
        }
        n_pos = labels.loc[labelled.index, codes].sum(axis=1)
        with_file, without = n_pos[labelled[modality]], n_pos[~labelled[modality]]
        entry["labelled_positives_per_patient"] = {
            "present": round(float(with_file.mean()), 3),
            "absent": round(float(without.mean()), 3),
            "mannwhitney_p": float(stats.mannwhitneyu(with_file, without).pvalue),
        }
        entry["labelled_age_at_anchor_mean"] = {
            "present": round(float(labelled.loc[labelled[modality], "age_at_anchor"].mean()), 1),
            "absent": round(float(labelled.loc[~labelled[modality], "age_at_anchor"].mean()), 1),
        }
        rows = []
        y = labels.loc[train.index, codes]
        for code in codes:
            a = int(y.loc[train[modality], code].sum())
            b = int(train[modality].sum()) - a
            c = int(y.loc[~train[modality], code].sum())
            d = int((~train[modality]).sum()) - c
            rows.append(
                {
                    "code": code,
                    "prevalence_present": round(a / max(1, a + b), 4),
                    "prevalence_absent": round(c / max(1, c + d), 4),
                    "fisher_p": float(stats.fisher_exact([[a, b], [c, d]])[1]),
                }
            )
        rows.sort(key=lambda r: r["fisher_p"])
        entry["train_targets_most_associated"] = rows[:6]
        entry["train_targets_fisher_p_below_0.01"] = sum(r["fisher_p"] < 0.01 for r in rows)
        result[modality] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "modality_presence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "scope"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
