"""D31 evidence: which date of a STOP < START medication row agrees with an encounter (train).

A date agrees when it is the calendar day of any encounter of that patient, start or stop.
Run from the repository root.
"""

import pandas as pd

root = "../research_intern_coding_test"
splits = pd.read_csv(f"{root}/patient_splits.csv")
train = set(splits.loc[splits["split"].eq("train"), "Id"])
m = pd.read_csv(f"{root}/train_val/medications.csv", dtype=str)
e = pd.read_csv(f"{root}/train_val/encounters.csv", dtype=str, usecols=["PATIENT", "START", "STOP"])
m = m[m["PATIENT"].isin(train) & m["STOP"].notna()].copy()
days = pd.concat([e[["PATIENT", "START"]].rename(columns={"START": "t"}),
                  e[["PATIENT", "STOP"]].rename(columns={"STOP": "t"})]).dropna()
days["day"] = pd.to_datetime(days["t"], utc=True).dt.floor("D")
known = set(zip(days["PATIENT"], days["day"], strict=True))
for column in ("START", "STOP"):
    m[column + "_day"] = pd.to_datetime(m[column], utc=True).dt.floor("D")
    pairs = zip(m["PATIENT"], m[column + "_day"], strict=True)
    m[column + "_known"] = [pair in known for pair in pairs]
bad = m["STOP_day"] < m["START_day"]
for name, rows in (("invalid", m[bad]), ("valid", m[~bad])):
    print(
        f"{name}: rows {len(rows)}, start agrees {int(rows['START_known'].sum())} "
        f"({rows['START_known'].mean():.3f}), stop agrees {int(rows['STOP_known'].sum())} "
        f"({rows['STOP_known'].mean():.3f})"
    )
