"""Column allowlists for every EHR table.

Each table is read through an explicit allowlist, so a column is used only when it is safe at
prediction time. Columns are left out for one of these reasons:

* They summarise the whole record, including the years after the anchor. Examples:
  ``patients.DEATHDATE``, ``patients.HEALTHCARE_EXPENSES``, ``patients.MARITAL`` (end-of-record
  state), and ``medications.DISPENSES`` / ``TOTALCOST``. The test files keep the untruncated
  dispense totals of medications that continue past the anchor, so these columns leak even there.
* They encode dates outside the event time (``devices.UDI`` holds manufacture and expiry dates).
* They are billing or administrative fields with no clinical content (costs, payer, provider).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSpec:
    """How one event table maps onto the shared event format."""

    name: str
    time_col: str
    columns: tuple[str, ...]
    stop_col: str | None = None
    code_cols: tuple[str, ...] = ("CODE",)

    @property
    def is_interval(self) -> bool:
        return self.stop_col is not None


PATIENT_COL = "PATIENT"

PATIENT_COLUMNS = ("Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY")

EVENT_TABLES: dict[str, TableSpec] = {
    spec.name: spec
    for spec in (
        TableSpec(
            name="encounters",
            time_col="START",
            stop_col="STOP",
            columns=("Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS", "CODE", "DESCRIPTION"),
        ),
        TableSpec(
            name="conditions",
            time_col="START",
            stop_col="STOP",
            columns=("START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"),
        ),
        TableSpec(
            name="medications",
            time_col="START",
            stop_col="STOP",
            columns=("START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE"),
        ),
        TableSpec(
            name="careplans",
            time_col="START",
            stop_col="STOP",
            columns=("START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE"),
        ),
        TableSpec(
            name="allergies",
            time_col="START",
            stop_col="STOP",
            columns=("START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"),
        ),
        TableSpec(
            name="devices",
            time_col="START",
            stop_col="STOP",
            columns=("START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"),
        ),
        TableSpec(
            name="procedures",
            time_col="DATE",
            columns=("DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE"),
        ),
        TableSpec(
            name="immunizations",
            time_col="DATE",
            columns=("DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"),
        ),
        TableSpec(
            name="observations",
            time_col="DATE",
            columns=(
                "DATE",
                "PATIENT",
                "ENCOUNTER",
                "CODE",
                "DESCRIPTION",
                "VALUE",
                "UNITS",
                "TYPE",
            ),
        ),
        TableSpec(
            name="imaging_studies",
            time_col="DATE",
            columns=(
                "Id",
                "DATE",
                "PATIENT",
                "ENCOUNTER",
                "BODYSITE_CODE",
                "BODYSITE_DESCRIPTION",
                "MODALITY_CODE",
                "MODALITY_DESCRIPTION",
            ),
            code_cols=("MODALITY_CODE", "BODYSITE_CODE"),
        ),
    )
}
