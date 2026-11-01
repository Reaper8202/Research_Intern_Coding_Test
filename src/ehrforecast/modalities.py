"""Optional modalities (D34): germline genomics and brain MRI volumes.

The task supplies both un-split and named by patient ``Id``: ``genomics/<Id>_dna.csv`` and
``imaging/<Id>.dcm``. This step reads every file once and writes, to the processed dataset:

* ``genomics.parquet``: one row per patient and variant site. ``count`` is the number of variant
  alleles (0, 1, or 2) over the two strands. Genotypes are fixed at birth and carry no dates.
* ``imaging.npz``: the study date and a ``size``-cubed float16 volume per readable DICOM file.
  Each volume is resampled with area averaging and standardized with its own mean and standard
  deviation, so no statistic is shared between patients.
* ``modalities.manifest.json``: versions, source and output hashes, and unreadable files.

The anchor is not applied here. ``sequences.build_histories`` keeps a scan only when its study
date is before the anchor day, the rule of every event table (``truncation``). The fold decides
later which patients enter training, exactly as for the event table. Official val and test files
are read only to fill the cache; training uses fit patients only.

Usage::

    python -m ehrforecast.modalities --raw data/raw --out data/processed \
        --reports reports/dataset --image-size 64
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

from ehrforecast.contracts import validate_ids
from ehrforecast.logging_utils import setup_logging
from ehrforecast.provenance import file_hash, object_hash, write_json

log = logging.getLogger("ehrforecast.modalities")

MODALITIES_VERSION = "modalities-v1"
GENOMICS_FILE = "genomics.parquet"
IMAGING_FILE = "imaging.npz"
MODALITIES_MANIFEST = "modalities.manifest.json"
GENOMICS_SUFFIX = "_dna.csv"
CLIP = 5.0  # standardized intensities are clipped to [-5, 5]


# ---------------------------------------------------------------------- genomics
def read_genotypes(path: Path) -> pd.Series:
    """Variant-allele count per site (``INDEX_PREFIX``) for one ``<Id>_dna.csv`` file."""
    frame = pd.read_csv(path, dtype=str, usecols=["INDEX_PREFIX", "STRAND", "VARIANT"])
    variant = frame["VARIANT"].str.strip().str.lower().map({"true": 1, "false": 0})
    if variant.isna().any():
        raise ValueError(f"{path}: VARIANT must be true or false")
    strands = frame.groupby("INDEX_PREFIX")["STRAND"].agg(["size", "nunique"])
    if not (strands["size"].eq(2) & strands["nunique"].eq(2)).all():
        raise ValueError(f"{path}: every site needs exactly two strands")
    return variant.groupby(frame["INDEX_PREFIX"]).sum().astype("int8")


def read_genomics(directory: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """All genotype files: ``patient_id``, ``site``, ``count``; and the file hashes."""
    parts, hashes = [], {}
    for path in sorted(directory.glob(f"*{GENOMICS_SUFFIX}")):
        counts = read_genotypes(path)
        parts.append(
            pd.DataFrame(
                {
                    "patient_id": path.name.removesuffix(GENOMICS_SUFFIX),
                    "site": counts.index.astype(str),
                    "count": counts.to_numpy(),
                }
            )
        )
        hashes[path.name] = file_hash(path)
    if not parts:
        raise ValueError(f"{directory} holds no {GENOMICS_SUFFIX} files")
    genomics = pd.concat(parts, ignore_index=True)
    validate_ids(pd.Index(genomics["patient_id"].unique()), "genomics patient IDs")
    return genomics, hashes


# ---------------------------------------------------------------------- imaging
def resample_volume(pixels: np.ndarray, size: int) -> np.ndarray:
    """Area-average a 3-D array to ``size`` cubed, then standardize it with its own statistics."""
    if pixels.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {pixels.shape}")
    tensor = torch.from_numpy(pixels.astype("float32"))[None, None]
    volume = F.interpolate(tensor, size=(size, size, size), mode="area")[0, 0].numpy()
    std = float(volume.std())
    if not np.isfinite(std) or std == 0:
        raise ValueError("the volume is constant or not finite")
    standardized = np.clip((volume - float(volume.mean())) / std, -CLIP, CLIP)
    return standardized.astype("float16")


def read_volume(data: bytes, size: int) -> tuple[pd.Timestamp, np.ndarray]:
    """Study date and resampled volume of one DICOM file's bytes."""
    dataset = pydicom.dcmread(io.BytesIO(data))
    study_date = pd.to_datetime(str(dataset.StudyDate), format="%Y%m%d")
    return study_date, resample_volume(dataset.pixel_array, size)


def read_imaging(directory: Path, size: int) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Study dates (index ``patient_id``), volumes in the same order, and file records."""
    rows, volumes, records = [], [], {}
    for path in sorted(directory.glob("*.dcm")):
        data = path.read_bytes()
        record: dict[str, Any] = {"sha256": hashlib.sha256(data).hexdigest()}
        try:
            study_date, volume = read_volume(data, size)
        except Exception as error:  # a damaged file is recorded and skipped, never guessed
            record["error"] = f"{type(error).__name__}: {str(error)[:200]}"
            log.warning("Skipping %s: %s", path.name, record["error"])
        else:
            rows.append({"patient_id": path.stem, "study_date": study_date})
            volumes.append(volume)
        records[path.name] = record
    dates = pd.DataFrame(rows).set_index("patient_id")
    validate_ids(dates.index, "imaging patient IDs")
    return dates, np.stack(volumes), records


# ---------------------------------------------------------------------- cache
@dataclass(frozen=True)
class Modalities:
    """Genotypes and resampled volumes of every patient who has them (all splits)."""

    genomics: pd.DataFrame  # patient_id, site, count
    study_dates: pd.Series  # index patient_id
    volumes: np.ndarray  # (patients, size, size, size) float16, in ``study_dates`` order
    image_size: int
    manifest_sha256: str
    _row: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_row", {p: i for i, p in enumerate(self.study_dates.index)})

    def volume(self, patient: str) -> np.ndarray:
        return self.volumes[self._row[patient]]


def build_modalities(raw_dir: Path, out_dir: Path, size: int) -> dict[str, Any]:
    genomics, genomics_hashes = read_genomics(raw_dir / "genomics")
    dates, volumes, image_records = read_imaging(raw_dir / "imaging", size)
    out_dir.mkdir(parents=True, exist_ok=True)
    genomics.to_parquet(out_dir / GENOMICS_FILE, index=False)
    tmp = out_dir / "imaging.tmp.npz"
    np.savez(
        tmp,
        patients=np.asarray(dates.index, dtype=str),
        study_dates=dates["study_date"].to_numpy("datetime64[D]"),
        volumes=volumes,
    )
    tmp.replace(out_dir / IMAGING_FILE)
    manifest = {
        "version": MODALITIES_VERSION,
        "image_size": size,
        "genomics_sources_sha256": object_hash(genomics_hashes),
        "imaging_sources_sha256": object_hash({k: v["sha256"] for k, v in image_records.items()}),
        "unreadable_images": {k: v["error"] for k, v in image_records.items() if "error" in v},
        "genomics_patients": int(genomics["patient_id"].nunique()),
        "genomics_sites": int(genomics["site"].nunique()),
        "image_patients": len(dates),
        "genomics_sha256": file_hash(out_dir / GENOMICS_FILE),
        "imaging_sha256": file_hash(out_dir / IMAGING_FILE),
    }
    write_json(out_dir / MODALITIES_MANIFEST, manifest)
    return manifest


def load_modalities(dataset_dir: Path, image_size: int) -> Modalities:
    """Load the cache after checking its manifest, versions, image size, and output hashes."""
    manifest_path = dataset_dir / MODALITIES_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path} is missing; run `make modalities`")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != MODALITIES_VERSION:
        raise ValueError(f"{manifest_path} has an unknown version; run `make modalities`")
    if int(manifest["image_size"]) != image_size:
        raise ValueError(
            f"the imaging cache holds {manifest['image_size']}-voxel volumes, not {image_size}"
        )
    for name, key in ((GENOMICS_FILE, "genomics_sha256"), (IMAGING_FILE, "imaging_sha256")):
        if file_hash(dataset_dir / name) != manifest[key]:
            raise ValueError(f"{dataset_dir / name} does not match its manifest")
    genomics = pd.read_parquet(dataset_dir / GENOMICS_FILE)
    with np.load(dataset_dir / IMAGING_FILE) as data:
        patients = pd.Index(data["patients"].astype(str), name="patient_id")
        study_dates = pd.Series(pd.to_datetime(data["study_dates"]), index=patients)
        volumes = data["volumes"]
    return Modalities(genomics, study_dates, volumes, image_size, file_hash(manifest_path))


def coverage_report(manifest: dict[str, Any], dataset_dir: Path, raw_dir: Path) -> dict[str, Any]:
    """Patients with each modality by split, and scans before and after the anchor day."""
    splits = pd.read_csv(raw_dir / "patient_splits.csv", dtype=str).set_index("Id")["split"]
    cohort = pd.read_parquet(dataset_dir / "cohort.parquet", columns=["anchor_date"])
    genomics = pd.read_parquet(dataset_dir / GENOMICS_FILE, columns=["patient_id"])
    with np.load(dataset_dir / IMAGING_FILE) as data:
        scans = pd.Series(
            pd.to_datetime(data["study_dates"]), index=pd.Index(data["patients"].astype(str))
        )
    anchor_day = cohort["anchor_date"].dt.normalize().reindex(scans.index)
    before = scans.lt(anchor_day)
    out: dict[str, Any] = {
        "image_size": manifest["image_size"],
        "unreadable_images": manifest["unreadable_images"],
        "files_without_split": sorted(
            set(scans.index).union(genomics["patient_id"]).difference(splits.index)
        ),
    }
    for split in ("train", "val", "test"):
        members = splits.index[splits.eq(split)]
        in_split = scans.index.isin(members)
        out[split] = {
            "patients": len(members),
            "genomics_patients": int(genomics["patient_id"].drop_duplicates().isin(members).sum()),
            "readable_scans": int(in_split.sum()),
            "scans_before_anchor_day": int((in_split & before.to_numpy()).sum()),
            "scans_on_or_after_anchor_day": int((in_split & ~before.to_numpy()).sum()),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    manifest = build_modalities(args.raw, args.out, args.image_size)
    report = coverage_report(manifest, args.out, args.raw)
    write_json(args.reports / "modalities.json", report)
    log.info(
        "Genomics: %d patients, %d sites. Imaging: %d readable volumes, %d unreadable.",
        manifest["genomics_patients"],
        manifest["genomics_sites"],
        manifest["image_patients"],
        len(manifest["unreadable_images"]),
    )


if __name__ == "__main__":
    main()
