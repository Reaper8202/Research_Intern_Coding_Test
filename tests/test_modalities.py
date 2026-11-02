"""D34: genomics and brain MRI inputs, their date rules, and the image encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

from ehrforecast.contracts import prediction_cohort
from ehrforecast.modalities import (
    GENOMICS_FILE,
    Modalities,
    build_modalities,
    load_modalities,
    read_genotypes,
    read_imaging,
    read_volume,
)
from ehrforecast.model import EHRTransformer, ModelConfig
from ehrforecast.sequences import SequenceConfig, build_histories, load_histories, save_histories
from ehrforecast.tokenizer import DNA, IMG, FoldTokenizer
from ehrforecast.training import collate
from tests.transformer_fixtures import ANCHOR, TARGETS, permitted_events, raw_tables

SIZE = 4


# ---------------------------------------------------------------------- helpers
def write_genotype_file(path: Path, alleles: dict[str, tuple[bool, bool]]) -> None:
    rows = [
        {"INDEX_PREFIX": site, "STRAND": strand + 1, "VARIANT": str(variant).lower()}
        for site, pair in alleles.items()
        for strand, variant in enumerate(pair)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_dicom(path: Path, volume: np.ndarray, study_date: str) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.4")
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = (
        meta.MediaStorageSOPClassUID,
        meta.MediaStorageSOPInstanceUID,
    )
    ds.StudyDate = study_date
    ds.NumberOfFrames, ds.Rows, ds.Columns = volume.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = volume.astype("<u2").tobytes()
    ds.save_as(path, enforce_file_format=True)


def fixture_modalities(
    patients: pd.Index, scans: dict[str, str], genomics: pd.DataFrame | None = None
) -> Modalities:
    rng = np.random.default_rng(3)
    if genomics is None:
        rows = []
        for k, p in enumerate(patients[:12]):  # half of the patients have a genotype file
            for site in ("rs1", "rs2", "rs3"):
                rows.append({"patient_id": p, "site": site, "count": (k + len(site)) % 3})
        genomics = pd.DataFrame(rows)
    dates = pd.Series(pd.to_datetime(list(scans.values())), index=pd.Index(list(scans)))
    volumes = rng.normal(size=(len(scans), SIZE, SIZE, SIZE)).astype("float16")
    return Modalities(genomics, dates, volumes, SIZE, "fixture")


def multimodal_setup(scans: dict[str, str]) -> tuple:
    cohort, tables = raw_tables()
    prepared = prediction_cohort(cohort)
    events = permitted_events(cohort, tables)
    modalities = fixture_modalities(prepared.index, scans)
    fit = prepared.index[:16]
    tokenizer = FoldTokenizer.fit(
        events, fit, min_patients=1, genomics=modalities.genomics, imaging=True
    )
    histories = build_histories(tokenizer.encode(events), prepared, tokenizer, TARGETS, modalities)
    return prepared, events, tokenizer, histories, modalities


# ---------------------------------------------------------------------- files
def test_read_genotypes_counts_variant_alleles_over_both_strands(tmp_path: Path) -> None:
    path = tmp_path / "p1_dna.csv"
    write_genotype_file(path, {"rs1": (False, False), "rs2": (True, False), "rs3": (True, True)})
    assert read_genotypes(path).to_dict() == {"rs1": 0, "rs2": 1, "rs3": 2}


def test_read_genotypes_rejects_a_site_without_two_strands(tmp_path: Path) -> None:
    path = tmp_path / "p1_dna.csv"
    pd.DataFrame({"INDEX_PREFIX": ["rs1"], "STRAND": [1], "VARIANT": ["true"]}).to_csv(path)
    with pytest.raises(ValueError, match="two strands"):
        read_genotypes(path)


def test_read_volume_resamples_and_standardizes_each_volume(tmp_path: Path) -> None:
    volume = np.arange(8**3, dtype="uint16").reshape(8, 8, 8) * 7
    write_dicom(tmp_path / "p1.dcm", volume, "20100102")
    study_date, out = read_volume((tmp_path / "p1.dcm").read_bytes(), SIZE)
    assert study_date == pd.Timestamp("2010-01-02")
    assert out.shape == (SIZE, SIZE, SIZE) and out.dtype == np.float16
    values = out.astype("float64")
    assert abs(values.mean()) < 1e-2 and abs(values.std() - 1) < 1e-2
    block_means = volume.reshape(4, 2, 4, 2, 4, 2).mean(axis=(1, 3, 5))  # area averaging
    expected = (block_means - block_means.mean()) / block_means.std()
    assert np.allclose(values, expected, atol=1e-2)


def test_a_truncated_dicom_is_recorded_and_skipped(tmp_path: Path) -> None:
    write_dicom(tmp_path / "good.dcm", np.ones((8, 8, 8), "uint16") * np.arange(8), "20100102")
    write_dicom(tmp_path / "bad.dcm", np.ones((8, 8, 8), "uint16") * np.arange(8), "20100102")
    data = (tmp_path / "bad.dcm").read_bytes()
    (tmp_path / "bad.dcm").write_bytes(data[:-100])
    dates, volumes, records = read_imaging(tmp_path, SIZE)
    assert list(dates.index) == ["good"] and volumes.shape == (1, SIZE, SIZE, SIZE)
    assert "error" in records["bad.dcm"] and "error" not in records["good.dcm"]


def test_built_modalities_load_only_with_matching_hashes_and_size(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    (raw / "genomics").mkdir(parents=True)
    (raw / "imaging").mkdir()
    write_genotype_file(raw / "genomics" / "p1_dna.csv", {"rs1": (True, False)})
    write_dicom(
        raw / "imaging" / "p1.dcm", np.arange(512, dtype="uint16").reshape(8, 8, 8), "20100102"
    )
    build_modalities(raw, out, SIZE)
    loaded = load_modalities(out, SIZE)
    assert loaded.genomics.to_dict("records") == [{"patient_id": "p1", "site": "rs1", "count": 1}]
    assert loaded.study_dates["p1"] == pd.Timestamp("2010-01-02")
    assert loaded.volume("p1").shape == (SIZE, SIZE, SIZE)
    with pytest.raises(ValueError, match="voxel"):
        load_modalities(out, 8)
    pd.DataFrame({"patient_id": ["p1"], "site": ["rs1"], "count": [2]}).to_parquet(
        out / GENOMICS_FILE, index=False
    )
    with pytest.raises(ValueError, match="does not match its manifest"):
        load_modalities(out, SIZE)


# ---------------------------------------------------------------------- tokenizer
def test_genotype_tokens_are_fitted_on_fit_patients_only() -> None:
    cohort, tables = raw_tables()
    prepared = prediction_cohort(cohort)
    events = permitted_events(cohort, tables)
    fit, held = prepared.index[:16], prepared.index[16:]
    genomics = pd.DataFrame(
        [{"patient_id": p, "site": "rs_fit", "count": 1} for p in fit[:5]]
        + [{"patient_id": p, "site": "rs_rare", "count": 2} for p in fit[:4]]
        + [{"patient_id": p, "site": "rs_held", "count": 1} for p in held]
    )
    tokenizer = FoldTokenizer.fit(events, fit, min_patients=5, genomics=genomics)
    genotype = {t for t in tokenizer.vocabulary if t.startswith("dna:")}
    assert genotype == {"dna:rs_fit:1"}  # rs_rare has 4 fit carriers; rs_held only held ones
    tokens = tokenizer.genomic_tokens(genomics)
    held_rows = tokens.loc[tokens["patient_id"].eq(held[0]), "token_id"].tolist()
    assert held_rows == [tokenizer.token_id(DNA)]  # file marker only; unknown genotype left out


def test_a_tokenizer_without_modalities_is_unchanged_and_modality_tokens_are_input_only(
    tmp_path: Path,
) -> None:
    _, events, tokenizer, _, _ = multimodal_setup({})
    plain = FoldTokenizer.fit(events, tokenizer_fit_patients(), min_patients=1)
    assert "modalities" not in plain.to_dict()
    assert DNA not in plain.vocabulary and IMG not in plain.vocabulary
    assert "modalities" in tokenizer.to_dict()
    tokenizer.save(tmp_path / "tokenizer.json")
    loaded = FoldTokenizer.load(tmp_path / "tokenizer.json")
    assert (loaded.genomics, loaded.imaging, loaded.sha256) == (True, True, tokenizer.sha256)
    input_only = {tokenizer.vocabulary[i] for i in tokenizer.input_only_ids()}
    assert {DNA, IMG} <= input_only
    assert {t for t in tokenizer.vocabulary if t.startswith("dna:")} <= input_only


def tokenizer_fit_patients() -> pd.Index:
    cohort, _ = raw_tables()
    return prediction_cohort(cohort).index[:16]


# ---------------------------------------------------------------------- sequences
def test_a_scan_on_or_after_the_anchor_day_is_never_used() -> None:
    cohort, _ = raw_tables()
    p_after, p_on = cohort.index[0], cohort.index[1]
    scans = {p_after: "2016-05-01", p_on: str(ANCHOR.date())}
    prepared, _, tokenizer, histories, _ = multimodal_setup(scans)
    config = SequenceConfig(48, 6)
    for p in (p_after, p_on):
        i = int(histories.index_of(pd.Index([p]))[0])
        assert histories.image_day[i] == -1
        sample = histories.prediction_sample(i, config)
        assert tokenizer.token_id(IMG) not in sample.tokens and sample.image is None
    # The same histories without these scans give identical prediction inputs.
    _, _, _, without, _ = multimodal_setup({})
    for i in range(len(prepared)):
        a, b = histories.prediction_sample(i, config), without.prediction_sample(i, config)
        assert np.array_equal(a.tokens, b.tokens) and np.array_equal(a.ages, b.ages)


def test_a_prescan_enters_prediction_but_a_window_only_after_the_scan_day() -> None:
    cohort, _ = raw_tables()
    _, _, tokenizer, histories, _ = multimodal_setup({})
    config = SequenceConfig(48, 6)
    # Pick a patient with at least three history days and scan between the second and third.
    i = next(
        i
        for i in range(len(histories))
        if histories.n_tokens(i) and len(histories._days(i)[0]) >= 3
    )
    days = histories._days(i)[1]
    p = str(histories.patients[i])
    scan_day = int(days[1]) + 1
    birth = pd.Timestamp(str(cohort.loc[p, "birthdate"]))
    scan = (birth + pd.Timedelta(days=scan_day)).strftime("%Y-%m-%d")
    _, _, _, with_scan, mods = multimodal_setup({p: scan})
    img = tokenizer.token_id(IMG)
    prediction = with_scan.prediction_sample(i, config)
    assert img in prediction.tokens and prediction.image is not None
    position = int(np.flatnonzero(prediction.tokens == img)[0])
    assert position < prediction.n_prefix
    assert prediction.ages[position] == pytest.approx(scan_day / 365.25)
    assert np.array_equal(prediction.image, mods.volume(p))
    early = with_scan.window_sample(i, 1, config)  # window starts before the scan day
    late = with_scan.window_sample(i, 2, config)  # window starts after the scan day
    assert img not in early.tokens and early.image is None
    assert img in late.tokens and late.image is not None
    # A scan on the window's first day is not before it, so that window leaves it out.
    same_day = (birth + pd.Timedelta(days=int(days[2]))).strftime("%Y-%m-%d")
    _, _, _, on_start, _ = multimodal_setup({p: same_day})
    assert img not in on_start.window_sample(i, 2, config).tokens
    assert img in on_start.prediction_sample(i, config).tokens


def test_genotype_tokens_lead_the_prefix_and_the_context_budget_holds() -> None:
    cohort, _ = raw_tables()
    scans = {p: "2010-06-01" for p in cohort.index[:6]}
    _, _, tokenizer, histories, _ = multimodal_setup(scans)
    config = SequenceConfig(48, 6)
    dna, img = tokenizer.token_id(DNA), tokenizer.token_id(IMG)
    seen_both = False
    for i in range(len(histories)):
        sample = histories.prediction_sample(i, config)
        assert len(sample.tokens) <= config.context_length
        prefix = sample.tokens[: sample.n_prefix].tolist()
        assert prefix[-1] == histories.sep_id
        if dna in prefix:
            assert prefix.index(dna) == 1
            genotype = [t for t in prefix if tokenizer.vocabulary[t].startswith("dna:")]
            assert prefix[2 : 2 + len(genotype)] == sorted(genotype)
            assert np.all(sample.ages[1 : 2 + len(genotype)] == 0)
            if img in prefix:
                assert prefix.index(img) == 2 + len(genotype)
                seen_both = True
        for first in range(histories.window_starts(i, config)):
            assert len(histories.window_sample(i, first, config).tokens) <= config.context_length
    assert seen_both


def test_a_scan_token_without_an_attached_volume_fails_loudly(tmp_path: Path) -> None:
    cohort, _ = raw_tables()
    p = cohort.index[2]
    _, _, _, histories, modalities = multimodal_setup({p: "2010-06-01"})
    path = tmp_path / "histories.npz"
    save_histories(histories, path, {"x": 1})
    loaded = load_histories(path, {"x": 1})
    assert loaded.image_id == histories.image_id
    assert np.array_equal(loaded.static_tokens, histories.static_tokens)
    assert np.array_equal(loaded.image_day, histories.image_day)
    i = int(loaded.index_of(pd.Index([p]))[0])
    with pytest.raises(ValueError, match="attach_volumes"):
        loaded.prediction_sample(i, SequenceConfig(48, 6))
    loaded.attach_volumes(modalities)
    assert loaded.prediction_sample(i, SequenceConfig(48, 6)).image is not None


# ---------------------------------------------------------------------- model
def test_image_volumes_reach_only_their_own_rows_and_train_the_encoder() -> None:
    cohort, _ = raw_tables()
    scans = {cohort.index[0]: "2010-06-01", cohort.index[3]: "2011-06-01"}
    _, _, tokenizer, histories, _ = multimodal_setup(scans)
    config = SequenceConfig(48, 6)
    samples = [histories.prediction_sample(i, config) for i in range(6)]
    assert [s.image is not None for s in samples] == [True, False, False, True, False, False]
    batch = collate(samples, torch.device("cpu"))
    assert batch["images"].shape == (2, 1, SIZE, SIZE, SIZE)
    assert torch.equal(batch["images"][1, 0], torch.from_numpy(samples[3].image.astype("float32")))
    torch.manual_seed(0)
    model = EHRTransformer(
        ModelConfig(
            tokenizer.size, 1, 16, 2, 2, 0.0, 0.0, 3, image_token_id=tokenizer.token_id(IMG)
        ),
        torch.from_numpy(tokenizer.input_only_ids()),
    ).eval()
    args = (batch["tokens"], batch["ages"], batch["readout"], batch["head"])
    base = model(*args, batch["images"])
    changed = model(*args, batch["images"] + 1.0)
    differs = (base - changed).abs().amax(dim=1) > 0
    assert differs.tolist() == [True, False, False, True, False, False]
    with pytest.raises(ValueError, match="image volumes"):
        model(*args, batch["images"][:1])
    with pytest.raises(ValueError, match="image volumes"):
        model(*args)
    model.train()
    model(*args, batch["images"]).sum().backward()
    assert model.image_encoder is not None
    assert all(p.grad is not None for p in model.image_encoder.parameters())
    assert any(p is q for p in model.image_encoder.parameters() for q in model.body_parameters())


def test_load_modalities_rejects_a_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="make modalities"):
        load_modalities(tmp_path, SIZE)
    (tmp_path / "modalities.manifest.json").write_text(json.dumps({"version": "old"}))
    with pytest.raises(ValueError, match="unknown version"):
        load_modalities(tmp_path, SIZE)


def test_dicom_fixture_is_valid(tmp_path: Path) -> None:
    write_dicom(tmp_path / "x.dcm", np.zeros((2, 3, 4), "uint16"), "20000101")
    assert pydicom.dcmread(tmp_path / "x.dcm").pixel_array.shape == (2, 3, 4)
