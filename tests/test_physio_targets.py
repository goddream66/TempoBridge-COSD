import csv
import json

import numpy as np
import pytest
import torch

from tempobridge.data import PairedDataset, make_synthetic, read_manifest
from tempobridge.inference import export_encoder, load_encoder
from tempobridge.model import TempoBridge, TransferHeads
from tempobridge.physio_targets import (
    FIELDS,
    ReferenceNormalizer,
    TargetTable,
    build_reference_targets,
    reference_statistics,
    reference_targets,
)
from tempobridge.quality import waveform_diagnostics
from tempobridge.training import train
from tempobridge.transfer import physiological_loss
from tempobridge.utils import load_checkpoint


def write_labels(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(records):
            writer.writerow(dict(sample_id=row.sample_id, target="hr_bpm", value=60 + i,
                                 valid=int(i % 3 != 0), reason="", units="bpm", reference="external_ecg",
                                 method="synthetic_test_only", reference_sha256="fixture",
                                 reference_fs=100, duration_seconds=2))


def test_diagnostics_preserve_missingness_and_detect_plateaus():
    x = np.r_[np.zeros(20), np.ones(20), np.nan * np.ones(10)]
    d = waveform_diagnostics(x, 100)
    assert d["missing_fraction"] == pytest.approx(.2)
    assert d["flatline_fraction"] > .95 and d["extreme_occupancy"] == 1
    assert waveform_diagnostics(None, 100)["missing_fraction"] == 1


def test_native_reference_targets_and_units(tmp_path):
    fs, duration = 250, 30
    t = np.arange(fs * duration) / fs
    ecg = sum(np.exp(-((t - peak) / .015) ** 2) for peak in np.arange(.5, duration, 1))
    signals = {"ecg": ecg, "ecg_fs": fs, "abp": 80 + 20 * np.sin(2 * np.pi * t),
               "abp_fs": fs, "abp_units": "mmHg", "resp": np.sin(2 * np.pi * .2 * t), "resp_fs": fs}
    targets = {r["target"]: r for r in reference_targets(signals, duration)}
    for key, expected in {"hr_bpm": 60, "rmssd_ms": 0, "sbp_mmhg": 100,
                          "dbp_mmhg": 60, "map_mmhg": 80, "rr_bpm": 12}.items():
        assert targets[key]["valid"] and targets[key]["value"] == pytest.approx(expected, abs=1)
    signals["abp_units"] = "normalized"
    targets = {r["target"]: r for r in reference_targets(signals, duration)}
    assert not targets["sbp_mmhg"]["valid"]
    signals["ecg"], signals["ecg_fs"] = ecg[::5], 50
    targets = {r["target"]: r for r in reference_targets(signals, duration)}
    assert not targets["rmssd_ms"]["valid"]
    np.savez(tmp_path / "ref.npz", **signals)
    manifest = tmp_path / "refs.jsonl"
    manifest.write_text(json.dumps({"sample_id": "one", "path": "ref.npz", "duration_seconds": 30}))
    output = build_reference_targets(manifest, tmp_path / "targets.csv")
    table = TargetTable(output)
    values, valid = table.arrays(["one", "absent"], ["hr_bpm", "sbp_mmhg"], 30)
    assert valid.tolist() == [[True, False], [False, False]]
    assert values[0, 0] == pytest.approx(60, abs=1)
    with pytest.raises(ValueError, match="duration"):
        table.arrays(["one"], ["hr_bpm"], 2)


def test_masked_physiological_loss_and_gradient_boundary(tiny_config):
    model = TempoBridge(tiny_config.model)
    heads = TransferHeads(tiny_config.model.dim, 2, [1])
    normalizer = ReferenceNormalizer({"mean": [0, 0], "scale": [1, 1]})
    ppg = torch.randn(3, 1, 100)
    labels = torch.randn(3, 2)
    valid = torch.tensor([[True, False], [False, False], [True, False]])
    loss, metrics = physiological_loss(model, heads, normalizer, ppg, labels, valid, 0, True)
    loss.backward()
    assert loss.isfinite() and metrics["active_transfer_fraction"] == pytest.approx(2 / 3)
    assert all(p.grad is None for p in model.stem.parameters())
    assert any(p.grad is not None for p in model.context.parameters())
    model.zero_grad(set_to_none=True)
    loss, _ = physiological_loss(model, heads, normalizer, ppg, labels, torch.zeros_like(valid), 0, True)
    assert loss == 0 and not loss.requires_grad


def test_physio_training_resume_export_and_train_only_statistics(tmp_path, tiny_config):
    cfg = tiny_config
    cfg.data.window_mode, cfg.data.scales_seconds = "full", [2.]
    manifest = make_synthetic(tmp_path / "data", subjects=8, windows=3, seconds=2)
    records = read_manifest(manifest)
    labels = tmp_path / "labels.csv"
    write_labels(labels, records)
    warmup = train(cfg, manifest, tmp_path / "warmup", stop_after=1)
    cfg.train.transfer_method, cfg.train.transfer_mode = "physio", "all"
    cfg.train.gradient_audit_interval = 1
    kwargs = dict(student_path=warmup, reference_targets_path=labels)
    full = train(cfg, manifest, tmp_path / "full", **kwargs)
    partial = train(cfg, manifest, tmp_path / "resume", stop_after=1, **kwargs)
    resumed = train(cfg, manifest, tmp_path / "resume", resume=partial, **kwargs)
    a, b = load_checkpoint(full), load_checkpoint(resumed)
    assert a["provenance"]["teacher_sha256"] is None and a["stage"] == "physio"
    for group in ("model", "heads"):
        for key in a[group]:
            torch.testing.assert_close(a[group][key], b[group][key], atol=0, rtol=0)
    encoder = load_encoder(export_encoder(full, tmp_path / "encoder.pt"))
    assert torch.isfinite(encoder(torch.randn(2, 1, 200))).all()
    dataset = PairedDataset(records, cfg.data)
    table = TargetTable(labels)
    before = reference_statistics(dataset, table, ["hr_bpm"])
    for row in records:
        if row.split != "train":
            table.rows[(row.sample_id, "hr_bpm")]["value"] = 99999
    after = reference_statistics(dataset, table, ["hr_bpm"])
    assert before == after
    dataset.close()
