import csv

import numpy as np
import pytest

from tempobridge.data import PairedDataset, make_synthetic, read_manifest, read_signal


def test_missing_ecg_preserves_ppg_training_sample(tmp_path, tiny_config):
    manifest = make_synthetic(tmp_path, subjects=8, windows=3, seconds=2)
    dataset = PairedDataset(read_manifest(manifest), tiny_config.data)
    first = dataset[0]
    assert not first["ecg_valid"]
    assert first["ppg"][1].abs().sum() > 0
    assert first["ecg"][1].abs().sum() == 0


def test_manifest_rejects_patient_leakage(tmp_path):
    manifest = make_synthetic(tmp_path, subjects=8, windows=2, seconds=2)
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["split"] = "test"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Patient leakage"):
        read_manifest(manifest)


def test_resampling_and_unknown_rate():
    wave = np.sin(np.arange(400) / 20).astype(np.float32)
    x = read_signal({"ecg": wave, "ecg_fs": np.array(200)}, "ecg", 100)
    assert x.shape == (200,)
    with pytest.raises(ValueError, match="ecg_fs"):
        read_signal({"ecg": wave}, "ecg", 100)


def test_epoch_sampling_reproducible(tmp_path, tiny_config):
    manifest = make_synthetic(tmp_path, subjects=8, windows=2, seconds=3)
    dataset = PairedDataset(read_manifest(manifest), tiny_config.data)
    a = dataset[1]["ppg"][0]
    assert (a == dataset[1]["ppg"][0]).all()
    dataset.set_epoch(1)
    assert not (a == dataset[1]["ppg"][0]).all()


def test_signal_length_mismatch_rejected(tmp_path, tiny_config):
    manifest = make_synthetic(tmp_path, subjects=8, windows=2, seconds=3)
    records = read_manifest(manifest)
    np.savez(records[1].path, ppg=np.ones(300), ppg_fs=100, ecg=np.ones(200), ecg_fs=100)
    with pytest.raises(ValueError, match="durations"):
        PairedDataset(records, tiny_config.data)[1]
