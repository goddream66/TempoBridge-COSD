import numpy as np
import pytest
import torch

from tempobridge.calibration import CalibrationWeights
from tempobridge.model import TempoBridge, TransferHeads


def test_transfer_feature_is_pool_and_layer_independent(tiny_config):
    model = TempoBridge(tiny_config.model).eval()
    x = torch.randn(2, 1, 100)
    expected = model.encode(x)["context"].mean(1)
    for pool in ("mean", "combined", "cls", "morph_context"):
        model.config.feature_pool = pool
        model.config.feature_layer = 0
        torch.testing.assert_close(model.transfer_features(x), expected)
    assert model.transfer_feature_spec()["layer"] == model.config.depth - 1
    heads = TransferHeads(model.config.dim, 7, [1])
    torch.testing.assert_close(heads(model.encode(x)["context"], 0), heads(expected, 0))


def test_transfer_feature_detachment_preserves_forward_value(tiny_config):
    model = TempoBridge(tiny_config.model).eval()
    x = torch.randn(2, 1, 100)
    protected = model.transfer_features(x, detach_shallow=True)
    torch.testing.assert_close(protected, model.transfer_features(x))
    protected.square().sum().backward()
    assert all(p.grad is None for p in model.stem.parameters())
    assert all(p.grad is None for p in model.shallow.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context.parameters())


def test_matching_legacy_calibration_still_requires_regeneration(tiny_config):
    cfg = tiny_config
    a = {"schema_version": 1, "manifest_sha256": "m", "teacher_sha256": "t",
         "student_sha256": "s", "teacher_kind": "native", "scales_seconds": cfg.data.scales_seconds,
         "quality_edges": cfg.data.quality_edges, "ppg_fs": cfg.data.ppg_fs,
         "ecg_fs": cfg.data.ecg_fs, "window_mode": cfg.data.window_mode,
         "scales": [{"weights": [1, 1, 1], "target_mean": [0], "target_scale": [1]}] * 2}
    with pytest.raises(ValueError, match="Legacy calibration"):
        CalibrationWeights(a).validate(cfg, "m", "t", "s", "native")


def test_hdf5_record_metadata_does_not_alias_patient(tmp_path):
    import h5py

    from tempobridge.config import DataConfig
    from tempobridge.data import PairedDataset, Record

    path = tmp_path / "windows.h5"
    with h5py.File(path, "w") as h:
        h["signals"] = np.tile(np.arange(100, dtype="f4"), (3, 1))
        h["record_key"] = np.asarray([b"record_a", b"record_b", b"record_a"])
        h["window_start_s"] = [0., 0., 30.]
        h["case_id"] = [7, 8, 7]
    records = [Record(str(i), "patient_1", "", "train", path, 1, 1,
                      h5_row=i, source="vitaldb", stored_fs=50, recording_known=False) for i in range(3)]
    ds = PairedDataset(records, DataConfig(scales_seconds=[2.]))
    a, b, c = [ds[i] for i in range(3)]
    assert a["subject_id"] == b["subject_id"]
    assert a["recording_id"] == c["recording_id"] != b["recording_id"]
    assert c["window_start_s"] == 30 and a["recording_known"]
    assert a["ppg_quality_source"] == a["ecg_quality_source"] == "unknown"
    ds.close()
    with h5py.File(path, "r+") as h:
        del h["record_key"]
    ds = PairedDataset(records, DataConfig(scales_seconds=[2.]))
    assert ds[0]["recording_id"] == "" and not ds[0]["recording_known"]
    ds.close()
