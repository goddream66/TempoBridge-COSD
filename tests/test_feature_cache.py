import json
import shutil

import numpy as np
import pytest

from tempobridge.calibration import CalibrationWeights, calibrate_cached
from tempobridge.data import make_synthetic, manifest_digest, read_manifest, subject_ids
from tempobridge.feature_cache import FeatureCache, cache_features
from tempobridge.model import TempoBridge
from tempobridge.utils import save_checkpoint


def fixtures(tmp_path, config):
    manifest = make_synthetic(tmp_path / "data", subjects=20, windows=3, seconds=2)
    records = read_manifest(manifest)
    model = TempoBridge(config.model)
    # Trained markers are used only as metadata fixtures, not as scientific evidence.
    state = {"global_step": 1, "manifest_sha256": manifest_digest(records),
             "model_config": model.model_config(), "config": config.to_dict(),
             "model": model.state_dict(), "train_subjects": sorted(subject_ids(records, "train"))}
    ppg, ecg = tmp_path / "ppg.pt", tmp_path / "ecg.pt"
    save_checkpoint(ppg, state | {"modality": "ppg"})
    save_checkpoint(ecg, state | {"modality": "ecg"})
    return manifest, ppg, ecg


def test_cache_is_portable_and_binds_content(tmp_path, tiny_config):
    manifest, ppg, ecg = fixtures(tmp_path, tiny_config)
    path = cache_features(tiny_config, manifest, ppg, ecg, tmp_path / "cache")
    cache = FeatureCache(path)
    x, y = cache.arrays(0)
    assert x.shape[1] == y.shape[1] == 16
    assert len(cache.rows) == cache.artifact["selection"]["reliable_pairs"]
    assert set(cache.column("subject_id")).issubset(cache.artifact["train_subjects"])
    shutil.copytree(path, tmp_path / "moved")
    relocated = FeatureCache(tmp_path / "moved")
    np.testing.assert_array_equal(x, relocated.arrays(0)[0])
    rows = json.loads((tmp_path / "moved" / "rows.json").read_text())
    rows[0]["recording_id"] = "tampered"
    (tmp_path / "moved" / "rows.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="content changed"):
        FeatureCache(tmp_path / "moved")


def test_cosd_and_vanilla_share_final_target_normalization(tmp_path, tiny_config):
    manifest, ppg, ecg = fixtures(tmp_path, tiny_config)
    path = cache_features(tiny_config, manifest, ppg, ecg, tmp_path / "cache")
    cache = FeatureCache(path)
    tiny_config.train.transfer_method = "vanilla"
    tiny_config.train.transfer_mode = "all"
    full = calibrate_cached(tiny_config, cache, tmp_path / "full.json")
    tiny_config.train.transfer_method = "cosd"
    tiny_config.cosd.rank = 4
    tiny_config.cosd.bootstrap_samples = 20
    projected = calibrate_cached(tiny_config, cache, tmp_path / "cosd.json")
    for a, b in zip(full["scales"], projected["scales"]):
        np.testing.assert_array_equal(a["target_mean"], b["subspace"]["teacher_mean"])
        np.testing.assert_array_equal(a["target_scale"], b["subspace"]["teacher_scale"])
    transform = CalibrationWeights(projected)
    assert all(0 <= r <= 4 for r in transform.output_dims)
    transform.validate(tiny_config, projected["manifest_sha256"], projected["teacher_sha256"],
                       projected["student_sha256"], projected["teacher_kind"])
