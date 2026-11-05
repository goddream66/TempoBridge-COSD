import numpy as np
import pytest
import torch

from tempobridge.calibration import CalibrationWeights, crossfit_scale, subject_weights
from tempobridge.config import CalibrationConfig


def test_crossfit_recovers_shared_signal_and_rejects_private_noise():
    rng = np.random.default_rng(5)
    subjects = np.repeat(np.arange(24).astype(str), 10)
    x = rng.normal(size=(240, 5))
    shared = x @ rng.normal(size=(5, 4)) + 0.01 * rng.normal(size=(240, 4))
    private = rng.normal(size=(240, 64))
    cfg = CalibrationConfig(folds=4, probe="ridge", min_subjects_per_bin=3)
    good = crossfit_scale(x, shared, subjects, np.ones(240), cfg, [])
    bad = crossfit_scale(x, private, subjects, np.ones(240), cfg, [])
    assert good["weights"][0] > 0.95
    assert bad["weights"][0] == 0
    scored = []
    for fold in good["folds"]:
        assert not set(fold["fit_subjects"]) & set(fold["score_subjects"])
        scored.extend(fold["score_subjects"])
    assert sorted(scored) == sorted(set(subjects))


def test_constant_teacher_does_not_get_positive_weight():
    rng = np.random.default_rng(8)
    report = crossfit_scale(rng.normal(size=(30, 4)), np.ones((30, 3)),
                            np.repeat(np.arange(6).astype(str), 5), np.ones(30),
                            CalibrationConfig(folds=3, probe="ridge"), [])
    assert report["weights"] == [0.0]
    assert np.isfinite(report["target_scale"]).all()


def test_insufficient_bin_abstains():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 4))
    report = crossfit_scale(x, x, np.repeat(np.arange(6).astype(str), 5), np.ones(30),
                            CalibrationConfig(folds=3, probe="ridge"), [0.5])
    assert report["weights"][0] == 0
    assert report["bins"][0]["sufficient_patients"] is False


def test_subject_balance_does_not_favor_long_recordings():
    ids = np.asarray(["a"] * 100 + ["b"] * 2)
    weights = subject_weights(ids)
    np.testing.assert_allclose(weights[ids == "a"].sum(), weights[ids == "b"].sum())


def test_consensus_uses_both_probes(tiny_config):
    cfg = tiny_config.calibration
    cfg.probe = "consensus"
    rng = np.random.default_rng(12)
    x = rng.normal(size=(40, 3))
    report = crossfit_scale(x, x[:, :2], np.repeat(np.arange(8).astype(str), 5),
                            np.ones(40), cfg, [])
    assert set(report["bins"][0]["r2"]) == {"ridge", "mlp"}
    assert report["weights"][0] == np.clip(min(report["bins"][0]["r2"].values()), 0, 1)


def test_quality_bin_boundary_matches_numpy():
    artifact = {"quality_edges": [0.33, 0.67], "scales": [
        {"weights": [0.1, 0.5, 0.9], "target_mean": [0], "target_scale": [1]}]}
    module = CalibrationWeights(artifact)
    weights = module.lookup(0, torch.tensor([0.0, 0.33, 0.67, 1]), torch.ones(4),
                            torch.tensor([True, True, True, False]))
    torch.testing.assert_close(weights, torch.tensor([0.1, 0.5, 0.9, 0.0]))
    assert not weights.requires_grad


def test_artifact_rejects_stale_teacher(tiny_config):
    artifact = {"schema_version": 1, "quality_edges": [], "scales": [
        {"weights": [1], "target_mean": [0], "target_scale": [1]}], "manifest_sha256": "data"}
    with pytest.raises(ValueError, match="teacher_sha256"):
        CalibrationWeights(artifact).validate(tiny_config, "data", "new_teacher", "student", "native")
