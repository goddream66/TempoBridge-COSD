import json

import pytest
import torch

from tempobridge.calibration import calibrate
from tempobridge.data import make_synthetic
from tempobridge.teacher import FrozenTeacher
from tempobridge.teacher_controls import teacher_variant
from tempobridge.training import train


def test_teacher_readout_and_random_control_are_explicit(tmp_path, tiny_config):
    cfg = tiny_config
    manifest = make_synthetic(tmp_path / "data", subjects=18, windows=3, seconds=2)
    original = train(cfg, manifest, tmp_path / "ecg", modality="ecg", stop_after=1)
    combined = teacher_variant(original, tmp_path / "combined.pt", layer=0, pool="combined", normalization="l2")
    teacher = FrozenTeacher(combined)
    wave = torch.randn(3, 1, 100)
    features = teacher(wave)
    assert features.shape == (3, 2 * cfg.model.dim) and not features.requires_grad
    torch.testing.assert_close(features.norm(dim=-1), torch.ones(3))
    assert teacher.feature_spec["layer"] == 0 and teacher.feature_spec["pool"] == "combined"
    random = teacher_variant(original, tmp_path / "random.pt", random_seed=18)
    with pytest.raises(ValueError, match="trained ECG"):
        FrozenTeacher(random)
    with pytest.raises(ValueError, match="explicitly marked"):
        FrozenTeacher(original, "random_control")
    control = FrozenTeacher(random, "random_control")
    assert control.training_subjects == control.training_sources == []
    assert control.feature_spec["diagnostic_random_teacher"] is True
    assert not torch.allclose(control(wave), FrozenTeacher(original)(wave))
    control.train()
    assert not control.training and not any(p.requires_grad for p in control.parameters())


def test_per_scale_rank_controls_use_actual_separate_dimensions(tmp_path, tiny_config):
    cfg = tiny_config
    manifest = make_synthetic(tmp_path / "data", subjects=18, windows=3, seconds=2)
    student = train(cfg, manifest, tmp_path / "ppg", stop_after=1)
    teacher = train(cfg, manifest, tmp_path / "ecg", modality="ecg", stop_after=1)
    cfg.train.transfer_method, cfg.train.transfer_mode = "cosd", "quality"
    cfg.cosd.rank, cfg.cosd.ranks_per_scale = 4, [0, 2]
    cfg.cosd.selection, cfg.cosd.bootstrap_samples = "all", 20
    artifact = tmp_path / "calibration.json"
    calibrate(cfg, manifest, student, teacher, artifact)
    report = json.loads(artifact.read_text())
    assert [s["subspace"]["rank"] for s in report["scales"]] == [0, 2]
    train(cfg, manifest, tmp_path / "continued", student_path=student, teacher_path=teacher,
          calibration_path=artifact, stop_after=1)
