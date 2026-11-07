import json

import numpy as np
import pytest

from tempobridge.config import CalibrationConfig
from tempobridge.representation_audit import (
    closed_set_identity,
    source_decodability,
    within_record_skill,
)


def test_record_centering_distinguishes_identity_from_state():
    subjects = np.repeat(["a", "b", "c"], 4)
    state = np.tile(np.asarray([-2., -1., 1., 2.]), 3)[:, None]
    identity = np.repeat([0., 100., 200.], 4)[:, None]
    known = np.ones(len(subjects), bool)
    wrong = within_record_skill(identity + state, identity, subjects, subjects, known)
    correct = within_record_skill(identity + state, identity + state, subjects, subjects, known)
    assert wrong["skill"] == pytest.approx(0)
    assert correct["skill"] == pytest.approx(1)
    assert not within_record_skill(state, state, subjects, subjects, ~known)["available"]


def test_identity_and_source_audits_have_explicit_scope():
    subjects = np.repeat([f"p{i}" for i in range(20)], 4)
    sources = np.repeat(["a"] * 10 + ["b"] * 10, 4)
    features = np.c_[np.repeat(np.arange(20), 4), (sources == "b") * 100.]
    times = np.tile(np.arange(4), 20)
    result = closed_set_identity(features, subjects, times)
    assert result["group_macro_accuracy"] == 1 and "closed-set" in result["scope"]
    decoded = source_decodability(features, subjects, sources, CalibrationConfig(folds=3), 7)
    assert decoded["patient_macro_accuracy"] > .95
    assert not source_decodability(features, subjects, np.full(len(subjects), "a"), CalibrationConfig(), 7)["available"]


def test_audit_cli_on_bound_cache(tmp_path, tiny_config):
    from tempobridge.calibration import calibrate
    from tempobridge.cli import main
    from tempobridge.data import make_synthetic
    from tempobridge.training import train
    from tempobridge.utils import file_sha256

    cfg = tiny_config
    cfg.train.epochs = 1
    manifest = make_synthetic(tmp_path / "data", subjects=18, windows=3, seconds=2)
    student = train(cfg, manifest, tmp_path / "student")
    teacher = train(cfg, manifest, tmp_path / "teacher", modality="ecg")
    cfg.train.transfer_method, cfg.train.transfer_mode = "cosd", "quality"
    cfg.cosd.rank, cfg.cosd.selection, cfg.cosd.bootstrap_samples = 4, "all", 20
    calibration = tmp_path / "cosd.json"
    calibrate(cfg, manifest, student, teacher, calibration)
    before = file_sha256(calibration)
    output = tmp_path / "audit.json"
    main(["audit-representation", "--cache", str(tmp_path / "cosd_features"),
          "--calibration", str(calibration), "--output", str(output)])
    report = json.loads(output.read_text())
    assert file_sha256(calibration) == before
    assert report["selection_fixed"]
    assert len(report["scales"]) == 2
    for scale in report["scales"]:
        assert scale["representations"]["teacher_selected"]["dimensions"] == 4
        assert scale["teacher_prediction"]["selected"]["used_for_selection"] is False
