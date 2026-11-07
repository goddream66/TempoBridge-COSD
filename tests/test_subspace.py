from dataclasses import replace

import numpy as np
import pytest
import torch

from tempobridge.config import CalibrationConfig, COSDConfig
from tempobridge.subspace import TeacherSubspace, fit_subspace, partition_patients


def example():
    rng = np.random.default_rng(102)
    subjects = np.repeat([f"p{i:03}" for i in range(80)], 12)
    x = rng.normal(size=(len(subjects), 4))
    y = np.column_stack([3 * x[:, 0], x[:, 1] - x[:, 2],
                         rng.normal(size=len(x)), np.ones(len(x))])
    sources = np.where(np.arange(len(x)) % 2, "mimic", "vitaldb")
    probe = CalibrationConfig(folds=3, probe="ridge")
    cfg = COSDConfig(rank=4, bootstrap_samples=40)
    return x, y, subjects, sources, probe, cfg


def test_shared_directions_selected_noise_and_constants_rejected():
    result = fit_subspace(*example(), seed=11)
    assert set(result["selected_directions"]) == {0, 1}
    assert result["rank"] == 2
    assert result["audit"]["used_for_selection"] is False
    groups = [set(v) for v in result["partitions"].values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    for fold in result["folds"]:
        assert not set(fold["fit_subjects"]) & set(fold["score_subjects"])
        assert set(fold["score_subjects"]).issubset(result["partitions"]["discovery"])


def test_audit_targets_cannot_change_normalization_or_selection():
    args = example()
    first = fit_subspace(*args, seed=11)
    x, y, ids, sources, probe, cfg = args
    y = y.copy()
    y[np.isin(ids, first["partitions"]["audit"])] += 1000
    second = fit_subspace(x, y, ids, sources, probe, cfg, seed=11)
    for key in ("teacher_mean", "teacher_scale", "projection", "selected_directions", "direction_skill"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["audit"]["skill"] != second["audit"]["skill"]


@pytest.mark.parametrize("basis,selection", [("pca", "variance"), ("random", "random"),
                                          ("standardized", "all")])
def test_projection_is_frozen_and_has_selected_coordinate_shape(basis, selection):
    x, y, ids, sources, probe, cfg = example()
    cfg = replace(cfg, basis=basis, selection=selection, rank=2)
    artifact = fit_subspace(x, y, ids, sources, probe, cfg, seed=11)
    transform = TeacherSubspace(artifact)
    teacher = torch.tensor(y[:5], dtype=torch.float32, requires_grad=True)
    target = transform(teacher)
    assert target.shape == (5, 2) and not target.requires_grad
    assert list(transform.parameters()) == []
    expected = ((y[:5] - artifact["teacher_mean"]) / artifact["teacher_scale"]) @ np.asarray(artifact["projection"])
    np.testing.assert_allclose(target, expected, atol=2e-6)


def test_no_directions_abstains_instead_of_filling_topk():
    x, y, ids, sources, probe, cfg = example()
    artifact = fit_subspace(x, np.ones_like(y), ids, sources, probe, cfg, seed=11)
    assert artifact["rank"] == 0
    transform = TeacherSubspace(artifact)
    assert transform(torch.ones(7, 4)).shape == (7, 0)


def test_subspace_invalid_rank_and_orthogonality_are_rejected():
    x, y, ids, sources, probe, cfg = example()
    with pytest.raises(ValueError, match="rank exceeds"):
        fit_subspace(x, y, ids, sources, probe, replace(cfg, rank=5), seed=11)
    artifact = fit_subspace(x, y, ids, sources, probe, cfg, seed=11)
    column = np.asarray(artifact["projection"])[:, 0]
    artifact["projection"][int(np.argmax(np.abs(column)))][0] *= 2
    with pytest.raises(ValueError, match="orthonormal"):
        TeacherSubspace(artifact)


def test_partition_depends_only_on_patient_ids():
    _, _, ids, _, _, cfg = example()
    assert partition_patients(ids, cfg, 4) == partition_patients(ids[::-1], cfg, 4)
