from dataclasses import replace

import numpy as np
import pytest
import torch

from tempobridge.calibration import CalibrationWeights
from tempobridge.config import CalibrationConfig, COSDConfig
from tempobridge.controls import WindowSubsetDataset
from tempobridge.data import PairedDataset, make_synthetic, read_manifest
from tempobridge.subspace import fit_subspace


def test_reliability_axes_are_independent():
    weights = CalibrationWeights({"quality_edges": [], "scales": [{"weights": [.5],
        "target_mean": [0], "target_scale": [1]}]})
    p, e, valid = torch.tensor([.2, .8]), torch.tensor([.5, 1.]), torch.tensor([True, False])
    assert weights.lookup(0, p, e, valid, "quality", True, False)[0] == pytest.approx(.2)
    assert weights.lookup(0, p, e, valid, "quality", False, True)[0] == pytest.approx(.5)
    assert weights.lookup(0, p, e, valid, "quality", True, True)[0] == pytest.approx(.1)
    assert weights.lookup(0, p, e, valid, "calibrated", True, True)[0] == pytest.approx(.05)
    torch.testing.assert_close(weights.lookup(0, p, e, valid, "all", True, True), torch.tensor([1., 0.]))


def test_quality_strata_selection_can_abstain():
    rng = np.random.default_rng(4)
    groups = np.repeat([f"s{i}" for i in range(30)], 5)
    x = rng.normal(size=(len(groups), 4))
    y = np.c_[x[:, 0], rng.normal(size=len(x))]
    cfg = COSDConfig(rank=2, bootstrap_samples=20, stability=False)
    probe = CalibrationConfig(folds=2, probe="ridge", min_subjects_per_bin=2)
    normal = fit_subspace(x, y, groups, np.full(len(groups), "a"), probe, cfg, 7)
    assert normal["rank"] >= 1
    stratified = fit_subspace(x, y, groups, np.full(len(groups), "a"), probe,
                               replace(cfg, quality_strata=True), 7, np.ones(len(groups)), [.5])
    assert stratified["rank"] == 0  # Empty quality stratum is unsupported, not evidence of stability.
    assert stratified["quality_strata"][0]["eligible"] is False


def test_subset_uses_valid_waveforms_and_preserves_original_ids(tmp_path, tiny_config):
    manifest = make_synthetic(tmp_path / "data", subjects=8, windows=2, seconds=2)
    records = read_manifest(manifest)
    paired = WindowSubsetDataset(PairedDataset(records, tiny_config.data), "paired")
    unpaired = WindowSubsetDataset(PairedDataset(records, tiny_config.data), "unpaired")
    assert len(paired) == 7 and len(unpaired) == 1
    assert all(paired[i]["ecg_valid"] for i in range(len(paired)))
    assert not unpaired[0]["ecg_valid"]
    assert set(paired.report()["sample_ids"]).isdisjoint(unpaired.report()["sample_ids"])
    paired.close()
    unpaired.close()
