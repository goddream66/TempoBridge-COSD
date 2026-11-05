import pytest

from tempobridge.config import ExperimentConfig


@pytest.mark.parametrize("values", [
    {"model": {"dim": 31}},
    {"model": {"protected_depth": 6}},
    {"model": {"dim": 24, "heads": 8}},
    {"data": {"scales_seconds": [0.1]}},
    {"data": {"scales_seconds": [2, 2]}},
    {"data": {"quality_edges": [0.7, 0.2]}},
    {"train": {"mask_ratio": 1}},
    {"train": {"transfer_mode": "typo"}},
    {"calibration": {"folds": 1}},
    {"unknown_section": {}},
])
def test_bad_config_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig.from_dict(values)


def test_config_serialization_is_lossless(tiny_config):
    restored = ExperimentConfig.from_dict(tiny_config.to_dict())
    assert restored.to_dict() == tiny_config.to_dict()
