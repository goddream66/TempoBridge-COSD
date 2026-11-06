import pytest
import torch

from tempobridge.config import ExperimentConfig


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(123)


@pytest.fixture(params=["local", "multiview", "selective"])
def tiny_config(request):
    return ExperimentConfig.from_dict({
        "model": {"encoder": request.param, "dim": 16, "depth": 3, "protected_depth": 1, "heads": 2,
                  "patch_size": 10, "stem_width": 4, "ff_multiplier": 2,
                  "token_kernel_size": 9, "scalar_dim": 8, "ff_hidden_dim": 32,
                  "selective_width": 8, "selective_kernel_sizes": [3, 7, 11]},
        "data": {"ppg_fs": 100, "ecg_fs": 100, "scales_seconds": [0.5, 1.0],
                 "quality_edges": [0.33, 0.67]},
        "train": {"epochs": 2, "batch_size": 6, "accumulation": 2, "device": "cpu",
                  "threads": 2, "lr": 0.002, "seed": 23},
        "calibration": {"folds": 2, "probe": "ridge", "min_subjects_per_bin": 1,
                        "mlp_steps": 20, "mlp_hidden": 8},
    })
