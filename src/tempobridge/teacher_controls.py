"""Declare native teacher readout ablations, with an explicit random negative control."""

from pathlib import Path

import torch

from .config import ModelConfig
from .model import TempoBridge
from .teacher import FrozenTeacher
from .utils import file_sha256, load_checkpoint, save_checkpoint


def teacher_variant(checkpoint, output, layer=-1, pool="context_mean", normalization="none", random_seed=None):
    if Path(output).exists():
        raise FileExistsError(output)
    original = FrozenTeacher(checkpoint)
    if type(layer) is not int or not -1 <= layer < original.encoder.config.depth:
        raise ValueError("Teacher layer must be -1 or a valid zero-based block index")
    if pool not in {"context_mean", "summary", "combined"} or normalization not in {"none", "l2"}:
        raise ValueError("Invalid teacher pooling or normalization")
    source = load_checkpoint(checkpoint)
    # Keep no optimizer or student state in a frozen teacher artifact.
    state = {k: source[k] for k in ("modality", "global_step", "model_config", "model", "config", "train_subjects")}
    state["train_sources"] = source.get("train_sources")
    state["teacher_feature_override"] = {"layer": layer, "pool": pool, "normalization": normalization}
    state["teacher_variant_provenance"] = {"source_sha256": file_sha256(checkpoint), "random_seed": random_seed,
        "scope": "diagnostic negative control only" if random_seed is not None else "predeclared frozen native readout"}
    if random_seed is not None:
        if type(random_seed) is not int or random_seed < 0:
            raise ValueError("Random teacher seed must be a nonnegative integer")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            state["model"] = TempoBridge(ModelConfig.from_dict(state["model_config"])).state_dict()
        state.update(global_step=0, train_subjects=[], train_sources=[], diagnostic_random_teacher=True)
    save_checkpoint(output, state)
    FrozenTeacher(output, "random_control" if random_seed is not None else "native")
    return Path(output)
