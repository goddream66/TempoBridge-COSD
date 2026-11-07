"""Frozen teachers: native ECG pretraining checkpoint or trusted TorchScript adapter."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .config import ModelConfig
from .model import TempoBridge
from .utils import file_sha256, load_checkpoint


class FrozenTeacher(nn.Module):
    def __init__(self, path, kind="native"):
        super().__init__()
        self.fingerprint, self.kind = file_sha256(path), kind
        self.training_subjects = None
        self.training_sources = None
        if kind in {"native", "random_control"}:
            checkpoint = load_checkpoint(path)
            random_control = checkpoint.get("diagnostic_random_teacher", False)
            if (checkpoint.get("modality") != "ecg" or
                    (kind == "native" and (random_control or checkpoint.get("global_step", 0) < 1))):
                raise ValueError("Native teacher requires a trained ECG checkpoint, not random/PPG weights")
            if kind == "random_control" and (not random_control or checkpoint.get("global_step") != 0):
                raise ValueError("random_control requires an explicitly marked untrained diagnostic teacher")
            self.encoder = TempoBridge(ModelConfig.from_dict(checkpoint["model_config"]))
            self.encoder.load_state_dict(checkpoint["model"])
            self.training_subjects = checkpoint["train_subjects"]
            self.training_sources = checkpoint.get("train_sources")
            self.input_fs = checkpoint["config"]["data"]["ecg_fs"]
            spec = checkpoint.get("teacher_feature_override", {})
            layer = spec.get("layer", self.encoder.config.feature_layer)
            self.pool, self.normalization = spec.get("pool", "context_mean"), spec.get("normalization", "none")
            if (type(layer) is not int or not -1 <= layer < self.encoder.config.depth or
                    self.pool not in {"context_mean", "summary", "combined"} or self.normalization not in {"none", "l2"}):
                raise ValueError("Invalid native teacher feature override")
            self.encoder.config.feature_layer = layer
            self.target_dim = self.encoder.config.dim * (2 if self.pool == "combined" else 1)
            self.feature_spec = {"layer": layer,
                                 "pool": self.pool, "dim": self.target_dim,
                                 "input_fs": self.input_fs, "kind": kind}
            if spec or random_control:
                self.feature_spec.update(normalization=self.normalization, diagnostic_random_teacher=random_control)
        elif kind == "torchscript":
            metadata_path = Path(str(path) + ".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not metadata.get("provenance"):
                raise ValueError("External teacher metadata requires provenance")
            self.input_fs = int(metadata["input_fs"])
            self.target_dim = int(metadata["target_dim"])
            if self.input_fs <= 0 or self.target_dim <= 0:
                raise ValueError("Invalid external teacher sample rate or feature dimension")
            self.training_subjects = metadata.get("training_subjects")
            self.training_sources = metadata.get("training_sources")
            self.feature_spec = {"adapter": metadata.get("feature_spec", "unspecified"),
                                 "dim": self.target_dim, "input_fs": self.input_fs,
                                 "kind": kind, "provenance": metadata["provenance"]}
            self.fingerprint = hashlib.sha256(
                (self.fingerprint + file_sha256(metadata_path)).encode()
            ).hexdigest()
            self.encoder = torch.jit.load(str(path), map_location="cpu")
        else:
            raise ValueError("Teacher kind must be native, torchscript or random_control")
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, ecg):
        if self.kind in {"native", "random_control"}:
            values = self.encoder.forward_features(ecg)
            result = (torch.cat((values["summary"], values["context"]), -1) if self.pool == "combined" else
                      values["summary"] if self.pool == "summary" else values["context"])
            if self.normalization == "l2":
                result = torch.nn.functional.normalize(result.float(), dim=-1)
        else:
            result = self.encoder(ecg)
        if (result.ndim != 2 or result.shape[0] != ecg.shape[0]
                or result.shape[1] != self.target_dim or not torch.isfinite(result).all()):
            raise ValueError("Teacher adapter must return finite [batch, features]")
        return result.detach()

    def verify_data(self, records, ecg_fs):
        from .data import subject_ids
        if self.input_fs is not None and self.input_fs != ecg_fs:
            raise ValueError("Teacher sampling rate differs from configured ECG sampling rate")
        if self.training_subjects is not None:
            heldout = subject_ids(records, "val") | subject_ids(records, "test")
            if heldout.intersection(self.training_subjects):
                raise ValueError("Native teacher saw a held-out patient")
