"""Export/load a PPG-only encoder without a teacher or calibration dependency."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig
from .data import PairedDataset, manifest_digest, read_manifest
from .model import TempoBridge
from .utils import file_sha256, load_checkpoint, save_checkpoint


def export_encoder(checkpoint, output):
    if Path(output).exists():
        raise FileExistsError(output)
    state = load_checkpoint(checkpoint)
    if state.get("modality") != "ppg" or state.get("global_step", 0) < 1:
        raise ValueError("Export requires trained PPG checkpoint")
    weights = {k: v for k, v in state["model"].items()
               if not k.startswith(("reconstruction.", "morph_reconstruction."))}
    save_checkpoint(output, {"format_version": 1, "kind": "ppg_encoder",
                    "model_config": state["model_config"], "model": weights,
                    "data_config": state["config"]["data"],
                    "pretraining_provenance": {
                        "train_subjects": state.get("train_subjects"), "train_sources": state.get("train_sources"),
                        "teacher_training_subjects": state.get("teacher_training_subjects"),
                        "teacher_training_sources": state.get("teacher_training_sources"),
                        "manifest_sha256": state.get("manifest_sha256")},
                    "source_sha256": file_sha256(checkpoint)})
    return Path(output)


def load_encoder(path, device="cpu"):
    state = load_checkpoint(path)
    if state.get("kind") != "ppg_encoder":
        raise ValueError("Use the export command to create a PPG-only encoder")
    model = TempoBridge(ModelConfig.from_dict(state["model_config"]))
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    allowed = {"reconstruction.weight", "reconstruction.bias", "morph_reconstruction.weight", "morph_reconstruction.bias"}
    if set(missing) != allowed or unexpected:
        raise ValueError(f"Invalid encoder weights: {missing}, {unexpected}")
    model.reconstruction = torch.nn.Identity()
    model.morph_reconstruction = torch.nn.Identity()
    return model.to(device).eval()


@torch.inference_mode()
def extract_features(checkpoint, manifest, output, split="test", device="cpu", batch_size=32,
                     noise_std=0., noise_seed=42):
    if Path(output).exists():
        raise FileExistsError(output)
    if not 0 <= noise_std < float("inf"):
        raise ValueError("Noise standard deviation must be finite and nonnegative")
    state = load_checkpoint(checkpoint)
    config = DataConfig.from_dict(state["data_config"])
    records = read_manifest(manifest, require_train=False, data_config=config)
    dataset = PairedDataset(records, config, split=split, augment=False, include_ecg=False)
    loader = DataLoader(dataset, batch_size=batch_size)
    model = load_encoder(checkpoint, device)
    scale = int(np.argmax(config.scales_seconds))
    values = []
    fields = {k: [] for k in ("sample_id", "subject_id", "source", "recording_id", "case_id",
                               "window_start_s", "ppg_quality", "ecg_quality", "subject_known",
                               "identity_level", "recording_known", "ppg_quality_source", "ecg_quality_source",
                               "timestamp_alignment_status", "device_id")}
    for batch in loader:
        waves = batch["ppg"][scale]
        if noise_std:
            waves = waves.clone()
            for i, sample in enumerate(batch["sample_id"]):
                seed = int.from_bytes(hashlib.sha256(f"{noise_seed}:{sample}".encode()).digest()[:8], "big")
                generator = torch.Generator().manual_seed(seed)
                noise = torch.randn(waves[i].shape, generator=generator)
                waves[i] += noise * noise_std * waves[i].std(correction=0)
        values.append(model(waves.to(device)).cpu().numpy())
        for key in fields:
            item = batch[key]
            fields[key].extend(item.tolist() if isinstance(item, torch.Tensor) else item)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not values:
        raise ValueError(f"No feature rows in split {split}")
    import json
    metadata = {"schema_version": 2, "split": split,
                "quality_edges": config.quality_edges,
                "perturbation": {"kind": "relative_white_gaussian", "std": noise_std, "seed": noise_seed},
                "encoder_sha256": file_sha256(checkpoint), "manifest_sha256": manifest_digest(records),
                "pretraining": state.get("pretraining_provenance", {})}
    availability = np.asarray([dataset.metadata(i)["ecg_available"] for i in range(len(dataset))])
    dataset.close()
    np.savez(output, embedding=np.concatenate(values), ecg_available=availability,
             **{k: np.asarray(v) for k, v in fields.items()},
             scale_seconds=config.scales_seconds[scale], metadata_json=json.dumps(metadata))
    return Path(output)
