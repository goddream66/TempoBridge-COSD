"""Export/load a PPG-only encoder without a teacher or calibration dependency."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig
from .data import PairedDataset, read_manifest
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
                    "source_sha256": file_sha256(checkpoint)})
    return Path(output)


def load_encoder(path, device="cpu"):
    state = load_checkpoint(path)
    if state.get("kind") != "ppg_encoder":
        raise ValueError("Use the export command to create a PPG-only encoder")
    model = TempoBridge(ModelConfig(**state["model_config"]))
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    allowed = {"reconstruction.weight", "reconstruction.bias", "morph_reconstruction.weight", "morph_reconstruction.bias"}
    if set(missing) != allowed or unexpected:
        raise ValueError(f"Invalid encoder weights: {missing}, {unexpected}")
    model.reconstruction = torch.nn.Identity()
    model.morph_reconstruction = torch.nn.Identity()
    return model.to(device).eval()


@torch.inference_mode()
def extract_features(checkpoint, manifest, output, split="test", device="cpu", batch_size=32):
    if Path(output).exists():
        raise FileExistsError(output)
    state = load_checkpoint(checkpoint)
    config = DataConfig(**state["data_config"])
    records = read_manifest(manifest, require_train=False, data_config=config)
    dataset = PairedDataset(records, config, split=split, augment=False, include_ecg=False)
    loader = DataLoader(dataset, batch_size=batch_size)
    model = load_encoder(checkpoint, device)
    scale = int(np.argmax(config.scales_seconds))
    values, ids, subjects = [], [], []
    for batch in loader:
        values.append(model(batch["ppg"][scale].to(device)).cpu().numpy())
        ids.extend(batch["sample_id"])
        subjects.extend(batch["subject_id"])
    dataset.close()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, embedding=np.concatenate(values), sample_id=np.asarray(ids),
             subject_id=np.asarray(subjects), scale_seconds=config.scales_seconds[scale])
    return Path(output)
