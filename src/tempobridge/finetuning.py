"""PPG-only downstream fine-tuning, with a random-initialization comparison.

Full windows and independent scalar labels; validation alone selects the epoch.
The model never reads ECG, even when it is present in the waveform archive.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .calibration import subject_weights
from .data import PairedDataset, manifest_digest, read_manifest
from .evaluation import cluster_intervals, select_label_groups, task_metrics, validate_splits
from .inference import load_encoder
from .model import TempoBridge
from .physio_targets import TargetTable
from .training import autocast_context
from .utils import device_for, file_sha256, load_checkpoint, save_checkpoint, seed_all, write_json


class LabeledPPG(Dataset):
    def __init__(self, dataset, indices, y, weights):
        self.dataset, self.indices, self.y, self.weights = dataset, indices, y, weights

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]["ppg"][0], self.y[index], self.weights[index]


def fine_tune(config, manifest, labels, target, output, encoder_path=None, task="regression",
              label_fraction=1., bootstrap_samples=200, protocol="group_disjoint", target_source=None):
    config.validate()
    if config.data.window_mode != "full":
        raise ValueError("Downstream window labels require full input windows")
    if bootstrap_samples < 20:
        raise ValueError("At least 20 cluster bootstrap replicates are required")
    if task not in {"regression", "classification"}:
        raise ValueError("Unknown downstream task")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fine-tuning output must be empty")
    seed_all(config.train.seed, config.train.threads)
    device = device_for(config.train.device)
    provenance = {"train_subjects": [], "train_sources": [], "teacher_training_subjects": [], "teacher_training_sources": []}
    if encoder_path:
        exported = load_checkpoint(encoder_path)
        if exported["model_config"] != config.to_dict()["model"]:
            raise ValueError("Fine-tuning configuration differs from exported encoder architecture")
        from .config import DataConfig
        if DataConfig.from_dict(exported["data_config"]).input_protocol() != config.data.input_protocol():
            raise ValueError("Fine-tuning input protocol differs from pretraining")
        model = load_encoder(encoder_path, device)
        provenance = exported.get("pretraining_provenance", {})
    else:
        model = TempoBridge(config.model).to(device)
        model.reconstruction = model.morph_reconstruction = torch.nn.Identity()
    records, table = read_manifest(manifest, data_config=config.data), TargetTable(labels)
    digest = manifest_digest(records)
    datasets, parts, prepared = {}, {}, {}
    for split in ("train", "val", "test"):
        dataset = PairedDataset(records, config.data, split=split, augment=False, include_ecg=False)
        datasets[split] = dataset
        rows = [dataset.metadata(i) for i in range(len(dataset))]
        part = {k: np.asarray([row[k] for row in rows]) for k in
                ("sample_id", "subject_id", "subject_known", "source", "identity_level")}
        part.update(embedding=np.zeros((len(rows), 1)), scale_seconds=config.data.scales_seconds[0],
                    metadata={"split": split, "encoder_sha256": file_sha256(encoder_path) if encoder_path else "random_init",
                              "pretraining": provenance})
        parts[split] = part
        values, valid = table.arrays(part["sample_id"], [target], config.data.scales_seconds[0])
        take = valid[:, 0]
        prepared[split] = {k: v[take] for k, v in part.items() if isinstance(v, np.ndarray) and v.ndim == 1}
        prepared[split].update(indices=np.flatnonzero(take), y=values[take, 0].astype(np.float64))
        if not take.any():
            raise ValueError(f"No valid downstream labels in {split}")
    audit = validate_splits(parts, protocol, target_source)
    training = prepared["train"]
    keep, groups = select_label_groups(training["subject_id"], label_fraction, config.train.seed)
    prepared["train"] = training = {k: v[keep] for k, v in training.items()}
    weights = subject_weights(training["subject_id"])
    classes = sorted(set(training["y"].tolist())) if task == "classification" else []
    if task == "classification":
        if len(classes) < 2:
            raise ValueError("Selected training groups contain fewer than two classes")
        for split, part in prepared.items():
            if not set(part["y"].tolist()).issubset(classes):
                raise ValueError(f"Unseen downstream class in {split}")
            part["y"] = np.asarray([classes.index(v) for v in part["y"]], dtype=np.int64)
        mean, scale = 0., 1.
    else:
        mean = float(np.average(training["y"], weights=weights))
        scale = float(max(np.sqrt(np.average((training["y"] - mean) ** 2, weights=weights)), 1e-4))
    head = torch.nn.Linear(model.feature_dim, len(classes) if classes else 1).to(device)
    parameters = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.train.lr, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.train.precision == "fp16")
    train_data = LabeledPPG(datasets["train"], training["indices"], training["y"], weights)
    output.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def predict(split):
        model.eval()
        head.eval()
        part = prepared[split]
        data = LabeledPPG(datasets[split], part["indices"], part["y"], np.ones(len(part["y"])))
        predictions = []
        for wave, _, _ in DataLoader(data, batch_size=config.train.batch_size, num_workers=config.data.workers):
            with autocast_context(config, device):
                out = head(model(wave.to(device))).float()
            value = out.softmax(-1) if classes else out[:, 0] * scale + mean
            predictions.append(value.cpu().numpy())
        return np.concatenate(predictions)

    best, history = float("inf"), []
    for epoch in range(config.train.epochs):
        model.train()
        head.train()
        loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(config.train.seed + epoch),
                            num_workers=config.data.workers)
        import itertools
        iterator = iter(loader)
        total_loss, examples = 0., 0
        while batches := list(itertools.islice(iterator, config.train.accumulation)):
            size = sum(len(wave) for wave, _, _ in batches)
            optimizer.zero_grad(set_to_none=True)
            for wave, y, w in batches:
                y, w = y.to(device), w.to(device)
                with autocast_context(config, device):
                    prediction = head(model(wave.to(device))).float()
                    loss = (torch.nn.functional.cross_entropy(prediction, y.long(), reduction="none") if classes else
                            (prediction[:, 0] - (y.float() - mean) / scale).square())
                    loss = (loss * w).sum() / size
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite downstream loss")
                scaler.scale(loss).backward()
                total_loss += float(loss.detach()) * size
                examples += len(wave)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.train.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
        validation = prepared["val"]
        pred = predict("val")
        if classes:
            losses = -np.log(pred[np.arange(len(pred)), validation["y"]].clip(1e-12))
            score = float(np.average(losses, weights=subject_weights(validation["subject_id"])))
        else:
            score = task_metrics(validation["y"], pred, validation["subject_id"], task)["patient_macro_mae"]
        history.append({"epoch": epoch + 1, "train_loss": total_loss / examples, "validation_score": score})
        if score < best:
            best = score
            save_checkpoint(output / "best.pt", {"kind": "ppg_downstream_model", "model": model.state_dict(),
                "head": head.state_dict(), "config": config.to_dict(), "task": task, "target": target,
                "classes": classes, "target_mean": mean, "target_scale": scale, "epoch": epoch + 1,
                "selected_training_groups": groups, "labels_sha256": table.fingerprint,
                "manifest_sha256": digest, "pretraining": provenance})
    selected = load_checkpoint(output / "best.pt")
    model.load_state_dict(selected["model"])
    head.load_state_dict(selected["head"])
    prediction, test = predict("test"), prepared["test"]
    report = {"schema_version": 1, "task": task, "target": target, "units": table.units[target], "classes": classes,
        "initialization": "pretrained" if encoder_path else "random", "label_fraction": label_fraction,
        "seed": config.train.seed, "selected_training_groups": groups, "selected_epoch": selected["epoch"],
        "history": history, "audit": audit, "test": task_metrics(test["y"], prediction, test["subject_id"], task),
        "by_source": {s: task_metrics(test["y"][test["source"] == s], prediction[test["source"] == s],
                                      test["subject_id"][test["source"] == s], task) for s in sorted(set(test["source"]))},
        "bootstrap": cluster_intervals(test["y"], prediction, test["subject_id"], task, bootstrap_samples, config.train.seed),
        "provenance": {"labels_sha256": table.fingerprint, "manifest_sha256": digest,
                       "encoder_sha256": file_sha256(encoder_path) if encoder_path else None}}
    np.savez(output / "predictions.npz", prediction=prediction, y=test["y"], sample_id=test["sample_id"],
             subject_id=test["subject_id"], source=test["source"], task=task, target=target, classes=np.asarray(classes))
    write_json(output / "report.json", report)
    for dataset in datasets.values():
        dataset.close()
    return output / "report.json"
