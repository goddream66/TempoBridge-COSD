"""Cross-fitted, subject-balanced predictability calibration.

The PPG warm-up representation is fixed; only the probes are out-of-fold.
This estimates learnability under these probes, not intrinsic observability or utility.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import PairedDataset, manifest_digest, read_manifest, subject_ids
from .model import ModelConfig, TempoBridge
from .sampling import calibration_indices
from .teacher import FrozenTeacher
from .utils import device_for, file_sha256, load_checkpoint, seed_all, write_json


def subject_weights(subjects):
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w / w.mean()


def balanced_mean(values, subjects):
    return float(np.mean([np.mean(values[subjects == s]) for s in np.unique(subjects)]))


def fit_probe(x_train, y_train, x_test, subjects_train, config, kind, seed):
    weights = subject_weights(subjects_train)
    xm = np.average(x_train, axis=0, weights=weights)
    xs = np.sqrt(np.average((x_train - xm) ** 2, axis=0, weights=weights)).clip(1e-4)
    ym = np.average(y_train, axis=0, weights=weights)
    ys = np.sqrt(np.average((y_train - ym) ** 2, axis=0, weights=weights)).clip(1e-4)
    x, xt, y = (x_train - xm) / xs, (x_test - xm) / xs, (y_train - ym) / ys
    if kind == "ridge":
        root = np.sqrt(weights)[:, None]
        a, b = x * root, y * root
        if a.shape[0] < a.shape[1]:
            coefficients = a.T @ np.linalg.solve(a @ a.T + config.ridge * np.eye(len(a)), b)
        else:
            coefficients = np.linalg.solve(a.T @ a + config.ridge * np.eye(a.shape[1]), a.T @ b)
        prediction = xt @ coefficients
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(x.shape[1], config.mlp_hidden), nn.GELU(),
                                nn.Linear(config.mlp_hidden, y.shape[1]))
            optimizer = torch.optim.AdamW(net.parameters(), lr=config.mlp_lr, weight_decay=0.01)
            tx, ty, tw = (torch.tensor(z, dtype=torch.float32) for z in (x, y, weights))
            for _ in range(config.mlp_steps):
                optimizer.zero_grad(set_to_none=True)
                # Uniform minibatches retain the subject-balanced weighted objective.
                indices = (torch.randint(len(tx), (config.probe_batch_size,))
                           if len(tx) > config.probe_batch_size else slice(None))
                loss = ((net(tx[indices]) - ty[indices]).square().mean(-1) * tw[indices]).mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                prediction = np.concatenate([net(chunk).numpy() for chunk in
                    torch.tensor(xt, dtype=torch.float32).split(config.probe_batch_size)])
    return prediction * ys + ym, ym, ys


def crossfit_scale(features, targets, subjects, quality, config, edges, seed=42):
    features, targets = np.asarray(features, np.float64), np.asarray(targets, np.float64)
    subjects, quality = np.asarray(subjects), np.asarray(quality)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Nonfinite calibration features/targets")
    unique = np.unique(subjects)
    if len(unique) < config.folds + 1:
        raise ValueError("Need at least folds + 1 paired training patients")
    shuffled = np.random.default_rng(seed).permutation(unique)
    partitions = np.array_split(shuffled, config.folds)
    kinds = ["ridge", "mlp"] if config.probe == "consensus" else [config.probe]
    errors = {kind: np.zeros(len(subjects)) for kind in kinds}
    baseline, visits, audit = np.zeros(len(subjects)), np.zeros(len(subjects), int), []
    for fold, heldout in enumerate(partitions):
        test = np.isin(subjects, heldout)
        fit = ~test
        audit.append({"fold": fold, "fit_subjects": sorted(set(subjects[fit].tolist())),
                      "score_subjects": sorted(set(subjects[test].tolist()))})
        visits[test] += 1
        for kind in kinds:
            pred, mean, std = fit_probe(features[fit], targets[fit], features[test],
                                        subjects[fit], config, kind, seed + fold)
            # Normalize with fit-fold statistics only. Score the mean baseline identically.
            errors[kind][test] = np.mean(((pred - targets[test]) / std) ** 2, axis=1)
            baseline[test] = np.mean(((mean - targets[test]) / std) ** 2, axis=1)
    if not np.all(visits == 1):
        raise RuntimeError("Every calibration example must be scored exactly once")
    bins = np.digitize(quality, edges)
    weights, reports = [], []
    for group in range(len(edges) + 1):
        select = bins == group
        count = len(np.unique(subjects[select]))
        r2 = {}
        if count >= config.min_subjects_per_bin:
            base_error = balanced_mean(baseline[select], subjects[select])
            for kind in kinds:
                error = balanced_mean(errors[kind][select], subjects[select])
                r2[kind] = 1 - error / base_error if base_error > 1e-8 else 0.0
            # Conservative agreement across probe families, not a claim of optimal recovery.
            weight = float(np.clip(min(r2.values()), 0, 1))
        else:
            weight = 0.0
        weights.append(weight)
        reports.append({"bin": group, "subjects": count, "samples": int(select.sum()),
                        "r2": r2, "weight": weight,
                        "sufficient_patients": count >= config.min_subjects_per_bin})
    w = subject_weights(subjects)
    mean = np.average(targets, axis=0, weights=w)
    scale = np.sqrt(np.average((targets - mean) ** 2, axis=0, weights=w)).clip(1e-4)
    return {"weights": weights, "bins": reports, "folds": audit,
            "target_mean": mean.tolist(), "target_scale": scale.tolist()}


def calibrate(config, manifest, student_path, teacher_path, output, teacher_kind="native"):
    if Path(output).exists():
        raise FileExistsError("Calibration output exists; choose a new artifact path")
    seed_all(config.train.seed, config.train.threads)
    config.validate()
    records = read_manifest(manifest, data_config=config.data)
    digest = manifest_digest(records)
    checkpoint = load_checkpoint(student_path)
    if checkpoint.get("modality") != "ppg" or checkpoint.get("global_step", 0) < 1:
        raise ValueError("Calibration requires a trained PPG warm-up checkpoint")
    if checkpoint["manifest_sha256"] != digest:
        raise ValueError("Student warm-up and calibration manifest differ")
    if asdict(ModelConfig.from_dict(checkpoint["model_config"])) != config.to_dict()["model"]:
        raise ValueError("Student architecture differs from config")
    device = device_for(config.train.device)
    model = TempoBridge(ModelConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    teacher = FrozenTeacher(teacher_path, teacher_kind).to(device)
    teacher.verify_data(records, config.data.ecg_fs)
    dataset = PairedDataset(records, config.data, augment=False)
    indices = calibration_indices(dataset.records, config.calibration.max_windows_per_subject,
                                  config.train.seed)
    loader = DataLoader(Subset(dataset, indices), batch_size=config.train.batch_size, shuffle=False,
                        num_workers=config.data.workers)
    features, targets = [[] for _ in config.data.scales_seconds], [[] for _ in config.data.scales_seconds]
    subjects, qualities = [], []
    with torch.inference_mode():
        for batch in loader:
            valid = (batch["subject_known"] & batch["ecg_valid"] &
                     (batch["ecg_quality"] >= config.calibration.min_ecg_quality))
            if not valid.any():
                continue
            subjects.extend(np.asarray(batch["subject_id"])[valid.numpy()].tolist())
            qualities.extend(batch["ppg_quality"][valid].tolist())
            for s in range(len(features)):
                x, e = batch["ppg"][s][valid].to(device), batch["ecg"][s][valid].to(device)
                features[s].append(model(x).cpu().numpy())
                targets[s].append(teacher(e).cpu().numpy())
    dataset.close()
    if not subjects:
        raise ValueError("No reliable paired ECG for calibration")
    reports = [crossfit_scale(np.concatenate(x), np.concatenate(y), subjects, qualities,
                             config.calibration, config.data.quality_edges, config.train.seed)
               for x, y in zip(features, targets)]
    result = {
        "schema_version": 1, "student_sha256": file_sha256(student_path),
        "teacher_sha256": teacher.fingerprint, "teacher_kind": teacher_kind,
        "manifest_sha256": digest, "train_subjects": sorted(subject_ids(records, "train")),
        "selection": {"seed": config.train.seed, "candidates": len(indices),
                      "reliable_pairs": len(subjects), "unknown_ids_excluded": True,
                      "max_windows_per_subject": config.calibration.max_windows_per_subject},
        "scales_seconds": config.data.scales_seconds, "ppg_fs": config.data.ppg_fs,
        "ecg_fs": config.data.ecg_fs, "quality_edges": config.data.quality_edges,
        "calibration_config": config.to_dict()["calibration"],
        "target_dim": len(reports[0]["target_mean"]), "scales": reports,
        "scope": "out-of-fold probes on a fixed train-patient-pretrained representation",
    }
    write_json(output, result)
    return result


class CalibrationWeights(nn.Module):
    def __init__(self, artifact):
        super().__init__()
        self.artifact = artifact
        self.register_buffer("edges", torch.tensor(artifact["quality_edges"], dtype=torch.float32))
        self.register_buffer("weights", torch.tensor([s["weights"] for s in artifact["scales"]]))
        self.register_buffer("mean", torch.tensor([s["target_mean"] for s in artifact["scales"]]))
        self.register_buffer("scale", torch.tensor([s["target_scale"] for s in artifact["scales"]]))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self, config, manifest_hash, teacher_hash, student_hash, teacher_kind):
        a = self.artifact
        if a.get("schema_version") != 1:
            raise ValueError("Unsupported calibration schema")
        expected = {"manifest_sha256": manifest_hash, "teacher_sha256": teacher_hash,
                    "student_sha256": student_hash, "teacher_kind": teacher_kind,
                    "scales_seconds": config.data.scales_seconds,
                    "quality_edges": config.data.quality_edges,
                    "ppg_fs": config.data.ppg_fs, "ecg_fs": config.data.ecg_fs}
        for key, value in expected.items():
            if a.get(key) != value:
                raise ValueError(f"Stale/incompatible calibration: {key}")
        if self.weights.shape != (len(config.data.scales_seconds), len(config.data.quality_edges) + 1):
            raise ValueError("Invalid calibration weight shape")
        if self.mean.shape != (len(config.data.scales_seconds), a["target_dim"]) or self.scale.shape != self.mean.shape:
            raise ValueError("Invalid target standardization shape")
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any() or (self.weights > 1).any():
            raise ValueError("Invalid calibration weights")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid target standardization")

    def lookup(self, scale_index, ppg_quality, ecg_quality, valid, mode="calibrated"):
        bins = torch.bucketize(ppg_quality.float(), self.edges, right=True)
        base = self.weights[scale_index, bins]
        if mode == "random":
            base = base[torch.randperm(len(base), device=base.device)]
        if mode in {"all", "quality"}:
            base = torch.ones_like(base)
        if mode == "none":
            base = torch.zeros_like(base)
        reliability = torch.ones_like(base) if mode == "all" else ecg_quality.float()
        return (base * reliability * valid.float()).detach()

    def normalize(self, target, scale_index):
        return (target.float() - self.mean[scale_index]) / self.scale[scale_index]
