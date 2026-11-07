"""Cross-fitted, subject-balanced predictability calibration.

The PPG warm-up representation is fixed; only the probes are out-of-fold.
This estimates learnability under these probes, not intrinsic observability or utility.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import manifest_digest, read_manifest
from .teacher import FrozenTeacher
from .utils import file_sha256, seed_all, write_json


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


def calibrate(config, manifest, student_path, teacher_path, output, teacher_kind="native", cache_path=None):
    """Build or verify a reusable frozen-feature cache, then fit calibration only."""
    from .feature_cache import FeatureCache, cache_features

    if Path(output).exists():
        raise FileExistsError("Calibration output exists; choose a new artifact path")
    if cache_path is None:
        cache_path = Path(output).with_name(Path(output).stem + "_features")
        cache_features(config, manifest, student_path, teacher_path, cache_path, teacher_kind)
    cache = FeatureCache(cache_path)
    records = read_manifest(manifest, data_config=config.data)
    teacher = FrozenTeacher(teacher_path, teacher_kind)
    teacher.verify_data(records, config.data.ecg_fs)
    cache.validate(config, manifest_digest(records), teacher.fingerprint,
                   file_sha256(student_path), teacher_kind)
    return calibrate_cached(config, cache, output)


def calibrate_cached(config, cache, output):
    """No encoder forward or waveform access; all transformations use train patients."""
    from .subspace import fit_subspace, partition_patients, patient_moments

    if Path(output).exists():
        raise FileExistsError("Calibration output exists; choose a new artifact path")
    config.validate()
    seed_all(config.train.seed, config.train.threads)
    a = cache.artifact
    cache.validate(config, a["manifest_sha256"], a["teacher_sha256"],
                   a["student_sha256"], a["teacher_kind"])
    subjects, qualities = cache.column("subject_id"), cache.column("ppg_quality")
    sources = cache.column("source")
    method = config.train.transfer_method
    if method in {"none", "physio"}:
        raise ValueError("SSL/physiological supervision does not require teacher calibration")
    partitions = (partition_patients(subjects, config.cosd, config.train.seed)
                  if len(np.unique(subjects)) >= 3 else None)
    normalizers = np.isin(subjects, partitions["basis"]) if partitions else np.ones(len(subjects), bool)
    reports = []
    for scale in range(len(config.data.scales_seconds)):
        x, y = cache.arrays(scale)
        if method == "pw_kd" or config.train.transfer_mode in {"calibrated", "random"}:
            report = crossfit_scale(x, y, subjects, qualities, config.calibration,
                                    config.data.quality_edges, config.train.seed)
        else:
            report = {"weights": [1.] * (len(config.data.quality_edges) + 1), "bins": [], "folds": []}
        # Same final target scaling for full and projected targets when using the same cache/config.
        mean, variance = patient_moments(y[normalizers], subjects[normalizers])
        report.update(target_mean=mean.tolist(), target_scale=np.sqrt(variance).clip(1e-4).tolist())
        if method == "cosd":
            subspace_config = (replace(config.cosd, rank=config.cosd.ranks_per_scale[scale])
                               if config.cosd.ranks_per_scale else config.cosd)
            report["subspace"] = fit_subspace(x, y, subjects, sources, config.calibration,
                                               subspace_config, config.train.seed, qualities, config.data.quality_edges)
        reports.append(report)
    result = {
        "schema_version": 3, "method": method, "student_sha256": a["student_sha256"],
        "teacher_sha256": a["teacher_sha256"], "teacher_kind": a["teacher_kind"],
        "manifest_sha256": a["manifest_sha256"], "train_subjects": a["train_subjects"],
        "feature_spec": a["feature_spec"], "teacher_feature_spec": a["teacher_feature_spec"],
        "selection": a["selection"], "cache_sha256": cache.fingerprint,
        **config.data.input_protocol(), "quality_edges": config.data.quality_edges,
        "calibration_config": asdict(config.calibration), "cosd_config": asdict(config.cosd),
        "normalization_subjects": sorted(np.unique(subjects[normalizers]).tolist()),
        "target_dim": a["target_dim"], "scales": reports,
        "scope": "OOF probes on fixed pretrained representations; COSD audit is independent of direction selection",
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
        from .subspace import TeacherSubspace
        self.transforms = nn.ModuleList([
            TeacherSubspace(s["subspace"]) if "subspace" in s else nn.Identity()
            for s in artifact["scales"]])
        self.output_dims = [s["subspace"]["rank"] if "subspace" in s else len(s["target_mean"])
                            for s in artifact["scales"]]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self, config, manifest_hash, teacher_hash, student_hash, teacher_kind):
        a = self.artifact
        if a.get("schema_version") not in {1, 2, 3}:
            raise ValueError("Unsupported calibration schema")
        expected = {"manifest_sha256": manifest_hash, "teacher_sha256": teacher_hash,
                    "student_sha256": student_hash, "teacher_kind": teacher_kind,
                    "scales_seconds": config.data.scales_seconds,
                    "window_mode": config.data.window_mode,
                    "quality_edges": config.data.quality_edges,
                    "ppg_fs": config.data.ppg_fs, "ecg_fs": config.data.ecg_fs}
        for key, value in expected.items():
            saved_value = a.get(key, "crop") if key == "window_mode" else a.get(key)
            if saved_value != value:
                raise ValueError(f"Stale/incompatible calibration: {key}")
        if a["schema_version"] == 1:
            raise ValueError("Legacy calibration used a different feature contract; recalibrate")
        if a["schema_version"] == 2 and config.train.transfer_method == "cosd":
            raise ValueError("COSD requires a schema v3 projected-target artifact")
        expected_spec = {"version": 1, "layer": config.model.depth - 1,
                         "pool": "context_mean", "dim": config.model.dim}
        if a.get("feature_spec") != expected_spec:
            raise ValueError("Stale/incompatible calibration: feature_spec")
        if a["schema_version"] == 3:
            from .config import COSDConfig
            saved_cosd = asdict(COSDConfig(**a["cosd_config"]))
            for key, expected_value in {"method": config.train.transfer_method,
                                        "cosd_config": asdict(config.cosd),
                                        "calibration_config": asdict(config.calibration)}.items():
                if (saved_cosd if key == "cosd_config" else a.get(key)) != expected_value:
                    raise ValueError(f"Stale/incompatible calibration: {key}")
            if config.train.transfer_method == "cosd" and any("subspace" not in s for s in a["scales"]):
                raise ValueError("COSD artifact is missing a subspace")
        if self.weights.shape != (len(config.data.scales_seconds), len(config.data.quality_edges) + 1):
            raise ValueError("Invalid calibration weight shape")
        if self.mean.shape != (len(config.data.scales_seconds), a["target_dim"]) or self.scale.shape != self.mean.shape:
            raise ValueError("Invalid target standardization shape")
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any() or (self.weights > 1).any():
            raise ValueError("Invalid calibration weights")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid target standardization")

    def lookup(self, scale_index, ppg_quality, ecg_quality, valid, mode="calibrated",
               ppg_quality_weight=False, ecg_quality_weight=True):
        bins = torch.bucketize(ppg_quality.float(), self.edges, right=True)
        base = self.weights[scale_index, bins]
        if mode == "random":
            base = base[torch.randperm(len(base), device=base.device)]
        if mode in {"all", "quality"}:
            base = torch.ones_like(base)
        if mode == "none":
            base = torch.zeros_like(base)
        reliability = torch.ones_like(base)
        if mode != "all":
            if ecg_quality_weight:
                reliability = reliability * ecg_quality.float()
            if ppg_quality_weight:
                reliability = reliability * ppg_quality.float()
        return (base * reliability * valid.float()).detach()

    def normalize(self, target, scale_index):
        if "subspace" in self.artifact["scales"][scale_index]:
            return self.transforms[scale_index](target)
        return (target.float() - self.mean[scale_index]) / self.scale[scale_index]

    def direction_weights(self, scale_index):
        if "subspace" in self.artifact["scales"][scale_index]:
            return self.transforms[scale_index].weights
        return torch.ones(self.output_dims[scale_index], device=self.mean.device)
