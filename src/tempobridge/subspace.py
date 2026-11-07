"""Patient-disjoint discovery and audit of teacher-coordinate targets.

Only probes are cross-fitted. Encoders are frozen, train-patient-pretrained models.
Basis patients fit all target transformations; discovery patients select directions;
audit patients never influence either. This does not identify causal physiology.
"""

from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from .calibration import fit_probe, subject_weights


def patient_moments(values, subjects):
    values = np.asarray(values, np.float64)
    weights = subject_weights(np.asarray(subjects))
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    return mean, np.maximum(variance, 0)


def partition_patients(subjects, config, seed):
    unique = np.random.default_rng(seed).permutation(np.unique(subjects))
    if len(unique) < 3:
        raise ValueError("COSD needs separate basis, discovery and audit patients")
    nb = max(1, int(len(unique) * config.basis_fraction))
    na = max(1, int(len(unique) * config.audit_fraction))
    return {"basis": sorted(unique[:nb].tolist()),
            "audit": sorted(unique[nb:nb + na].tolist()),
            "discovery": sorted(unique[nb + na:].tolist())}


def make_basis(targets, subjects, kind, seed):
    mean, variance = patient_moments(targets, subjects)
    scale = np.sqrt(variance).clip(1e-4)
    standardized = (targets - mean) / scale
    d = targets.shape[1]
    weights = subject_weights(np.asarray(subjects))
    covariance = (standardized * weights[:, None]).T @ standardized / weights.sum()
    if kind == "pca":
        eigenvalues, vectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        vectors = vectors[:, order]
        # Deterministic signs; repeated eigenvalues still define subspaces, not unique axes.
        signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(d)])
        vectors *= np.where(signs == 0, 1, signs)
    elif kind == "random":
        vectors, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    else:
        vectors = np.eye(d)
    projected_variance = np.maximum(np.diag(vectors.T @ covariance @ vectors), 0)
    return mean, scale, vectors, projected_variance


def crossfit_directions(features, targets, subjects, config, seed, repeats=1):
    """Return errors per coordinate; all probe statistics use fit-fold patients only."""
    x, y = np.asarray(features, np.float64), np.asarray(targets, np.float64)
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    if len(unique) < config.folds + 1:
        raise ValueError("Need at least folds + 1 discovery patients")
    kinds = ["ridge", "mlp"] if config.probe == "consensus" else [config.probe]
    errors = {kind: np.zeros_like(y) for kind in kinds}
    null = np.zeros_like(y)
    folds = []
    for repeat in range(repeats):
        partitions = np.array_split(np.random.default_rng(seed + repeat).permutation(unique), config.folds)
        visits = np.zeros(len(y), int)
        for fold, heldout in enumerate(partitions):
            score = np.isin(subjects, heldout)
            fit = ~score
            visits[score] += 1
            folds.append({"repeat": repeat, "fold": fold,
                          "fit_subjects": sorted(np.unique(subjects[fit]).tolist()),
                          "score_subjects": sorted(heldout.tolist())})
            for kind in kinds:
                pred, mean, scale = fit_probe(x[fit], y[fit], x[score], subjects[fit],
                                              config, kind, seed + 1009 * repeat + fold)
                errors[kind][score] += ((pred - y[score]) / scale) ** 2 / repeats
            null[score] += ((mean - y[score]) / scale) ** 2 / repeats
        if not np.all(visits == 1):
            raise RuntimeError("OOF visits must equal one for each repeat")
    return errors, null, folds


def direction_skill(errors, baseline, subjects):
    weights = subject_weights(np.asarray(subjects))
    denominator = np.average(baseline, axis=0, weights=weights)
    skills = []
    for error in errors.values():
        numerator = np.average(error, axis=0, weights=weights)
        skill = np.zeros_like(denominator)
        np.divide(numerator, denominator, out=skill, where=denominator > 1e-8)
        skill = np.where(denominator > 1e-8, 1 - skill, 0.)
        skills.append(skill)
    return np.min(skills, axis=0)


def bootstrap_skill(errors, baseline, subjects, count, confidence, seed):
    """Patient-cluster bootstrap conditional on fitted OOF probes and fixed basis."""
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    by_patient = {key: np.stack([value[subjects == p].mean(0) for p in unique])
                  for key, value in {**errors, "__null__": baseline}.items()}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(count):
        take = rng.integers(len(unique), size=len(unique))
        den = by_patient["__null__"][take].mean(0)
        scores = []
        for kind in errors:
            ratio = np.zeros_like(den)
            np.divide(by_patient[kind][take].mean(0), den, out=ratio, where=den > 1e-8)
            scores.append(np.where(den > 1e-8, 1 - ratio, 0.))
        draws.append(np.min(scores, axis=0))
    alpha = (1 - confidence) / 2
    return np.quantile(draws, [alpha, 1 - alpha], axis=0)


def fit_subspace(features, targets, subjects, sources, probe_config, config, seed, qualities=None, quality_edges=()):
    x, y = np.asarray(features, np.float64), np.asarray(targets, np.float64)
    subjects, sources = np.asarray(subjects), np.asarray(sources)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(subjects) != len(y) or len(sources) != len(y):
        raise ValueError("Invalid feature/target/metadata shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Subspace calibration requires finite features/targets")
    if config.rank > y.shape[1]:
        raise ValueError("Requested COSD rank exceeds teacher dimension")
    partitions = partition_patients(subjects, config, seed)
    masks = {key: np.isin(subjects, ids) for key, ids in partitions.items()}
    basis, discover, audit = (masks[key] for key in ("basis", "discovery", "audit"))
    mean, scale, vectors, variance = make_basis(y[basis], subjects[basis], config.basis, seed)
    scores = ((y - mean) / scale) @ vectors
    errors, null, folds = crossfit_directions(x[discover], scores[discover], subjects[discover],
                                             probe_config, seed, config.repeats)
    skill = direction_skill(errors, null, subjects[discover])
    intervals = bootstrap_skill(errors, null, subjects[discover], config.bootstrap_samples,
                                 config.confidence, seed + 101)
    criterion = intervals[0] if config.stability else skill
    quality_reports = []
    if config.quality_strata:
        if qualities is None or len(qualities) != len(y):
            raise ValueError("Quality-stratified selection requires quality metadata")
        bins = np.digitize(np.asarray(qualities)[discover], quality_edges)
        for group in range(len(quality_edges) + 1):
            mask = bins == group
            count = len(np.unique(subjects[discover][mask]))
            if count < probe_config.min_subjects_per_bin:
                quality_reports.append({"bin": group, "groups": count, "eligible": False})
                criterion = np.minimum(criterion, np.zeros_like(criterion))
                continue
            group_errors = {k: v[mask] for k, v in errors.items()}
            group_skill = direction_skill(group_errors, null[mask], subjects[discover][mask])
            group_ci = bootstrap_skill(group_errors, null[mask], subjects[discover][mask],
                                       config.bootstrap_samples, config.confidence, seed + 201 + group)
            criterion = np.minimum(criterion, group_ci[0] if config.stability else group_skill)
            quality_reports.append({"bin": group, "groups": count, "eligible": True,
                                    "skill": group_skill.tolist(), "ci": group_ci.T.tolist()})
    eligible = variance > 1e-8
    if config.selection == "topk_oof_skill":
        eligible &= criterion > config.min_skill
        order = np.argsort(-criterion, kind="stable")
    elif config.selection == "random":
        order = np.random.default_rng(seed + 707).permutation(y.shape[1])
    elif config.selection == "variance":
        order = np.argsort(-variance, kind="stable")
    else:
        order = np.arange(y.shape[1])
    selected = order[eligible[order]][:config.rank]
    # The independently held-out audit is evaluated AFTER the transformation is fixed.
    kinds = ["ridge", "mlp"] if probe_config.probe == "consensus" else [probe_config.probe]
    audit_errors = {}
    fit = basis | discover
    for kind in kinds:
        pred, amean, astd = fit_probe(x[fit], scores[fit], x[audit], subjects[fit],
                                     probe_config, kind, seed + 65537)
        audit_errors[kind] = ((pred - scores[audit]) / astd) ** 2
    audit_null = ((amean - scores[audit]) / astd) ** 2
    audit_skill = direction_skill(audit_errors, audit_null, subjects[audit])
    source_reports = {}
    for source in sorted(set(sources[audit].tolist())):
        mask = sources[audit] == source
        source_reports[source] = {
            "patients": len(np.unique(subjects[audit][mask])), "windows": int(mask.sum()),
            "skill": direction_skill({k: v[mask] for k, v in audit_errors.items()},
                                      audit_null[mask], subjects[audit][mask]).tolist()}
    loss_weights = np.clip(skill[selected], 0, 1) if config.soft else np.ones(len(selected))
    return {
        "version": 1, "basis_kind": config.basis, "selection": config.selection,
        "teacher_mean": mean.tolist(), "teacher_scale": scale.tolist(),
        "projection": vectors[:, selected].tolist(), "rank": len(selected),
        "requested_rank": config.rank, "selected_directions": selected.tolist(),
        "direction_weights": loss_weights.tolist(), "direction_skill": skill.tolist(),
        "direction_ci": intervals.T.tolist(), "candidate_variance": variance.tolist(),
        "quality_strata": quality_reports,
        "retained_energy_fraction": float(variance[selected].sum() / max(variance.sum(), 1e-8)),
        "partitions": partitions, "folds": folds,
        "audit": {"skill": audit_skill.tolist(), "by_source": source_reports,
                  "used_for_selection": False},
        "config": asdict(config), "probe_config": asdict(probe_config), "seed": seed,
        "scope": "probe OOF; bootstrap conditional on fixed probes; basis/discovery/audit patients disjoint",
    }


class TeacherSubspace(nn.Module):
    """A frozen affine teacher target transform; never part of the exported encoder."""
    def __init__(self, artifact):
        super().__init__()
        if artifact.get("version") != 1:
            raise ValueError("Unsupported subspace version")
        self.artifact = artifact
        self.rank = int(artifact["rank"])
        mean = torch.tensor(artifact["teacher_mean"], dtype=torch.float32)
        scale = torch.tensor(artifact["teacher_scale"], dtype=torch.float32)
        projection = torch.tensor(artifact["projection"], dtype=torch.float32)
        weights = torch.tensor(artifact["direction_weights"], dtype=torch.float32)
        if mean.ndim != 1 or scale.shape != mean.shape or projection.shape != (len(mean), self.rank):
            raise ValueError("Invalid subspace shape")
        if weights.shape != (self.rank,):
            raise ValueError("Invalid direction weights")
        if any(not torch.isfinite(t).all() for t in (mean, scale, projection, weights)) or (scale <= 0).any():
            raise ValueError("Nonfinite or nonpositive subspace normalization")
        if (weights < 0).any() or (weights > 1).any():
            raise ValueError("Direction weights must be in [0,1]")
        if not torch.allclose(projection.T @ projection, torch.eye(self.rank), atol=1e-4, rtol=1e-4):
            raise ValueError("Subspace columns must be orthonormal")
        for name, value in (("mean", mean), ("scale", scale), ("projection", projection), ("weights", weights)):
            self.register_buffer(name, value)

    @torch.no_grad()
    def forward(self, teacher_features):
        if teacher_features.ndim != 2 or teacher_features.shape[1] != len(self.mean):
            raise ValueError("Teacher feature dimension differs from subspace")
        if not torch.isfinite(teacher_features).all():
            raise ValueError("Teacher features must be finite")
        return ((teacher_features.float() - self.mean) / self.scale) @ self.projection
