"""Descriptive audits of frozen features; no claim of causal shared physiology.

Within-record centering is retrospective analysis, never a deployment input.
Identity tests are closed-set diagnostics, not unseen-patient task evaluation.
"""

import json
from pathlib import Path

import numpy as np

from .calibration import fit_probe, subject_weights
from .config import CalibrationConfig
from .feature_cache import FeatureCache
from .physio_targets import TargetTable
from .subspace import crossfit_directions, direction_skill
from .utils import file_sha256, write_json


def within_record_skill(target, prediction, subjects, recordings, known):
    """Compare changes after removing each record's own mean in held-out rows."""
    residual, estimated = np.zeros_like(target), np.zeros_like(prediction)
    keep = np.zeros(len(target), bool)
    keys = sorted({(str(p), str(r)) for p, r, k in zip(subjects, recordings, known) if k})
    used = 0
    for patient, record in keys:
        take = (subjects == patient) & (recordings == record) & known
        if take.sum() < 2:
            continue
        keep[take] = True
        residual[take] = target[take] - target[take].mean(0)
        estimated[take] = prediction[take] - prediction[take].mean(0)
        used += 1
    if not keep.any():
        return {"available": False, "reason": "no_known_record_with_two_windows"}
    weights = subject_weights(subjects[keep])
    denominator = float(np.average(residual[keep] ** 2, axis=0, weights=weights).mean())
    error = float(np.average((estimated[keep] - residual[keep]) ** 2, axis=0, weights=weights).mean())
    return {"available": denominator > 1e-8, "skill": 1 - error / denominator if denominator > 1e-8 else None,
            "error": error, "null_error": denominator, "records": used, "windows": int(keep.sum()),
            "groups": len(np.unique(subjects[keep])), "baseline": "retrospective_record_mean",
            "scope": "held-out probe predictions; both prediction and target centered within record"}


def closed_set_identity(features, labels, times, seed=42, max_groups=256):
    """Nearest-centroid diagnostic on disjoint ordered windows of the SAME identities."""
    unique, counts = np.unique(labels, return_counts=True)
    candidates = unique[counts >= 2]
    if len(candidates) < 2 or features.shape[1] == 0:
        return {"available": False, "reason": "insufficient_repeated_identities_or_dimensions"}
    selected = np.random.default_rng(seed).permutation(candidates)[:max_groups]
    centers, test, truth = [], [], []
    for j, group in enumerate(selected):
        indices = np.flatnonzero(labels == group)
        indices = indices[np.argsort(times[indices], kind="stable")]
        half = max(1, len(indices) // 2)
        centers.append(features[indices[:half]].mean(0))
        test.extend(indices[half:])
        truth.extend([j] * (len(indices) - half))
    centers, values = np.asarray(centers), features[test]
    scale = centers.std(0).clip(1e-4)
    centers, values = centers / scale, values / scale
    predicted = []
    for start in range(0, len(values), 256):
        batch = values[start:start + 256]
        distance = (batch ** 2).sum(1)[:, None] + (centers ** 2).sum(1) - 2 * batch @ centers.T
        predicted.extend(distance.argmin(1).tolist())
    correct = np.asarray(predicted) == truth
    truth = np.asarray(truth)
    return {"available": True, "group_macro_accuracy": float(np.mean([correct[truth == j].mean() for j in range(len(selected))])),
            "groups": len(selected), "windows": len(test), "uniform_chance": 1 / len(selected),
            "time_order_known": bool(np.all(times[test] >= 0)),
            "scope": "closed-set centroid diagnostic; same identities, different windows; not downstream generalization"}


def source_decodability(features, subjects, sources, config, seed):
    classes = sorted(set(sources.tolist()))
    if len(classes) < 2 or features.shape[1] == 0:
        return {"available": False, "reason": "one_source_or_no_dimensions"}
    target = np.asarray([classes.index(s) for s in sources])
    folds = np.array_split(np.random.default_rng(seed).permutation(np.unique(subjects)), config.folds)
    prediction = np.full(len(target), -1)
    for fold, heldout in enumerate(folds):
        score = np.isin(subjects, heldout)
        fit = ~score
        if not score.any() or set(target[fit]) != set(range(len(classes))):
            continue
        values, _, _ = fit_probe(features[fit], np.eye(len(classes))[target[fit]], features[score],
                                 subjects[fit], config, "ridge", seed + fold)
        prediction[score] = values.argmax(1)
    available = prediction >= 0
    if not available.any():
        return {"available": False, "reason": "folds_lack_source_support"}
    weights = subject_weights(subjects[available])
    return {"available": True, "patient_macro_accuracy": float(np.average(prediction[available] == target[available], weights=weights)),
            "coverage": float(available.mean()), "sources": classes,
            "scope": "patient-OOF ridge source classifier; source information alone is not proof of harmful shortcut"}


def audit_representations(cache_path, calibration_path, output, labels=None, targets=(), seed=42, max_identity_groups=256):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if max_identity_groups < 2:
        raise ValueError("Identity diagnostic requires max_identity_groups >=2")
    cache = FeatureCache(cache_path)
    artifact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if artifact.get("cache_sha256") != cache.fingerprint or artifact.get("method") != "cosd":
        raise ValueError("Audit requires a COSD artifact bound to this exact feature cache")
    subjects, records, sources = (cache.column(k) for k in ("subject_id", "recording_id", "source"))
    known, times = cache.column("recording_known").astype(bool), cache.column("window_start_s").astype(float)
    config = CalibrationConfig(**artifact["calibration_config"])
    table = TargetTable(labels) if labels else None
    if targets and table is None:
        raise ValueError("Physiological target audits require --labels")
    reports = []
    for index, scale in enumerate(artifact["scales"]):
        x, raw = (np.asarray(a, np.float64) for a in cache.arrays(index))
        subspace = scale["subspace"]
        normalized = (raw - subspace["teacher_mean"]) / np.asarray(subspace["teacher_scale"])
        projection = np.asarray(subspace["projection"])
        selected = normalized @ projection
        residual = normalized - selected @ projection.T
        audit = np.isin(subjects, subspace["partitions"]["audit"])
        fit = np.isin(subjects, subspace["partitions"]["basis"] + subspace["partitions"]["discovery"])
        report = {"scale_seconds": artifact["scales_seconds"][index], "rank": subspace["rank"],
                  "representations": {}, "teacher_prediction": {}}
        representations = {"ppg": x, "teacher_full": normalized, "teacher_selected": selected,
                           "teacher_residual": residual}
        for name, values in representations.items():
            description = {"dimensions": values.shape[1],
                "source": source_decodability(values, subjects, sources, config, seed),
                "subject_identity": closed_set_identity(values, subjects, times, seed, max_identity_groups)}
            if known.any():
                record_labels = np.asarray([json.dumps([str(p), str(r)]) for p, r in zip(subjects[known], records[known])])
                description["record_identity"] = closed_set_identity(values[known], record_labels, times[known], seed, max_identity_groups)
            if values.shape[1] > 0:
                q = np.stack([cache.column("ppg_quality"), cache.column("ecg_quality")], axis=1).astype(float)
                errors, null, _ = crossfit_directions(values, q, subjects, config, seed)
                description["quality_oof_skill"] = dict(zip(("ppg_quality", "ecg_quality"), direction_skill(errors, null, subjects).tolist()))
            physiology = {}
            for target in targets:
                y, valid = table.arrays(cache.column("sample_id"), [target], report["scale_seconds"])
                mask = valid[:, 0]
                if values.shape[1] == 0 or len(np.unique(subjects[mask])) < config.folds + 1:
                    physiology[target] = {"available": False, "reason": "insufficient_groups_or_dimensions"}
                    continue
                errors, null, _ = crossfit_directions(values[mask], y[mask], subjects[mask], config, seed)
                physiology[target] = {"available": True, "skill": float(direction_skill(errors, null, subjects[mask])[0]),
                                       "windows": int(mask.sum()), "units": table.units[target]}
            description["physiological_probes"] = physiology
            report["representations"][name] = description
        for name, target in (("full", normalized), ("selected", selected), ("residual", residual)):
            if not target.shape[1]:
                report["teacher_prediction"][name] = {"available": False, "reason": "rank_zero"}
                continue
            predicted, mean, std = fit_probe(x[fit], target[fit], x[audit], subjects[fit], config, "ridge", seed)
            y, p, baseline = target[audit] / std, predicted / std, mean / std
            error, null = (p - y) ** 2, (baseline - y) ** 2
            weights = subject_weights(subjects[audit])
            denominator = float(np.average(null, axis=0, weights=weights).mean())
            score = 1 - float(np.average(error, axis=0, weights=weights).mean()) / denominator if denominator > 1e-8 else None
            quality_reports = {}
            bins = np.digitize(cache.column("ppg_quality")[audit], artifact["quality_edges"])
            for quality in range(len(artifact["quality_edges"]) + 1):
                take = bins == quality
                if not take.any():
                    quality_reports[str(quality)] = {"windows": 0}
                    continue
                w = subject_weights(subjects[audit][take])
                den = float(np.average(null[take], axis=0, weights=w).mean())
                quality_reports[str(quality)] = {"windows": int(take.sum()),
                    "skill": 1 - float(np.average(error[take], axis=0, weights=w).mean()) / den if den > 1e-8 else None}
            report["teacher_prediction"][name] = {"available": True, "pooled_audit_skill": score,
                "within_record": within_record_skill(y, p, subjects[audit], records[audit], known[audit]),
                "quality_strata": quality_reports, "used_for_selection": False}
        reports.append(report)
    write_json(output, {"schema_version": 1, "scope": "descriptive audits on frozen training features; probe OOF only",
        "selection_fixed": True, "no_causal_or_utility_guarantee": True, "scales": reports,
        "cache_sha256": cache.fingerprint, "calibration_sha256": file_sha256(calibration_path),
        "labels_sha256": table.fingerprint if table else None, "seed": seed,
        "max_identity_groups": max_identity_groups})
    return output
