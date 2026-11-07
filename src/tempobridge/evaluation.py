"""Patient-disjoint frozen probes and paired patient-cluster uncertainty.

Hyperparameters use validation only. Label fractions select whole training groups.
Source-held-out evaluation checks encoder AND teacher AND downstream-head sources.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr

from .calibration import subject_weights
from .physio_targets import TargetTable
from .utils import file_sha256, write_json


def load_features(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files if k != "metadata_json"}
        if "metadata_json" not in archive:
            raise ValueError("Features lack provenance; regenerate with the features command")
        result["metadata"] = json.loads(str(archive["metadata_json"].item()))
    required = {"embedding", "sample_id", "subject_id", "source", "subject_known", "scale_seconds"}
    if not required.issubset(result):
        raise ValueError("Incomplete feature metadata")
    x = result["embedding"]
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Feature matrix must be nonempty, finite, and two-dimensional")
    for key in required - {"embedding", "scale_seconds"}:
        if result[key].shape != (len(x),):
            raise ValueError(f"Invalid feature column: {key}")
    if len(set(result["sample_id"].tolist())) != len(x):
        raise ValueError("Duplicate feature sample IDs")
    result["file_sha256"] = file_sha256(path)
    return result


def validate_splits(parts, protocol="group_disjoint", target_source=None):
    if protocol not in {"group_disjoint", "source_held_out"}:
        raise ValueError("Unknown evaluation protocol")
    for name, part in parts.items():
        if part["metadata"].get("split") != name:
            raise ValueError(f"Feature split metadata must be {name}")
        if not part["subject_known"].all():
            raise ValueError("Evaluation requires known group IDs; unknown rows must be audited separately")
    for i, a in enumerate(parts):
        for b in list(parts)[i + 1:]:
            for key in ("subject_id", "sample_id"):
                if set(parts[a][key]) & set(parts[b][key]):
                    raise ValueError(f"{key} leakage between {a} and {b}")
    hashes = {p["metadata"]["encoder_sha256"] for p in parts.values()}
    if len(hashes) != 1 or len({p["embedding"].shape[1] for p in parts.values()}) != 1:
        raise ValueError("Every split must use the identical frozen encoder")
    if len({float(p["scale_seconds"]) for p in parts.values()}) != 1:
        raise ValueError("Evaluation splits use different input durations")
    provenance = parts["train"]["metadata"].get("pretraining", {})
    heldout = set(parts["val"]["subject_id"]) | set(parts["test"]["subject_id"])
    unknown = []
    for key in ("train_subjects", "teacher_training_subjects"):
        ids = provenance.get(key)
        if ids is None:
            unknown.append(key)
        elif heldout & set(ids):
            raise ValueError(f"Pretraining overlap with downstream held-out groups: {key}")
    if protocol == "source_held_out":
        if not target_source or set(parts["test"]["source"]) != {target_source}:
            raise ValueError("Source-held-out test must contain exactly target_source")
        for name in ("train", "val"):
            if target_source in parts[name]["source"]:
                raise ValueError("Target source appears in downstream fitting/model selection")
        for key in ("train_sources", "teacher_training_sources"):
            sources = provenance.get(key)
            if sources is None or target_source in sources:
                raise ValueError(f"Cannot establish source-held-out pretraining: {key}")
        if unknown:
            raise ValueError("Cannot establish source-held-out group separation with unknown pretraining provenance")
    levels = sorted({str(level) for p in parts.values() for level in p.get("identity_level", ["unspecified"])})
    return {"protocol": protocol, "target_source": target_source,
            "identity_levels": levels, "unknown_pretraining_provenance": unknown,
            "claim": "group-disjoint evaluation; patient-disjoint only where identity mapping is verified"}


def select_label_groups(subjects, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("Label fraction must lie in (0,1]")
    unique = np.random.default_rng(seed).permutation(np.unique(subjects))
    chosen = unique[:max(1, int(np.ceil(len(unique) * fraction)))]
    return np.isin(subjects, chosen), sorted(chosen.tolist())


def classification_scores(y, probabilities):
    y = np.asarray(y, int)
    probabilities = np.asarray(probabilities)
    classes = probabilities.shape[1]
    predicted = probabilities.argmax(1)
    aucs, aps, f1s = [], [], []
    for c in range(classes):
        positive = y == c
        count = positive.sum()
        tp = np.sum(positive & (predicted == c))
        fp = np.sum(~positive & (predicted == c))
        fn = np.sum(positive & (predicted != c))
        f1s.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
        if not count or count == len(y):
            continue
        score = probabilities[:, c]
        ranks = rankdata(score)
        aucs.append(float((ranks[positive].sum() - count * (count + 1) / 2) / (count * (len(y) - count))))
        order = np.argsort(-score, kind="stable")
        # Endpoints of tied score groups: AP must not depend on input row order.
        endpoints = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(y) - 1]
        cumulative = np.cumsum(positive[order])[endpoints]
        recall = cumulative / count
        aps.append(float(np.sum(np.diff(np.r_[0., recall]) * cumulative / (endpoints + 1))))
    result = {"accuracy": float((predicted == y).mean()), "macro_f1": float(np.mean(f1s)),
              "macro_auroc": float(np.mean(aucs)) if len(aucs) == classes else None,
              "macro_auprc": float(np.mean(aps)) if len(aps) == classes else None,
              "supported_classes": len(aucs)}
    if classes == 2:
        positive = y == 1
        result.update(sensitivity=float((predicted[positive] == 1).mean()) if positive.any() else None,
                      specificity=float((predicted[~positive] == 0).mean()) if (~positive).any() else None)
    return result


def task_metrics(y, prediction, subjects, task):
    if task == "classification":
        return classification_scores(y, prediction)
    if task != "regression":
        raise ValueError("Task must be regression or classification")
    error = np.asarray(prediction).reshape(-1) - y
    result = {"mae": float(abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
              "patient_macro_mae": float(np.mean([abs(error[subjects == p]).mean() for p in np.unique(subjects)])),
              "pearson": None, "spearman": None}
    if len(y) >= 2 and np.std(y) > 0 and np.std(prediction) > 0:
        result.update(pearson=float(np.corrcoef(y, np.asarray(prediction).reshape(-1))[0, 1]),
                      spearman=float(spearmanr(y, np.asarray(prediction).reshape(-1)).statistic))
    return result


def cluster_intervals(y, prediction, subjects, task, count=200, seed=42, other=None):
    """Resample groups, optionally reporting paired (prediction - other) metrics."""
    if count < 20:
        raise ValueError("At least 20 cluster bootstrap replicates are required")
    unique = np.unique(subjects)
    if len(unique) < 2:
        return {"reason": "fewer_than_two_groups", "intervals": {}}
    pools = [np.flatnonzero(subjects == p) for p in unique]
    rng, draws = np.random.default_rng(seed), {}
    for _ in range(count):
        chosen = rng.integers(len(pools), size=len(pools))
        indices = np.concatenate([pools[i] for i in chosen])
        # Each resampled occurrence is a separate cluster for macro averages.
        groups = np.concatenate([np.full(len(pools[i]), j) for j, i in enumerate(chosen)])
        metrics = task_metrics(y[indices], prediction[indices], groups, task)
        reference = task_metrics(y[indices], other[indices], groups, task) if other is not None else None
        for key, value in metrics.items():
            if key == "supported_classes" or value is None or (reference is not None and reference[key] is None):
                continue
            draws.setdefault(key, []).append(value - reference[key] if reference is not None else value)
    return {"replicates": count, "confidence": .95, "unit": "subject/group",
            "difference": "first_minus_second" if other is not None else None,
            "intervals": {key: {"lower": float(np.quantile(values, .025)),
                                  "upper": float(np.quantile(values, .975)), "valid_replicates": len(values)}
                          for key, values in draws.items()}}


def fit_linear_probe(x, y, subjects, task, regularization, seed, classes=None, steps=300):
    if not np.isfinite(regularization) or regularization <= 0 or steps < 1:
        raise ValueError("Invalid probe regularization/steps")
    weights = subject_weights(subjects)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)).clip(1e-6)
    standardized = (x - mean) / scale
    if task == "regression":
        ym = np.average(y, weights=weights)
        a = standardized * np.sqrt(weights)[:, None]
        b = (y - ym) * np.sqrt(weights)
        if len(a) < a.shape[1]:
            coef = a.T @ np.linalg.solve(a @ a.T + regularization * np.eye(len(a)), b)
        else:
            coef = np.linalg.solve(a.T @ a + regularization * np.eye(a.shape[1]), a.T @ b)
        return {"mean": mean, "scale": scale, "coef": coef, "intercept": np.asarray(ym), "task": task}
    if task != "classification" or classes is None or classes < 2:
        raise ValueError("Classification requires >=2 training classes")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = torch.nn.Linear(x.shape[1], classes)
        optimizer = torch.optim.Adam(layer.parameters(), lr=.03)
        tx, ty = torch.tensor(standardized, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        tw = torch.tensor(weights, dtype=torch.float32)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (torch.nn.functional.cross_entropy(layer(tx), ty, reduction="none") * tw).mean()
            loss = loss + regularization * layer.weight.square().sum() / (2 * len(x))
            loss.backward()
            optimizer.step()
        return {"mean": mean, "scale": scale, "coef": layer.weight.detach().numpy().T,
                "intercept": layer.bias.detach().numpy(), "task": task}


def predict_linear(probe, x):
    values = ((x - probe["mean"]) / probe["scale"]) @ probe["coef"] + probe["intercept"]
    return softmax(values, axis=1) if probe["task"] == "classification" else values


def evaluate_probe(train_features, val_features, test_features, labels, target, output,
                   task="regression", label_fraction=1., seed=42, regularizations=(.1, 1., 10.),
                   bootstrap_samples=200, protocol="group_disjoint", target_source=None, steps=300):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output must be empty")
    parts = {key: load_features(path) for key, path in
             zip(("train", "val", "test"), (train_features, val_features, test_features))}
    audit = validate_splits(parts, protocol, target_source)
    table = TargetTable(labels)
    prepared = {}
    for name, part in parts.items():
        values, valid = table.arrays(part["sample_id"], [target], float(part["scale_seconds"]))
        take = valid[:, 0]
        prepared[name] = {k: v[take] for k, v in part.items()
                          if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(take)}
        prepared[name]["y"] = values[take, 0].astype(np.float64)
        if not take.any():
            raise ValueError(f"No valid {target} labels in {name}")
    training = prepared["train"]
    keep, groups = select_label_groups(training["subject_id"], label_fraction, seed)
    prepared["train"] = training = {k: v[keep] for k, v in training.items()}
    classes = sorted(set(training["y"].tolist())) if task == "classification" else None
    if classes is not None:
        if len(classes) < 2:
            raise ValueError("Selected training groups contain fewer than two classes")
        for name, part in prepared.items():
            if not set(part["y"].tolist()).issubset(classes):
                raise ValueError(f"{name} contains a class missing from selected training groups")
            part["y"] = np.asarray([classes.index(value) for value in part["y"]])
    candidates = []
    for regularization in regularizations:
        probe = fit_linear_probe(training["embedding"], training["y"], training["subject_id"],
                                 task, regularization, seed, len(classes) if classes else None, steps)
        validation = prepared["val"]
        pred = predict_linear(probe, validation["embedding"])
        if task == "regression":
            score = task_metrics(validation["y"], pred, validation["subject_id"], task)["patient_macro_mae"]
        else:
            losses = -np.log(pred[np.arange(len(pred)), validation["y"]].clip(1e-12))
            score = np.average(losses, weights=subject_weights(validation["subject_id"]))
        candidates.append((float(score), float(regularization), probe))
    if not candidates:
        raise ValueError("At least one regularization candidate is required")
    score, regularization, probe = min(candidates, key=lambda item: item[0])
    test = prepared["test"]
    prediction = predict_linear(probe, test["embedding"])
    metrics = task_metrics(test["y"], prediction, test["subject_id"], task)
    by_source = {}
    for source in sorted(set(test["source"])):
        take = test["source"] == source
        by_source[source] = task_metrics(test["y"][take], prediction[take], test["subject_id"][take], task)
    report = {"schema_version": 1, "task": task, "target": target, "units": table.units[target],
              "classes": classes, "seed": seed, "label_fraction": label_fraction,
              "selected_training_groups": groups, "selection_metric": "validation patient-balanced error",
              "selected_regularization": regularization, "validation_score": score,
              "candidates": [{"regularization": r, "validation_score": s} for s, r, _ in candidates],
              "counts": {k: {"windows": len(v["y"]), "groups": len(set(v["subject_id"]))} for k, v in prepared.items()},
              "test": metrics, "by_source": by_source, "audit": audit,
              "bootstrap": cluster_intervals(test["y"], prediction, test["subject_id"], task, bootstrap_samples, seed),
              "provenance": {"labels_sha256": table.fingerprint,
                             "features": {k: v["file_sha256"] for k, v in parts.items()}}}
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "probe.npz", **probe, classes=np.asarray(classes if classes else []))
    np.savez(output / "predictions.npz", prediction=prediction, y=test["y"],
             sample_id=test["sample_id"], subject_id=test["subject_id"], source=test["source"],
             task=task, target=target, classes=np.asarray(classes if classes else []))
    write_json(output / "report.json", report)
    return output / "report.json"


def compare_predictions(first, second, output, bootstrap_samples=200, seed=42):
    if Path(output).exists():
        raise FileExistsError(output)
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if (set(a["sample_id"]) != set(b["sample_id"]) or
                any(len(set(p["sample_id"])) != len(p["sample_id"]) for p in (a, b))):
            raise ValueError("Paired comparison requires identical unique sample support")
        position = {sid: i for i, sid in enumerate(b["sample_id"])}
        order = [position[sid] for sid in a["sample_id"]]
        for key in ("subject_id", "source", "y"):
            if not np.array_equal(a[key], b[key][order]):
                raise ValueError(f"Paired comparison differs in {key}")
        for key in ("task", "target", "classes"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"Paired comparison differs in {key}")
        task = str(a["task"].item())
        report = cluster_intervals(a["y"], a["prediction"], a["subject_id"], task,
                                   bootstrap_samples, seed, other=b["prediction"][order])
        report.update(first_sha256=file_sha256(first), second_sha256=file_sha256(second),
                      first=task_metrics(a["y"], a["prediction"], a["subject_id"], task),
                      second=task_metrics(a["y"], b["prediction"][order], a["subject_id"], task))
    write_json(output, report)
    return Path(output)
