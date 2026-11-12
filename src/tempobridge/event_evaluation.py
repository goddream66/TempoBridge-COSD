"""PPG-only event inference and held-out V3 evaluation."""

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import PairedDataset, manifest_digest, read_manifest
from .dynamic_transfer import METRIC_SCALES, EventTransferHeads, decode_event_times
from .evaluation import cluster_intervals, task_metrics
from .event_targets import EventDataset, EventTable, collate_events, match_event_times
from .model import TempoBridge
from .utils import device_for, file_sha256, load_checkpoint, write_json


def load_event_model(checkpoint, device="cpu"):
    state = load_checkpoint(checkpoint)
    if state.get("stage") != "event" or state.get("global_step", 0) < 1:
        raise ValueError("Event inference requires a trained V3 checkpoint")
    config = ExperimentConfig.from_dict(state["config"])
    model = TempoBridge(config.model)
    model.load_state_dict(state["model"])
    heads = EventTransferHeads(config, state["event_dynamic_dim"])
    heads.load_state_dict(state["heads"])
    return model.to(device).eval(), heads.to(device).eval(), config


INTERVAL_LEVELS = {level: float(norm.ppf((1 + level) / 2)) for level in (.50, .80, .90, .95)}


@torch.inference_mode()
def predict_events(model, heads, ppg, fs, threshold=.5):
    if not 0 < threshold < 1:
        raise ValueError("Event threshold must be in (0,1)")
    outputs = heads.encode_physiological(model, ppg)
    feet = decode_event_times(outputs["foot_logits"], fs, threshold)
    peaks = decode_event_times(outputs["peak_logits"], fs, threshold)
    anchors = torch.nn.utils.rnn.pad_sequence([ppg.new_tensor(row, dtype=torch.float32) for row in feet],
                                              batch_first=True)
    lengths = torch.as_tensor([len(row) for row in feet], device=ppg.device)
    valid = torch.arange(anchors.shape[1], device=ppg.device)[None] < lengths[:, None]
    intervals = heads.events.intervals(outputs["patch_tokens"], outputs["foot_logits"], anchors, valid, fs)
    return {**outputs, **intervals, "feet_s": feet, "peaks_s": peaks}


def _summary(totals, dynamic_metrics, probabilistic):
    result = {}
    for name in ("foot", "peak"):
        tp, pred, ref = (totals[f"{name}_{key}"] for key in ("tp", "pred", "ref"))
        result[name] = {"tp": tp, "predicted": pred, "reference": ref,
                        "precision": tp / pred if pred else None, "recall": tp / ref if ref else None,
                        "f1": 2 * tp / (pred + ref) if pred + ref else None}
    count, available = totals["rr_count"], totals["rr_available"]
    result["rr"] = {"matched": count, "available": available, "coverage": count / available if available else None,
                    "mae_ms": 1000 * totals["rr_abs"] / count if count else None,
                    "rmse_ms": 1000 * (totals["rr_sq"] / count) ** .5 if count else None,
                    "gaussian_nll_without_constant": totals["rr_nll"] / count if count and probabilistic else None}
    if probabilistic and count:
        result["rr"]["gaussian_nll"] = totals["rr_nll"] / count + .5 * np.log(2 * np.pi)
        result["rr"]["interval_calibration"] = [{"nominal": level, "observed": totals[f"rr_hit/{level}"] / count,
                                                 "mean_width_ms": 2000 * z * totals["rr_std_sum"] / count}
                                                for level, z in INTERVAL_LEVELS.items()]
        result["rr"]["mean_abs_coverage_error"] = float(np.mean([
            abs(row["nominal"] - row["observed"]) for row in result["rr"]["interval_calibration"]]))
    for name in dynamic_metrics:
        count = totals[f"{name}_count"]
        result[f"delta_{name}"] = {"pairs": count, "mae": totals[f"{name}_abs"] / count if count else None,
                                  "zero_change_mae": totals[f"{name}_zero"] / count if count else None}
    return result


def _dynamic_analysis(rows, name):
    if not rows:
        return None
    prediction = np.asarray([row[1] for row in rows])
    target = np.asarray([row[2] for row in rows])
    edges = (1., 5., 10.) if name == "hr_bpm" else (5., 10., 25.)
    neutral = edges[0]
    def direction(values):
        return np.where(abs(values) < neutral, 0., np.sign(values))

    strata = np.searchsorted(edges, abs(target), side="right")
    known = np.asarray([row[0] is not None for row in rows])
    ids = np.asarray([row[0] for row in rows if row[0] is not None])
    correlation = task_metrics(target, prediction, np.zeros(len(target)), "regression")
    return {"pearson": correlation["pearson"], "spearman": correlation["spearman"],
            "direction_accuracy": float(np.mean(direction(prediction) == direction(target))),
            "direction_neutral_threshold": neutral,
            "magnitude_strata": [{"lower": lower, "upper": upper, "pairs": int((strata == i).sum()),
                                  "mae": float(abs(prediction[strata == i] - target[strata == i]).mean())
                                  if (strata == i).any() else None}
                                 for i, (lower, upper) in enumerate(zip((0., *edges), (*edges, None)))],
            "group_paired_bootstrap_vs_zero": cluster_intervals(
                target[known], prediction[known], ids, "regression", 200, 42,
                other=np.zeros(int(known.sum()))) if len(set(ids)) >= 2 else None}


@torch.inference_mode()
def evaluate_events(checkpoint, manifest, targets, output, split="test", device="cpu", threshold=None,
                    tolerance=.08, require_manual=False):
    if Path(output).exists():
        raise FileExistsError(output)
    if split not in {"val", "test"} or not 0 < tolerance <= .25:
        raise ValueError("Event evaluation requires a held-out split and a valid time tolerance")
    device = device_for(device)
    model, heads, config = load_event_model(checkpoint, device)
    state = load_checkpoint(checkpoint)
    records = read_manifest(manifest, data_config=config.data)
    digest = manifest_digest(records)
    if digest != state["manifest_sha256"]:
        raise ValueError("Event evaluation must use the original corpus and group splits")
    settings = config.event
    threshold = settings.rr_anchor_threshold if threshold is None else threshold
    probabilistic = settings.rr_loss == "gaussian" and settings.rr_weight > 0
    metric_mode = settings.dynamic_weight > 0 and settings.dynamic_target == "metrics"
    dynamic_metrics = settings.dynamic_metrics if metric_mode else []
    if not metric_mode:
        settings = replace(settings, dynamic_weight=0.)
    table = EventTable(targets, digest, config.data.scales_seconds[0], manual_only=require_manual)
    base = PairedDataset(records, config.data, split=split, augment=False)
    dataset = EventDataset(base, table, settings)
    if not dataset.covered:
        dataset.close()
        raise ValueError("No held-out event reference rows")
    totals, groups, sources, identity = Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
    change_rows = defaultdict(list)
    reference_kinds = Counter()
    try:
        for batch in DataLoader(dataset, batch_size=config.train.batch_size, collate_fn=collate_events):
            ppg = batch["ppg"][0].to(device)
            predictions = predict_events(model, heads, ppg, config.data.ppg_fs, threshold)
            delta = None
            if metric_mode:
                use = batch["pair_weight"] > 0
                if use.any():
                    current = predictions["patch_tokens"].mean(1)[use.to(device)]
                    after = model.transfer_features(batch["next_ppg"][use].to(device))
                    raw = heads.predict_change(current, after).cpu()
                    delta = torch.zeros_like(batch["delta_metric"])
                    delta[use] = raw * raw.new_tensor([METRIC_SCALES[key] for key in dynamic_metrics])
            rr_mean, logvar = predictions["rr_mean"].cpu(), predictions["rr_logvar"].cpu()
            for i, key in enumerate(batch["subject_id"]):
                row = Counter()
                if batch["event_labelled"][i] or batch["manual_events"][i]:
                    reference_kinds["manual" if batch["manual_events"][i] else "heuristic"] += 1
                    valid = batch["beat_valid"][i]
                    feet, peaks = (batch[name][i, valid].numpy() for name in ("feet_s", "peaks_s"))
                    foot_pairs = match_event_times(predictions["feet_s"][i], feet, tolerance)
                    peak_pairs = match_event_times(predictions["peaks_s"][i], peaks, tolerance)
                    for name, matches, reference, column in (("foot", foot_pairs, feet, "feet_s"),
                                                             ("peak", peak_pairs, peaks, "peaks_s")):
                        row.update({f"{name}_tp": len(matches), f"{name}_pred": len(predictions[column][i]),
                                    f"{name}_ref": len(reference)})
                    eligible = batch["rr_valid"][i] & (batch["rr_weight"][i] > 0) & (
                        batch["rr_weight"][i] >= settings.minimum_quality)
                    row["rr_available"] = int(eligible.sum())
                    for (a, r), (b, s) in zip(foot_pairs, foot_pairs[1:]):
                        if b == a + 1 and s == r + 1 and eligible[r]:
                            error = float(rr_mean[i, a] - batch["rr_s"][i, r])
                            variance = float(logvar[i, a])
                            row.update(rr_count=1, rr_abs=abs(error), rr_sq=error ** 2)
                            if probabilistic:
                                std = float(np.exp(.5 * variance))
                                residual = abs(error) / std
                                row["rr_nll"] += .5 * (residual ** 2 + variance)
                                row["rr_std_sum"] += std
                                for level, z in INTERVAL_LEVELS.items():
                                    row[f"rr_hit/{level}"] += int(residual <= z)
                if delta is not None:
                    for j, name in enumerate(dynamic_metrics):
                        if batch["delta_valid"][i, j] and batch["pair_weight"][i] > 0:
                            target = float(batch["delta_metric"][i, j])
                            prediction = float(delta[i, j])
                            group = repr((batch["source"][i], key)) if batch["subject_known"][i] else None
                            change_rows[name].append((group, prediction, target))
                            row.update({f"{name}_count": 1, f"{name}_abs": abs(prediction - target),
                                        f"{name}_zero": abs(target)})
                totals.update(row)
                sources[batch["source"][i]].update(row)
                if batch["subject_known"][i]:
                    groups[(batch["source"][i], key)].update(row)
                    identity[batch["identity_level"][i]] += 1
    finally:
        dataset.close()
    group_results = [_summary(row, dynamic_metrics, probabilistic) for row in groups.values()]
    macro = {}
    for target, metric in (("foot", "f1"), ("peak", "f1"), ("rr", "mae_ms"),
                           *((f"delta_{name}", "mae") for name in dynamic_metrics)):
        values = [group[target][metric] for group in group_results if group[target][metric] is not None]
        macro[f"{target}/{metric}"] = {"value": float(np.mean(values)) if values else None, "groups": len(values)}
    result = {"split": split, "checkpoint_sha256": file_sha256(checkpoint), "event_targets_sha256": table.fingerprint,
              "threshold": threshold, "event_tolerance_s": tolerance, "reference_windows": dataset.covered,
              "identity_levels": dict(identity), "metrics": _summary(totals, dynamic_metrics, probabilistic),
              "reference_kinds": dict(reference_kinds), "manual_only": require_manual,
              "dynamic_analysis": {name: _dynamic_analysis(change_rows[name], name) for name in dynamic_metrics},
              "group_macro": macro,
              "by_source": {key: _summary(row, dynamic_metrics, probabilistic) for key, row in sources.items()},
              "scope": "PPG-only retrospective estimates; event references and interval eligibility used only in scoring"}
    write_json(output, result)
    return Path(output)
