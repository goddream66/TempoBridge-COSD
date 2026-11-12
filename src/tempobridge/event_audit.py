"""Read-only audit of native events, actual model input, and independent annotations."""

import random
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from .data import PairedDataset, manifest_digest, read_manifest, split_size
from .event_targets import (
    EventTable,
    _pulse_candidates,
    beat_rr_targets,
    match_event_times,
    unique_beat_pairs,
)
from .utils import write_json


def _score(counter, key, candidates, reference, tolerance):
    matches = match_event_times(candidates, reference, tolerance)
    counter.update({f"{key}/detected": len(candidates), f"{key}/reference": len(reference),
                    f"{key}/matched": len(matches),
                    f"{key}/abs_ms": sum(1000 * abs(float(candidates[i] - reference[j])) for i, j in matches)})
    return matches


def _compare_manual(counter, auto, manual, config, tolerance):
    r, foot = auto["rpeaks_s"], auto["feet_s"]
    ref_r, ref_foot = manual["rpeaks_s"], manual["feet_s"]
    r_matches = _score(counter, "native_vs_manual_rpeak", r, ref_r, tolerance)
    foot_matches = _score(counter, "native_vs_manual_foot", foot, ref_foot, tolerance)
    _score(counter, "native_vs_manual_peak", auto["peaks_s"], manual["peaks_s"], tolerance)
    counter["rr_reference"] += int(np.count_nonzero(manual["rr_valid"]))
    for (a, b), (c, d) in zip(r_matches, r_matches[1:]):
        if c == a + 1 and d == b + 1 and auto["rr_valid"][a] and manual["rr_valid"][b]:
            counter.update(rr_matched=1, rr_abs_ms=1000 * abs(float((r[c] - r[a]) - (ref_r[d] - ref_r[b]))))
    if not (auto["pairing_verified"] and manual["pairing_verified"] and len(manual["pat_r_index"])):
        return
    auto_pairs = unique_beat_pairs(r, foot, config)
    manual_pairs = manual["pat_r_index"]
    r_map = dict(r_matches)
    counter["pat_reference"] += int((manual_pairs >= 0).sum())
    counter["pat_detected"] += int((auto_pairs >= 0).sum())
    for a, b in foot_matches:
        if manual_pairs[b] >= 0 and r_map.get(auto_pairs[a], -1) == manual_pairs[b]:
            delay = foot[a] - r[auto_pairs[a]]
            ref_delay = ref_foot[b] - ref_r[manual_pairs[b]]
            counter.update(pat_matched=1, pat_abs_ms=1000 * abs(float(delay - ref_delay)))


def _summarize(counts):
    result = {key: value for key, value in counts.items() if "/" not in key}
    for key in sorted({key.split("/")[0] for key in counts if "/" in key}):
        matched, detected, reference = (counts[f"{key}/{name}"] for name in ("matched", "detected", "reference"))
        result[key] = {"matched": matched, "detected": detected, "reference": reference,
                       "precision": matched / detected if detected else None,
                       "recall": matched / reference if reference else None,
                       "timing_mae_ms": counts[f"{key}/abs_ms"] / matched if matched else None}
    for name in ("rr", "pat"):
        n = counts[f"{name}_matched"]
        result[f"{name}_mae_ms"] = counts[f"{name}_abs_ms"] / n if n else None
    windows = counts["windows"]
    result["invalid_event_window_fraction"] = 1 - counts["pulse_windows"] / windows if windows else None
    return result


def _stratum(meta, native, manual):
    quality = meta["ppg_quality"]
    quality_bin = "low" if quality < .33 else "mid" if quality < .67 else "high"
    hr = native["metrics"][0]
    rate = "unknown" if not np.isfinite(hr) else "slow" if hr < 60 else "fast" if hr > 100 else "middle"
    return (meta["source"], quality_bin, rate, manual["rhythm"], manual.get("label_origin") == "manual")


def audit_events(manifest, targets, config, output, manual_targets=None, split="all", max_per_source=500,
                 tolerance=.08, seed=42):
    if Path(output).exists():
        raise FileExistsError(output)
    if config.data.window_mode != "full" or not 0 < tolerance <= .25 or max_per_source < 1:
        raise ValueError("Audit requires full windows, a valid tolerance and a positive source budget")
    records = read_manifest(manifest, data_config=config.data)
    digest = manifest_digest(records)
    seconds = config.data.scales_seconds[0]
    native = EventTable(targets, digest, seconds)
    manual = EventTable(manual_targets, digest, seconds, manual_only=True) if manual_targets else None
    by_stratum, datasets, windows = defaultdict(list), {}, []
    totals, by_source, strata = Counter(), defaultdict(Counter), defaultdict(Counter)
    try:
        for name in ("train", "val", "test") if split == "all" else (split,):
            if not split_size(records, name):
                continue
            data = datasets[name] = PairedDataset(records, config.data, split=name, augment=False, include_ecg=False)
            for index, record in enumerate(data.records):
                if record.sample_id not in native.index:
                    continue
                row = native.row(record.sample_id)
                if row["label_origin"] != "heuristic":
                    continue
                ref = manual.row(record.sample_id) if manual else {"rhythm": "unknown"}
                group = _stratum(data.metadata(index), row, ref)
                by_stratum[group].append((name, index))
        rng = random.Random(seed)
        queues = defaultdict(list)
        for key, values in sorted(by_stratum.items()):
            rng.shuffle(values)
            queues[key[0]].append((key, deque(values)))
        for source, groups in sorted(queues.items()):
            rng.shuffle(groups)
            active = deque(groups)
            for _ in range(max_per_source):
                if not active:
                    break
                group, remaining = active.popleft()
                name, index = remaining.popleft()
                if remaining:
                    active.append((group, remaining))
                sample = datasets[name][index]
                key = sample["sample_id"]
                row = native.row(key)
                feet, peaks = _pulse_candidates(sample["ppg"][0][0].numpy(), config.data.ppg_fs)
                counts = Counter(windows=1, pulse_windows=int(len(row["feet_s"]) >= 2),
                                 verified_windows=int(row["pairing_verified"]),
                                 rr_valid=int(np.count_nonzero(row["rr_valid"])),
                                 eligible_crossmodal_rr=int(beat_rr_targets(row, config.event)[1].sum()))
                _score(counts, "model_input_vs_native_foot", feet, row["feet_s"], tolerance)
                _score(counts, "model_input_vs_native_peak", peaks, row["peaks_s"], tolerance)
                if manual and key in manual.index:
                    ref = manual.row(key)
                    if ref["reference_sha256"] != row["reference_sha256"]:
                        raise ValueError(f"Manual and heuristic references use different native waveforms: {key}")
                    counts.update(manual_windows=1, annotated_nn_intervals=int(np.count_nonzero(ref["nn_valid"])),
                                  annotated_rr_intervals=len(ref["rr_valid"]))
                    _compare_manual(counts, row, ref, config.event, tolerance)
                totals.update(counts)
                by_source[source].update(counts)
                strata[group].update(counts)
                windows.append({"sample_id": key, "split": name, "source": source, "stratum": list(group[1:]),
                                "native_foot_s": row["feet_s"].tolist(), "model_input_foot_s": feet.tolist(),
                                "native_rpeak_s": row["rpeaks_s"].tolist(), "provenance": row["provenance"],
                                "reference_sha256": row["reference_sha256"],
                                "has_manual_reference": bool(manual and key in manual.index)})
    finally:
        native.close()
        if manual:
            manual.close()
        for data in datasets.values():
            data.close()
    if not windows:
        raise ValueError("No heuristic reference windows available for audit")
    result = {"scope": "Model/native agreement and separate comparison with supplied manual annotations",
              "manifest_sha256": digest, "native_targets_sha256": native.fingerprint,
              "manual_targets_sha256": manual.fingerprint if manual else None,
              "tolerance_s": tolerance, "max_per_source": max_per_source, "seed": seed,
              "overall": _summarize(totals), "by_source": {key: _summarize(value) for key, value in by_source.items()},
              "by_stratum": [{"source": key[0], "quality": key[1], "hr": key[2], "rhythm": key[3],
                              "manual_available": key[4], **_summarize(value)}
                             for key, value in sorted(strata.items())], "review_windows": windows}
    write_json(output, result)
    return Path(output)
