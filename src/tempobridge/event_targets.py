"""Native-rate event references and strict same-record pairs for TempoBridge V3."""

import json
import os
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch.utils.data import Dataset, default_collate

from .data import manifest_digest, read_manifest
from .h5data import H5Reader, h5py_module
from .physio_targets import ecg_interval_metrics, ecg_peak_candidates
from .utils import file_sha256

METRICS = ("hr_bpm", "rmssd_ms")
SEQUENCES = ("rpeaks_s", "feet_s", "peaks_s", "rr_valid", "nn_valid", "pat_r_index")
LEGACY_SEQUENCES = SEQUENCES[:4]
PROVENANCE_STRINGS = ("ecg_channel", "ppg_channel", "native_timestamp_origin", "clock_evidence")


def _native_wave(signals, key, duration):
    if key not in signals:
        return None, 0.0
    fs = float(signals[f"{key}_fs"])
    wave = np.asarray(signals[key], dtype=np.float64)
    if not np.isfinite(fs) or fs <= 0 or wave.ndim != 1 or len(wave) != round(duration * fs):
        raise ValueError(f"{key} reference must cover the exact declared window at a valid rate")
    # V3-A abstains on gaps. Interpolation could create artificial events or bridge missed beats.
    if fs < 50 or duration < 10 or not np.isfinite(wave).all() or np.ptp(wave) < 1e-8:
        return None, fs
    return wave, fs


def _pulse_candidates(wave, fs):
    smooth = sosfiltfilt(butter(2, 8, fs=fs, output="sos"), wave)
    peaks, props = find_peaks(smooth, distance=round(.27 * fs), prominence=max(.2 * smooth.std(), 1e-8))
    feet, kept = [], []
    for i, peak in enumerate(peaks):
        start = max(1, peak - round(.8 * fs), peaks[i - 1] + 1 if i else 1)
        end = peak - round(.06 * fs)
        if end <= start or peak >= len(wave) - round(.1 * fs):
            continue
        foot = start + np.argmin(smooth[start:end + 1])
        if not start < foot < end or smooth[peak] - smooth[foot] < .5 * props["prominences"][i]:
            continue
        feet.append(foot)
        kept.append(peak)
    return np.asarray(feet) / fs, np.asarray(kept) / fs


def _annotated_times(annotation, name, duration):
    values = np.asarray(annotation[name], dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all() or (np.diff(values) <= 0).any()
            or ((values < 0) | (values >= duration)).any()):
        raise ValueError(f"Invalid annotated {name}")
    return values


def extract_event_reference(signals, duration, annotation=None):
    ecg, efs = _native_wave(signals, "ecg", duration)
    ppg, pfs = _native_wave(signals, "ppg", duration)
    rpeaks = np.empty(0) if ecg is None else ecg_peak_candidates(ecg, efs) / efs
    feet, peaks = (np.empty(0), np.empty(0)) if ppg is None else _pulse_candidates(ppg, pfs)
    if annotation is not None:
        rpeaks, feet, peaks = (_annotated_times(annotation, key, duration)
                               for key in ("rpeaks_s", "feet_s", "peaks_s"))
        if (len(feet) != len(peaks) or (feet >= peaks).any() or (peaks[:-1] >= feet[1:]).any()
                or (len(rpeaks) and not efs) or (len(feet) and not pfs)):
            raise ValueError("Annotated pulse feet/peaks or reference channel availability disagree")
    rr = np.diff(rpeaks)
    valid = (rr >= .27) & (rr <= 2.4)
    # Suppress edge detections affected by finite-window filtering.
    if len(rr):
        valid &= (rpeaks[:-1] >= .15) & (rpeaks[1:] < duration - .15)
    hr, rmssd = ecg_interval_metrics(rr, valid, efs, duration)
    nn = np.zeros(len(rr), dtype=bool)
    pat = np.empty(0, dtype=np.int32)
    rmssd_kind = "rr_proxy"
    rhythm = "unknown"
    if annotation is not None:
        rhythm = annotation.get("rhythm", "unknown")
        if rhythm not in {"unknown", "regular", "irregular"}:
            raise ValueError("Expected annotated rhythm: regular, irregular or unknown")
        if "nn_valid" in annotation:
            declared = annotation["nn_valid"]
            if not isinstance(declared, list) or len(declared) != len(rr) or any(type(x) is not bool for x in declared):
                raise ValueError("nn_valid requires one boolean per adjacent annotated R-peak pair")
            nn = np.asarray(declared, dtype=bool) & valid
            pairs = nn[:-1] & nn[1:]
            rmssd = (float(1000 * np.sqrt(np.mean(np.diff(rr)[pairs] ** 2)))
                     if efs >= 100 and duration >= 30 and pairs.sum() >= 20 else None)
            rmssd_kind = "annotated_nn"
        if "pat_r_index" in annotation:
            declared = annotation["pat_r_index"]
            if (not isinstance(declared, list) or len(declared) != len(feet)
                    or any(type(x) is not int or not -1 <= x < len(rpeaks) for x in declared)):
                raise ValueError("pat_r_index requires one annotated R index or -1 per foot")
            pat = np.asarray(declared, dtype=np.int32)
            match = pat >= 0
            if (np.diff(pat[match]) <= 0).any() or (feet[match] <= rpeaks[pat[match]]).any():
                raise ValueError("Annotated PAT pairs must be ordered and have positive delays")
    return {"rpeaks_s": rpeaks, "rr_valid": valid, "feet_s": feet, "peaks_s": peaks,
            "metrics": [np.nan if value is None else value for value in (hr, rmssd)],
            "rates": [efs, pfs], "nn_valid": nn, "pat_r_index": pat, "rmssd_kind": rmssd_kind,
            "rhythm": rhythm, "label_origin": "manual" if annotation is not None else "heuristic"}


def _reference_provenance(row, record, reader, rates, duration, digest):
    if "reference_sha256" in row and row["reference_sha256"] != digest:
        raise ValueError(f"Reference SHA-256 mismatch: {record.sample_id}")
    status = row.get("clock_alignment_status", "unverified")
    if status not in {"unverified", "verified_common_clock", "verified_offset_corrected"}:
        raise ValueError(f"Unknown clock alignment status: {status}")
    fields = {name: getattr(record, name) for name in ("recording_id", "case_id", "window_start_s", "recording_known")}
    if record.h5_row is not None and status != "unverified":
        fields.update(reader.metadata(record))
    expected = {"source_id": record.source, "subject_id": record.subject_id, "record_id": fields["recording_id"],
                "case_id": fields["case_id"], "window_start_s": fields["window_start_s"],
                "window_end_s": fields["window_start_s"] + duration,
                "ecg_native_fs": rates[0], "ppg_native_fs": rates[1]}
    verified = status != "unverified"
    if verified:
        if not fields["recording_known"] or fields["window_start_s"] < 0 or not all(rates):
            raise ValueError(f"Verified clock requires indexed recording, timestamp and both signals: {record.sample_id}")
        for key, value in expected.items():
            supplied = row.get(key)
            matches = (type(supplied) in {int, float} and abs(supplied - value) <= 1e-6
                       if isinstance(value, (int, float)) else supplied == value)
            if not matches:
                raise ValueError(f"Reference {key} disagrees with indexed window: {record.sample_id}")
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in PROVENANCE_STRINGS):
            raise ValueError(f"Verified clock requires channels, timestamp origin and clock evidence: {record.sample_id}")
        bound = row.get("max_alignment_error_s")
        if type(bound) not in {int, float} or not 0 <= bound <= .02:
            raise ValueError(f"Verified clock requires a declared residual error bound <=20 ms: {record.sample_id}")
    evidence = {key: row[key] for key in (*expected, *PROVENANCE_STRINGS, "max_alignment_error_s") if key in row}
    evidence.update(clock_alignment_status=status, reference_sha256=digest, legacy_pairing_verified=row.get("pairing_verified"))
    return verified, json.dumps(evidence, allow_nan=False, ensure_ascii=False)


def _annotations(path):
    rows = {}
    if path is not None:
        with Path(path).open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    key = row["sample_id"]
                    if key in rows or not isinstance(row.get("annotation_source"), str) or not row["annotation_source"].strip():
                        raise ValueError(f"Duplicate annotation or missing annotation_source: {key}")
                    rows[key] = row
    return rows


def build_event_targets(references, manifest, config, output, annotations=None):
    """Stream independent reference NPZ windows into a variable-length HDF5 table."""
    references, output = Path(references), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if config.data.window_mode != "full":
        raise ValueError("Event targets require full windows")
    duration = config.data.scales_seconds[0]
    records = read_manifest(manifest, data_config=config.data)
    expected = {record.sample_id: i for i, record in enumerate(records)}
    annotated = _annotations(annotations)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    h5py = h5py_module()
    seen = set()
    reader = H5Reader(config.data.max_open_shards)
    try:
        with h5py.File(temporary, "x") as table, references.open(encoding="utf-8-sig") as handle:
            table.attrs.update(schema=2, duration_seconds=duration, manifest_sha256=manifest_digest(records),
                               method="native_qrs_energy_and_ppg_trough_v2", references_sha256=file_sha256(references),
                               annotations_sha256=file_sha256(annotations) if annotations else "")
            sequences = {key: h5py.vlen_dtype(np.dtype("uint8" if key.endswith("_valid") else
                                                       "int32" if key == "pat_r_index" else "float64"))
                         for key in SEQUENCES}
            string_fields = {name: h5py.string_dtype() for name in
                             ("sample_id", "provenance", "label_origin", "annotation_source", "rmssd_kind", "rhythm")}
            fields = {name: table.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
                      for name, dtype in {**string_fields, "reference_sha256": "S64", "pairing_verified": "?",
                                          **sequences}.items()}
            for key in ("metrics", "rates"):
                fields[key] = table.create_dataset(key, shape=(0, 2), maxshape=(None, 2), dtype="f8")
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row["sample_id"]
                if key not in expected or key in seen:
                    raise ValueError(f"Unknown or duplicate sample: {key}")
                if abs(float(row["duration_seconds"]) - duration) > 1e-6 or not np.isfinite(float(row["duration_seconds"])):
                    raise ValueError(f"Reference duration differs from input protocol: {key}")
                path = references.parent / row["path"]
                digest = file_sha256(path)
                annotation = annotated.get(key)
                if annotation is not None and annotation.get("reference_sha256") != digest:
                    raise ValueError(f"Annotation must bind to the native waveform SHA-256: {key}")
                with np.load(path, allow_pickle=False) as signals:
                    values = extract_event_reference(signals, duration, annotation)
                verified, provenance = _reference_provenance(row, records[expected[key]], reader, values["rates"], duration, digest)
                values.update(sample_id=key, reference_sha256=digest, pairing_verified=verified, provenance=provenance,
                              annotation_source=annotation["annotation_source"] if annotation else "")
                i = len(seen)
                for name, field in fields.items():
                    field.resize(i + 1, axis=0)
                    field[i] = np.asarray(values[name], dtype=np.uint8) if name.endswith("_valid") else values[name]
                seen.add(key)
            if not seen:
                raise ValueError("Empty event reference manifest")
            if annotated.keys() - seen:
                raise ValueError("Annotation table includes sample IDs without native references")
        os.link(temporary, output)
    finally:
        reader.close()
        temporary.unlink(missing_ok=True)
    return output


class EventTable:
    def __init__(self, path, digest, duration, manual_only=False):
        self.path = Path(path).resolve()
        self.fingerprint = file_sha256(path)
        self.handle = None
        self.pid = None
        with h5py_module().File(path, "r") as table:
            self.schema = table.attrs.get("schema")
            if (self.schema not in {1, 2} or table.attrs.get("manifest_sha256") != digest
                    or table.attrs.get("duration_seconds") != duration):
                raise ValueError("Event table schema, corpus, or input duration differs")
            keys = table["sample_id"].asstr()[:]
            self.index = {key: i for i, key in enumerate(keys)}
            if len(self.index) != len(keys) or any(len(field) != len(keys) for field in table.values()):
                raise ValueError("Inconsistent event table rows")
            if manual_only:
                if self.schema == 1:
                    raise ValueError("Manual evaluation requires a schema-2 annotated event table")
                origins = table["label_origin"].asstr()[:]
                self.index = {key: i for i, key in enumerate(keys) if origins[i] == "manual"}
            self.verified = table["pairing_verified"][:].astype(bool) if self.schema == 2 else np.zeros(len(keys), dtype=bool)

    def __getstate__(self):
        return {**vars(self), "handle": None, "pid": None}

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def row(self, key):
        if key not in self.index:
            return {name: np.empty(0) for name in SEQUENCES} | {
                "metrics": np.full(2, np.nan), "pairing_verified": False, "rmssd_kind": "unknown",
                "label_origin": "missing", "rhythm": "unknown", "provenance": {}, "reference_sha256": ""}
        if self.handle is None or self.pid != os.getpid():
            self.close()
            self.handle = h5py_module().File(self.path, "r")
            self.pid = os.getpid()
        i = self.index[key]
        row = {name: self.handle[name][i] for name in (*LEGACY_SEQUENCES, "metrics", "rates")}
        row["reference_sha256"] = self.handle["reference_sha256"].asstr()[i]
        if self.schema == 1:
            row.update(pairing_verified=False, rmssd_kind="rr_proxy", label_origin="heuristic", rhythm="unknown",
                       provenance={}, nn_valid=np.zeros(len(row["rr_valid"]), dtype=bool), pat_r_index=np.empty(0, dtype=int))
        else:
            row.update({name: self.handle[name][i] for name in (*SEQUENCES[4:], "pairing_verified")})
            row.update({name: self.handle[name].asstr()[i] for name in ("rmssd_kind", "label_origin", "rhythm", "annotation_source")})
            row["provenance"] = json.loads(self.handle["provenance"].asstr()[i])
        return row


def match_event_times(predicted, reference, tolerance):
    """Maximum-cardinality ordered tolerance matching."""
    i, j, pairs = 0, 0, []
    while i < len(predicted) and j < len(reference):
        difference = predicted[i] - reference[j]
        if difference < -tolerance:
            i += 1
        elif difference > tolerance:
            j += 1
        else:
            pairs.append((i, j))
            i, j = i + 1, j + 1
    return pairs


def unique_beat_pairs(rpeaks, feet, config):
    if not len(rpeaks):
        return np.full(len(feet), -1, dtype=int)
    lag = feet[:, None] - rpeaks[None, :]
    allowed = (lag >= config.delay_min_s) & (lag <= config.delay_max_s)
    unique = allowed & (allowed.sum(0, keepdims=True) == 1) & (allowed.sum(1, keepdims=True) == 1)
    return np.where(unique.any(1), unique.argmax(1), -1)


def beat_rr_targets(row, config):
    """Conservative hard reference for A/B. Ambiguous and skipped beats abstain."""
    r, feet = row["rpeaks_s"], row["feet_s"]
    values = np.zeros(max(0, len(feet) - 1), dtype=np.float32)
    valid = np.zeros(len(values), dtype=bool)
    if not row["pairing_verified"] or len(r) < 2 or not len(values):
        return values, valid
    if row.get("label_origin") == "manual":
        # Independent interval evaluation must use explicit correspondences from the reviewer.
        matching = row["pat_r_index"]
        if not len(matching):
            return values, valid
    else:
        matching = unique_beat_pairs(r, feet, config)
    start = matching[:-1].clip(min=0, max=len(r) - 2)
    valid = (matching[:-1] >= 0) & (matching[1:] == matching[:-1] + 1) & row["rr_valid"][start].astype(bool)
    values[valid] = np.diff(r)[start[valid]]
    return values, valid


def adjacent_indices(rows, duration, tolerance):
    """Use only chronological, non-overlapping neighbors within a known record."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["subject_known"] and row["recording_known"] and row["window_start_s"] >= 0:
            key = (row["source"], row["subject_id"], row["recording_id"], row["case_id"])
            groups[key].append((row["window_start_s"], i))
    result = np.full(len(rows), -1, dtype=np.int64)
    for group in groups.values():
        ordered = sorted(group)
        counts = defaultdict(int)
        for start, _ in ordered:
            counts[start] += 1
        for (start, i), (end, j) in zip(ordered, ordered[1:]):
            if (counts[start] == counts[end] == 1 and duration - 1e-6 <= end - start <= duration + tolerance + 1e-6
                    and rows[i]["device_id"] == rows[j]["device_id"]):
                result[i] = j
    return result


class EventDataset(Dataset):
    def __init__(self, dataset, table, event_config):
        self.dataset, self.table, self.event = dataset, table, event_config
        self.records, self.config = dataset.records, dataset.config
        self.metric_indices = [METRICS.index(name) for name in event_config.dynamic_metrics]
        self.neighbors = None
        self.covered = sum(record.sample_id in table.index for record in self.records)
        self.verified_covered = sum(bool(table.verified[table.index[record.sample_id]])
                                    for record in self.records if record.sample_id in table.index)
        if not self.covered and event_config.dynamic_target != "latent":
            raise ValueError("This split has no event reference rows")
        if event_config.dynamic_weight:
            rows = [dataset.metadata(i) for i in range(len(dataset))]
            self.neighbors = adjacent_indices(rows, self.config.scales_seconds[0], event_config.pair_tolerance_s)

    def __len__(self):
        return len(self.dataset)

    def metadata(self, index):
        return self.dataset.metadata(index)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def close(self):
        self.dataset.close()
        self.table.close()

    def __getitem__(self, index):
        sample = self.dataset[index]
        row = self.table.row(sample["sample_id"])
        quality = sample["ppg_quality"]
        feet = row["feet_s"]
        rr, rr_valid = beat_rr_targets(row, self.event)
        sample.update(feet_s=torch.as_tensor(feet, dtype=torch.float32),
                      peaks_s=torch.as_tensor(row["peaks_s"], dtype=torch.float32),
                      manual_events=row["label_origin"] == "manual",
                      event_labelled=len(feet) >= 2 and quality >= self.event.minimum_quality,
                      rr_s=torch.from_numpy(rr), rr_valid=torch.from_numpy(rr_valid),
                      rr_weight=min(quality, sample["ecg_quality"]) if sample["ecg_valid"] else 0.)
        if self.neighbors is not None:
            self._add_neighbor(sample, row, int(self.neighbors[index]))
        return sample

    def _add_neighbor(self, sample, row, neighbor):
        metric = self.event.dynamic_target == "metrics"
        delta = np.zeros(len(self.metric_indices), dtype=np.float32)
        valid = np.zeros_like(delta, dtype=bool)
        pair_weight = 0.
        other = None
        if neighbor >= 0 and sample["ecg_valid"]:
            other = self.dataset[neighbor]
            pair_weight = min(sample["ppg_quality"], sample["ecg_quality"],
                              other["ppg_quality"], other["ecg_quality"])
            if not other["ecg_valid"] or pair_weight < self.event.minimum_quality:
                pair_weight = 0.
            elif metric:
                other_row = self.table.row(other["sample_id"])
                change = other_row["metrics"] - row["metrics"]
                if (row["rmssd_kind"] != other_row["rmssd_kind"] or
                        (self.event.rmssd_reference == "nn_only" and row["rmssd_kind"] != "annotated_nn")):
                    change[METRICS.index("rmssd_ms")] = np.nan
                change = change[self.metric_indices]
                valid = np.isfinite(change)
                delta[valid] = change[valid]
                if not valid.any():
                    pair_weight = 0.
        sample.update(pair_weight=pair_weight, delta_metric=torch.from_numpy(delta),
                      delta_valid=torch.from_numpy(valid),
                      next_ppg=other["ppg"][0] if pair_weight else torch.zeros_like(sample["ppg"][0]))
        if not metric:
            sample["next_ecg"] = other["ecg"][0] if pair_weight else torch.zeros_like(sample["ecg"][0])


def collate_events(samples):
    keys = ("feet_s", "peaks_s", "rr_s", "rr_valid")
    lengths = torch.tensor([len(sample["feet_s"]) for sample in samples])
    batch = default_collate([{key: value for key, value in sample.items() if key not in keys}
                             for sample in samples])
    for key in keys:
        batch[key] = torch.nn.utils.rnn.pad_sequence([sample[key] for sample in samples], batch_first=True)
    batch["beat_valid"] = torch.arange(batch["feet_s"].shape[1])[None, :] < lengths[:, None]
    return batch
