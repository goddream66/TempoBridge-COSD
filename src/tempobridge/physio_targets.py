"""Independent reference targets and masked continuous physiological supervision.

The built-in detectors are reproducible heuristic proxies, NOT clinical gold
standards. Rhythm diagnoses must be supplied as external adjudicated labels.
No target is generated from PPG or from a learned model's predictions.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch.utils.data import Dataset

from .quality import waveform_diagnostics
from .utils import file_sha256

TARGET_UNITS = {"hr_bpm": "bpm", "rmssd_ms": "ms", "sbp_mmhg": "mmHg",
                "dbp_mmhg": "mmHg", "map_mmhg": "mmHg", "rr_bpm": "breaths/min"}
FIELDS = ["sample_id", "target", "value", "valid", "reason", "units", "reference",
          "method", "reference_sha256", "reference_fs", "duration_seconds"]


def reference_targets(signals, duration):
    """NPZ keys: ecg/ecg_fs, abp/abp_fs/abp_units='mmHg', resp/resp_fs.

    Every supplied waveform must cover exactly this window at its declared rate.
    ECG targets use QRS-energy peaks; ABP uses beat maxima/intervening minima;
    respiration uses a spectral maximum. Missing/invalid channels produce masks.
    """
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Reference duration must be positive")
    output = {}
    for name, units in TARGET_UNITS.items():
        modality = "ecg" if name in {"hr_bpm", "rmssd_ms"} else "resp" if name == "rr_bpm" else "abp"
        output[name] = dict(target=name, value="", valid=0, reason="reference_missing", units=units,
                            reference=modality, method="heuristic_reference_v1", reference_fs=0,
                            duration_seconds=float(duration))

    def prepare(key):
        if key not in signals:
            return None
        fs = float(signals[key + "_fs"])
        wave = np.asarray(signals[key], np.float64)
        if wave.ndim != 1 or not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"Invalid reference waveform/rate: {key}")
        if abs(len(wave) / fs - duration) > 1 / fs + 1e-6:
            raise ValueError(f"Reference duration differs from window: {key}")
        targets = [r for r in output.values() if r["reference"] == key]
        diagnostics = waveform_diagnostics(wave, fs)
        for row in targets:
            row.update(reference_fs=fs, reason="reference_unreliable")
        if diagnostics["missing_fraction"] > .02 or diagnostics["flatline_fraction"] > .95 or len(wave) < 4:
            return None
        valid = np.isfinite(wave)
        wave = np.interp(np.arange(len(wave)), np.flatnonzero(valid), wave[valid])
        if np.ptp(wave) < 1e-8:
            return None
        return wave, fs

    def assign(name, value, valid=True, reason=""):
        output[name].update(value=float(value) if valid else "", valid=int(valid), reason=reason)

    ecg = prepare("ecg")
    if ecg is not None:
        wave, fs = ecg
        if fs < 50 or duration < 10:
            for name in ("hr_bpm", "rmssd_ms"):
                output[name]["reason"] = "insufficient_ecg_rate_or_duration"
        else:
            filtered = sosfiltfilt(butter(3, [5., min(25., .4 * fs)], fs=fs, btype="band", output="sos"), wave)
            width = max(1, round(.08 * fs))
            energy = np.convolve(filtered ** 2, np.ones(width) / width, mode="same")
            peaks, _ = find_peaks(energy, distance=max(1, round(.25 * fs)), prominence=max(energy.std() * .5, 1e-12))
            # Energy smoothing locates QRS regions, not precise R events. Refine
            # against absolute bandpassed ECG before deriving RR differences.
            radius = max(1, round(.06 * fs))
            peaks = np.unique([max(0, p - radius) + np.argmax(abs(filtered[max(0, p - radius):p + radius + 1]))
                               for p in peaks]).astype(int)
            intervals = np.diff(peaks) / fs
            plausible = (intervals >= .27) & (intervals <= 2.4)
            if len(intervals) >= 4 and plausible.mean() >= .8:
                assign("hr_bpm", 60 / np.median(intervals[plausible]))
                # Never bridge rejected beats when computing adjacent RR differences.
                pairs = plausible[:-1] & plausible[1:]
                if fs >= 100 and duration >= 30 and pairs.sum() >= 20:
                    assign("rmssd_ms", 1000 * np.sqrt(np.mean(np.diff(intervals)[pairs] ** 2)))
                else:
                    output["rmssd_ms"]["reason"] = "insufficient_native_rate_duration_or_rr_pairs"

    abp = prepare("abp")
    if abp is not None:
        wave, fs = abp
        if str(np.asarray(signals.get("abp_units", "")).item()) != "mmHg":
            for name in ("sbp_mmhg", "dbp_mmhg", "map_mmhg"):
                output[name]["reason"] = "abp_units_must_be_mmHg"
        elif duration >= 10 and fs >= 20:
            peaks, _ = find_peaks(wave, distance=max(1, round(.25 * fs)), prominence=.15 * np.ptp(wave))
            if len(peaks) >= 5:
                systolic = np.median(wave[peaks])
                diastolic = np.median([wave[a:b + 1].min() for a, b in zip(peaks[:-1], peaks[1:])])
                mean = wave[peaks[0]:peaks[-1]].mean()
                if 30 <= systolic <= 300 and 5 <= diastolic < systolic and diastolic < mean < systolic:
                    for name, value in zip(("sbp_mmhg", "dbp_mmhg", "map_mmhg"), (systolic, diastolic, mean)):
                        assign(name, value)

    resp = prepare("resp")
    if resp is not None:
        wave, fs = resp
        if duration >= 30 and fs >= 2:
            power = abs(np.fft.rfft((wave - wave.mean()) * np.hanning(len(wave)))) ** 2
            freq = np.fft.rfftfreq(len(wave), 1 / fs)
            band = (freq >= .08) & (freq <= .7)
            if band.any() and power[band].sum() / max(power.sum(), 1e-12) >= .2:
                assign("rr_bpm", 60 * freq[band][np.argmax(power[band])])
    return list(output.values())


def build_reference_targets(manifest, output):
    """JSONL rows: sample_id, path (window NPZ), duration_seconds; paths are relative."""
    manifest, output = Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, seen = [], set()
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in seen:
            raise ValueError("Reference sample IDs must be unique and nonempty")
        seen.add(sample_id)
        path = Path(row["path"])
        path = path if path.is_absolute() else manifest.parent / path
        with np.load(path, allow_pickle=False) as signals:
            targets = reference_targets(signals, float(row["duration_seconds"]))
        digest = file_sha256(path)
        rows.extend(dict(sample_id=sample_id, reference_sha256=digest, **t) for t in targets)
    if not rows:
        raise ValueError("Empty reference manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


class TargetTable:
    """Long-format scalar labels, shared by pretraining and downstream evaluation.

    External labels may use any target name, but require units, reference, method,
    validity, and window duration. Missing labels are always masked, never zero labels.
    """
    def __init__(self, path):
        self.fingerprint = file_sha256(path)
        self.rows, self.units = {}, {}
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "target", "value", "valid", "units", "reference", "method", "duration_seconds"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Target CSV requires {sorted(required)}")
            for row in reader:
                key = (row["sample_id"], row["target"])
                if not all(key) or key in self.rows or row["valid"] not in {"0", "1"}:
                    raise ValueError("Invalid/duplicate target key or validity")
                valid = row["valid"] == "1"
                value = float(row["value"]) if valid else 0.
                duration = float(row["duration_seconds"])
                if not np.isfinite(value) or not np.isfinite(duration) or duration <= 0:
                    raise ValueError("Nonfinite label or invalid window duration")
                if not row["reference"] or not row["method"] or not row["units"]:
                    raise ValueError("Labels require explicit reference, method and units")
                previous = self.units.setdefault(row["target"], row["units"])
                if previous != row["units"]:
                    raise ValueError("A target cannot mix units")
                self.rows[key] = {**row, "value": value, "valid": valid, "duration_seconds": duration}

    def arrays(self, ids, targets, duration=None):
        values = np.zeros((len(ids), len(targets)), np.float32)
        valid = np.zeros_like(values, dtype=bool)
        for i, sample in enumerate(ids):
            for j, target in enumerate(targets):
                row = self.rows.get((str(sample), target))
                if row is None:
                    continue
                if duration is not None and abs(row["duration_seconds"] - duration) > 1e-6:
                    raise ValueError("Label duration differs from input window; cropping labels is not supported")
                values[i, j], valid[i, j] = row["value"], row["valid"]
        return values, valid


class ReferenceDataset(Dataset):
    def __init__(self, dataset, table, targets):
        if dataset.config.window_mode != "full":
            raise ValueError("Window reference targets require full input windows")
        self.dataset, self.records, self.table, self.targets = dataset, dataset.records, table, targets

    def __len__(self):
        return len(self.dataset)

    def metadata(self, index):
        return self.dataset.metadata(index)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        sample = self.dataset[index]
        values, valid = self.table.arrays([sample["sample_id"]], self.targets,
                                         self.dataset.config.scales_seconds[0])
        sample.update(reference_values=torch.from_numpy(values[0]), reference_valid=torch.from_numpy(valid[0]))
        return sample

    def close(self):
        self.dataset.close()


class ReferenceNormalizer(torch.nn.Module):
    def __init__(self, statistics):
        super().__init__()
        self.statistics = statistics
        self.register_buffer("mean", torch.tensor(statistics["mean"], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(statistics["scale"], dtype=torch.float32))

    def forward(self, values):
        return (values.float() - self.mean) / self.scale


def reference_statistics(dataset, table, targets):
    """Patient-balanced target scaling fitted on known TRAIN groups only."""
    from .calibration import subject_weights

    rows = [dataset.metadata(i) for i in range(len(dataset))]
    ids = [r["sample_id"] for r in rows]
    subjects = np.asarray([r["subject_id"] for r in rows])
    known = np.asarray([r["subject_known"] for r in rows])
    values, valid = table.arrays(ids, targets, dataset.config.scales_seconds[0])
    mean, scale, counts = [], [], []
    for j, target in enumerate(targets):
        take = valid[:, j] & known
        if not take.any():
            raise ValueError(f"No known training groups with valid target: {target}")
        weights = subject_weights(subjects[take])
        mu = np.average(values[take, j], weights=weights)
        sd = np.sqrt(np.average((values[take, j] - mu) ** 2, weights=weights))
        mean.append(float(mu))
        scale.append(float(max(sd, 1e-4)))
        counts.append({"target": target, "windows": int(take.sum()), "groups": len(set(subjects[take]))})
    return {"version": 1, "targets": targets, "mean": mean, "scale": scale, "counts": counts,
            "labels_sha256": table.fingerprint, "units": {t: table.units[t] for t in targets},
            "scope": "known training groups; patient-balanced target normalization"}
