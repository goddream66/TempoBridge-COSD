"""Patient-disjoint manifests and paired crops in physical time.

Each NPZ is one synchronous recording/window, not a serialized Python object.
Resampling is offline and anti-aliased; this is not a streaming-causal pipeline.
"""

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .config import DataConfig


@dataclass(frozen=True)
class Record:
    sample_id: str
    subject_id: str
    recording_id: str
    split: str
    path: Path
    ppg_quality: float
    ecg_quality: float
    h5_row: int | None = None
    source: str = "default"
    subject_known: bool = True
    stored_fs: float | None = None


def read_manifest(path, check_files=True, require_train=True, data_config=None):
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        from .h5data import H5RecordIndex
        config = data_config or DataConfig()
        index = H5RecordIndex(path, config.source_roots, config.integrity)
        if require_train and not index.counts["train"]:
            raise ValueError("No training rows in HDF5 manifest")
        return index
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "subject_id", "recording_id", "split", "path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires {sorted(required)}")
        records, sample_ids, patient_splits, recording_subjects, file_splits = [], set(), {}, {}, {}
        for row in reader:
            if any(not row.get(k, "").strip() for k in required):
                raise ValueError("Manifest contains empty required field")
            sid, patient, rec, split = (row[k].strip() for k in (
                "sample_id", "subject_id", "recording_id", "split"
            ))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unknown split {split}")
            if sid in sample_ids:
                raise ValueError(f"Duplicate sample_id {sid}")
            sample_ids.add(sid)
            if patient in patient_splits and patient_splits[patient] != split:
                raise ValueError(f"Patient leakage across splits: {patient}")
            patient_splits[patient] = split
            if rec in recording_subjects and recording_subjects[rec] != patient:
                raise ValueError(f"recording_id shared by multiple patients: {rec}")
            recording_subjects[rec] = patient
            file_path = (path.parent / row["path"]).resolve()
            if str(file_path) in file_splits and file_splits[str(file_path)] != split:
                raise ValueError("Same source file appears across splits")
            file_splits[str(file_path)] = split
            if check_files and not file_path.is_file():
                raise FileNotFoundError(file_path)
            pq, eq = float(row.get("ppg_quality") or 1), float(row.get("ecg_quality") or 1)
            if not (0 <= pq <= 1 and 0 <= eq <= 1):
                raise ValueError("Quality scores must be finite values in [0,1]")
            records.append(Record(sid, patient, rec, split, file_path, pq, eq))
    if not records or (require_train and not any(r.split == "train" for r in records)):
        raise ValueError("Manifest needs nonempty train split")
    return records


def manifest_digest(records):
    """Bind artifacts to content, IDs and split assignments, independently of location."""
    if hasattr(records, "digest"):
        return records.digest
    digest = hashlib.sha256()
    for r in sorted(records, key=lambda r: r.sample_id):
        digest.update(json.dumps([r.sample_id, r.subject_id, r.recording_id, r.split,
                                  r.ppg_quality, r.ecg_quality]).encode())
        with r.path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def split_records(records, split):
    return records.subset(split) if hasattr(records, "subset") else [r for r in records if r.split == split]


def split_size(records, split):
    return records.counts[split] if hasattr(records, "counts") else sum(r.split == split for r in records)


def subject_ids(records, split):
    if hasattr(records, "subjects"):
        return records.subjects(split)
    return {r.subject_id for r in records if r.split == split and r.subject_known}


def read_signal(container, key, target_fs):
    if key not in container:
        return None
    raw = np.asarray(container[key], dtype=np.float32)
    if raw.ndim == 2 and 1 in raw.shape:
        raw = raw.reshape(-1)
    if raw.ndim != 1 or raw.size < 2 or not np.isfinite(raw).all():
        raise ValueError(f"{key} must be a finite 1D single-lead waveform")
    rate_key = key + "_fs"
    if rate_key not in container:
        raise ValueError(f"Missing {rate_key}; sample rate is never inferred")
    source_fs = float(np.asarray(container[rate_key]).item())
    if not np.isfinite(source_fs) or source_fs <= 0 or source_fs != round(source_fs):
        raise ValueError("Sample rates must be positive integers")
    source_fs = int(source_fs)
    if source_fs != target_fs:
        divisor = math.gcd(source_fs, target_fs)
        raw = resample_poly(raw, target_fs // divisor, source_fs // divisor).astype(np.float32)
    return raw


class PairedDataset(Dataset):
    def __init__(self, records, config: DataConfig, split="train", seed=42, augment=True,
                 include_ecg=True):
        self.records = split_records(records, split)
        self.config, self.seed, self.epoch, self.augment = config, seed, 0, augment
        self.include_ecg = include_ecg
        from .h5data import H5Reader
        self.h5_reader = H5Reader(config.max_open_shards)
        if not self.records:
            raise ValueError(f"Empty {split} split")

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        record = self.records[index]
        pq, eq = record.ppg_quality, record.ecg_quality
        if record.h5_row is not None:
            data, pq, eq = self.h5_reader.read(record, self.include_ecg)
            ppg = read_signal(data, "ppg", self.config.ppg_fs)
            ecg = read_signal(data, "ecg", self.config.ecg_fs) if self.include_ecg else None
        else:
            with np.load(record.path, allow_pickle=False) as data:
                ppg = read_signal(data, "ppg", self.config.ppg_fs)
                ecg = read_signal(data, "ecg", self.config.ecg_fs) if self.include_ecg else None
        if ppg is None:
            raise ValueError(f"Missing PPG: {record.sample_id}")
        duration = len(ppg) / self.config.ppg_fs
        if ecg is not None:
            ecg_duration = len(ecg) / self.config.ecg_fs
            if abs(ecg_duration - duration) > max(1 / self.config.ppg_fs, 1 / self.config.ecg_fs) + 1e-6:
                raise ValueError(f"Unequal synchronized durations: {record.sample_id}")
            duration = min(duration, ecg_duration)
        if duration + 1e-6 < max(self.config.scales_seconds):
            raise ValueError(f"Recording too short: {record.sample_id}")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        ppg_crops, ecg_crops = [], []
        for seconds in self.config.scales_seconds:
            available = max(0, duration - seconds)
            start = rng.uniform(0, available) if self.augment else available / 2
            p_start, e_start = round(start * self.config.ppg_fs), round(start * self.config.ecg_fs)
            p_len, e_len = round(seconds * self.config.ppg_fs), round(seconds * self.config.ecg_fs)
            ppg_crops.append(torch.from_numpy(ppg[p_start:p_start + p_len].copy())[None])
            e = ecg[e_start:e_start + e_len].copy() if ecg is not None else np.zeros(e_len, np.float32)
            ecg_crops.append(torch.from_numpy(e)[None])
        return {
            "ppg": ppg_crops, "ecg": ecg_crops, "ecg_valid": ecg is not None,
            "ppg_quality": pq, "ecg_quality": eq,
            "subject_id": record.subject_id, "recording_id": record.recording_id,
            "sample_id": record.sample_id,
            "source": record.source, "subject_known": record.subject_known,
        }

    def close(self):
        self.h5_reader.close()


def make_synthetic(output, subjects=18, windows=6, seconds=10, fs=100, seed=42):
    """Integration fixtures, NOT physiologically validated signals or benchmarks."""
    output = Path(output)
    if (output / "manifest.csv").exists():
        raise FileExistsError("Synthetic manifest already exists; use another output directory")
    if subjects < 8 or windows < 1 or seconds <= 0 or fs < 10:
        raise ValueError("Synthetic fixture needs >=8 subjects, >=1 windows and valid duration/rate")
    output.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(seed), []
    t = np.arange(round(seconds * fs)) / fs
    for person in range(subjects):
        split = "train" if person < subjects - 4 else ("val" if person < subjects - 2 else "test")
        base_rate = rng.uniform(0.9, 1.7)
        for window in range(windows):
            rate = base_rate + rng.uniform(-0.15, 0.15)
            phase = 2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi)
            quality = [0.2, 0.5, 0.9][window % 3]
            latent = np.sin(phase) + 0.25 * np.sin(2 * phase)
            ppg = np.sin(phase - 0.5) + 0.2 * np.sin(2 * phase - 1) + (1 - quality) * rng.normal(0, 0.3, len(t))
            ecg = latent + 0.1 * np.sin(11 * phase) + rng.normal(0, 0.03, len(t))
            name = f"s{person:03d}_w{window:03d}"
            fields = {"ppg": ppg.astype(np.float32), "ppg_fs": np.array(fs)}
            # Exercise missing-ECG fallback without dropping the PPG sample.
            if not (person == 0 and window == 0):
                fields.update(ecg=ecg.astype(np.float32), ecg_fs=np.array(fs))
            np.savez(output / f"{name}.npz", **fields)
            rows.append(dict(sample_id=name, subject_id=f"s{person:03d}",
                             recording_id=f"r{person:03d}", split=split, path=f"{name}.npz",
                             ppg_quality=quality, ecg_quality=0.95))
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output / "manifest.csv"
