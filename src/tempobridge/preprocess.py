"""Offline waveform cleaning and transactional PulseBound-layout HDF5 output.

Same nominal PPG filter/window/resampling as the released multimodal converter;
missingness is deliberately checked BEFORE interpolation, and ECG is row-wise.
No ECG-derived diagnostic labels are fabricated for the TempoBridge objective.
"""

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt
from scipy.special import expit
from scipy.stats import skew

from .h5data import canonical_hash, h5py_module
from .utils import write_json


@dataclass
class PreprocessConfig:
    target_fs: int = 50
    window_seconds: float = 30
    quality_patch_seconds: float = 1
    max_missing_fraction: float = 0.2
    windows_per_shard: int = 50000
    max_windows_per_record: int = 200
    ppg_channel: str | None = None
    ecg_channel: str | None = None

    def validate(self):
        if self.target_fs <= 16 or self.window_seconds <= 0 or self.quality_patch_seconds <= 0:
            raise ValueError("Invalid window/rate; target Nyquist must exceed 8 Hz")
        n, patch = self.target_fs * self.window_seconds, self.target_fs * self.quality_patch_seconds
        if n != round(n) or patch != round(patch) or patch < 4 or n % patch:
            raise ValueError("Window/quality patch lengths must be integral and divisible")
        if not 0 <= self.max_missing_fraction < 1 or self.windows_per_shard < 1 or self.max_windows_per_record < 0:
            raise ValueError("Invalid missingness/shard/record cap")
        return self


def clean_finite(wave, max_missing):
    if wave is None:
        return None, 1.0, "absent"
    wave = np.asarray(wave, np.float64)
    if wave.ndim != 1 or len(wave) < 4:
        return None, 1.0, "shape"
    finite = np.isfinite(wave)
    fraction = 1 - finite.mean()
    if not finite.any() or fraction > max_missing:
        return None, float(fraction), "missing_fraction"
    if not finite.all():
        locations = np.arange(len(wave))
        wave = np.interp(locations, locations[finite], wave[finite])
    if np.ptp(wave) <= 1e-8:
        return None, float(fraction), "constant"
    return wave, float(fraction), None


def minmax(wave):
    span = np.ptp(wave)
    if not np.isfinite(wave).all() or span <= 1e-8:
        raise ValueError("Cannot normalize a constant/nonfinite waveform")
    return (2 * (wave - wave.min()) / span - 1).astype(np.float32)


def spectral_sqi(wave, fs):
    if np.std(wave) < 1e-8:
        return 0.0
    power = abs(np.fft.rfft(wave - np.mean(wave))) ** 2
    frequencies = np.fft.rfftfreq(len(wave), 1 / fs)
    return float(np.clip(power[(frequencies >= 0.5) & (frequencies <= 8)].sum() / (power.sum() + 1e-12), 0, 1))


def clean_window(ppg, ecg, fs, config):
    config.validate()
    if not np.isfinite(fs) or fs <= 16:
        return None, "source_fs"
    if len(ppg) != round(config.window_seconds * fs):
        return None, "incomplete_window"
    ppg, missing_ppg, reason = clean_finite(ppg, config.max_missing_fraction)
    if reason:
        return None, "ppg_" + reason
    target_len = round(config.window_seconds * config.target_fs)
    try:
        filtered = sosfiltfilt(butter(2, [0.5, 8], btype="band", fs=fs, output="sos"), ppg)
        normalized = minmax(resample(filtered, target_len) if len(filtered) != target_len else filtered)
    except ValueError:
        return None, "ppg_filter_or_normalization"
    ecg, missing_ecg, ecg_reason = clean_finite(ecg, config.max_missing_fraction)
    if ecg is not None and len(ecg) != len(ppg):
        ecg, ecg_reason = None, "length"
    ecg_norm = np.zeros(target_len, np.float32)
    if ecg is not None:
        try:
            ecg_norm = minmax(resample(ecg, target_len) if len(ecg) != target_len else ecg)
        except ValueError:
            ecg, ecg_reason = None, "normalization"
    patch = round(config.quality_patch_seconds * config.target_fs)
    patches = normalized.reshape(-1, patch)
    std = np.std(patches, axis=1)
    median = np.median(std)
    mad = np.median(abs(std - median))
    relative = np.ones(len(patches)) if mad < 1e-9 else np.exp(-0.2 * (0.6745 * (std - median) / mad) ** 2)
    amplitude = relative * expit(50 * (std - 0.05)) * expit(-5 * (std - 2))
    skewness = np.tanh(abs(np.nan_to_num(skew(patches, axis=1, bias=False), nan=0)))
    quality = np.asarray([spectral_sqi(p, config.target_fs) for p in patches], np.float32)
    return {"signals": normalized, "ecg_wave": ecg_norm, "has_ecg": ecg is not None,
            "feat_amp": amplitude.astype(np.float32), "feat_skew": skewness.astype(np.float32),
            "feat_avg": (0.5 * amplitude + 0.5 * skewness).astype(np.float32),
            "patch_sqi": quality, "ppg_quality": spectral_sqi(ppg, fs),
            "ecg_quality": 1 - missing_ecg if ecg is not None else 0.0,
            "ppg_missing_fraction": missing_ppg, "ecg_missing_fraction": missing_ecg,
            "ecg_rejection": ecg_reason or ""}, None


def discover_wfdb(root):
    root = Path(root).resolve()
    records_file = root / "RECORDS"
    paths = set()
    if records_file.exists():
        for value in records_file.read_text().splitlines():
            value = value.strip().rstrip("/")
            if not value or value.startswith("#"):
                continue
            item = (root / value).resolve()
            if not item.is_relative_to(root):
                raise ValueError("RECORDS entry escapes input root")
            if item.is_dir():
                paths.update(item.glob("*.hea"))
            elif item.with_suffix(".hea").is_file():
                paths.add(item.with_suffix(".hea"))
    else:
        paths.update(root.rglob("*.hea"))
    return sorted(p for p in paths if not p.stem.endswith(("_layout", "n")))


def read_id_map(path):
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = row["record_path"].replace("\\", "/")
        if key in result:
            raise ValueError(f"Duplicate record_path in ID map: {key}")
        result[key] = {k: int(row[k]) if row.get(k, "").strip() else -1
                       for k in ("subject_id", "case_id")}
    return result


def pick_channel(names, explicit, kind):
    if explicit:
        if explicit not in names:
            raise ValueError(f"Requested {kind} track {explicit!r} not found")
        return names.index(explicit)
    aliases = {"ppg": ["pleth", "ppg"], "ecg": ["ii", "i", "iii", "v", "avr", "avl", "avf", "mcl", "mcl1", "ecg", "ecg_ii", "ecg1", "ecg2"]}[kind]
    for i, name in enumerate(names):
        tail = name.lower().split("/")[-1].strip()
        if tail in aliases:
            return i
    return None


def raw_windows(path, kind, config, start_window=0):
    """Yield (window number, ppg, ecg, common fs); input stays on disk for WFDB."""
    if kind == "mimic":
        try:
            import wfdb
        except ImportError as exc:
            raise ImportError("WFDB conversion requires pip install -e '.[data]'") from exc
        record_path = str(path.with_suffix(""))
        header = wfdb.rdheader(record_path)
        if not getattr(header, "sig_name", None):
            # Match the released leaf-record pathway; do not also duplicate parent MultiRecords.
            raise ValueError("no_single_segment_channels")
        names, fs = header.sig_name, float(header.fs)
        if not np.isfinite(fs) or fs <= 16:
            raise ValueError("invalid_wfdb_sampling_rate")
        p = pick_channel(names, config.ppg_channel, "ppg")
        e = pick_channel(names, config.ecg_channel, "ecg")
        if p is None:
            raise ValueError("no_ppg_channel")
        size = round(config.window_seconds * fs)
        count = header.sig_len // size
        if config.max_windows_per_record:
            count = min(count, config.max_windows_per_record)
        selected = [p] + ([e] if e is not None else [])
        for i in range(start_window, count):
            record = wfdb.rdrecord(record_path, sampfrom=i * size, sampto=(i + 1) * size,
                                   channels=selected, physical=True)
            yield i, record.p_signal[:, 0], record.p_signal[:, 1] if e is not None else None, fs
    else:
        try:
            import vitaldb
        except ImportError as exc:
            raise ImportError("VitalDB conversion requires pip install -e '.[data]'") from exc
        names = vitaldb.vital_trks(str(path))
        p = pick_channel(names, config.ppg_channel, "ppg")
        e = pick_channel(names, config.ecg_channel, "ecg")
        if p is None:
            raise ValueError("no_ppg_channel")
        selected = [names[p]] + ([names[e]] if e is not None else [])
        vf = vitaldb.VitalFile(str(path), track_names=selected)
        # A common absolute timeline; never independently strip NaNs from tracks.
        fs = max(float(vf.trks[name].srate) for name in selected)
        if not np.isfinite(fs) or fs <= 16:
            raise ValueError("invalid_vital_waveform_sampling_rate")
        wave = vf.to_numpy(selected, interval=1 / fs)
        size = round(config.window_seconds * fs)
        count = len(wave) // size
        if config.max_windows_per_record:
            count = min(count, config.max_windows_per_record)
        for i in range(start_window, count):
            part = wave[i * size:(i + 1) * size]
            yield i, part[:, 0], part[:, 1] if e is not None else None, fs


class ShardWriter:
    """Append rows to a bounded, chunked shard; publish only completed files."""
    def __init__(self, output, source, config, contract, shard_number):
        self.output, self.source, self.config, self.contract = output, source, config, contract
        self.number, self.handle, self.rows, self.last_cursor = shard_number, None, 0, None

    def add(self, sample, cursor):
        if self.handle is None:
            self.path = self.output / f"{self.source}_mm_part_{self.number:04d}.h5"
            if self.path.exists():
                raise FileExistsError(self.path)
            self.partial = self.path.with_suffix(".h5.partial")
            self.handle = h5py_module().File(self.partial, "w")
            self.handle.attrs.update(target_fs=self.config.target_fs,
                                     window_length=self.config.window_seconds,
                                     patch_length_sec=self.config.quality_patch_seconds,
                                     preprocessing_contract=self.contract,
                                     converter="tempobridge_strict_v1")
        for key, value in sample.items():
            array = np.asarray(value)
            if key not in self.handle:
                dtype = h5py_module().string_dtype("utf-8") if isinstance(value, str) else array.dtype
                self.handle.create_dataset(key, shape=(0, *array.shape), maxshape=(None, *array.shape),
                                           chunks=(min(128, self.config.windows_per_shard), *array.shape),
                                           dtype=dtype, compression="lzf")
            dataset = self.handle[key]
            dataset.resize(self.rows + 1, axis=0)
            dataset[self.rows] = value
        self.rows += 1
        self.last_cursor = cursor
        if self.rows >= self.config.windows_per_shard:
            self.publish()

    def publish(self):
        if self.handle is None:
            return
        self.handle.attrs["last_cursor"] = json.dumps(self.last_cursor)
        self.handle.flush()
        self.handle.close()
        self.handle = None
        self.partial.replace(self.path)
        self.number += 1
        self.rows = 0

    def close_incomplete(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def preprocess_raw(kind, root, output, config=None, source=None, id_map=None, resume=False,
                   max_records=None):
    config = (config or PreprocessConfig()).validate()
    source = source or ("mimic" if kind == "mimic" else "vitaldb")
    if kind not in {"mimic", "vitaldb"} or not re.fullmatch(r"[A-Za-z0-9_-]+", source):
        raise ValueError("Invalid converter kind/source name")
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir() or output == root or output.is_relative_to(root):
        raise ValueError("Use a separate output directory outside the raw-data root")
    paths = discover_wfdb(root) if kind == "mimic" else sorted(root.rglob("*.vital"))
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        paths = paths[:max_records]
    if not paths:
        raise FileNotFoundError("No local raw records found")
    identities = read_id_map(id_map)
    inventory = [(p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    # WFDB samples live in companion .dat files. Include their inventory in resume identity.
    companions = [(p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in sorted(root.rglob("*.dat"))] if kind == "mimic" else []
    contract_data = {"kind": kind, "source": source, "config": asdict(config),
                     "records": inventory, "companions": companions, "id_map": identities}
    contract = canonical_hash(contract_data)
    contract_path = output / "conversion_contract.json"
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Output is nonempty; use --resume with the identical input/config")
    output.mkdir(parents=True, exist_ok=True)
    if resume and contract_path.exists():
        if json.loads(contract_path.read_text())["sha256"] != contract:
            raise ValueError("Resume input/config changed")
    elif resume and any(output.glob("*.h5")):
        raise ValueError("Cannot resume shards without conversion contract")
    write_json(contract_path, {"sha256": contract, **contract_data})
    report_path = output / "conversion_report.json"
    previous_report = json.loads(report_path.read_text()) if resume and report_path.exists() else None
    cursor, existing_rows = (-1, -1), 0
    existing = sorted(output.glob(f"{source}_mm_part_*.h5"))
    for number, shard in enumerate(existing):
        if shard.name != f"{source}_mm_part_{number:04d}.h5":
            raise ValueError("Noncontiguous conversion shards")
        with h5py_module().File(shard, "r") as handle:
            if handle.attrs.get("preprocessing_contract") != contract:
                raise ValueError("Shard belongs to another conversion")
            last = tuple(json.loads(handle.attrs["last_cursor"]))
            if last <= cursor:
                raise ValueError("Invalid conversion cursor")
            cursor = last
            existing_rows += len(handle["signals"])
    if previous_report and previous_report.get("complete"):
        if previous_report["total_saved_windows"] != existing_rows:
            raise ValueError("Completed conversion is missing published windows")
        return previous_report
    writer = ShardWriter(output, source, config, contract, len(existing))
    counts, issues, processed = Counter(), [], 0
    try:
        for number, path in enumerate(paths):
            if number < cursor[0]:
                continue
            relative = path.relative_to(root).as_posix()
            identity = identities.get(relative, {})
            found = re.search(r"p(\d{5,7})", relative) if kind == "mimic" else None
            subject = identity.get("subject_id", int(found.group(1)) if found else -1)
            case = identity.get("case_id", int(path.stem) if kind == "vitaldb" and path.stem.isdigit() else -1)
            try:
                start = cursor[1] + 1 if number == cursor[0] else 0
                for window, ppg, ecg, fs in raw_windows(path, kind, config, start):
                    counts["attempted_windows"] += 1
                    sample, reason = clean_window(ppg, ecg, fs, config)
                    if sample is None:
                        counts["rejected_" + reason] += 1
                        continue
                    sample.update(subject_id=np.int64(subject), case_id=np.int64(case),
                                  record_key=relative, window_start_s=window * config.window_seconds,
                                  raw_fs=float(fs), dataset_id=np.int64(0 if kind == "mimic" else 1))
                    try:
                        writer.add(sample, (number, window))
                    except Exception as exc:
                        # Storage errors must abort, never masquerade as a rejected raw record.
                        raise RuntimeError("Shard write failed; resume after fixing storage") from exc
                    counts["accepted_windows"] += 1
                    counts["paired_windows"] += int(sample["has_ecg"])
                    if sample["ecg_rejection"]:
                        counts["ecg_" + sample["ecg_rejection"]] += 1
                processed += 1
            except (OSError, ValueError, KeyError) as exc:
                counts["record_errors"] += 1
                if len(issues) < 100:
                    issues.append({"record": relative, "error": str(exc)})
        writer.publish()
    finally:
        writer.close_incomplete()
    report = {"complete": existing_rows + counts["accepted_windows"] > 0,
              "contract_sha256": contract, "kind": kind,
              "records_in_inventory": len(paths), "records_processed_this_invocation": processed,
              "existing_windows_on_resume": existing_rows, "counts_this_invocation": dict(counts),
              "total_saved_windows": existing_rows + counts["accepted_windows"],
              "shards": writer.number, "issues_first_100": issues,
              "differences_from_pulsebound": ["missingness checked before interpolation",
                  "nonfinite/constant ECG marked unavailable per row", "ECG stored even in mixed-availability shards",
                  "VitalDB converter is a new synchronized local-file implementation",
                  "no unused scaffold/event labels fabricated", "ECG quality is finite-coverage, not calibrated confidence"]}
    write_json(report_path, report)
    if report["total_saved_windows"] == 0:
        raise ValueError(f"No usable windows; inspect {report_path}")
    return report
