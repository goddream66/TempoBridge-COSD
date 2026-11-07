"""PulseBound-compatible HDF5 indexing without materializing millions of rows.

The group-ID/hash/ranking wire protocol matches the supplied v6 implementation.
Storage roots are relocatable; integrity is content-based rather than mtime-based.
"""

import bisect
import hashlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .utils import file_sha256, write_json

ID_KEYS = ("subject_id", "patient_id", "case_id", "record_id")
SALT = "pulsebound-iclr2027-source-split-v1"
SPLITS = ("train", "validation", "test")


def h5py_module():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 support requires: pip install -e '.[data]'") from exc
    return h5py


def canonical_hash(payload, digest_key="manifest_sha256"):
    unsigned = {k: v for k, v in payload.items() if k != digest_key}
    value = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_identifier(value):
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(()).item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return f"bytes:{value.hex()}" if value else None
    if isinstance(value, str):
        value = value.strip()
        return None if value.lower() in {"", "-1", "nan", "none", "null", "unknown"} else f"str:{value}"
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"int:{value}" if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        return f"int:{int(value)}" if value.is_integer() else f"float:{value:.17g}"
    return None


def group_hash(source, key, value, salt=SALT):
    value = normalize_identifier(value)
    if value is None:
        return None
    return hashlib.sha256("\0".join(("pulsebound-source-group-v1", salt, source, key, value)).encode()).hexdigest()


def split_counts(n):
    fractions = {"train": 0.90, "validation": 0.05, "test": 0.05}
    exact = {k: n * v for k, v in fractions.items()}
    counts = {k: math.floor(v) for k, v in exact.items()}
    priority = {"validation": 0, "test": 1, "train": 2}
    order = sorted(SPLITS, key=lambda k: (-(exact[k] - counts[k]), priority[k]))
    for k in order[:n - sum(counts.values())]:
        counts[k] += 1
    return counts


def assign_groups(groups, source, salt=SALT):
    ranked = sorted(set(groups), key=lambda g: (hashlib.sha256(
        "\0".join(("pulsebound-source-split-v1", salt, source, g)).encode()).digest(), g))
    counts, assigned, start = split_counts(len(ranked)), {}, 0
    for split in SPLITS:
        for g in ranked[start:start + counts[split]]:
            assigned[g] = "val" if split == "validation" else split
        start += counts[split]
    return assigned


def scan_shard(path, source, spec, salt):
    with h5py_module().File(path, "r") as handle:
        if "signals" not in handle or handle["signals"].ndim != 2:
            raise ValueError(f"{path}: expected signals [N,L]")
        n, length = handle["signals"].shape
        if n < 1 or length < 2:
            raise ValueError(f"{path}: empty/invalid signals")
        fs = handle.attrs.get("target_fs", handle.attrs.get("fs", spec.get("fs")))
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"{path}: supply explicit fs in source spec when attributes are absent")
        if spec.get("fs") is not None and float(fs) != float(spec["fs"]):
            raise ValueError(f"{path}: source fs conflicts with HDF5 metadata")
        for key in (*ID_KEYS, "ecg_wave", "has_ecg", "patch_sqi", "metrics", "ecg_quality", "ppg_quality"):
            if key in handle and (not handle[key].shape or handle[key].shape[0] != n):
                raise ValueError(f"{path}: {key} row count differs")
        if "ecg_wave" in handle and handle["ecg_wave"].shape != (n, length):
            raise ValueError(f"{path}: ECG must align with stored PPG [N,L]")
        available = [k for k in ID_KEYS if k in handle]
        intervals, usage = [], {}
        for begin in range(0, n, 65536):
            end = min(n, begin + 65536)
            ids = {k: handle[k][begin:end] for k in available}
            for offset in range(end - begin):
                g = None
                for key in available:
                    g = group_hash(source, key, ids[key][offset], salt)
                    if g is not None:
                        usage[key] = usage.get(key, 0) + 1
                        break
                row = begin + offset
                if intervals and intervals[-1][2] == g:
                    intervals[-1][1] = row + 1
                else:
                    intervals.append([row, row + 1, g])
        metadata = {"rows": n, "length": length, "fs": float(fs), "intervals": intervals,
                    "id_key_usage": usage, "has_ecg_dataset": "ecg_wave" in handle}
    return metadata


def build_h5_manifest(spec_path, output, pulsebound_split=None, salt=SALT):
    """Index metadata only; existing waveforms are never rewritten/re-filtered."""
    output, spec_path = Path(output).resolve(), Path(spec_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if not spec.get("sources"):
        raise ValueError("Source specification needs nonempty sources")
    original = None
    if pulsebound_split:
        original = json.loads(Path(pulsebound_split).read_text(encoding="utf-8-sig"))
        if (original.get("format") != "pulsebound_group_split_manifest" or original.get("version") != 1
                or original.get("manifest_sha256") != canonical_hash(original)
                or original.get("group_id_priority") != list(ID_KEYS)
                or original.get("split_ratios") != {"train": 0.9, "validation": 0.05, "test": 0.05}):
            raise ValueError("Invalid PulseBound split manifest")
        if set(original["sources"]) != set(spec["sources"]):
            raise ValueError("Source set differs from PulseBound split manifest")
        salt = original["salt"]
    result = {"format": "tempobridge_h5_manifest", "version": 1, "salt": salt,
              "group_id_priority": list(ID_KEYS), "sources": {}}
    seen_paths, seen_hashes = set(), set()
    for source, settings in sorted(spec["sources"].items()):
        root = (spec_path.parent / settings["root"]).resolve()
        pattern = settings.get("pattern", "*.h5")
        paths = sorted(root.rglob(pattern) if settings.get("recursive", False) else root.glob(pattern))
        paths = [p.resolve() for p in paths if p.is_file()]
        if not paths:
            raise FileNotFoundError(f"No {pattern} in {root}")
        files, groups = [], set()
        for path in paths:
            if path in seen_paths or not path.is_relative_to(root):
                raise ValueError("Duplicate shard or shard outside source root")
            seen_paths.add(path)
            item = scan_shard(path, source, settings, salt)
            item.update(relative_path=path.relative_to(root).as_posix(),
                        size_bytes=path.stat().st_size, sha256=file_sha256(path))
            if item["sha256"] in seen_hashes:
                raise ValueError("Identical HDF5 files appear more than once in the corpus")
            seen_hashes.add(item["sha256"])
            groups.update(g for _, _, g in item["intervals"] if g is not None)
            files.append(item)
        assignments = assign_groups(groups, source, salt)
        if original is not None:
            reference = original["sources"].get(source)
            if reference is None:
                raise ValueError(f"PulseBound manifest has no source {source}")
            registered = {g: ("val" if split == "validation" else split)
                          for split, values in reference["groups"].items() for g in values}
            if len(registered) != sum(len(v) for v in reference["groups"].values()) or registered != assignments:
                raise ValueError("Source groups/assignments differ from PulseBound manifest")
            existing = {f["relative_path"]: f for f in reference["files"]}
            if set(existing) != {f["relative_path"] for f in files}:
                raise ValueError("Source file set differs from PulseBound manifest")
            for item in files:
                old = existing[item["relative_path"]]
                if old["sha256"] != item["sha256"] or old["rows"] != item["rows"]:
                    raise ValueError("Shard content differs from PulseBound split")
                old_rows = sorted((a, b, None if g == "__unknown__" else g)
                                  for g, ranges in old["group_rows"].items() for a, b in ranges)
                # Compare row membership even if an upstream manifest uses different range merging.
                _assert_same_intervals(item["intervals"], old_rows)
        try:
            root_string = Path(os.path.relpath(root, output.parent)).as_posix()
        except ValueError:
            root_string = root.as_posix()  # Cross-drive layout; can override via config source_roots.
        result["sources"][source] = {"root": root_string, "assignments": assignments, "files": files}
    result["pulsebound_manifest_sha256"] = original["manifest_sha256"] if original else None
    result["manifest_sha256"] = canonical_hash(result)
    write_json(output, result)
    return result


def _assert_same_intervals(a, b):
    i = j = position = 0
    while i < len(a) and j < len(b):
        if a[i][0] > position or b[j][0] > position or a[i][2] != b[j][2]:
            raise ValueError("Row-to-group mapping differs")
        position = min(a[i][1], b[j][1])
        if a[i][1] == position:
            i += 1
        if b[j][1] == position:
            j += 1
    if i != len(a) or j != len(b):
        raise ValueError("Row-to-group coverage differs")


class H5RecordIndex(Sequence):
    """O(number of group runs) metadata, O(log runs) row lookup."""
    def __init__(self, manifest, source_roots=None, integrity="sha256", split=None, _parent=None):
        self.split = split
        self.segments, self.ends = [], []
        self._subjects, self.counts = {}, {s: 0 for s in ("train", "val", "test")}
        if _parent is not None:
            self.payload, self.files, self.digest = _parent.payload, _parent.files, _parent.digest
        else:
            path = Path(manifest).resolve()
            self.payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if (self.payload.get("format") != "tempobridge_h5_manifest" or self.payload.get("version") != 1
                    or self.payload.get("manifest_sha256") != canonical_hash(self.payload)):
                raise ValueError("Corrupt HDF5 index manifest")
            if integrity not in {"sha256", "metadata"}:
                raise ValueError("Unknown HDF5 integrity policy")
            self.files = {}
            for source, spec in self.payload["sources"].items():
                root = (path.parent / (source_roots or {}).get(source, spec["root"])).resolve()
                for entry in spec["files"]:
                    file_path = (root / entry["relative_path"]).resolve()
                    if not file_path.is_relative_to(root):
                        raise ValueError("Shard escapes source root")
                    if file_path.stat().st_size != entry["size_bytes"]:
                        raise ValueError(f"Shard size changed: {file_path}")
                    if integrity == "sha256" and file_sha256(file_path) != entry["sha256"]:
                        raise ValueError(f"Shard hash changed: {file_path}")
                    self.files[(source, entry["relative_path"])] = file_path
            # Keep identity stable under root overrides / copying the corpus.
            semantic = json.loads(json.dumps(self.payload))
            for spec in semantic["sources"].values():
                spec.pop("root", None)
            self.digest = canonical_hash(semantic)
        total = 0
        for source, spec in self.payload["sources"].items():
            assignments = spec["assignments"]
            if assignments != assign_groups(assignments, source, self.payload["salt"]):
                raise ValueError("Invalid deterministic group assignment")
            observed = set()
            for item in spec["files"]:
                cursor = 0
                file_path = self.files[(source, item["relative_path"])]
                for begin, end, group in item["intervals"]:
                    if begin != cursor or not begin < end <= item["rows"]:
                        raise ValueError("Invalid HDF5 row partition")
                    cursor = end
                    if group is not None:
                        if group not in assignments:
                            raise ValueError("Unassigned HDF5 group")
                        observed.add(group)
                    partition = assignments[group] if group is not None else "train"
                    self.counts[partition] += end - begin
                    if group is not None:
                        self._subjects.setdefault(partition, set()).add(f"{source}:{group}")
                    if split is None or split == partition:
                        self.segments.append((total, begin, end, group, partition, source, file_path, item["fs"]))
                        total += end - begin
                        self.ends.append(total)
                if cursor != item["rows"]:
                    raise ValueError("HDF5 row partition is incomplete")
            if observed != set(assignments):
                raise ValueError("Groups without rows in HDF5 index")
        self.length = total
        self.relative_names = {file_path: relative for (_, relative), file_path in self.files.items()}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        from .data import Record
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        offset, begin, _, group, split, source, path, fs = self.segments[bisect.bisect_right(self.ends, index)]
        row = begin + index - offset
        # Unknown IDs remain one unknown pool, never fictitious independent patients.
        return Record(f"{source}:{self.relative_names[path]}:{row}", f"{source}:{group or 'unknown'}",
                      "", split, path, 1.0, 1.0,
                      h5_row=row, source=source, subject_known=group is not None, stored_fs=fs,
                      recording_known=False, ppg_quality_source="unknown", ecg_quality_source="unknown")

    def subset(self, split):
        return H5RecordIndex(None, split=split, _parent=self)

    def subjects(self, split):
        return set(self._subjects.get(split, set()))


class H5Reader:
    """LRU handles are per process and excluded from DataLoader pickle state."""
    def __init__(self, maximum=4):
        self.maximum, self.pid = maximum, os.getpid()
        self.handles = OrderedDict()

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        return {"maximum": self.maximum, "pid": None, "handles": OrderedDict()}

    def __del__(self):
        self.close()

    def _handle(self, record):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        key = str(record.path)
        if key in self.handles:
            handle = self.handles.pop(key)
        else:
            handle = h5py_module().File(key, "r")
        self.handles[key] = handle
        while len(self.handles) > self.maximum:
            self.handles.popitem(last=False)[1].close()
        return handle

    def metadata(self, record):
        handle = self._handle(record)
        i = record.h5_row
        metadata = {"recording_id": "", "recording_known": False, "window_start_s": -1.0,
                    "case_id": "", "ppg_quality_source": "unknown", "ecg_quality_source": "unknown"}
        for field in ("record_key", "record_id"):
            if field in handle:
                value = normalize_identifier(handle[field][i])
                if value is not None:
                    metadata.update(recording_id=f"{record.source}:{field}:{value}", recording_known=True)
                    break
        if "case_id" in handle:
            value = normalize_identifier(handle["case_id"][i])
            metadata["case_id"] = f"{record.source}:{value}" if value is not None else ""
        if "window_start_s" in handle:
            value = float(handle["window_start_s"][i])
            if np.isfinite(value) and value >= 0:
                metadata["window_start_s"] = value
        metadata["ecg_available"] = "ecg_wave" in handle and ("has_ecg" not in handle or bool(handle["has_ecg"][i]))
        metadata["ppg_quality"] = float(handle["ppg_quality"][i]) if "ppg_quality" in handle else 1.0
        if "ppg_quality" not in handle:
            names = handle.attrs.get("metric_keys", "")
            names = names.decode() if isinstance(names, bytes) else names
            names = names.split(",") if isinstance(names, str) else list(names)
            if "metrics" in handle and "sqi" in names:
                metadata["ppg_quality"] = float(handle["metrics"][i, names.index("sqi")])
            elif "patch_sqi" in handle:
                metadata["ppg_quality"] = float(np.mean(handle["patch_sqi"][i]))
        if not np.isfinite(metadata["ppg_quality"]) or not 0 <= metadata["ppg_quality"] <= 1:
            metadata["ppg_quality"] = 0.
        metadata["identity_level"] = "unknown"
        for field in ID_KEYS:
            if field in handle and normalize_identifier(handle[field][i]) is not None:
                metadata["identity_level"] = field
                break
        return metadata

    def read(self, record, include_ecg=True):
        handle = self._handle(record)
        i = record.h5_row
        metadata = self.metadata(record)
        result = {"ppg": handle["signals"][i], "ppg_fs": record.stored_fs, "metadata": metadata}
        if not np.isfinite(result["ppg"]).all() or np.ptp(result["ppg"]) <= 1e-8:
            raise ValueError(f"Invalid PPG in {record.sample_id}; repair/reconvert the shard before training")
        available = include_ecg and "ecg_wave" in handle and ("has_ecg" not in handle or bool(handle["has_ecg"][i]))
        if available:
            ecg = np.asarray(handle["ecg_wave"][i], np.float32)
            if np.isfinite(ecg).all() and np.ptp(ecg) > 1e-8:
                result.update(ecg=ecg, ecg_fs=record.stored_fs)
        pq = float(handle["ppg_quality"][i]) if "ppg_quality" in handle else None
        if pq is not None:
            metadata["ppg_quality_source"] = str(handle.attrs.get("ppg_quality_definition", "stored_unspecified"))
        if pq is None and "metrics" in handle:
            names = handle.attrs.get("metric_keys", "")
            if isinstance(names, bytes):
                names = names.decode()
            names = names.split(",") if isinstance(names, str) else list(names)
            if "sqi" in names:
                pq = float(handle["metrics"][i, names.index("sqi")])
                metadata["ppg_quality_source"] = "metrics:sqi"
        if pq is None and "patch_sqi" in handle:
            pq = float(np.mean(handle["patch_sqi"][i]))
            metadata["ppg_quality_source"] = "patch_sqi_mean"
        eq = float(handle["ecg_quality"][i]) if include_ecg and "ecg_quality" in handle else 1.0
        if include_ecg and "ecg_quality" in handle:
            metadata["ecg_quality_source"] = str(handle.attrs.get("ecg_quality_definition", "stored_unspecified"))
        pq = 1.0 if pq is None else pq
        if not np.isfinite(pq) or not 0 <= pq <= 1:
            pq = 0.0
        if not np.isfinite(eq) or not 0 <= eq <= 1:
            eq = 0.0
        return result, pq, eq
