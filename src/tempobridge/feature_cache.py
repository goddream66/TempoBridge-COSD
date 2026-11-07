"""Content-bound, memory-mapped frozen encoder features for calibration.

The cache contains training patients only. It is portable and never stores raw
waveforms. A cache is complete only once its manifest has been atomically written.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import DataConfig, ModelConfig
from .data import PairedDataset, manifest_digest, read_manifest, subject_ids
from .model import TempoBridge
from .sampling import calibration_indices
from .teacher import FrozenTeacher
from .utils import device_for, file_sha256, load_checkpoint, seed_all, write_json


def cache_features(config, manifest, student_path, teacher_path, output, teacher_kind="native"):
    config.validate()
    seed_all(config.train.seed, config.train.threads)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Feature cache exists; choose a new directory or explicitly reuse it")
    records = read_manifest(manifest, data_config=config.data)
    digest = manifest_digest(records)
    checkpoint = load_checkpoint(student_path)
    if checkpoint.get("modality") != "ppg" or checkpoint.get("global_step", 0) < 1:
        raise ValueError("Calibration requires a trained PPG warm-up checkpoint")
    if checkpoint["manifest_sha256"] != digest:
        raise ValueError("Student warm-up and calibration manifest differ")
    if asdict(ModelConfig.from_dict(checkpoint["model_config"])) != config.to_dict()["model"]:
        raise ValueError("Student architecture differs from config")
    if DataConfig.from_dict(checkpoint["config"]["data"]).input_protocol() != config.data.input_protocol():
        raise ValueError("Student warm-up input protocol differs from config; warm up again")
    device = device_for(config.train.device)
    model = TempoBridge(ModelConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    teacher = FrozenTeacher(teacher_path, teacher_kind).to(device)
    teacher.verify_data(records, config.data.ecg_fs)
    dataset = PairedDataset(records, config.data, augment=False)
    indices = calibration_indices(dataset.records, config.calibration.max_windows_per_subject,
                                  config.train.seed)
    if not indices:
        dataset.close()
        raise ValueError("No identified training patients for calibration")
    output.mkdir(parents=True)
    arrays, files, metadata = [], {}, []
    for s in range(len(config.data.scales_seconds)):
        pair = []
        for name, dim in (("student", model.config.dim), ("teacher", teacher.target_dim)):
            filename = f"{name}_{s}.npy"
            pair.append(np.lib.format.open_memmap(output / filename, mode="w+", dtype="float32",
                                                  shape=(len(indices), dim)))
            files[filename] = None
        arrays.append(pair)
    loader = DataLoader(Subset(dataset, indices), batch_size=config.train.batch_size, shuffle=False,
                        num_workers=config.data.workers)
    cursor = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                valid = (batch["subject_known"] & batch["ecg_valid"] &
                         (batch["ecg_quality"] >= config.calibration.min_ecg_quality))
                selected = valid.nonzero().flatten().tolist()
                if not selected:
                    continue
                count = len(selected)
                for s, (xout, yout) in enumerate(arrays):
                    x = model.transfer_features(batch["ppg"][s][valid].to(device)).cpu().numpy()
                    y = teacher(batch["ecg"][s][valid].to(device)).cpu().numpy()
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise ValueError("Nonfinite cache features")
                    xout[cursor:cursor + count], yout[cursor:cursor + count] = x, y
                for i in selected:
                    row = {key: batch[key][i] for key in
                           ("sample_id", "subject_id", "recording_id", "source", "case_id",
                            "ppg_quality_source", "ecg_quality_source")}
                    row.update(recording_known=bool(batch["recording_known"][i]),
                               window_start_s=float(batch["window_start_s"][i]),
                               ppg_quality=float(batch["ppg_quality"][i]),
                               ecg_quality=float(batch["ecg_quality"][i]),
                               crop_offsets_s=[float(offset[i]) for offset in batch["crop_offsets_s"]])
                    metadata.append(row)
                cursor += count
        if not cursor:
            raise ValueError("No reliable paired ECG for calibration")
        for pair in arrays:
            for array in pair:
                array.flush()
    finally:
        dataset.close()
        # Explicitly release mmap handles, including local loop references, on Windows.
        for pair in arrays:
            for array in pair:
                array._mmap.close()
    for filename in files:
        files[filename] = file_sha256(output / filename)
    write_json(output / "rows.json", metadata)
    files["rows.json"] = file_sha256(output / "rows.json")
    artifact = {
        "schema_version": 1, "kind": "tempobridge_calibration_features", "rows": cursor,
        "capacity": len(indices), "files": files,
        "student_sha256": file_sha256(student_path), "teacher_sha256": teacher.fingerprint,
        "teacher_kind": teacher_kind, "manifest_sha256": digest,
        "train_subjects": sorted(subject_ids(records, "train")),
        "feature_spec": model.transfer_feature_spec(), "teacher_feature_spec": teacher.feature_spec,
        "target_dim": teacher.target_dim, "input_protocol": config.data.input_protocol(),
        "model_config": model.model_config(),
        "selection": {"seed": config.train.seed, "candidates": len(indices), "reliable_pairs": cursor,
                      "unknown_ids_excluded": True,
                      "max_windows_per_subject": config.calibration.max_windows_per_subject,
                      "min_ecg_quality": config.calibration.min_ecg_quality},
        "scope": "fixed encoders; train split only; deterministic unaugmented input",
    }
    write_json(output / "manifest.json", artifact)
    return output


class FeatureCache:
    def __init__(self, path):
        self.path = Path(path)
        self.artifact = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        a = self.artifact
        if a.get("schema_version") != 1 or a.get("kind") != "tempobridge_calibration_features":
            raise ValueError("Unsupported feature cache")
        self.fingerprint = file_sha256(self.path / "manifest.json")
        scales = a["input_protocol"]["scales_seconds"]
        expected_files = {f"{name}_{s}.npy" for s in range(len(scales)) for name in ("student", "teacher")} | {"rows.json"}
        if set(a["files"]) != expected_files:
            raise ValueError("Invalid feature cache file list")
        for name, digest in a["files"].items():
            if file_sha256(self.path / name) != digest:
                raise ValueError(f"Feature cache content changed: {name}")
        self.rows = json.loads((self.path / "rows.json").read_text(encoding="utf-8"))
        if len(self.rows) != a["rows"] or not 0 < a["rows"] <= a["capacity"]:
            raise ValueError("Invalid feature cache row count")
        if len({r["sample_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate feature cache sample ID")
        if not {r["subject_id"] for r in self.rows}.issubset(a["train_subjects"]):
            raise ValueError("Feature cache contains non-training patients")

    def validate(self, config, manifest_hash, teacher_hash, student_hash, teacher_kind):
        expected = {"manifest_sha256": manifest_hash, "teacher_sha256": teacher_hash,
                    "student_sha256": student_hash, "teacher_kind": teacher_kind,
                    "input_protocol": config.data.input_protocol(), "model_config": config.to_dict()["model"]}
        for key, value in expected.items():
            if self.artifact.get(key) != value:
                raise ValueError(f"Stale/incompatible feature cache: {key}")
        for key, value in {"seed": config.train.seed,
                           "max_windows_per_subject": config.calibration.max_windows_per_subject,
                           "min_ecg_quality": config.calibration.min_ecg_quality}.items():
            if self.artifact["selection"].get(key) != value:
                raise ValueError(f"Stale/incompatible feature cache selection: {key}")

    def column(self, key):
        return np.asarray([row[key] for row in self.rows])

    def arrays(self, scale_index):
        if not 0 <= scale_index < len(self.artifact["input_protocol"]["scales_seconds"]):
            raise ValueError("Invalid cache scale")
        result = []
        for name, dim in (("student", self.artifact["feature_spec"]["dim"]),
                          ("teacher", self.artifact["target_dim"])):
            a = np.load(self.path / f"{name}_{scale_index}.npy", mmap_mode="r", allow_pickle=False)
            if a.shape != (self.artifact["capacity"], dim) or a.dtype != np.float32:
                raise ValueError("Invalid cached feature shape/dtype")
            result.append(a[:self.artifact["rows"]])
        return tuple(result)
