"""CPU-only CLI verification; synthetic waveforms are not efficacy evidence."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Use a new smoke output directory")
    out.mkdir(parents=True)
    cfg = {
        "model": {"encoder": "selective", "dim": 16, "depth": 3, "protected_depth": 1,
                  "heads": 2, "patch_size": 10, "scalar_dim": 8, "ff_hidden_dim": 32,
                  "selective_width": 8, "selective_kernel_sizes": [3, 7, 11], "dropout": .1},
        "data": {"ppg_fs": 50, "ecg_fs": 50, "scales_seconds": [30.], "window_mode": "full"},
        "train": {"device": "cpu", "precision": "fp32", "threads": 2, "epochs": 1,
                  "batch_size": 8, "seed": 42, "checkpoint_selection": "last", "gradient_audit_interval": 4},
        "calibration": {"probe": "ridge", "folds": 2, "min_subjects_per_bin": 1,
                        "max_windows_per_subject": 4},
        "cosd": {"rank": 4, "bootstrap_samples": 20},
    }
    base = out / "config.json"
    base.write_text(json.dumps(cfg), encoding="utf-8")

    def run(*arguments):
        command = [sys.executable, "-m", "tempobridge", *map(str, arguments)]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    manifest = out / "data" / "manifest.csv"
    run("synthetic", "--output", out / "data", "--subjects", 24, "--windows", 4, "--seconds", 30, "--fs", 50)
    for modality in ("ppg", "ecg"):
        run("pretrain", "--config", base, "--manifest", manifest, "--modality", modality, "--output", out / modality)
    student, teacher, cache = out / "ppg/last.pt", out / "ecg/last.pt", out / "cache"
    run("cache-features", "--config", base, "--manifest", manifest, "--student", student,
        "--teacher", teacher, "--output", cache)
    reports = {}
    for name, method, mode in (("ssl", "none", "none"), ("vanilla", "vanilla", "all"),
                                ("pw_kd", "pw_kd", "calibrated"), ("cosd", "cosd", "quality"),
                                ("cosd_all", "cosd", "quality")):
        setting = copy.deepcopy(cfg)
        setting["train"].update(transfer_method=method, transfer_mode=mode)
        if name == "cosd_all":
            setting["cosd"]["selection"] = "all"
        config = out / f"{name}.json"
        config.write_text(json.dumps(setting), encoding="utf-8")
        common = ["--config", config, "--manifest", manifest, "--student", student]
        artifact = out / f"{name}_calibration.json"
        if method == "none":
            run("pretrain", *common, "--modality", "ppg", "--output", out / name)
        else:
            run("calibrate-cache", "--config", config, "--cache", cache, "--output", artifact)
            run("distill", *common, "--teacher", teacher, "--calibration", artifact, "--output", out / name)
        exported = out / f"{name}_encoder.pt"
        run("export", "--checkpoint", out / name / "last.pt", "--output", exported)
        features = out / f"{name}_features.npz"
        run("features", "--checkpoint", exported, "--manifest", manifest, "--output", features, "--device", "cpu")
        with np.load(features, allow_pickle=False) as data:
            assert data["embedding"].shape == (8, 32)
            assert np.isfinite(data["embedding"]).all()
        metrics = json.loads((out / name / "metrics.jsonl").read_text().splitlines()[-1])
        reports[name] = {key: metrics[key] for key in
                         ("target_rank", "mean_transfer_weight", "active_transfer_fraction", "kd_gradient_norm")}
    summary = {"completed": True, "kind": "CPU synthetic integration only", "methods": reports}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
