"""Run the actual CLI end-to-end on explicitly synthetic data."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tempobridge.config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/smoke")
    parser.add_argument("--config", default="configs/smoke.json")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a fresh smoke output directory")
    out.mkdir(parents=True)

    def run(*arguments):
        command = [sys.executable, "-m", "tempobridge", *map(str, arguments)]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    manifest = out / "data" / "manifest.csv"
    common = ["--config", args.config, "--manifest", manifest]
    config = ExperimentConfig.load(args.config)
    seconds = max(config.data.scales_seconds)
    # Match the configured full-window contract, including 30-second validation.
    run("synthetic", "--output", out / "data", "--seconds", seconds,
        "--fs", config.data.ppg_fs)
    run("audit-data", *common)
    for modality in ("ppg", "ecg"):
        run("pretrain", *common, "--modality", modality, "--output", out / modality)
    student, teacher, artifact = out / "ppg/last.pt", out / "ecg/last.pt", out / "calibration.json"
    run("calibrate", *common, "--student", student, "--teacher", teacher, "--output", artifact)
    run("distill", *common, "--student", student, "--teacher", teacher,
        "--calibration", artifact, "--output", out / "distill")
    run("export", "--checkpoint", out / "distill/last.pt", "--output", out / "encoder.pt")
    run("features", "--checkpoint", out / "encoder.pt", "--manifest", manifest,
        "--output", out / "features.npz")
    with np.load(out / "features.npz", allow_pickle=False) as features:
        assert np.isfinite(features["embedding"]).all()
        shape = list(features["embedding"].shape)
    calibration = json.loads(artifact.read_text())
    report = {"kind": "synthetic integration, not clinical evidence", "feature_shape": shape,
              "weights": [s["weights"] for s in calibration["scales"]],
              "completed": True}
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
