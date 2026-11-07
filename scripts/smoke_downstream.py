"""CPU synthetic CLI verification of references, auxiliary training and evaluation.

All inputs are generated nonclinical fixtures. No efficacy assertion is made.
Run from any working directory after installing this package.
"""

import argparse
import csv
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
    cfg = {"model": {"encoder": "selective", "dim": 16, "depth": 3, "protected_depth": 1,
            "heads": 2, "patch_size": 10, "scalar_dim": 8, "ff_hidden_dim": 32,
            "selective_width": 8, "selective_kernel_sizes": [3, 7, 11]},
        "data": {"ppg_fs": 50, "ecg_fs": 50, "scales_seconds": [30.], "window_mode": "full"},
        "train": {"device": "cpu", "threads": 2, "epochs": 1, "batch_size": 6, "seed": 42,
                  "checkpoint_selection": "last", "transfer_method": "none", "transfer_mode": "none"}}
    config = out / "config.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")

    def run(*arguments):
        command = [sys.executable, "-m", "tempobridge", *map(str, arguments)]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    run("synthetic", "--output", out / "data", "--subjects", 12, "--windows", 3, "--seconds", 30, "--fs", 50)
    manifest = out / "data/manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    references = []
    reference_dir = out / "reference_waveforms"
    reference_dir.mkdir()
    for i, sample in enumerate(samples):
        fs, duration, rate = 250, 30, .9 + (i % 7) * .1
        t = np.arange(fs * duration) / fs
        ecg = sum(np.exp(-((t - p) / .015) ** 2) for p in np.arange(.5, duration, 1 / rate))
        path = reference_dir / (sample["sample_id"] + ".npz")
        np.savez(path, ecg=ecg, ecg_fs=fs, abp=80 + 20 * np.sin(2 * np.pi * rate * t),
                 abp_fs=fs, abp_units="mmHg", resp=np.sin(2 * np.pi * .2 * t), resp_fs=fs)
        references.append({"sample_id": sample["sample_id"], "path": path.relative_to(out).as_posix(),
                           "duration_seconds": duration})
    reference_manifest = out / "references.jsonl"
    reference_manifest.write_text("\n".join(json.dumps(r) for r in references), encoding="utf-8")
    labels = out / "targets.csv"
    run("reference-targets", "--references", reference_manifest, "--output", labels)
    # External synthetic classification label, distinct from heuristic physiology.
    with labels.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for i, sample in enumerate(samples):
            writer.writerow(dict(sample_id=sample["sample_id"], target="synthetic_class", value=i % 2,
                                 valid=1, units="class", reference="nonclinical_fixture", method="fixture_index",
                                 duration_seconds=30))
    run("pretrain", "--config", config, "--manifest", manifest, "--modality", "ppg", "--output", out / "warmup")
    cfg["train"].update(transfer_method="physio", transfer_mode="all", physio_targets=["hr_bpm", "map_mmhg"])
    physio_config = out / "physio.json"
    physio_config.write_text(json.dumps(cfg), encoding="utf-8")
    run("pretrain", "--config", physio_config, "--manifest", manifest, "--modality", "ppg",
        "--student", out / "warmup/last.pt", "--reference-targets", labels, "--output", out / "physio")
    encoder = out / "encoder.pt"
    run("export", "--checkpoint", out / "physio/last.pt", "--output", encoder)
    for split in ("train", "val", "test"):
        run("features", "--checkpoint", encoder, "--manifest", manifest, "--split", split,
            "--output", out / f"{split}.npz", "--device", "cpu")
    reports = {}
    for task, target in (("regression", "hr_bpm"), ("classification", "synthetic_class")):
        run("evaluate-probe", "--train-features", out / "train.npz", "--val-features", out / "val.npz",
            "--test-features", out / "test.npz", "--labels", labels, "--target", target, "--task", task,
            "--bootstrap-samples", 20, "--steps", 30, "--output", out / f"probe_{task}")
        reports[f"probe_{task}"] = json.loads((out / f"probe_{task}/report.json").read_text())["test"]
    pred = out / "probe_regression/predictions.npz"
    run("compare-predictions", "--first", pred, "--second", pred, "--bootstrap-samples", 20,
        "--output", out / "paired.json")
    for name, extra in (("pretrained", ["--encoder", encoder]), ("random", [])):
        run("finetune", "--config", config, "--manifest", manifest, "--labels", labels, "--target", "hr_bpm",
            "--bootstrap-samples", 20, "--output", out / f"finetune_{name}", *extra)
        reports[f"finetune_{name}"] = json.loads((out / f"finetune_{name}/report.json").read_text())["test"]
    summary = {"completed": True, "kind": "CPU synthetic integration only; no efficacy evidence", "reports": reports}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
