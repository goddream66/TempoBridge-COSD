"""Small CPU CLI matrix, including rank-matched controls and a diagnostic teacher."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError("Choose a new smoke directory")
    root.mkdir(parents=True)
    config = {
        "model": {"encoder": "selective", "dim": 16, "depth": 3, "protected_depth": 1,
                  "heads": 2, "patch_size": 50, "scalar_dim": 8, "ff_hidden_dim": 32,
                  "selective_width": 8, "selective_kernel_sizes": [3, 7, 11]},
        "data": {"ppg_fs": 50, "ecg_fs": 50, "window_mode": "full", "scales_seconds": [30.],
                 "quality_edges": [], "samples_per_epoch": 42},
        "train": {"device": "cpu", "epochs": 1, "batch_size": 6, "threads": 2,
                  "gradient_audit_interval": 4},
        "calibration": {"probe": "ridge", "folds": 2, "min_subjects_per_bin": 1},
        "cosd": {"rank": 4, "selection": "all", "bootstrap_samples": 20},
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    def run(*arguments):
        subprocess.run([sys.executable, "-m", "tempobridge", *map(str, arguments)], check=True)

    run("synthetic", "--output", root / "data", "--subjects", 18, "--windows", 3, "--seconds", 30, "--fs", 50)
    run("plan-experiments", "--config", path, "--manifest", root / "data/manifest.csv", "--output", root / "matrix",
        "--seeds", 23, "--families", "core", "teacher")
    selected = ["ssl", "pw_kd", "cosd", "cosd_pw", "random_rank", "pca_rank", "teacher_random_diagnostic"]
    plan = root / "matrix/plan.json"
    run("run-experiments", "--plan", plan, "--only", *selected)
    assert not (root / "matrix/artifacts").exists()
    run("run-experiments", "--plan", plan, "--execute", "--only", *selected)
    # A second invocation must verify and reuse completed jobs, not train again.
    run("run-experiments", "--plan", plan, "--execute", "--only", *selected)
    report = json.loads((root / "matrix/summary.json").read_text(encoding="utf-8"))
    runs = {r["variant"]: r for r in report["executed_variants"]}
    assert len(runs) == len(selected)
    assert {r["optimizer_steps"] for r in runs.values()} == {7}
    assert runs["cosd"]["actual_ranks"] == runs["random_rank"]["actual_ranks"] == runs["pca_rank"]["actual_ranks"] == [4]
    assert runs["teacher_random_diagnostic"]["diagnostic_only"]
    summary = {"completed": True, "scope": "CPU synthetic engineering check; all selection forces rank for route coverage",
               "tested_variants": selected, "full_scientific_matrix_completed": False, "matrix_report": "matrix/summary.json"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
