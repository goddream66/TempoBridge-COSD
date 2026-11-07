import csv
import json

import pytest

from tempobridge.cli import main
from tempobridge.config import ExperimentConfig
from tempobridge.data import make_synthetic, read_manifest
from tempobridge.experiments import ExperimentRunner, plan_experiments, run_experiments
from tempobridge.h5data import build_h5_manifest
from tempobridge.manifest_tools import filter_sources
from tempobridge.physio_targets import FIELDS
from tempobridge.utils import write_json


def configuration(tmp_path):
    cfg = ExperimentConfig.from_dict({
        "model": {"encoder": "local", "dim": 16, "depth": 3, "protected_depth": 1,
                  "heads": 2, "patch_size": 50, "stem_width": 4, "ff_multiplier": 2},
        "data": {"window_mode": "full", "scales_seconds": [30.], "ppg_fs": 50, "ecg_fs": 50,
                 "quality_edges": [], "samples_per_epoch": 42},
        "train": {"epochs": 1, "batch_size": 6, "device": "cpu", "threads": 2},
        "calibration": {"folds": 2, "probe": "ridge", "min_subjects_per_bin": 1,
                        "mlp_steps": 2, "mlp_hidden": 8},
        "cosd": {"rank": 4, "selection": "all", "bootstrap_samples": 20},
    })
    path = tmp_path / "config.json"
    write_json(path, cfg.to_dict())
    return path


def label_fixture(manifest, output):
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, record in enumerate(read_manifest(manifest)):
            writer.writerow(dict(sample_id=record.sample_id, target="hr_bpm", value=60 + i % 9,
                valid=1, units="bpm", reference="independent_synthetic_fixture", method="fixture", duration_seconds=30.))
    return output


def test_all_families_plan_without_training_and_preserve_source_splits(tmp_path, capsys):
    config = configuration(tmp_path)
    manifest = make_synthetic(tmp_path / "data", subjects=18, windows=3, seconds=30, fs=50)
    records = read_manifest(manifest)
    view = filter_sources(manifest, tmp_path / "view.csv", ["default"])
    assert [(r.sample_id, r.split, r.subject_id) for r in records] == [
        (r.sample_id, r.split, r.subject_id) for r in read_manifest(view)]
    plan = plan_experiments(config, manifest, tmp_path / "plan", seeds=[7], families=["all"])
    payload = json.loads(plan.read_text())
    assert len(payload["variants"]) > 40
    assert not (plan.parent / "artifacts").exists()
    main(["run-experiments", "--plan", str(plan), "--only", "random_rank"])
    result = json.loads(capsys.readouterr().out)
    assert result["variants"] == ["cosd", "random_rank"] and result["execute"] is False
    assert payload["budget"]["continuation_optimizer_steps"] == 7


def test_real_runner_matches_rank_budget_and_verifies_completed_outputs(tmp_path):
    config = configuration(tmp_path)
    manifest = make_synthetic(tmp_path / "data", subjects=18, windows=3, seconds=30, fs=50)
    labels = label_fixture(manifest, tmp_path / "labels.csv")
    plan = plan_experiments(config, manifest, tmp_path / "plan", seeds=[7], labels=labels,
        targets=[{"name": "hr_bpm", "task": "regression"}], label_fractions=[1.],
        include_finetune=False, bootstrap_samples=20)
    chosen = ["ssl", "pw_kd", "cosd", "cosd_pw", "random_rank", "pca_rank"]
    report_path = run_experiments(plan, execute=True, only=chosen)
    report = json.loads(report_path.read_text())
    assert report["complete"] is False
    runs = {r["variant"]: r for r in report["executed_variants"]}
    assert len(runs) == 6
    assert runs["cosd"]["actual_ranks"] == runs["random_rank"]["actual_ranks"] == runs["pca_rank"]["actual_ranks"] == [4]
    assert {r["optimizer_steps"] for r in runs.values()} == {7}
    assert {r["windows"] for r in runs.values()} == {42}
    state = json.loads((plan.parent / "state.json").read_text())
    assert len([k for k in state["jobs"] if "/warmup_" in k]) == 1
    assert all(r["evaluations"] for r in runs.values())
    assert len(report["paired_downstream"]) == 5 and report["seed_aggregates"]
    assert all(r["sample_std"] is None for r in report["seed_aggregates"])
    run_experiments(plan, execute=True, only=chosen)
    prediction = next((plan.parent / "artifacts").rglob("predictions.npz"))
    prediction.write_bytes(prediction.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="output changed"):
        run_experiments(plan, execute=True, only=chosen)


def test_failed_stage_retries_without_deleting_partial_outputs(tmp_path):
    config = configuration(tmp_path)
    manifest = make_synthetic(tmp_path / "data", subjects=18, windows=3, seconds=30, fs=50)
    runner = ExperimentRunner(plan_experiments(config, manifest, tmp_path / "plan", seeds=[7]))
    output = runner.output("partial.txt")

    def fail():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        runner.fresh_job("test", {"input": 1}, [output], fail)
    with pytest.raises(ValueError, match="input changed"):
        runner.fresh_job("test", {"input": 2}, [output], lambda: output)

    def succeed():
        output.write_text("complete")
        return output
    runner.fresh_job("test", {"input": 1}, [output], succeed)
    assert output.read_text() == "complete"
    assert next(runner.output("interrupted").rglob("partial.txt")).read_text() == "partial"


def test_source_heldout_runner_and_finetuning_keep_separate_corpora(tmp_path):
    config = configuration(tmp_path)
    cfg = ExperimentConfig.load(config)
    cfg.data.samples_per_epoch = 60
    write_json(config, cfg.to_dict())
    manifest = make_synthetic(tmp_path / "data", subjects=24, windows=3, seconds=30, fs=50)
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for i, row in enumerate(rows):
        row["source"] = "a" if (i // 3) % 2 == 0 else "b"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, "source"])
        writer.writeheader()
        writer.writerows(rows)
    labels = label_fixture(manifest, tmp_path / "labels.csv")
    plan = plan_experiments(config, manifest, tmp_path / "plan", seeds=[7], families=["data"],
        labels=labels, targets=[{"name": "hr_bpm", "task": "regression"}], label_fractions=[1.], bootstrap_samples=20)
    report = json.loads(run_experiments(plan, execute=True, only=["source_only_a"]).read_text())
    summaries = report["executed_variants"]
    assert len(summaries) == 1 and len(summaries[0]["evaluations"]) == 6
    for path in summaries[0]["evaluations"]:
        result = json.loads((plan.parent / path).read_text())
        assert result["audit"]["unknown_pretraining_provenance"] == []
        if "/evaluation/b/" in path:
            assert result["audit"]["protocol"] == "source_held_out"
            assert list(result["by_source"]) == ["b"]


def test_hdf5_source_view_preserves_groups_and_relocates_roots(tmp_path):
    import h5py
    import numpy as np
    for source in ("a", "b"):
        folder = tmp_path / source
        folder.mkdir()
        with h5py.File(folder / "part.h5", "w") as h:
            h["signals"] = np.random.default_rng(2 if source == "a" else 3).normal(size=(40, 100)).astype("f4")
            h["subject_id"] = np.arange(40)
            h.attrs["target_fs"] = 50
    spec, manifest = tmp_path / "sources.json", tmp_path / "index.json"
    write_json(spec, {"sources": {s: {"root": s, "fs": 50} for s in ("a", "b")}})
    build_h5_manifest(spec, manifest)
    output = filter_sources(manifest, tmp_path / "nested" / "a.json", ["a"])
    original, selected = read_manifest(manifest), read_manifest(output)
    assert [(r.sample_id, r.subject_id, r.split) for r in original if r.source == "a"] == [
        (r.sample_id, r.subject_id, r.split) for r in selected]


def test_teacher_matrix_routes_random_control_without_claiming_training(tmp_path):
    config = configuration(tmp_path)
    manifest = make_synthetic(tmp_path / "data", subjects=18, windows=3, seconds=30, fs=50)
    plan = plan_experiments(config, manifest, tmp_path / "plan", seeds=[7], families=["teacher"])
    report = json.loads(run_experiments(plan, execute=True,
        only=["teacher_random_diagnostic", "teacher_pool_combined", "teacher_l2"]).read_text())
    runs = {r["variant"]: r for r in report["executed_variants"]}
    assert runs["teacher_random_diagnostic"]["diagnostic_only"] is True
    assert runs["teacher_pool_combined"]["diagnostic_only"] is False
