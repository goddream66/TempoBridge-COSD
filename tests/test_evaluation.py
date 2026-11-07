import csv
import json

import numpy as np
import pytest

from tempobridge.evaluation import (
    classification_scores,
    cluster_intervals,
    compare_predictions,
    evaluate_probe,
    load_features,
    select_label_groups,
    task_metrics,
    validate_splits,
)
from tempobridge.physio_targets import FIELDS


def fixture_features(tmp_path, classification=False):
    rng = np.random.default_rng(7)
    paths, rows = {}, []
    for name, count in (("train", 12), ("val", 4), ("test", 4)):
        subjects = np.repeat([f"{name}_{i}" for i in range(count)], 3)
        ids = np.asarray([f"{p}_{i}" for i, p in enumerate(subjects)])
        x = rng.normal(size=(len(ids), 4))
        y = 60 + 3 * x[:, 0] - 2 * x[:, 1]
        if classification:
            y = (x[:, 0] > 0).astype(int)
        metadata = {"split": name, "encoder_sha256": "identical_encoder", "pretraining": {
            "train_subjects": ["pretraining_patient"], "train_sources": ["a"],
            "teacher_training_subjects": [], "teacher_training_sources": []}}
        paths[name] = tmp_path / f"{name}.npz"
        np.savez(paths[name], embedding=x, sample_id=ids, subject_id=subjects, subject_known=np.ones(len(ids), bool),
                 source=np.full(len(ids), "b" if name == "test" else "a"), identity_level=np.full(len(ids), "subject_id"),
                 scale_seconds=2., metadata_json=json.dumps(metadata))
        for sid, value in zip(ids, y):
            rows.append(dict(sample_id=sid, target="target", value=value, valid=1, units="class" if classification else "bpm",
                             reference="synthetic_reference", method="fixture", duration_seconds=2.))
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return paths, labels


def test_group_fraction_and_patient_macro():
    subjects = np.asarray(["long"] * 20 + ["short"] * 2)
    take, chosen = select_label_groups(subjects, .5, 19)
    assert len(chosen) == 1 and np.all(take[subjects == chosen[0]])
    metrics = task_metrics(np.zeros(22), np.r_[np.zeros(20), [10, 10]], subjects, "regression")
    assert metrics["patient_macro_mae"] == 5 and metrics["mae"] < 1
    result = cluster_intervals(np.zeros(22), np.ones(22), subjects, "regression", 20,
                               other=np.ones(22))
    assert result["intervals"]["mae"]["lower"] == result["intervals"]["mae"]["upper"] == 0


def test_classification_metrics_handle_ties_and_missing_classes():
    y = np.asarray([0, 0, 1, 1])
    pred = np.full((4, 2), .5)
    scores = classification_scores(y, pred)
    assert scores["macro_auroc"] == scores["macro_auprc"] == .5
    perfect = classification_scores(y, np.eye(2)[y])
    assert perfect["macro_auroc"] == perfect["macro_f1"] == 1
    assert classification_scores(np.zeros(4), pred)["macro_auroc"] is None


def test_source_holdout_and_patient_leakage_guards(tmp_path):
    paths, _ = fixture_features(tmp_path)
    parts = {k: load_features(v) for k, v in paths.items()}
    validate_splits(parts, "source_held_out", "b")
    parts["train"]["metadata"]["pretraining"]["train_sources"].append("b")
    with pytest.raises(ValueError, match="source-held-out"):
        validate_splits(parts, "source_held_out", "b")
    parts["train"]["metadata"]["pretraining"]["train_sources"] = ["a"]
    parts["test"]["subject_id"] = parts["test"]["subject_id"].astype(object)
    parts["test"]["subject_id"][0] = parts["train"]["subject_id"][0]
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(parts)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_probe_pipeline_and_paired_comparison(tmp_path, task):
    paths, labels = fixture_features(tmp_path, classification=task == "classification")
    report_path = evaluate_probe(paths["train"], paths["val"], paths["test"], labels, "target",
                                 tmp_path / "evaluation", task=task, bootstrap_samples=20,
                                 protocol="source_held_out", target_source="b", steps=60)
    report = json.loads(report_path.read_text())
    assert report["audit"]["protocol"] == "source_held_out" and report["counts"]["test"]["groups"] == 4
    if task == "regression":
        assert report["test"]["mae"] < .1
    else:
        assert report["test"]["macro_auroc"] > .8
    predictions = report_path.parent / "predictions.npz"
    comparison = compare_predictions(predictions, predictions, tmp_path / "comparison.json", 20)
    assert all(row["lower"] == row["upper"] == 0 for row in json.loads(comparison.read_text())["intervals"].values())


def test_test_labels_do_not_select_probe_hyperparameters(tmp_path):
    paths, labels = fixture_features(tmp_path)
    args = (paths["train"], paths["val"], paths["test"], labels, "target")
    first = evaluate_probe(*args, tmp_path / "first", bootstrap_samples=20)
    with labels.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["sample_id"].startswith("test_"):
            row["value"] = 99999
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    second = evaluate_probe(*args, tmp_path / "second", bootstrap_samples=20)
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    assert a["selected_regularization"] == b["selected_regularization"]
    assert a["candidates"] == b["candidates"]
    assert a["test"]["mae"] != b["test"]["mae"]
