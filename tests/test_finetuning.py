import csv
import json

import numpy as np
import pytest
import torch

from tempobridge.data import make_synthetic, read_manifest
from tempobridge.finetuning import fine_tune
from tempobridge.inference import export_encoder
from tempobridge.physio_targets import FIELDS
from tempobridge.training import train
from tempobridge.utils import load_checkpoint


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_finetuning_and_random_init(tmp_path, tiny_config, task):
    cfg = tiny_config
    cfg.data.window_mode, cfg.data.scales_seconds = "full", [2.]
    cfg.train.epochs = 2
    manifest = make_synthetic(tmp_path / "data", subjects=8, windows=3, seconds=2)
    checkpoint = train(cfg, manifest, tmp_path / "pretraining", stop_after=1)
    encoder = export_encoder(checkpoint, tmp_path / "encoder.pt")
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, record in enumerate(read_manifest(manifest)):
            writer.writerow(dict(sample_id=record.sample_id, target="target", value=i % 2 if task == "classification" else 60 + i,
                                 valid=1, units="class" if task == "classification" else "bpm",
                                 reference="independent_fixture", method="synthetic", duration_seconds=2.))
    for name, initial in (("pretrained", encoder), ("random", None)):
        result = fine_tune(cfg, manifest, labels, "target", tmp_path / name, initial, task=task,
                           bootstrap_samples=20, label_fraction=.5)
        report = json.loads(result.read_text())
        assert report["initialization"] == name and len(report["selected_training_groups"]) == 2
        assert report["selected_epoch"] in {1, 2}
        best = load_checkpoint(result.parent / "best.pt")
        assert best["kind"] == "ppg_downstream_model"
        assert all(torch.isfinite(v).all() for v in best["head"].values())
        with np.load(result.parent / "predictions.npz", allow_pickle=False) as predictions:
            assert len(predictions["sample_id"]) == 6 and np.isfinite(predictions["prediction"]).all()
    assert report["audit"]["unknown_pretraining_provenance"] == []
