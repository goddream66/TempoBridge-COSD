import json

import numpy as np
import pytest
import torch

from tempobridge.calibration import calibrate
from tempobridge.data import make_synthetic
from tempobridge.inference import export_encoder, extract_features, load_encoder
from tempobridge.model import TempoBridge
from tempobridge.teacher import FrozenTeacher
from tempobridge.training import train
from tempobridge.utils import load_checkpoint, save_checkpoint


def test_complete_pipeline_freeze_export_and_features(tmp_path, tiny_config):
    manifest = make_synthetic(tmp_path / "data", subjects=10, windows=3, seconds=2)
    ppg = train(tiny_config, manifest, tmp_path / "ppg")
    ecg = train(tiny_config, manifest, tmp_path / "ecg", modality="ecg")
    teacher = FrozenTeacher(ecg)
    teacher.train(True)
    assert not teacher.training
    assert not teacher.encoder.training
    assert all(not p.requires_grad for p in teacher.parameters())
    artifact = calibrate(tiny_config, manifest, ppg, ecg, tmp_path / "calibration.json")
    assert len(artifact["scales"]) == 2
    result = train(tiny_config, manifest, tmp_path / "distill", student_path=ppg,
                   teacher_path=ecg, calibration_path=tmp_path / "calibration.json")
    trained = load_checkpoint(result)
    assert trained["global_step"] > 0
    exported = export_encoder(result, tmp_path / "encoder.pt")
    encoder = load_encoder(exported)
    original = TempoBridge(tiny_config.model).eval()
    original.load_state_dict(trained["model"])
    x = torch.randn(3, 1, 100)
    torch.testing.assert_close(encoder(x), original(x), rtol=0, atol=0)
    payload = load_checkpoint(exported)
    assert not any(k in payload for k in ("heads", "optimizer", "teacher", "calibration"))
    assert not any(k.startswith(("reconstruction.", "morph_reconstruction.")) for k in payload["model"])
    feature_file = extract_features(exported, manifest, tmp_path / "features.npz")
    with np.load(feature_file, allow_pickle=False) as data:
        assert data["embedding"].shape == (6, encoder.feature_dim)
        assert np.isfinite(data["embedding"]).all()
    for line in (tmp_path / "distill" / "metrics.jsonl").read_text().splitlines():
        assert np.isfinite(json.loads(line)["loss"])


def test_epoch_resume_is_identical_to_uninterrupted(tmp_path, tiny_config):
    tiny_config.model.dropout = 0.1
    manifest = make_synthetic(tmp_path / "data", subjects=8, windows=2, seconds=2)
    full = train(tiny_config, manifest, tmp_path / "full")
    partial = train(tiny_config, manifest, tmp_path / "resume", stop_after=1)
    resumed = train(tiny_config, manifest, tmp_path / "resume", resume=partial)
    a, b = load_checkpoint(full), load_checkpoint(resumed)
    assert a["global_step"] == b["global_step"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)


def test_random_or_ppg_teacher_is_rejected(tmp_path, tiny_config):
    path = tmp_path / "random.pt"
    save_checkpoint(path, {"modality": "ecg", "global_step": 0})
    with pytest.raises(ValueError, match="trained ECG"):
        FrozenTeacher(path)


def test_teacher_cannot_have_seen_heldout_subject(tmp_path, tiny_config):
    from tempobridge.data import read_manifest
    manifest = make_synthetic(tmp_path / "data", subjects=8, windows=2, seconds=2)
    checkpoint = train(tiny_config, manifest, tmp_path / "ecg", modality="ecg")
    state = load_checkpoint(checkpoint)
    records = read_manifest(manifest)
    state["train_subjects"].append(next(r.subject_id for r in records if r.split == "test"))
    save_checkpoint(tmp_path / "leaky.pt", state)
    teacher = FrozenTeacher(tmp_path / "leaky.pt")
    with pytest.raises(ValueError, match="held-out"):
        teacher.verify_data(records, tiny_config.data.ecg_fs)


def test_external_teacher_metadata_and_freeze(tmp_path):
    class External(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x.mean(-1), x.std(-1, correction=0)], dim=1)
    path = tmp_path / "external.ts"
    torch.jit.trace(External(), torch.randn(2, 1, 100)).save(str(path))
    metadata = {"input_fs": 100, "target_dim": 2, "training_subjects": None,
                "provenance": {"purpose": "unit test only, not a pretrained model"}}
    (tmp_path / "external.ts.json").write_text(json.dumps(metadata))
    teacher = FrozenTeacher(path, "torchscript")
    assert teacher(torch.randn(3, 1, 200)).shape == (3, 2)
    assert not teacher.training


def test_feature_cli_does_not_access_ecg(tmp_path, tiny_config):
    import csv
    manifest = make_synthetic(tmp_path / "data", subjects=8, windows=2, seconds=2)
    checkpoint = train(tiny_config, manifest, tmp_path / "ppg")
    exported = export_encoder(checkpoint, tmp_path / "encoder.pt")
    # Test-only manifest and deliberately malformed ECG must not affect PPG inference.
    with manifest.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "test"]
    for row in rows:
        path = manifest.parent / row["path"]
        with np.load(path) as data:
            ppg = data["ppg"]
        np.savez(path, ppg=ppg, ppg_fs=100, ecg=np.full(3, np.nan))
    only_test = manifest.parent / "test_only.csv"
    with only_test.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = extract_features(exported, only_test, tmp_path / "features.npz")
    with np.load(output) as data:
        assert np.isfinite(data["embedding"]).all()
