import json

import numpy as np
import torch

from tempobridge.calibration import CalibrationWeights, calibrate
from tempobridge.data import make_synthetic
from tempobridge.inference import export_encoder, load_encoder
from tempobridge.model import TempoBridge, TransferHeads
from tempobridge.training import batch_loss, train, transfer_loss
from tempobridge.utils import load_checkpoint


def test_identity_projection_matches_full_kd_loss_and_gradients(tiny_config):
    dim = 4
    full = {"quality_edges": [], "scales": [{"weights": [1.],
             "target_mean": [1.] * dim, "target_scale": [2.] * dim}]}
    subspace = {"version": 1, "rank": dim, "teacher_mean": [1.] * dim,
                "teacher_scale": [2.] * dim, "projection": np.eye(dim).tolist(),
                "direction_weights": [1.] * dim}
    projected = {"quality_edges": [], "scales": [{**full["scales"][0], "subspace": subspace}]}
    model = TempoBridge(tiny_config.model).eval()
    heads = TransferHeads(tiny_config.model.dim, dim, [1])

    class Teacher(torch.nn.Module):
        def forward(self, x):
            return x[:, 0, :dim]

    ppg, ecg = torch.randn(3, 1, 100), torch.randn(3, 1, 100)
    valid, quality = torch.ones(3, dtype=torch.bool), torch.ones(3)
    args = (ppg, ecg, valid, quality, quality, 0, "all", False)
    a, _ = transfer_loss(model, heads, Teacher(), CalibrationWeights(full), *args)
    ga = torch.autograd.grad(a, model.parameters(), allow_unused=True)
    b, _ = transfer_loss(model, heads, Teacher(), CalibrationWeights(projected), *args)
    gb = torch.autograd.grad(b, model.parameters(), allow_unused=True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for x, y in zip(ga, gb):
        if x is not None:
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_cosd_training_resume_and_export(tmp_path, tiny_config):
    cfg = tiny_config
    cfg.model.dropout = 0.1
    manifest = make_synthetic(tmp_path / "data", subjects=18, windows=3, seconds=2)
    ppg = train(cfg, manifest, tmp_path / "ppg", stop_after=1)
    ecg = train(cfg, manifest, tmp_path / "ecg", modality="ecg", stop_after=1)
    cfg.train.transfer_method = "cosd"
    cfg.train.transfer_mode = "quality"
    cfg.train.checkpoint_selection = "last"
    cfg.train.gradient_audit_interval = 2
    cfg.cosd.rank, cfg.cosd.selection, cfg.cosd.bootstrap_samples = 4, "all", 20
    artifact = tmp_path / "cosd.json"
    result = calibrate(cfg, manifest, ppg, ecg, artifact)
    assert [s["subspace"]["rank"] for s in result["scales"]] == [4, 4]
    kwargs = dict(student_path=ppg, teacher_path=ecg, calibration_path=artifact)
    full = train(cfg, manifest, tmp_path / "full", **kwargs)
    partial = train(cfg, manifest, tmp_path / "resume", stop_after=1, **kwargs)
    resumed = train(cfg, manifest, tmp_path / "resume", resume=partial, **kwargs)
    a, b = load_checkpoint(full), load_checkpoint(resumed)
    for group in ("model", "heads"):
        for key in a[group]:
            torch.testing.assert_close(a[group][key], b[group][key], rtol=0, atol=0)
    encoder = load_encoder(export_encoder(full, tmp_path / "encoder.pt"))
    original = TempoBridge(cfg.model).eval()
    original.load_state_dict(a["model"])
    x = torch.randn(2, 1, 100)
    torch.testing.assert_close(encoder(x), original(x), rtol=0, atol=0)
    metrics = [json.loads(line) for line in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()]
    assert all(m["target_rank"] == 4 and m["active_transfer_fraction"] > 0 for m in metrics)
    assert all(np.isfinite(m["kd_gradient_norm"]) and m["gradient_audit_fraction"] > 0 for m in metrics)
    assert all(m["kd_gradient_norm/stem"] == 0 and m["kd_gradient_norm/shallow.0"] == 0 for m in metrics)
    assert all(m["ssl_gradient_norm/stem"] > 0 and m["parameter_update_norm/stem"] > 0 for m in metrics)
    assert load_checkpoint(tmp_path / "full" / "best.pt")["global_step"] == a["global_step"]


def test_rank_zero_skips_teacher_and_transfer_head(tiny_config):
    d = 4
    artifact = {"quality_edges": [], "scales": [{"weights": [1.], "target_mean": [0.] * d,
        "target_scale": [1.] * d, "subspace": {"version": 1, "rank": 0, "teacher_mean": [0.] * d,
        "teacher_scale": [1.] * d, "projection": [[] for _ in range(d)], "direction_weights": []}}]}

    class MustNotRun(torch.nn.Module):
        def forward(self, *_):
            raise AssertionError("rank-zero transfer should not run")

    x = torch.randn(2, 1, 100)
    loss, weight = transfer_loss(MustNotRun(), MustNotRun(), MustNotRun(), CalibrationWeights(artifact),
                                 x, x, torch.ones(2, dtype=torch.bool), torch.ones(2), torch.ones(2),
                                 0, "all", True)
    assert loss == weight == 0


def test_vanilla_does_not_require_oof_calibration(tmp_path, tiny_config):
    cfg = tiny_config
    manifest = make_synthetic(tmp_path / "data", subjects=8, windows=2, seconds=2)
    ppg = train(cfg, manifest, tmp_path / "ppg", stop_after=1)
    ecg = train(cfg, manifest, tmp_path / "ecg", modality="ecg", stop_after=1)
    cfg.train.transfer_method, cfg.train.transfer_mode = "vanilla", "all"
    cfg.calibration.folds = 50  # Deliberately impossible; vanilla must not train probes.
    output = tmp_path / "vanilla"
    train(cfg, manifest, output, student_path=ppg, teacher_path=ecg, stop_after=1)
    stats = json.loads((output / "target_statistics.json").read_text())
    assert all(not s["folds"] for s in stats["scales"])


def test_ssl_reports_available_pairs_without_claiming_active_transfer(tiny_config):
    model = TempoBridge(tiny_config.model)
    batch = {"ppg": [torch.randn(3, 1, 50), torch.randn(3, 1, 100)],
             "ecg_valid": torch.tensor([True, False, True])}
    _, metrics = batch_loss(tiny_config, model, batch, torch.device("cpu"), "ppg")
    assert abs(metrics["paired_fraction"] - 2 / 3) < 1e-6
    assert metrics["active_transfer_fraction"] == metrics["mean_transfer_weight"] == 0
