import pytest
import torch

from tempobridge.calibration import CalibrationWeights
from tempobridge.model import TempoBridge, TransferHeads, random_mask, reconstruction_loss
from tempobridge.training import transfer_loss


def test_hidden_targets_cannot_change_encoder(tiny_config):
    model = TempoBridge(tiny_config.model).eval()
    x = torch.randn(3, 1, 100)
    mask = random_mask(x, 10, 0.5)
    changed = x.clone().reshape(3, 10, 10)
    changed[mask] = torch.randn_like(changed[mask]) * 1000 + 500
    a, b = model.encode(x, mask), model.encode(changed.reshape_as(x), mask)
    torch.testing.assert_close(a["context"], b["context"], rtol=0, atol=0)
    torch.testing.assert_close(a["morphology"], b["morphology"], rtol=0, atol=0)
    assert not torch.equal(a["target"], b["target"])


def test_dynamic_lengths_and_constant_waveform(tiny_config):
    model = TempoBridge(tiny_config.model).eval()
    for n in [50, 100, 230]:
        result = model(torch.ones(2, 1, n))
        assert result.shape == (2, 32)
        assert torch.isfinite(result).all()


def test_invalid_inputs_are_rejected(tiny_config):
    model = TempoBridge(tiny_config.model)
    for x in [torch.randn(2, 100), torch.randn(2, 2, 100), torch.randn(2, 1, 99),
              torch.full((2, 1, 100), float("nan"))]:
        with pytest.raises(ValueError):
            model(x)
    with pytest.raises(ValueError, match="visible"):
        model.encode(torch.randn(2, 1, 100), torch.ones(2, 10, dtype=torch.bool))


class DummyFrozenTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 4)
        self.requires_grad_(False)

    def forward(self, x):
        return self.linear(x[:, 0])


def artifact():
    return {"quality_edges": [0.33, 0.67], "scales": [
        {"weights": [0.2, 0.5, 1.0], "target_mean": [0.0] * 4, "target_scale": [1.0] * 4}
    ]}


def test_kd_gradient_is_confined_to_context(tiny_config):
    model = TempoBridge(tiny_config.model)
    heads = TransferHeads(16, 4, [1])
    teacher = DummyFrozenTeacher().eval()
    calibration = CalibrationWeights(artifact())
    loss, _ = transfer_loss(model, heads, teacher, calibration, torch.randn(4, 1, 100),
                           torch.randn(4, 1, 100), torch.ones(4, dtype=torch.bool),
                           torch.ones(4), torch.ones(4), 0, "calibrated", True)
    loss.backward()
    for module in [model.stem, model.shallow, model.morphology, model.morph_norm, teacher]:
        assert all(p.grad is None for p in module.parameters())
    assert sum(p.grad.abs().sum() for p in model.context.parameters() if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in heads.parameters() if p.grad is not None) > 0


def test_no_protection_ablation_allows_shallow_gradients(tiny_config):
    model = TempoBridge(tiny_config.model)
    heads = TransferHeads(16, 4, [1])
    loss, _ = transfer_loss(model, heads, DummyFrozenTeacher(), CalibrationWeights(artifact()),
                           torch.randn(3, 1, 100), torch.randn(3, 1, 100),
                           torch.ones(3, dtype=torch.bool), torch.ones(3), torch.ones(3),
                           0, "all", False)
    loss.backward()
    assert model.stem.projection.weight.grad.abs().sum() > 0


def test_zero_weights_skip_teacher_and_keep_ssl_trainable(tiny_config):
    model = TempoBridge(tiny_config.model)
    class MustNotRun(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("zero-weight teacher was called")
    x = torch.randn(3, 1, 100)
    kd, weight = transfer_loss(model, TransferHeads(16, 4, [1]), MustNotRun(),
                              CalibrationWeights(artifact()), x, x,
                              torch.zeros(3, dtype=torch.bool), torch.ones(3), torch.ones(3),
                              0, "calibrated", True)
    assert kd == 0 and weight == 0
    loss = reconstruction_loss(model, x, random_mask(x, 10, 0.5)) + kd
    loss.backward()
    assert model.stem.projection.weight.grad.abs().sum() > 0


def test_weight_scale_does_not_cancel_in_denominator(tiny_config):
    model = TempoBridge(tiny_config.model).eval()
    teacher, heads = DummyFrozenTeacher(), TransferHeads(16, 4, [1])
    a = artifact()
    b = artifact()
    b["scales"][0]["weights"] = [0.1, 0.25, 0.5]
    p, e = torch.randn(4, 1, 100), torch.randn(4, 1, 100)
    kwargs = (p, e, torch.ones(4, dtype=torch.bool), torch.ones(4), torch.ones(4), 0, "calibrated", True)
    first, _ = transfer_loss(model, heads, teacher, CalibrationWeights(a), *kwargs)
    second, _ = transfer_loss(model, heads, teacher, CalibrationWeights(b), *kwargs)
    torch.testing.assert_close(second, first * 0.5)


def test_fixed_batch_reconstruction_can_be_optimized(tiny_config):
    model = TempoBridge(tiny_config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    x = torch.sin(torch.linspace(0, 12, 100))[None, None].repeat(4, 1, 1)
    mask = random_mask(x, 10, 0.5)
    initial = float(reconstruction_loss(model, x, mask).detach())
    for _ in range(30):
        optimizer.zero_grad()
        loss = reconstruction_loss(model, x, mask)
        loss.backward()
        optimizer.step()
    final = float(reconstruction_loss(model, x, mask).detach())
    assert final < initial * 0.5


def test_short_crop_cannot_access_outer_context(tiny_config):
    model = TempoBridge(tiny_config.model).eval()
    long = torch.randn(2, 1, 200)
    changed = long.clone()
    changed[:, :, 100:] += 1000
    torch.testing.assert_close(model(long[:, :, :100]), model(changed[:, :, :100]), rtol=0, atol=0)
