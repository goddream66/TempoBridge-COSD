"""Selective tokenizer behavior, masking boundaries, and checkpoint compatibility."""

import pytest
import torch

from tempobridge.config import ModelConfig
from tempobridge.encoding import TempoBridgeSelectiveKernel, TempoBridgeSelectiveTokenizer
from tempobridge.inference import load_encoder
from tempobridge.model import TempoBridge
from tempobridge.utils import save_checkpoint


def selective_config(**changes):
    values = dict(encoder='selective', dim=16, depth=3, protected_depth=1,
                  heads=2, patch_size=10, stem_width=4, ff_multiplier=2,
                  ff_hidden_dim=32, scalar_dim=8, token_kernel_size=9,
                  selective_width=8, selective_kernel_sizes=[3, 7, 11])
    return ModelConfig(**(values | changes))


@pytest.mark.parametrize('local_context', [False, True])
def test_hidden_samples_cannot_affect_tokens_or_global_selection(local_context):
    tokenizer = TempoBridgeSelectiveTokenizer(
        selective_config(selective_local_context=local_context)).eval()
    patches = torch.randn(2, 7, 10)
    mask = torch.tensor([[True, False, True, False, False, True, False],
                         [False, True, False, True, True, False, True]])
    changed = patches.clone()
    changed[mask] = torch.randn_like(changed[mask]) * 1e6 + 5e5
    with torch.no_grad():
        first, before = tokenizer.forward_with_diagnostics(patches, mask)
        second, after = tokenizer.forward_with_diagnostics(changed, mask)
        plain = tokenizer(patches, mask)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, plain, rtol=0, atol=0)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    # Kernel 11 crosses a patch boundary; this also covers derivative endpoints.
    differentiable = patches.clone().requires_grad_()
    tokens, diagnostics = tokenizer.forward_with_diagnostics(differentiable, mask)
    loss = (tokens * torch.randn_like(tokens)).sum()
    loss = loss + (diagnostics['weights_waveform'] *
                   torch.randn_like(diagnostics['weights_waveform'])).sum()
    loss.backward()
    assert torch.count_nonzero(differentiable.grad[mask]) == 0
    assert differentiable.grad[~mask].abs().sum() > 0
    assert torch.isfinite(differentiable.grad).all()


@pytest.mark.parametrize('constant', [None, 0.0, 1000.0])
@pytest.mark.parametrize('masked', [False, True])
def test_normal_and_constant_signals_have_finite_forward_and_backward(constant, masked):
    tokenizer = TempoBridgeSelectiveTokenizer(selective_config())
    patches = (torch.randn(2, 5, 10) if constant is None else
               torch.full((2, 5, 10), constant)).requires_grad_()
    mask = torch.zeros(2, 5, dtype=torch.bool)
    if masked:
        mask[:, [1, 3]] = True
    tokens, diagnostics = tokenizer.forward_with_diagnostics(patches, mask)
    assert torch.isfinite(tokens).all()
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    (tokens * torch.randn_like(tokens)).sum().backward()
    assert torch.isfinite(patches.grad).all()
    assert torch.count_nonzero(patches.grad[mask]) == 0
    gradients = [parameter.grad for parameter in tokenizer.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_each_scale_branch_and_selector_receive_learning_signal():
    tokenizer = TempoBridgeSelectiveTokenizer(selective_config())
    patches = torch.randn(3, 8, 10, requires_grad=True)
    tokens = tokenizer(patches, torch.zeros(3, 8, dtype=torch.bool))
    (tokens * torch.randn_like(tokens)).sum().backward()
    for kernel in (tokenizer.waveform_kernel, tokenizer.difference_kernel):
        for branch in list(kernel.depthwise_convs) + list(kernel.pointwise_convs):
            assert branch.weight.grad is not None
            assert torch.isfinite(branch.weight.grad).all()
            assert branch.weight.grad.abs().sum() > 0
        selectors = [parameter for name, parameter in kernel.named_parameters()
                     if name.startswith('selector_')]
        assert selectors and all(parameter.grad is not None for parameter in selectors)
        assert all(torch.isfinite(parameter.grad).all() for parameter in selectors)
        assert any(parameter.grad.abs().sum() > 0 for parameter in selectors)


@pytest.mark.parametrize('fusion', ['adaptive', 'mean'])
def test_weights_normalize_only_observed_patches_and_mean_is_equal(fusion):
    tokenizer = TempoBridgeSelectiveTokenizer(selective_config(selective_fusion=fusion)).eval()
    mask = torch.tensor([[False, True, False, False, True, False],
                         [True, False, False, True, False, True]])
    _, diagnostics = tokenizer.forward_with_diagnostics(torch.randn(2, 6, 10), mask)
    for view in ('waveform', 'difference'):
        weights, coverage = diagnostics[f'weights_{view}'], diagnostics[f'coverage_{view}']
        assert weights.shape == (2, 6, 3, 8)
        assert coverage.shape == (2, 6, 3)
        assert torch.all((coverage >= 0) & (coverage <= 1))
        assert torch.count_nonzero(weights[mask]) == 0
        assert torch.count_nonzero(coverage[mask]) == 0
        sums = weights[~mask].sum(1)
        torch.testing.assert_close(sums, torch.ones_like(sums))
        if fusion == 'mean':
            torch.testing.assert_close(weights[~mask], torch.full_like(weights[~mask], 1 / 3))


def test_adaptive_selection_depends_on_visible_content():
    tokenizer = TempoBridgeSelectiveTokenizer(selective_config()).eval()
    time = torch.arange(120, dtype=torch.float32)
    smooth = torch.sin(time / 10).reshape(1, 12, 10)
    abrupt = ((time % 7) < 2).float().reshape(1, 12, 10)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    _, first = tokenizer.forward_with_diagnostics(smooth, mask)
    _, second = tokenizer.forward_with_diagnostics(abrupt, mask)
    for view in ('waveform', 'difference'):
        assert not torch.allclose(first[f'weights_{view}'], second[f'weights_{view}'])


def test_global_only_selection_is_shared_across_visible_patches():
    kernel = TempoBridgeSelectiveKernel(selective_config(selective_local_context=False)).eval()
    values = torch.randn(2, 1, 100)
    _, diagnostics = kernel.forward_with_diagnostics(values, torch.ones_like(values, dtype=torch.bool), 10)
    weights = diagnostics['weights']
    torch.testing.assert_close(weights, weights[:, :1].expand_as(weights), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(diagnostics['coverage'], torch.ones(2, 10, 3), rtol=0, atol=0)


def test_coverage_reports_observation_support_without_boundary_penalty():
    kernel = TempoBridgeSelectiveKernel(selective_config()).eval()
    values = torch.randn(1, 1, 50)
    visible = torch.ones_like(values, dtype=torch.bool)
    _, complete = kernel.forward_with_diagnostics(values, visible, 10)
    torch.testing.assert_close(complete['coverage'], torch.ones(1, 5, 3), rtol=0, atol=0)
    visible[:, :, 10:20] = False
    _, incomplete = kernel.forward_with_diagnostics(values, visible, 10)
    assert torch.count_nonzero(incomplete['coverage'][:, 1]) == 0
    assert torch.all(incomplete['coverage'][:, 2] < 1)
    assert torch.all(incomplete['coverage'][:, 2, 0] > incomplete['coverage'][:, 2, 2])
    torch.testing.assert_close(incomplete['coverage'][:, 4], torch.ones(1, 3), rtol=0, atol=0)


def test_local_selector_can_choose_different_scales_in_one_recording():
    kernel = TempoBridgeSelectiveKernel(selective_config(selective_local_context=True)).eval()
    values = torch.randn(2, 1, 100)
    _, diagnostics = kernel.forward_with_diagnostics(values, torch.ones_like(values, dtype=torch.bool), 10)
    weights = diagnostics['weights']
    assert not torch.allclose(weights, weights[:, :1].expand_as(weights))


def test_coverage_ablation_changes_fusion_but_keeps_observation_diagnostics():
    enabled = TempoBridgeSelectiveKernel(selective_config(selective_coverage=True)).eval()
    disabled = TempoBridgeSelectiveKernel(selective_config(selective_coverage=False)).eval()
    disabled.load_state_dict(enabled.state_dict(), strict=True)
    values = torch.randn(2, 1, 70)
    visible = torch.ones_like(values, dtype=torch.bool)
    visible[:, :, 20:40] = False
    first, first_diagnostics = enabled.forward_with_diagnostics(values, visible, 10)
    second, second_diagnostics = disabled.forward_with_diagnostics(values, visible, 10)
    torch.testing.assert_close(first_diagnostics['coverage'], second_diagnostics['coverage'],
                               rtol=0, atol=0)
    assert not torch.allclose(first_diagnostics['weights'], second_diagnostics['weights'])
    assert not torch.allclose(first, second)


@pytest.mark.parametrize('fusion', ['adaptive', 'mean'])
def test_kernel_with_no_observations_has_zero_features_and_allocations(fusion):
    kernel = TempoBridgeSelectiveKernel(selective_config(selective_fusion=fusion))
    values = torch.randn(2, 1, 40, requires_grad=True)
    features, diagnostics = kernel.forward_with_diagnostics(values, torch.zeros_like(values, dtype=torch.bool), 10)
    assert torch.count_nonzero(features) == 0
    assert all(torch.count_nonzero(value) == 0 for value in diagnostics.values())
    features.sum().backward()
    assert torch.count_nonzero(values.grad) == 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in kernel.parameters()
               if parameter.grad is not None)


@pytest.mark.parametrize('local_context', [False, True])
@pytest.mark.parametrize('coverage', [False, True])
def test_ablation_switches_accept_changing_token_lengths(local_context, coverage):
    tokenizer = TempoBridgeSelectiveTokenizer(selective_config(
        selective_local_context=local_context, selective_coverage=coverage)).eval()
    for count in (2, 7, 31):
        patches = torch.randn(2, count, 10)
        mask = torch.zeros(2, count, dtype=torch.bool)
        mask[:, 1] = True
        tokens, diagnostics = tokenizer.forward_with_diagnostics(patches, mask)
        assert tokens.shape == (2, count, 16)
        assert torch.isfinite(tokens).all()
        assert diagnostics['weights_waveform'].shape == (2, count, 3, 8)


def test_mean_fusion_bypasses_selector_and_preserves_checkpoint_layout():
    adaptive = TempoBridgeSelectiveTokenizer(selective_config())
    mean = TempoBridgeSelectiveTokenizer(selective_config(selective_fusion='mean'))
    mean.load_state_dict(adaptive.state_dict(), strict=True)
    patches = torch.randn(2, 5, 10)
    tokens = mean(patches, torch.zeros(2, 5, dtype=torch.bool))
    (tokens * torch.randn_like(tokens)).sum().backward()
    selectors = [parameter for name, parameter in mean.named_parameters() if '.selector_' in name]
    assert selectors and all(parameter.grad is None for parameter in selectors)
    assert mean.waveform_kernel.depthwise_convs[0].weight.grad.abs().sum() > 0


def test_prior_multiview_export_remains_identical_without_selective_config_fields(tmp_path):
    model = TempoBridge(selective_config(encoder='multiview')).eval()
    historical_config = {name: value for name, value in model.model_config().items()
                         if not name.startswith('selective_')}
    weights = {name: value for name, value in model.state_dict().items()
               if not name.startswith(('reconstruction.', 'morph_reconstruction.'))}
    artifact = tmp_path / 'multiview_v2.pt'
    save_checkpoint(artifact, {'format_version': 1, 'kind': 'ppg_encoder',
                               'model_config': historical_config, 'model': weights})
    restored = load_encoder(artifact)
    assert restored.config.encoder == 'multiview'
    for length in (50, 100):
        waveform = torch.randn(2, 1, length)
        with torch.no_grad():
            torch.testing.assert_close(model(waveform), restored(waveform), rtol=0, atol=0)
