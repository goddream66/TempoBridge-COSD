"""Behavioral checks for the waveform/statistics tokenizer and temporal blocks."""

from types import SimpleNamespace

import pytest
import torch

from tempobridge.encoding import (
    TempoBridgeAttentionBlock,
    TempoBridgeScalarEncoder,
    TempoBridgeTokenizer,
)


def encoding_config(**overrides):
    values = dict(dim=16, heads=2, token_kernel_size=5, scalar_dim=8,
                  scalar_scales=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000],
                  scalar_epsilon=1.1, ff_hidden_dim=32, dropout=0.0,
                  xpos_scale_base=512)
    return SimpleNamespace(**(values | overrides))


def test_hidden_samples_cannot_change_neighbor_tokens_or_attention():
    cfg = encoding_config(token_kernel_size=11)
    tokenizer, block = TempoBridgeTokenizer(cfg).eval(), TempoBridgeAttentionBlock(cfg).eval()
    # The convolution reaches across each boundary. Hidden endpoints would also
    # contaminate the first difference of an adjacent visible patch if unmasked.
    patches = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, False, True, False, True],
                         [False, True, False, True, False]])
    changed = patches.clone()
    changed[mask] = torch.randn_like(changed[mask]) * 10000 + 500
    original_tokens, changed_tokens = tokenizer(patches, mask), tokenizer(changed, mask)
    torch.testing.assert_close(original_tokens, changed_tokens, rtol=0, atol=0)
    torch.testing.assert_close(block(original_tokens), block(changed_tokens), rtol=0, atol=0)


def test_hidden_input_gradient_is_zero_but_visible_signal_can_learn():
    tokenizer = TempoBridgeTokenizer(encoding_config(token_kernel_size=11))
    patches = torch.randn(2, 5, 4, requires_grad=True)
    mask = torch.tensor([[False, True, False, True, False],
                         [True, False, True, False, True]])
    tokens = tokenizer(patches, mask)
    objective = (tokens * torch.randn_like(tokens)).sum()
    objective.backward()
    assert torch.isfinite(patches.grad).all()
    assert torch.count_nonzero(patches.grad[mask]) == 0
    assert patches.grad[~mask].abs().sum() > 0
    assert tokenizer.mask_token.grad.abs().sum() > 0


@pytest.mark.parametrize('constant', [0.0, 1000.0])
@pytest.mark.parametrize('hide_middle', [False, True])
def test_constant_waveforms_have_finite_forward_and_backward(constant, hide_middle):
    cfg = encoding_config()
    tokenizer, block = TempoBridgeTokenizer(cfg), TempoBridgeAttentionBlock(cfg)
    patches = torch.full((2, 3, 4), constant, requires_grad=True)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mask[:, 1] = hide_middle
    tokens = tokenizer(patches, mask)
    result = block(tokens)
    (result * torch.linspace(-1, 1, result.numel()).reshape_as(result)).sum().backward()
    assert torch.isfinite(result).all()
    assert torch.isfinite(patches.grad).all()
    for module in (tokenizer, block):
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_statistics_receive_original_visible_patch_scale():
    tokenizer = TempoBridgeTokenizer(encoding_config())
    captured = {}

    def capture(name):
        def hook(module, args):
            captured[name] = args[0].detach().clone()
        return hook

    hooks = [tokenizer.mean_encoder.register_forward_pre_hook(capture('mean')),
             tokenizer.std_encoder.register_forward_pre_hook(capture('std'))]
    patches = torch.tensor([[[10., 12., 14., 16.], [500., 700., 900., 1100.],
                             [-8., -4., 0., 4.]]])
    mask = torch.tensor([[False, True, False]])
    tokenizer(patches, mask)
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(captured['mean'][~mask], patches.mean(-1, keepdim=True)[~mask])
    torch.testing.assert_close(captured['std'][~mask], patches.std(-1, keepdim=True)[~mask])
    assert torch.count_nonzero(captured['mean'][mask]) == 0
    assert torch.count_nonzero(captured['std'][mask]) == 0


def test_scalar_encoder_handles_sign_zero_and_wide_magnitudes():
    cfg = encoding_config()
    encoder = TempoBridgeScalarEncoder(cfg.scalar_scales, cfg.scalar_dim, cfg.scalar_epsilon)
    values = torch.tensor([-1e6, -1., -1e-6, 0., 1e-6, 1., 1e6]).reshape(1, -1, 1)
    values.requires_grad_(True)
    result = encoder(values)
    assert result.shape == (1, 7, cfg.scalar_dim)
    assert torch.isfinite(result).all()
    assert not torch.allclose(result[:, 0], result[:, -1])
    (result * torch.randn_like(result)).sum().backward()
    assert torch.isfinite(values.grad).all()
    assert encoder.slopes.grad.abs().sum() > 0
    assert encoder.offsets.grad.abs().sum() > 0


def test_scalar_scale_buffer_survives_checkpoint_loading():
    first = TempoBridgeScalarEncoder([0.01, 1., 100.], 8)
    restored = TempoBridgeScalarEncoder([1., 2., 3.], 8)
    restored.load_state_dict(first.state_dict())
    values = torch.tensor([0., 0.02, 5., 200.]).reshape(1, -1, 1)
    torch.testing.assert_close(first(values), restored(values), rtol=0, atol=0)


def test_tokenizer_requires_a_visible_patch_per_example():
    tokenizer = TempoBridgeTokenizer(encoding_config())
    patches = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match='visible'):
        tokenizer(patches, torch.tensor([[False, True, False], [True, True, True]]))


def test_attention_eval_disables_dropout_and_train_enables_it():
    block = TempoBridgeAttentionBlock(encoding_config(dropout=0.4))
    tokens = torch.randn(2, 9, 16)
    block.eval()
    torch.testing.assert_close(block(tokens), block(tokens), rtol=0, atol=0)
    block.train()
    assert not torch.equal(block(tokens), block(tokens))


def test_position_encoding_handles_alternating_sequence_lengths():
    block = TempoBridgeAttentionBlock(encoding_config()).eval()
    short, long = torch.randn(2, 4, 16), torch.randn(2, 65, 16)
    before = block(short)
    assert torch.isfinite(block(long)).all()
    torch.testing.assert_close(before, block(short), rtol=0, atol=0)
