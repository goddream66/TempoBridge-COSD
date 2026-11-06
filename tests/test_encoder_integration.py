"""Public encoder semantics and compatibility across the two architectures."""

import pytest
import torch

from tempobridge.config import ModelConfig
from tempobridge.inference import load_encoder
from tempobridge.model import TempoBridge
from tempobridge.utils import save_checkpoint


def small_model_config(**overrides):
    values = dict(encoder='multiview', dim=16, depth=3, protected_depth=1,
                  heads=2, patch_size=10, stem_width=4, ff_multiplier=2,
                  ff_hidden_dim=32, scalar_dim=8, token_kernel_size=9)
    return ModelConfig(**(values | overrides))


def test_new_default_and_historical_config_have_distinct_architectures():
    assert ModelConfig().encoder == 'selective'
    assert ModelConfig.from_dict({'dim': 16, 'heads': 2}).encoder == 'local'
    assert ModelConfig.from_dict({'encoder': 'multiview'}).encoder == 'multiview'


@pytest.mark.parametrize('pool,dimension', [('cls', 16), ('mean', 16),
                                         ('combined', 32), ('morph_context', 32)])
def test_public_pooling_uses_the_documented_token_content(pool, dimension):
    model = TempoBridge(small_model_config(feature_pool=pool)).eval()
    captured = {}

    def final_layer(module, inputs, output):
        captured['tokens'] = output.detach().clone()

    handle = model.context[-1].register_forward_hook(final_layer)
    with torch.inference_mode():
        features = model.forward_features(torch.randn(2, 1, 50))
    handle.remove()
    encoded = captured['tokens']
    assert encoded.shape == (2, 6, 16)  # Five patches followed by one summary token.
    assert features['tokens'].shape == (2, 5, 16)
    torch.testing.assert_close(features['tokens'], encoded[:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(features['summary'], encoded[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(features['context'], encoded[:, :-1].mean(1), rtol=0, atol=0)
    expected = {
        'cls': encoded[:, -1],
        'mean': encoded[:, :-1].mean(1),
        'combined': torch.cat((encoded[:, -1], encoded[:, :-1].mean(1)), dim=-1),
        'morph_context': torch.cat((features['morphology'], encoded[:, :-1].mean(1)), dim=-1),
    }[pool]
    assert model.feature_dim == dimension
    assert features['embedding'].shape == (2, dimension)
    torch.testing.assert_close(features['embedding'], expected, rtol=0, atol=0)


def test_intermediate_features_ignore_changes_to_later_layers():
    model = TempoBridge(small_model_config(feature_layer=1, feature_pool='combined')).eval()
    captured = {}

    def selected_layer(module, inputs, output):
        captured['tokens'] = output.detach().clone()

    handle = model.context[0].register_forward_hook(selected_layer)
    waveform = torch.randn(2, 1, 70)
    with torch.inference_mode():
        before = model.forward_features(waveform)
    handle.remove()
    torch.testing.assert_close(before['tokens'], captured['tokens'][:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(before['summary'], captured['tokens'][:, -1], rtol=0, atol=0)
    with torch.no_grad():
        for parameter in model.context[1].parameters():
            parameter.add_(torch.randn_like(parameter) * 10)
    with torch.inference_mode():
        after = model.forward_features(waveform)
    torch.testing.assert_close(before['embedding'], after['embedding'], rtol=0, atol=0)


def test_fixed_token_count_rejects_otherwise_valid_waveform_length():
    model = TempoBridge(small_model_config(num_tokens=5)).eval()
    assert model(torch.randn(2, 1, 50)).shape == (2, 32)
    with pytest.raises(ValueError, match='fixed num_tokens'):
        model(torch.randn(2, 1, 60))


def test_reconstruction_mask_and_targets_exclude_summary_token():
    model = TempoBridge(small_model_config()).eval()
    waveform = torch.randn(2, 1, 50)
    mask = torch.tensor([[False, True, False, True, False],
                         [True, False, True, False, False]])
    encoded = model.encode(waveform, mask)
    assert encoded['context'].shape == encoded['morphology'].shape == (2, 5, 16)
    assert encoded['target'].shape == (2, 5, 10)
    assert encoded['mask'].shape == (2, 5)
    assert torch.equal(encoded['mask'], mask)
    # A caller must never allocate or mask a sixth patch for the summary token.
    with pytest.raises(ValueError, match='mask must be boolean'):
        model.encode(waveform, torch.zeros(2, 6, dtype=torch.bool))


def test_historical_export_without_new_config_fields_loads_identically(tmp_path):
    model = TempoBridge(small_model_config(encoder='local')).eval()
    historical_keys = {'dim', 'depth', 'protected_depth', 'heads', 'patch_size',
                       'stem_width', 'ff_multiplier', 'dropout', 'morphology_branch'}
    exported_weights = {name: value for name, value in model.state_dict().items()
                        if not name.startswith(('reconstruction.', 'morph_reconstruction.'))}
    historical_config = {name: value for name, value in model.model_config().items()
                         if name in historical_keys}
    artifact = tmp_path / 'historical_encoder.pt'
    save_checkpoint(artifact, {'format_version': 1, 'kind': 'ppg_encoder',
                               'model_config': historical_config, 'model': exported_weights})
    restored = load_encoder(artifact)
    assert restored.config.encoder == 'local'
    waveform = torch.randn(2, 1, 90)
    with torch.inference_mode():
        torch.testing.assert_close(model(waveform), restored(waveform), rtol=0, atol=0)


def test_morphology_pool_without_branch_retains_context_dimension():
    model = TempoBridge(small_model_config(feature_pool='morph_context', morphology_branch=False)).eval()
    features = model.forward_features(torch.randn(2, 1, 50))
    assert model.feature_dim == 16
    torch.testing.assert_close(features['embedding'], features['context'], rtol=0, atol=0)


@pytest.mark.parametrize('encoder', ['local', 'multiview'])
@pytest.mark.parametrize('selected_layer', [0, 1])
def test_selected_shallow_snapshot_respects_gradient_protection(encoder, selected_layer):
    model = TempoBridge(small_model_config(encoder=encoder, protected_depth=2)).eval()
    waveform = torch.randn(2, 1, 50)
    head = torch.nn.Linear(16, 1)
    protected = model.encode(waveform, detach_shallow=True, return_layer=selected_layer)['context']
    unprotected = model.encode(waveform, detach_shallow=False, return_layer=selected_layer)['context']
    torch.testing.assert_close(protected, unprotected, rtol=0, atol=0)
    head(protected).sum().backward()
    assert head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.stem.parameters())
    assert all(parameter.grad is None for parameter in model.shallow.parameters())
    model.zero_grad(set_to_none=True)
    head(unprotected).sum().backward()
    assert model.stem.projection.weight.grad.abs().sum() > 0
