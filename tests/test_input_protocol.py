"""Physical input duration and synchronized full-window/crop protocol behavior."""

import csv

import numpy as np
import pytest
import torch

from tempobridge.calibration import CalibrationWeights, calibrate
from tempobridge.config import DataConfig, ExperimentConfig, ModelConfig
from tempobridge.data import PairedDataset, Record, make_synthetic, manifest_digest, read_manifest
from tempobridge.inference import extract_features
from tempobridge.model import TempoBridge
from tempobridge.training import train
from tempobridge.utils import save_checkpoint


def paired_record(tmp_path, ppg_length=1500, ecg_length=1500, split='train'):
    # Integer sample identities make any crop offset or ECG misalignment observable.
    ppg = np.arange(ppg_length, dtype=np.float32)
    ecg = 2 * np.arange(ecg_length, dtype=np.float32) + 7
    path = tmp_path / f'{split}.npz'
    np.savez(path, ppg=ppg, ppg_fs=50, ecg=ecg, ecg_fs=50)
    return Record(split, f'person_{split}', f'record_{split}', split, path, 0.9, 0.9), ppg, ecg


def test_new_defaults_use_one_complete_thirty_second_window():
    config = ExperimentConfig().validate()
    assert config.data.window_mode == 'full'
    assert config.data.scales_seconds == [30.0]
    assert config.data.ppg_fs == config.data.ecg_fs == 50
    assert config.model.patch_size == 10
    assert ExperimentConfig.from_dict({}).data.window_mode == 'full'


def test_old_explicit_data_config_retains_crop_protocol():
    config = ExperimentConfig.from_dict({'data': {'ppg_fs': 50, 'ecg_fs': 50,
                                                 'scales_seconds': [2.0, 8.0]}})
    assert config.data.window_mode == 'crop'
    assert config.data.scales_seconds == [2.0, 8.0]
    legacy = ExperimentConfig.from_dict({'model': {}, 'data': {}})
    assert legacy.model.patch_size == 25
    assert legacy.data.input_protocol() == {
        'ppg_fs': 100, 'ecg_fs': 100, 'scales_seconds': [2.0, 8.0], 'window_mode': 'crop',
    }
    assert ModelConfig.from_dict({'encoder': 'multiview'}).patch_size == 25


@pytest.mark.parametrize('model', [{}, {'encoder': 'local'}, {'encoder': 'multiview'}])
def test_legacy_model_without_data_section_retains_historical_defaults(model):
    config = ExperimentConfig.from_dict({'model': model})
    assert config.model.patch_size == 25
    assert config.data.input_protocol() == {
        'ppg_fs': 100, 'ecg_fs': 100, 'scales_seconds': [2.0, 8.0], 'window_mode': 'crop',
    }
    new = ExperimentConfig.from_dict({'model': {'encoder': 'selective'}})
    assert new.data.input_protocol() == DataConfig().input_protocol()


def test_full_input_preserves_every_sample_in_training_and_evaluation(tmp_path):
    train_record, ppg, ecg = paired_record(tmp_path)
    test_record, _, _ = paired_record(tmp_path, split='test')
    config = DataConfig(ppg_fs=50, ecg_fs=50, scales_seconds=[30.0], window_mode='full')
    train = PairedDataset([train_record], config, seed=17, augment=True)
    evaluation = PairedDataset([test_record], config, split='test', seed=91, augment=False)
    for epoch in (0, 1, 7):
        train.set_epoch(epoch)
        for item in (train[0], evaluation[0]):
            assert len(item['ppg']) == len(item['ecg']) == 1
            assert item['ppg'][0].shape == item['ecg'][0].shape == (1, 1500)
            torch.testing.assert_close(item['ppg'][0][0], torch.from_numpy(ppg), rtol=0, atol=0)
            torch.testing.assert_close(item['ecg'][0][0], torch.from_numpy(ecg), rtol=0, atol=0)
            assert item['ecg_valid']
    train.close()
    evaluation.close()


@pytest.mark.parametrize('ppg_length,ecg_length', [(1499, 1499), (1501, 1501),
                                                (1500, 1499), (1500, 1501)])
def test_full_input_rejects_even_one_missing_or_extra_sample(tmp_path, ppg_length, ecg_length):
    record, _, _ = paired_record(tmp_path, ppg_length, ecg_length)
    config = DataConfig(ppg_fs=50, ecg_fs=50, scales_seconds=[30.0], window_mode='full')
    dataset = PairedDataset([record], config)
    with pytest.raises(ValueError, match='full window_mode requires exactly 30 seconds'):
        dataset[0]
    dataset.close()


def test_full_protocol_rejects_ambiguous_multiple_durations(tmp_path):
    with pytest.raises(ValueError, match='exactly one duration'):
        ExperimentConfig.from_dict({'data': {'window_mode': 'full', 'scales_seconds': [2, 8]}})
    record, _, _ = paired_record(tmp_path)
    with pytest.raises(ValueError, match='exactly one duration'):
        PairedDataset([record], DataConfig(ppg_fs=50, ecg_fs=50,
                                          scales_seconds=[2, 8], window_mode='full'))


def test_explicit_two_and_eight_second_crops_are_reproducible_and_synchronized(tmp_path):
    record, _, _ = paired_record(tmp_path)
    config = DataConfig(ppg_fs=50, ecg_fs=50, scales_seconds=[2.0, 8.0], window_mode='crop')
    first = PairedDataset([record], config, seed=101, augment=True)
    second = PairedDataset([record], config, seed=101, augment=True)
    first.set_epoch(3)
    second.set_epoch(3)
    a, b = first[0], second[0]
    for index, length in enumerate((100, 400)):
        assert a['ppg'][index].shape == a['ecg'][index].shape == (1, length)
        torch.testing.assert_close(a['ppg'][index], b['ppg'][index], rtol=0, atol=0)
        torch.testing.assert_close(a['ecg'][index], b['ecg'][index], rtol=0, atol=0)
        torch.testing.assert_close(a['ecg'][index], 2 * a['ppg'][index] + 7, rtol=0, atol=0)
        torch.testing.assert_close(a['ppg'][index].diff(), torch.ones(1, length - 1), rtol=0, atol=0)
    second.set_epoch(4)
    changed = second[0]
    assert any(not torch.equal(a['ppg'][index], changed['ppg'][index]) for index in (0, 1))
    first.close()
    second.close()


def test_crop_evaluation_uses_the_same_center_for_both_modalities(tmp_path):
    record, ppg, ecg = paired_record(tmp_path)
    config = DataConfig(ppg_fs=50, ecg_fs=50, scales_seconds=[2.0, 8.0], window_mode='crop')
    dataset = PairedDataset([record], config, augment=False)
    item = dataset[0]
    for index, (start, length) in enumerate(((700, 100), (550, 400))):
        torch.testing.assert_close(item['ppg'][index][0], torch.from_numpy(ppg[start:start + length]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(item['ecg'][index][0], torch.from_numpy(ecg[start:start + length]),
                                   rtol=0, atol=0)
    dataset.close()


def test_historical_export_feature_extraction_preserves_crop_protocol(tmp_path):
    record, ppg, _ = paired_record(tmp_path, split='test')
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id', 'subject_id', 'recording_id',
                                                   'split', 'path'])
        writer.writeheader()
        writer.writerow({'sample_id': record.sample_id, 'subject_id': record.subject_id,
                         'recording_id': record.recording_id, 'split': record.split,
                         'path': record.path.name})
    model = TempoBridge(ModelConfig(encoder='multiview', dim=16, depth=3, protected_depth=1,
                                    heads=2, patch_size=10, scalar_dim=8, token_kernel_size=9,
                                    ff_hidden_dim=32)).eval()
    historical_config = {name: value for name, value in model.model_config().items()
                         if not name.startswith('selective_')}
    weights = {name: value for name, value in model.state_dict().items()
               if not name.startswith(('reconstruction.', 'morph_reconstruction.'))}
    artifact = tmp_path / 'historical_multiview.pt'
    save_checkpoint(artifact, {'format_version': 1, 'kind': 'ppg_encoder',
                               'model_config': historical_config, 'model': weights,
                               'data_config': {'ppg_fs': 50, 'ecg_fs': 50,
                                               'scales_seconds': [2.0, 8.0]}})
    output = tmp_path / 'features.npz'
    extract_features(artifact, manifest, output, split='test', device='cpu', batch_size=1)
    with torch.no_grad():
        expected = model(torch.from_numpy(ppg[550:950])[None, None]).numpy()
    with np.load(output, allow_pickle=False) as features:
        np.testing.assert_allclose(features['embedding'], expected, rtol=0, atol=0)
        assert features['scale_seconds'].item() == 8.0


def protocol_config():
    return ExperimentConfig.from_dict({
        'model': {'encoder': 'selective', 'dim': 16, 'heads': 2, 'depth': 3,
                  'protected_depth': 1, 'scalar_dim': 8, 'ff_hidden_dim': 32,
                  'selective_width': 8, 'selective_kernel_sizes': [3, 7, 11]},
        'data': {'window_mode': 'full', 'scales_seconds': [30.0], 'ppg_fs': 50, 'ecg_fs': 50},
        'train': {'device': 'cpu', 'threads': 2, 'epochs': 1, 'batch_size': 4},
        'calibration': {'folds': 2, 'probe': 'ridge', 'min_subjects_per_bin': 1},
    })


@pytest.mark.parametrize('operation', ['calibrate', 'train'])
@pytest.mark.parametrize('field,replacement', [('ppg_fs', 100), ('ecg_fs', 100),
                                              ('scales_seconds', [20.0]), ('window_mode', 'crop')])
def test_warmup_protocol_mismatch_is_rejected_before_learning(tmp_path, operation, field, replacement):
    config = protocol_config()
    manifest = make_synthetic(tmp_path / 'data', subjects=8, windows=1, seconds=30, fs=50)
    records = read_manifest(manifest)
    checkpoint = tmp_path / 'warmup.pt'
    # A synthetic trained marker tests the preflight boundary. No weights or
    # optimizer are needed: an incompatible protocol must fail before loading them.
    save_checkpoint(checkpoint, {'modality': 'ppg', 'global_step': 1,
                                 'manifest_sha256': manifest_digest(records),
                                 'model_config': config.to_dict()['model'],
                                 'config': config.to_dict(), 'model': {}})
    setattr(config.data, field, replacement)
    config.validate()
    with pytest.raises(ValueError, match='input protocol differs'):
        if operation == 'calibrate':
            calibrate(config, manifest, checkpoint, tmp_path / 'unused_teacher.pt',
                      tmp_path / 'calibration.json')
        else:
            train(config, manifest, tmp_path / 'run', student_path=checkpoint)
    assert not (tmp_path / 'calibration.json').exists()
    assert not (tmp_path / 'run' / 'last.pt').exists()


def test_calibration_persists_mode_and_treats_legacy_artifact_as_crop(tmp_path):
    config = protocol_config()
    manifest = make_synthetic(tmp_path / 'data', subjects=8, windows=2, seconds=30, fs=50)
    records = read_manifest(manifest)
    # These random-weight fixtures carry a trained marker solely to exercise
    # protocol metadata. No pretraining or model optimizer runs in this test.
    model = TempoBridge(config.model)
    state = {'global_step': 1, 'manifest_sha256': manifest_digest(records),
             'model_config': model.model_config(), 'config': config.to_dict(),
             'model': model.state_dict(),
             'train_subjects': sorted({record.subject_id for record in records if record.split == 'train'})}
    student, teacher = tmp_path / 'student.pt', tmp_path / 'teacher.pt'
    save_checkpoint(student, state | {'modality': 'ppg'})
    save_checkpoint(teacher, state | {'modality': 'ecg'})
    output = tmp_path / 'calibration.json'
    artifact = calibrate(config, manifest, student, teacher, output)
    assert artifact['window_mode'] == 'full'
    restored = CalibrationWeights.load(output)
    assert restored.artifact['window_mode'] == 'full'
    hashes = (artifact['manifest_sha256'], artifact['teacher_sha256'], artifact['student_sha256'],
              artifact['teacher_kind'])
    restored.validate(config, *hashes)
    config.data.window_mode = 'crop'
    with pytest.raises(ValueError, match='window_mode'):
        restored.validate(config, *hashes)
    historical = CalibrationWeights({key: value for key, value in artifact.items() if key != 'window_mode'})
    historical.validate(config, *hashes)
    config.data.window_mode = 'full'
    with pytest.raises(ValueError, match='window_mode'):
        historical.validate(config, *hashes)
