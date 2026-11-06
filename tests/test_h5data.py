import json
import pickle
import shutil
from collections import Counter

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tempobridge.calibration import calibrate
from tempobridge.config import DataConfig
from tempobridge.data import PairedDataset, read_manifest, subject_ids
from tempobridge.h5data import (
    SALT,
    H5RecordIndex,
    assign_groups,
    build_h5_manifest,
    canonical_hash,
    group_hash,
    normalize_identifier,
    split_counts,
)
from tempobridge.inference import export_encoder, extract_features
from tempobridge.sampling import SourceBlockSampler, calibration_indices, source_counts
from tempobridge.training import train
from tempobridge.utils import load_checkpoint


def write_shard(path, ids, seed=12, fs=50, seconds=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    waves = np.random.default_rng(seed).normal(size=(len(ids), fs * seconds)).astype('f4')
    with h5py.File(path, 'w') as h:
        h['signals'] = waves
        h['ecg_wave'] = waves * 0.5
        h['has_ecg'] = np.ones(len(ids), bool)
        h['subject_id'] = np.asarray(ids)
        h['metrics'] = np.full((len(ids), 2), [123, 0.8])
        h.attrs.update(target_fs=fs, metric_keys='hr,sqi')
    return path


def make_index(tmp_path, sources=('mimic',), subjects=40, windows=2, unknown=True):
    specs = {}
    for i, source in enumerate(sources):
        ids = np.repeat(np.arange(subjects), windows).tolist() + ([-1] if unknown else [])
        write_shard(tmp_path / source / 'part.h5', ids, seed=12+i)
        specs[source] = {'root': source, 'fs': 50}
    spec = tmp_path / 'sources.json'
    spec.write_text(json.dumps({'sources': specs}))
    manifest = tmp_path / 'index.json'
    build_h5_manifest(spec, manifest)
    return manifest


@pytest.mark.parametrize('value,expected', [(-1, None), ('unknown', None), (True, None),
    (float('nan'), None), (12, 'int:12'), (12.0, 'int:12'), (' 12 ', 'str:12'),
    (b'12', 'str:12'), (b'\xff', 'bytes:ff'), (np.array([12]), 'int:12')])
def test_id_wire_types(value, expected):
    assert normalize_identifier(value) == expected


def test_grouping_and_unknown_pool(tmp_path):
    # Golden value independently checked against the supplied PulseBound v6 code.
    assert group_hash('mimic', 'subject_id', 12) == '7dfbc27bc409e064e771afc1b026306ac399d42a167df88bf5b742b8a685eee2'
    manifest = make_index(tmp_path)
    index = read_manifest(manifest)
    assert index.counts == {'train': 73, 'val': 4, 'test': 4}
    assert split_counts(40) == {'train': 36, 'validation': 2, 'test': 2}
    known = [subject_ids(index, s) for s in ('train', 'val', 'test')]
    assert not (known[0] & known[1] or known[0] & known[2] or known[1] & known[2])
    assert index[-1].split == 'train' and not index[-1].subject_known
    groups = {group_hash('mimic', 'subject_id', i) for i in range(40)}
    assert assign_groups(reversed(sorted(groups)), 'mimic') == assign_groups(groups, 'mimic')
    train_index = index.subset('train')
    selected = calibration_indices(train_index, 1, 42)
    assert len(selected) == 36
    assert len({train_index[i].subject_id for i in selected}) == 36
    assert all(train_index[i].subject_known for i in selected)
    assert selected == calibration_indices(train_index, 1, 42)


def test_mixed_ecg_and_quality_reading_without_reprocessing(tmp_path):
    path = write_shard(tmp_path / 'shards' / 'part.h5', [-1] * 5)
    with h5py.File(path, 'r+') as h:
        h['has_ecg'][1] = False
        h['ecg_wave'][2] = np.nan
        h['ecg_wave'][3] = 0
        h['metrics'][4, 1] = np.nan
        expected = h['signals'][0, 25:75]
    spec = tmp_path / 'sources.json'
    spec.write_text(json.dumps({'sources': {'mimic': {'root': 'shards'}}}))
    build_h5_manifest(spec, tmp_path / 'index.json')
    records = read_manifest(tmp_path / 'index.json')
    cfg = DataConfig(ppg_fs=50, ecg_fs=50, scales_seconds=[1.0], window_mode='crop')
    ds = PairedDataset(records, cfg, augment=False)
    assert [ds[i]['ecg_valid'] for i in range(5)] == [True, False, False, False, True]
    np.testing.assert_array_equal(ds[0]['ppg'][0].numpy().ravel(), expected)
    assert ds[0]['ppg_quality'] == 0.8 and ds[4]['ppg_quality'] == 0
    copied = pickle.loads(pickle.dumps(ds))
    assert len(copied.h5_reader.handles) == 0
    assert copied[0]['ecg_valid']
    ds.close()
    copied.close()
    # Historical mixed shards may have has_ecg=True but no ECG dataset.
    with h5py.File(path, 'r+') as h:
        del h['ecg_wave']
    reader = PairedDataset(records, cfg, augment=False)
    assert not reader[0]['ecg_valid']
    reader.close()


def test_content_integrity_and_relocation(tmp_path):
    manifest = make_index(tmp_path)
    original = H5RecordIndex(manifest)
    shutil.copytree(tmp_path / 'mimic', tmp_path / 'relocated')
    relocated = H5RecordIndex(manifest, source_roots={'mimic': 'relocated'})
    assert original.digest == relocated.digest
    assert original[0].sample_id == relocated[0].sample_id
    with h5py.File(tmp_path / 'mimic' / 'part.h5', 'r+') as h:
        h['signals'][0, 0] += 10
    with pytest.raises(ValueError, match='hash changed'):
        H5RecordIndex(manifest)


def test_identifier_fallback_across_files_and_unique_relative_sample_id(tmp_path):
    for subdir, subject in [('a', 7), ('b', 8)]:
        path = write_shard(tmp_path / 'mimic' / subdir / 'part.h5', [subject, -1, -1])
        with h5py.File(path, 'r+') as h:
            h['case_id'] = np.array([100, 201, -1])
    spec = tmp_path / 'sources.json'
    spec.write_text(json.dumps({'sources': {'mimic': {'root': 'mimic', 'recursive': True}}}))
    build_h5_manifest(spec, tmp_path / 'index.json')
    index = H5RecordIndex(tmp_path / 'index.json')
    assert len({r.sample_id for r in index}) == 6
    assert index[1].subject_id == index[4].subject_id  # repeated case belongs to one split
    assert index[1].split == index[4].split
    assert index[0].subject_id.endswith(group_hash('mimic', 'subject_id', 7))


def test_imported_pulsebound_split_requires_exact_shards(tmp_path):
    manifest = make_index(tmp_path)
    payload = json.loads(manifest.read_text())
    spec = payload['sources']['mimic']
    groups = {name: [] for name in ('train', 'validation', 'test')}
    for group, split in spec['assignments'].items():
        groups['validation' if split == 'val' else split].append(group)
    files = []
    for entry in spec['files']:
        rows = {}
        for a, b, g in entry['intervals']:
            rows.setdefault(g or '__unknown__', []).append([a, b])
        files.append({**entry, 'group_rows': rows})
    pb = {'format': 'pulsebound_group_split_manifest', 'version': 1, 'salt': SALT,
          'split_ratios': {'train': 0.9, 'validation': 0.05, 'test': 0.05},
          'group_id_priority': ['subject_id', 'patient_id', 'case_id', 'record_id'],
          'sources': {'mimic': {'groups': groups, 'files': files}}}
    pb['manifest_sha256'] = canonical_hash(pb)
    reference = tmp_path / 'pulsebound_split.json'
    reference.write_text(json.dumps(pb))
    imported = build_h5_manifest(tmp_path / 'sources.json', tmp_path / 'imported.json', reference)
    assert imported['sources']['mimic']['assignments'] == spec['assignments']
    pb['sources']['mimic']['files'][0]['sha256'] = '0' * 64
    pb['manifest_sha256'] = canonical_hash(pb)
    reference.write_text(json.dumps(pb))
    with pytest.raises(ValueError, match='content differs'):
        build_h5_manifest(tmp_path / 'sources.json', tmp_path / 'bad.json', reference)


def test_exact_source_mix_no_replacement_and_epoch_seed(tmp_path):
    index = H5RecordIndex(make_index(tmp_path, sources=('mimic', 'vitaldb'))).subset('train')
    cfg = DataConfig(source_weights={'mimic': 0.8, 'vitaldb': 0.2}, shuffle_block_size=7)
    sampler = SourceBlockSampler(index, cfg, 5, 2, 42)
    order = list(sampler)
    assert len(order) == len(set(order)) == 90
    assert Counter(index[i].source for i in order) == {'mimic': 72, 'vitaldb': 18}
    assert list(sampler) == order
    sampler.epoch = 1
    assert list(sampler) != order
    with pytest.raises(ValueError, match='without oversampling'):
        source_counts({'mimic': 73, 'vitaldb': 73}, cfg.source_weights, requested=100, update_size=10)


def test_h5_spawn_workers_and_lru(tmp_path):
    manifest = make_index(tmp_path)
    cfg = DataConfig(ppg_fs=50, ecg_fs=50, scales_seconds=[1.0], max_open_shards=1,
                     window_mode='crop')
    dataset = PairedDataset(read_manifest(manifest), cfg, split='val', augment=False)
    # Parent opens a handle before spawn; workers must never inherit/pickle it.
    dataset[0]
    loader = DataLoader(dataset, batch_size=2, num_workers=2, multiprocessing_context='spawn')
    assert sum(len(b['sample_id']) for b in loader) == 4
    dataset.close()


def test_h5_complete_pipeline_and_resume(tmp_path, tiny_config):
    manifest = make_index(tmp_path, subjects=20, windows=1)
    tiny_config.data.ppg_fs = tiny_config.data.ecg_fs = 50
    tiny_config.data.scales_seconds = [0.4, 1.0]
    tiny_config.calibration.max_windows_per_subject = 1
    tiny_config.model.dropout = 0.1
    ppg = train(tiny_config, manifest, tmp_path / 'ppg')
    partial = train(tiny_config, manifest, tmp_path / 'resume', stop_after=1)
    resumed = train(tiny_config, manifest, tmp_path / 'resume', resume=partial)
    for k, value in load_checkpoint(ppg)['model'].items():
        torch.testing.assert_close(value, load_checkpoint(resumed)['model'][k], rtol=0, atol=0)
    ecg = train(tiny_config, manifest, tmp_path / 'ecg', modality='ecg', stop_after=1)
    calibration = tmp_path / 'calibration.json'
    report = calibrate(tiny_config, manifest, ppg, ecg, calibration)
    assert report['selection']['reliable_pairs'] == 18
    distill = train(tiny_config, manifest, tmp_path / 'distill', student_path=ppg,
                    teacher_path=ecg, calibration_path=calibration, stop_after=1)
    exported = export_encoder(distill, tmp_path / 'encoder.pt')
    features = extract_features(exported, manifest, tmp_path / 'features.npz')
    with np.load(features) as data:
        assert data['embedding'].shape == (1, 32)
        assert np.isfinite(data['embedding']).all()


def test_cli_index_and_sampled_audit(tmp_path, tiny_config, capsys):
    from tempobridge.cli import main
    make_index(tmp_path, subjects=20, windows=1)
    config = tmp_path / 'config.json'
    tiny_config.data.ppg_fs = tiny_config.data.ecg_fs = 50
    tiny_config.data.scales_seconds = [0.4, 1.0]
    config.write_text(json.dumps(tiny_config.to_dict()))
    main(['index-h5', '--sources', str(tmp_path / 'sources.json'),
          '--output', str(tmp_path / 'cli.json')])
    capsys.readouterr()
    main(['audit-data', '--config', str(config), '--manifest', str(tmp_path / 'cli.json'),
          '--max-windows', '2'])
    audit = json.loads(capsys.readouterr().out)
    assert audit['train']['checked_windows'] == 2
    assert audit['train']['samples'] == 19
    assert audit['test']['checked_windows'] == 1
