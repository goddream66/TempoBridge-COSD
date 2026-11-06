import json

import h5py
import numpy as np
import pytest

from tempobridge.preprocess import (
    PreprocessConfig,
    ShardWriter,
    clean_window,
    preprocess_raw,
    raw_windows,
)


def waves(fs=100, seconds=30):
    t = np.arange(fs * seconds) / fs
    return np.sin(2*np.pi*1.2*t) + 0.2*np.sin(2*np.pi*2.4*t), np.sin(2*np.pi*1.2*t + 0.5)


def wfdb_fixture(root, seconds=95):
    import wfdb
    root.mkdir(parents=True)
    p, e = waves(seconds=seconds)
    wfdb.wrsamp('p000012_segment', 100, ['mV', 'mV'], ['PLETH', 'II'],
                p_signal=np.stack([p, e], axis=1), fmt=['16', '16'], write_dir=str(root))
    (root / 'RECORDS').write_text('p000012_segment\n')


def test_cleaning_rejects_missing_before_interpolation_and_keeps_ppg():
    p, e = waves()
    bad = p.copy()
    bad[:601] = np.nan
    assert clean_window(bad, e, 100, PreprocessConfig())[1] == 'ppg_missing_fraction'
    bad = e.copy()
    bad[:800] = np.inf
    row, reason = clean_window(p, bad, 100, PreprocessConfig())
    assert reason is None and not row['has_ecg']
    assert np.isfinite(row['signals']).all() and np.count_nonzero(row['ecg_wave']) == 0
    assert row['signals'].shape == (1500,) and row['patch_sqi'].shape == (30,)
    assert row['signals'].min() == -1 and row['signals'].max() == 1
    assert clean_window(np.ones(3000), e, 100, PreprocessConfig())[1] == 'ppg_constant'
    assert clean_window(p[:2999], e, 100, PreprocessConfig())[1] == 'incomplete_window'


def test_wfdb_conversion_cursor_resume_and_no_duplicate_windows(tmp_path, monkeypatch):
    raw, out = tmp_path / 'raw', tmp_path / 'processed'
    wfdb_fixture(raw)
    cfg = PreprocessConfig(windows_per_shard=1)
    original = ShardWriter.add

    def interrupted(self, sample, cursor):
        if cursor[1] == 1:
            raise OSError('simulated disk full')
        return original(self, sample, cursor)

    monkeypatch.setattr(ShardWriter, 'add', interrupted)
    with pytest.raises(RuntimeError, match='Shard write failed'):
        preprocess_raw('mimic', raw, out, cfg)
    assert len(list(out.glob('*.h5'))) == 1
    monkeypatch.setattr(ShardWriter, 'add', original)
    report = preprocess_raw('mimic', raw, out, cfg, resume=True)
    assert report['existing_windows_on_resume'] == 1 and report['total_saved_windows'] == 3
    starts = []
    for path in sorted(out.glob('*.h5')):
        with h5py.File(path) as h:
            assert h['signals'].shape == h['ecg_wave'].shape == (1, 1500)
            assert int(h['subject_id'][0]) == 12 and bool(h['has_ecg'][0])
            starts.extend(h['window_start_s'][:].tolist())
    assert starts == [0, 30, 60]  # trailing 5 seconds dropped
    assert preprocess_raw('mimic', raw, out, cfg, resume=True) == report
    with pytest.raises(ValueError, match='input/config changed'):
        preprocess_raw('mimic', raw, out, PreprocessConfig(target_fs=100), resume=True)
    with pytest.raises(FileExistsError):
        preprocess_raw('mimic', raw, out, cfg)


def test_actual_vital_file_common_timeline_and_case_id(tmp_path):
    import vitaldb
    raw = tmp_path / 'vital'
    raw.mkdir()
    vf = vitaldb.VitalFile()
    vf.dtstart, vf.dtend = 1700000000.0, 1700000060.0
    p, _ = waves(fs=100, seconds=60)
    _, e = waves(fs=200, seconds=59)
    vf.add_track('SNUADC/PLETH', [{'dt': vf.dtstart, 'val': p.astype('f4')}], srate=100)
    # ECG begins one second later; this gap must remain at the common start.
    vf.add_track('SNUADC/ECG_II', [{'dt': vf.dtstart + 1, 'val': e.astype('f4')}], srate=200)
    path = raw / '00042.vital'
    assert vf.to_vital(str(path))
    first = next(raw_windows(path, 'vitaldb', PreprocessConfig()))
    assert first[3] == 200 and len(first[1]) == 6000
    assert np.isnan(first[2][:200]).all()
    assert np.isfinite(first[2][200:]).all()
    report = preprocess_raw('vitaldb', raw, tmp_path / 'converted')
    assert report['total_saved_windows'] == 2
    with h5py.File(next((tmp_path / 'converted').glob('*.h5'))) as h:
        assert h['case_id'][:].tolist() == [42, 42]
        assert h['subject_id'][:].tolist() == [-1, -1]
        assert h['has_ecg'][:].tolist() == [True, True]
        assert h['ecg_missing_fraction'][0] == pytest.approx(1/30)


def test_no_usable_data_cannot_resume_as_success(tmp_path, monkeypatch):
    raw, out = tmp_path / 'raw', tmp_path / 'processed'
    wfdb_fixture(raw, seconds=30)
    monkeypatch.setattr('tempobridge.preprocess.clean_window', lambda *args: (None, 'ppg_constant'))
    for resume in (False, True):
        with pytest.raises(ValueError, match='No usable windows'):
            preprocess_raw('mimic', raw, out, resume=resume)
    assert not json.loads((out / 'conversion_report.json').read_text())['complete']


def test_preprocess_cli_and_explicit_id_map(tmp_path, capsys):
    from tempobridge.cli import main
    raw, out = tmp_path / 'raw', tmp_path / 'processed'
    wfdb_fixture(raw, seconds=30)
    mapping = tmp_path / 'ids.csv'
    mapping.write_text('record_path,subject_id,case_id\np000012_segment.hea,567,123\n')
    main(['preprocess', '--kind', 'mimic', '--input-root', str(raw), '--output', str(out),
          '--id-map', str(mapping), '--max-records', '1'])
    report = json.loads(capsys.readouterr().out)
    assert report['total_saved_windows'] == 1
    with h5py.File(next(out.glob('*.h5'))) as h:
        assert int(h['subject_id'][0]) == 567
