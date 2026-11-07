import numpy as np
import pytest
import torch

from tempobridge.config import DataConfig, TrainConfig
from tempobridge.controls import PairingControlDataset, pairing_map
from tempobridge.data import PairedDataset, Record
from tempobridge.sampling import PatientSourceSampler


def rows():
    return [{"sample_id": str(i), "subject_id": f"p{i // 4}", "subject_known": True,
             "recording_id": f"r{i // 4}", "recording_known": True, "ecg_available": True,
             "source": "a" if i < 8 else "b", "ppg_quality": .9, "window_start_s": (i % 4) * 30.}
            for i in range(16)]


@pytest.mark.parametrize("mode", ["within_record", "cross_patient", "cross_source", "time_shift"])
def test_pairing_invariants_and_seed(mode):
    metadata = rows()
    donors = pairing_map(metadata, mode, 23, shift_seconds=30)
    np.testing.assert_array_equal(donors, pairing_map(metadata, mode, 23))
    for i, donor in enumerate(donors):
        if donor < 0:
            continue
        a, b = metadata[i], metadata[donor]
        assert i != donor
        if mode in {"within_record", "time_shift"}:
            assert a["recording_id"] == b["recording_id"]
        elif mode == "cross_patient":
            assert a["subject_id"] != b["subject_id"] and a["source"] == b["source"]
        else:
            assert a["source"] != b["source"]
        if mode == "time_shift":
            assert b["window_start_s"] - a["window_start_s"] == 30


def test_unknown_record_or_time_never_fabricated():
    metadata = rows()
    for r in metadata:
        r["recording_known"] = False
    assert (pairing_map(metadata, "within_record", 1) == -1).all()
    metadata = rows()
    for r in metadata:
        r["window_start_s"] = -1
    assert (pairing_map(metadata, "time_shift", 1) == -1).all()


def test_matched_correct_pairs_have_identical_coverage(tmp_path):
    metadata, records = rows(), []
    for i, r in enumerate(metadata):
        path = tmp_path / f"{i}.npz"
        np.savez(path, ppg=np.arange(100, dtype="f4"), ppg_fs=50,
                  ecg=np.arange(100, dtype="f4") + i * 1000, ecg_fs=50)
        records.append(Record(str(i), r["subject_id"], r["recording_id"], "train", path, .9, .3 + .03 * i,
                              source=r["source"], window_start_s=r["window_start_s"]))
    cfg = DataConfig(scales_seconds=[2.])
    wrong = PairingControlDataset(PairedDataset(records, cfg), TrainConfig(pairing_control="time_shift"), [])
    correct = PairingControlDataset(PairedDataset(records, cfg),
                                    TrainConfig(pairing_control="time_shift", pairing_apply=False), [])
    for i in range(len(records)):
        a, b = wrong[i], correct[i]
        assert a["ecg_valid"] == b["ecg_valid"] == (i % 4 != 3)
        assert a["ecg_quality"] == b["ecg_quality"]
        torch.testing.assert_close(a["ppg"][0], b["ppg"][0])
        if a["ecg_valid"]:
            assert a["ecg"][0][0, 0] == b["ecg"][0][0, 0] + 1000
    assert wrong.report()["mapping_sha256"] == correct.report()["mapping_sha256"]
    wrong.close()
    correct.close()


def test_patient_sampler_balances_groups_despite_record_lengths():
    class MetadataOnly:
        def __init__(self):
            self.rows = []
            for source in ("a", "b"):
                for patient, count in (("long", 50), ("short", 2)):
                    self.rows.extend({"source": source, "subject_id": patient, "subject_known": True,
                                      "recording_id": "r", "recording_known": True} for _ in range(count))
        def __len__(self):
            return len(self.rows)
        def metadata(self, i):
            return self.rows[i]

    dataset = MetadataOnly()
    cfg = DataConfig(source_weights={"a": .8, "b": .2}, samples_per_epoch=100, sampler="patient")
    sampler = PatientSourceSampler(dataset, cfg, 5, 1, 7)
    first = list(sampler)
    assert sampler.last_report["source_counts"] == {"a": 80, "b": 20}
    assert sampler.last_report["group_counts"] == {"a": {"long": 40, "short": 40}, "b": {"long": 10, "short": 10}}
    assert sampler.last_report["repeated_windows"] > 0
    assert first == list(sampler)
    sampler.epoch = 1
    assert first != list(sampler)


def test_pairing_and_patient_sampler_resume(tmp_path, tiny_config):
    import json

    from tempobridge.data import make_synthetic
    from tempobridge.training import train
    from tempobridge.utils import load_checkpoint

    cfg = tiny_config
    cfg.data.window_mode, cfg.data.scales_seconds = "full", [2.]
    cfg.data.sampler = "patient"
    manifest = make_synthetic(tmp_path / "data", subjects=8, windows=3, seconds=2)
    student = train(cfg, manifest, tmp_path / "student", stop_after=1)
    teacher = train(cfg, manifest, tmp_path / "teacher", modality="ecg", stop_after=1)
    cfg.train.transfer_method, cfg.train.transfer_mode = "vanilla", "quality"
    cfg.train.pairing_control, cfg.train.pairing_shift_seconds = "time_shift", 2.
    kwargs = dict(student_path=student, teacher_path=teacher)
    full = train(cfg, manifest, tmp_path / "full", **kwargs)
    partial = train(cfg, manifest, tmp_path / "partial", stop_after=1, **kwargs)
    resumed = train(cfg, manifest, tmp_path / "partial", resume=partial, **kwargs)
    a, b = load_checkpoint(full), load_checkpoint(resumed)
    for group in ("model", "heads"):
        for key in a[group]:
            torch.testing.assert_close(a[group][key], b[group][key], atol=0, rtol=0)
    report = json.loads((tmp_path / "full" / "pairing_epoch_0001.json").read_text())
    assert report["matched_ecg_reliability"] and report["metadata_eligible"] > 0
    assert (tmp_path / "full" / "sampling_epoch_0001.json").exists()
