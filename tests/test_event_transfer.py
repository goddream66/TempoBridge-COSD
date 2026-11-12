import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from tempobridge.cli import main
from tempobridge.config import ExperimentConfig
from tempobridge.dynamic_transfer import select_rr_anchors
from tempobridge.event_evaluation import _summary, load_event_model, predict_events
from tempobridge.event_targets import (
    EventTable,
    adjacent_indices,
    beat_rr_targets,
    build_event_targets,
)
from tempobridge.inference import load_encoder
from tempobridge.utils import file_sha256, load_checkpoint


def test_rr_rejects_skipped_beats_and_unverified_timebase():
    settings = ExperimentConfig().event
    row = {"rpeaks_s": np.array([1., 2., 3., 4., 5.]), "rr_valid": np.ones(4, bool),
           "feet_s": np.array([1.2, 2.2, 4.2, 5.2]), "pairing_verified": True}
    values, valid = beat_rr_targets(row, settings)
    np.testing.assert_array_equal(valid, [True, False, True])
    np.testing.assert_allclose(values[valid], 1.)
    row["pairing_verified"] = False
    assert not beat_rr_targets(row, settings)[1].any()
    row.update(pairing_verified=True, feet_s=np.array([1.2, 1.3, 2.2, 3.2]))
    np.testing.assert_array_equal(beat_rr_targets(row, settings)[1], [False, False, True])


def test_dynamic_pairs_require_contiguity_identity_and_device():
    common = dict(subject_known=True, recording_known=True, source="mimic", subject_id="1",
                  recording_id="r", case_id="c", device_id="d")
    rows = [dict(common, window_start_s=start) for start in (0., 30., 59., 90., 120., 150., 180.)]
    rows[4]["device_id"] = "other"
    rows[6]["recording_id"] = "other"
    np.testing.assert_array_equal(adjacent_indices(rows, 30., .02), [1, -1, -1, -1, -1, -1, -1])


def test_predicted_anchors_do_not_bridge_extra_or_missing_pulses():
    settings = replace(ExperimentConfig().event, rr_anchor_mode="predicted")
    logits = torch.full((1, 300), -10.)
    logits[0, [51, 101, 125, 151, 251]] = 10
    feet = torch.tensor([[1., 2., 3., 4., 5.]])
    anchors, beats, rr, selected = select_rr_anchors(logits, feet, torch.ones_like(feet, dtype=torch.bool),
                                                    torch.ones(1, 4, dtype=torch.bool), 50, settings, 0)
    torch.testing.assert_close(anchors[0, :3], torch.tensor([1.02, 2.02, 3.02]))
    assert selected == 1 and beats.tolist() == [[True, True, True, False, True]]
    assert rr.tolist() == [[True, False, False, False]]


def test_rr_calibration_has_conditional_denominator():
    from collections import Counter

    values = Counter(rr_count=2, rr_available=4, rr_std_sum=.2, rr_nll=-1.)
    values["rr_hit/0.5"] = 1
    scores = _summary(values, [], True)["rr"]
    assert scores["coverage"] == .5
    assert scores["interval_calibration"][0]["observed"] == .5
    assert scores["interval_calibration"][0]["mean_width_ms"] > 100
    assert "interval_calibration" not in _summary(values, [], False)["rr"]


def _fixture(path, train_groups=2):
    fs, duration = 125, 30
    t = np.arange(fs * duration) / fs
    manifest, references = path / "manifest.csv", path / "refs.jsonl"
    with manifest.open("w", newline="", encoding="utf-8") as files, references.open("w", encoding="utf-8") as labels:
        writer = csv.DictWriter(files, fieldnames=["sample_id", "subject_id", "recording_id", "split",
                                                   "path", "window_start_s", "ppg_quality", "ecg_quality"])
        writer.writeheader()
        for person, split in enumerate(["train"] * train_groups + ["val", "test"]):
            for window in range(2):
                name = f"p{person}_w{window}"
                period = .83 + .12 * window + .015 * (person % 5)
                phase = (t - .2) / period
                ppg = -np.cos(2 * np.pi * phase)
                ecg = sum(np.exp(-.5 * ((t - r) / .012) ** 2) for r in np.arange(0., duration, period))
                wave = path / f"{name}.npz"
                np.savez(wave, ecg=ecg, ppg=ppg, ppg_fs=fs, ecg_fs=fs)
                writer.writerow(dict(sample_id=name, subject_id=str(person), recording_id=f"r{person}", split=split,
                                     path=wave.name, window_start_s=window * duration, ppg_quality=.9, ecg_quality=.9))
                labels.write(json.dumps(dict(sample_id=name, path=wave.name, duration_seconds=duration,
                                             source_id="default", subject_id=str(person), record_id=f"r{person}",
                                             case_id="", window_start_s=window * duration,
                                             window_end_s=(window + 1) * duration, ecg_native_fs=fs, ppg_native_fs=fs,
                                             ecg_channel="ECG", ppg_channel="PPG", native_timestamp_origin="synthetic t=0",
                                             clock_alignment_status="verified_common_clock", max_alignment_error_s=0.,
                                             clock_evidence="shared synthetic generator grid")) + "\n")
    return str(manifest), str(references)


def _write_annotation(path, reference, period):
    r = np.arange(0., 29., period)
    annotation = {"sample_id": reference["sample_id"], "annotation_source": "independent synthetic timings",
                  "reference_sha256": file_sha256(path / reference["path"]),
                  "rpeaks_s": r.tolist(), "feet_s": (r + .2).tolist(),
                  "peaks_s": (r + .2 + period / 2).tolist(),
                  "nn_valid": [True] * (len(r) - 2) + [False], "rhythm": "regular",
                  "pat_r_index": list(range(len(r)))}
    output = path / "annotations.jsonl"
    output.write_text(json.dumps(annotation), encoding="utf-8")
    return output


def test_v3_cli_beat_dynamic_resume_and_ppg_inference(tmp_path):
    pytest.importorskip("h5py")
    manifest, references = _fixture(tmp_path)
    config = ExperimentConfig.from_dict({"model": {"encoder": "local", "dim": 16, "depth": 2, "heads": 2,
                                                   "patch_size": 10, "protected_depth": 1},
                                         "data": {"window_mode": "full"},
                                         "train": {"epochs": 2, "threads": 1, "batch_size": 2,
                                                   "device": "cpu", "transfer_method": "none",
                                                   "gradient_audit_interval": 1},
                                         "event": {"hidden_dim": 8, "rmssd_reference": "allow_rr_proxy"}})
    a, b, latent = (str(tmp_path / name) for name in ("a.json", "b.json", "latent.json"))
    with open(a, "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle)
    with open(b, "w", encoding="utf-8") as handle:
        json.dump(replace(config, event=replace(config.event, dynamic_weight=1.)).to_dict(), handle)
    with open(latent, "w", encoding="utf-8") as handle:
        json.dump(replace(config, event=replace(config.event, dynamic_weight=1., dynamic_target="latent")).to_dict(),
                  handle)
    table = str(tmp_path / "events.h5")
    warm, beat, dynamic = (str(tmp_path / name) for name in ("warm", "beat", "dynamic"))
    main(["pretrain", "--manifest", manifest, "--config", a, "--modality", "ppg", "--output", warm])
    main(["event-targets", "--references", references, "--manifest", manifest, "--config", a, "--output", table])
    shared = ["--manifest", manifest, "--event-targets", table]
    a_run = ["train-events", *shared, "--config", a, "--student", f"{warm}/last.pt", "--output", beat]
    main([*a_run, "--stop-after", "1"])
    main([*a_run, "--resume", f"{beat}/last.pt"])
    b_run = ["train-events", *shared, "--config", b, "--student", f"{beat}/last.pt", "--output", dynamic]
    main(b_run)
    teacher, latent_run = str(tmp_path / "teacher"), str(tmp_path / "latent")
    main(["pretrain", "--manifest", manifest, "--config", a, "--modality", "ecg", "--output", teacher])
    main(["train-events", *shared, "--config", latent, "--student", f"{warm}/last.pt", "--output", latent_run,
          "--teacher", f"{teacher}/last.pt", "--stop-after", "1"])
    assert load_checkpoint(f"{latent_run}/last.pt")["event_dynamic_dim"] == 16
    assert load_checkpoint(f"{beat}/last.pt")["epoch"] == 1
    row = json.loads((tmp_path / "dynamic" / "metrics.jsonl").read_text().splitlines()[-1])
    assert row["v3_rr_labels_per_window"] > 0 and row["v3_dynamic_pair_fraction"] > 0
    assert row["v3_gradient_norm/dynamic/context.0"] > 0
    assert "v3_gradient_cosine/ssl/dynamic/backbone" in row
    report = str(tmp_path / "test.json")
    checkpoint = f"{dynamic}/last.pt"
    main(["evaluate-events", *shared, "--checkpoint", checkpoint, "--output", report])
    scores = json.loads((tmp_path / "test.json").read_text())["metrics"]
    assert scores["foot"]["reference"] > 0 and scores["rr"]["available"] > 0
    assert scores["delta_hr_bpm"]["pairs"] == 1 and scores["delta_rmssd_ms"]["pairs"] == 1
    refs = [json.loads(line) for line in (tmp_path / "refs.jsonl").read_text().splitlines()]
    annotations = _write_annotation(tmp_path, refs[-2], .875)
    manual = build_event_targets(references, manifest, config, tmp_path / "reviewed.h5", annotations)
    manual_report = tmp_path / "manual_test.json"
    main(["evaluate-events", "--manifest", manifest, "--event-targets", str(manual), "--checkpoint", checkpoint,
          "--output", str(manual_report), "--require-manual"])
    independent = json.loads(manual_report.read_text())
    assert independent["reference_kinds"] == {"manual": 1}
    assert independent["metrics"]["rr"]["available"] > 20
    encoder = tmp_path / "encoder.pt"
    main(["export", "--checkpoint", checkpoint, "--output", str(encoder)])
    signal = torch.zeros(1, 1, 1500)
    assert load_encoder(encoder)(signal).shape == (1, 32)
    model, heads, _ = load_event_model(checkpoint)
    assert predict_events(model, heads, signal, 50, threshold=.999)["rr_mean"].shape == (1, 0)


def test_clock_provenance_and_independent_audit(tmp_path):
    pytest.importorskip("h5py")
    from tempobridge.data import manifest_digest, read_manifest

    manifest, references = _fixture(tmp_path)
    refs = [json.loads(line) for line in (tmp_path / "refs.jsonl").read_text().splitlines()]
    cfg = ExperimentConfig()
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(dict(refs[0], window_start_s=1.)), encoding="utf-8")
    with pytest.raises(ValueError, match="window_start_s disagrees"):
        build_event_targets(bad, manifest, cfg, tmp_path / "wrong.h5")
    assert not (tmp_path / "wrong.h5").exists()
    legacy = dict(refs[0], pairing_verified=True)
    legacy.pop("clock_alignment_status")
    bad.write_text(json.dumps(legacy), encoding="utf-8")
    digest = manifest_digest(read_manifest(manifest))
    unsafe = EventTable(build_event_targets(bad, manifest, cfg, tmp_path / "unverified.h5"), digest, 30.)
    assert not unsafe.row(refs[0]["sample_id"])["pairing_verified"]
    unsafe.close()

    key = refs[0]["sample_id"]
    annotations = _write_annotation(tmp_path, refs[0], .83)
    native = build_event_targets(references, manifest, cfg, tmp_path / "native.h5")
    manual = build_event_targets(references, manifest, cfg, tmp_path / "manual.h5", annotations)
    annotated = EventTable(manual, digest, 30., manual_only=True)
    assert list(annotated.index) == [key]
    assert annotated.row(key)["rmssd_kind"] == "annotated_nn"
    assert not annotated.row(key)["nn_valid"][-1]
    annotated.close()
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    audit = tmp_path / "audit.json"
    main(["audit-events", "--manifest", manifest, "--config", str(config), "--event-targets", str(native),
          "--manual-targets", str(manual), "--output", str(audit)])
    report = json.loads(audit.read_text())["overall"]
    assert report["manual_windows"] == 1
    assert report["native_vs_manual_rpeak"]["matched"] > 20
    assert report["model_input_vs_native_foot"]["matched"] > 20
    assert report["rr_mae_ms"] is not None and report["pat_mae_ms"] is not None
