import argparse
import json

from .calibration import calibrate
from .config import ExperimentConfig
from .data import make_synthetic, read_manifest
from .inference import export_encoder, extract_features
from .training import train


def main(argv=None):
    parser = argparse.ArgumentParser(description="TempoBridge research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("synthetic", help="Generate nonclinical integration fixtures")
    p.add_argument("--output", required=True)
    p.add_argument("--subjects", type=int, default=18)
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--fs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("finetune", help="PPG-only full downstream fine-tuning or random-init baseline")
    for name in ("config", "manifest", "labels", "target", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--encoder", help="Exported pretrained encoder; omit for random initialization")
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--label-fraction", type=float, default=1.)
    p.add_argument("--bootstrap-samples", type=int, default=200)
    p.add_argument("--protocol", choices=["group_disjoint", "source_held_out"], default="group_disjoint")
    p.add_argument("--target-source")
    p = sub.add_parser("audit-data", help="Validate manifest identities and signal shapes")
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--max-windows", type=int, default=32,
                   help="Waveforms checked per split (0 = all); all shard hashes are verified by default")
    p = sub.add_parser("preprocess", help="Convert local MIMIC WFDB or VitalDB .vital to synchronized HDF5")
    p.add_argument("--kind", choices=["mimic", "vitaldb"], required=True)
    p.add_argument("--input-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source")
    p.add_argument("--id-map", help="CSV: record_path,subject_id,case_id")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-records", type=int)
    p.add_argument("--target-fs", type=int, default=50)
    p.add_argument("--window-seconds", type=float, default=30)
    p.add_argument("--quality-patch-seconds", type=float, default=1)
    p.add_argument("--max-missing-fraction", type=float, default=0.2)
    p.add_argument("--windows-per-shard", type=int, default=50000)
    p.add_argument("--max-windows-per-record", type=int, default=200, help="0 = unlimited")
    p.add_argument("--ppg-channel")
    p.add_argument("--ecg-channel")
    p = sub.add_parser("index-h5", help="Index existing shards with PulseBound source/group split protocol")
    p.add_argument("--sources", required=True, help="JSON source roots, patterns and optional sample rates")
    p.add_argument("--output", required=True)
    p.add_argument("--pulsebound-split", help="Verify/reuse an original group_split_manifest.json")
    p = sub.add_parser("reference-targets", help="Extract masked heuristic targets from independent native-rate reference windows")
    p.add_argument("--references", required=True, help="JSONL: sample_id,path,duration_seconds; NPZ with ECG/ABP/RESP")
    p.add_argument("--output", required=True, help="Long-format target CSV")
    for name in ("pretrain", "distill", "calibrate", "cache-features"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument("--output", required=True)
        if name == "pretrain":
            p.add_argument("--modality", choices=["ppg", "ecg"], required=True)
            p.add_argument("--student", help="Continue PPG-only SSL from the shared warm-up checkpoint")
            p.add_argument("--reference-targets", help="Independent target CSV; requires transfer_method=physio")
        else:
            p.add_argument("--student", required=True)
            p.add_argument("--teacher", required=True)
            p.add_argument("--teacher-kind", choices=["native", "torchscript"], default="native")
        if name == "distill":
            p.add_argument("--calibration", help="Required for PW-KD/COSD; vanilla computes target statistics only")
        if name == "calibrate":
            p.add_argument("--cache", help="Reuse a fingerprint-verified frozen-feature cache")
        if name in {"pretrain", "distill"}:
            p.add_argument("--resume")
            p.add_argument("--stop-after", type=int, help="Stop after this epoch; keep the configured schedule for exact resume")
    p = sub.add_parser("calibrate-cache", help="Calibrate from an existing verified training feature cache")
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("features")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p = sub.add_parser("evaluate-probe", help="Patient-disjoint frozen linear probe with validation-only selection")
    for name in ("train-features", "val-features", "test-features", "labels", "target", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--label-fraction", type=float, default=1.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--regularizations", type=float, nargs="+", default=[.1, 1., 10.])
    p.add_argument("--bootstrap-samples", type=int, default=200)
    p.add_argument("--protocol", choices=["group_disjoint", "source_held_out"], default="group_disjoint")
    p.add_argument("--target-source")
    p.add_argument("--steps", type=int, default=300, help="Classification probe optimizer steps")
    p = sub.add_parser("compare-predictions", help="Paired patient-cluster comparison on identical test support")
    p.add_argument("--first", required=True)
    p.add_argument("--second", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        result = make_synthetic(args.output, args.subjects, args.windows, args.seconds, args.fs, args.seed)
    elif args.command == "reference-targets":
        from .physio_targets import build_reference_targets
        result = build_reference_targets(args.references, args.output)
    elif args.command == "preprocess":
        from .preprocess import PreprocessConfig, preprocess_raw
        config = PreprocessConfig(**{name: getattr(args, name) for name in PreprocessConfig.__dataclass_fields__})
        result = json.dumps(preprocess_raw(args.kind, args.input_root, args.output, config,
                                          args.source, args.id_map, args.resume, args.max_records))
    elif args.command == "index-h5":
        from .h5data import build_h5_manifest
        result = build_h5_manifest(args.sources, args.output, args.pulsebound_split)
        result = json.dumps({"output": args.output, "sha256": result["manifest_sha256"],
                             "sources": list(result["sources"])})
    elif args.command == "audit-data":
        import random

        from .data import PairedDataset, split_size, subject_ids
        cfg = ExperimentConfig.load(args.config)
        if args.max_windows < 0:
            raise ValueError("max-windows must be nonnegative")
        records = read_manifest(args.manifest, data_config=cfg.data)
        counts = {}
        for split in ("train", "val", "test"):
            if split_size(records, split):
                dataset = PairedDataset(records, cfg.data, split=split, augment=False)
                indices = (range(len(dataset)) if not args.max_windows else
                           random.Random(cfg.train.seed).sample(range(len(dataset)),
                                                               min(args.max_windows, len(dataset))))
                missing = sum(not dataset[i]["ecg_valid"] for i in indices)
                counts[split] = {"samples": len(dataset), "checked_windows": len(indices),
                                 "missing_ecg_in_checked_windows": missing,
                                 "known_groups": len(subject_ids(records, split))}
                dataset.close()
        result = json.dumps(counts)
    elif args.command == "calibrate":
        artifact = calibrate(ExperimentConfig.load(args.config), args.manifest, args.student,
                             args.teacher, args.output, args.teacher_kind, args.cache)
        result = json.dumps({"output": args.output, "weights": [s["weights"] for s in artifact["scales"]],
                             "ranks": [s.get("subspace", {}).get("rank", artifact["target_dim"])
                                       for s in artifact["scales"]]})
    elif args.command == "calibrate-cache":
        from .calibration import calibrate_cached
        from .feature_cache import FeatureCache
        artifact = calibrate_cached(ExperimentConfig.load(args.config), FeatureCache(args.cache), args.output)
        result = json.dumps({"output": args.output, "method": artifact["method"]})
    elif args.command == "cache-features":
        from .feature_cache import cache_features
        result = cache_features(ExperimentConfig.load(args.config), args.manifest, args.student,
                                args.teacher, args.output, args.teacher_kind)
    elif args.command in {"pretrain", "distill"}:
        result = train(ExperimentConfig.load(args.config), args.manifest, args.output,
                       modality=getattr(args, "modality", "ppg"),
                       student_path=getattr(args, "student", None),
                       teacher_path=getattr(args, "teacher", None),
                       calibration_path=getattr(args, "calibration", None),
                       teacher_kind=getattr(args, "teacher_kind", "native"),
                       resume=args.resume, stop_after=args.stop_after,
                       reference_targets_path=getattr(args, "reference_targets", None))
    elif args.command == "export":
        result = export_encoder(args.checkpoint, args.output)
    elif args.command == "evaluate-probe":
        from .evaluation import evaluate_probe
        result = evaluate_probe(args.train_features, args.val_features, args.test_features, args.labels,
                                args.target, args.output, args.task, args.label_fraction, args.seed,
                                args.regularizations, args.bootstrap_samples, args.protocol,
                                args.target_source, args.steps)
    elif args.command == "compare-predictions":
        from .evaluation import compare_predictions
        result = compare_predictions(args.first, args.second, args.output, args.bootstrap_samples, args.seed)
    elif args.command == "finetune":
        from .finetuning import fine_tune
        result = fine_tune(ExperimentConfig.load(args.config), args.manifest, args.labels, args.target,
                           args.output, args.encoder, args.task, args.label_fraction, args.bootstrap_samples,
                           args.protocol, args.target_source)
    else:
        result = extract_features(args.checkpoint, args.manifest, args.output,
                                  args.split, args.device, args.batch_size)
    print(result, flush=True)


if __name__ == "__main__":
    main()
