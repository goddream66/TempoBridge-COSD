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
    p.add_argument("--test-manifest", help="Optional separate test corpus; original group assignments are retained")
    p = sub.add_parser("filter-sources", help="Create a corpus view without changing group splits")
    for name in ("manifest", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--config")
    p = sub.add_parser("plan-experiments", help="Write a reviewable ablation plan without training")
    for name in ("config", "manifest", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--families", nargs="+", default=["core"])
    p.add_argument("--labels")
    p.add_argument("--targets", nargs="*", default=[], help="name:regression or name:classification")
    p.add_argument("--label-fractions", type=float, nargs="+", default=[.01, .05, .1, .25, 1.])
    p.add_argument("--teacher")
    p.add_argument("--teacher-kind", choices=["native", "torchscript", "random_control"], default="native")
    p.add_argument("--skip-finetune", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=200)
    p = sub.add_parser("run-experiments", help="Inspect a plan; --execute explicitly runs its stages")
    p.add_argument("--plan", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--only", nargs="*", default=[])
    p = sub.add_parser("teacher-variant", help="Declare a frozen native teacher readout or an explicit random diagnostic")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--pool", choices=["context_mean", "summary", "combined"], default="context_mean")
    p.add_argument("--normalization", choices=["none", "l2"], default="none")
    p.add_argument("--random-seed", type=int, help="Create untrained negative-control weights; requires teacher-kind=random_control downstream")
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
    p = sub.add_parser("event-targets", help="Extract native-rate V3 beat references into HDF5")
    for name in ("references", "manifest", "config", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--annotations", help="Optional independent native event annotations (JSONL)")
    p = sub.add_parser("audit-events", help="Compare native references, model-input events and manual labels")
    for name in ("manifest", "event-targets", "config", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--manual-targets", help="Optional schema-2 HDF5 containing annotated events")
    p.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    p.add_argument("--max-per-source", type=int, default=500)
    p.add_argument("--tolerance-s", type=float, default=.08)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("train-events", help="Continue PPG SSL with V3 event and optional dynamics losses")
    for name in ("manifest", "config", "event-targets", "student", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--teacher", help="Frozen ECG checkpoint for latent dynamics only")
    p.add_argument("--teacher-kind", choices=["native", "torchscript", "random_control"], default="native")
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p = sub.add_parser("evaluate-events", help="Score PPG-only event predictions on held-out windows")
    for name in ("checkpoint", "manifest", "event-targets", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float)
    p.add_argument("--tolerance-s", type=float, default=.08)
    p.add_argument("--require-manual", action="store_true", help="Use only independently annotated windows")
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
            p.add_argument("--teacher-kind", choices=["native", "torchscript", "random_control"], default="native")
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
    p = sub.add_parser("audit-representation", help="Frozen teacher/student identity, source, quality and within-record audits")
    for name in ("cache", "calibration", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--labels")
    p.add_argument("--targets", nargs="*", default=[])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-identity-groups", type=int, default=256)
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
    p.add_argument("--noise-std", type=float, default=0., help="White Gaussian noise relative to each PPG window standard deviation")
    p.add_argument("--noise-seed", type=int, default=42)
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
    elif args.command == "filter-sources":
        from .manifest_tools import filter_sources
        result = filter_sources(args.manifest, args.output, args.sources,
                                ExperimentConfig.load(args.config).data if args.config else None)
    elif args.command == "teacher-variant":
        from .teacher_controls import teacher_variant
        result = teacher_variant(args.checkpoint, args.output, args.layer, args.pool, args.normalization, args.random_seed)
    elif args.command == "plan-experiments":
        from .experiments import plan_experiments
        if any(len(t.split(":")) != 2 for t in args.targets):
            parser.error("--targets expects name:regression or name:classification")
        targets = [dict(zip(("name", "task"), t.split(":"), strict=True)) for t in args.targets]
        result = plan_experiments(args.config, args.manifest, args.output, args.seeds, args.families,
                                 args.labels, targets, args.label_fractions, args.teacher,
                                 args.teacher_kind, not args.skip_finetune, args.bootstrap_samples)
    elif args.command == "run-experiments":
        from .experiments import run_experiments
        result = run_experiments(args.plan, args.execute, args.only)
        if isinstance(result, dict):
            result = json.dumps(result)
    elif args.command == "reference-targets":
        from .physio_targets import build_reference_targets
        result = build_reference_targets(args.references, args.output)
    elif args.command == "event-targets":
        from .event_targets import build_event_targets
        result = build_event_targets(args.references, args.manifest, ExperimentConfig.load(args.config), args.output,
                                     args.annotations)
    elif args.command == "audit-events":
        from .event_audit import audit_events
        result = audit_events(args.manifest, args.event_targets, ExperimentConfig.load(args.config), args.output,
                              args.manual_targets, args.split, args.max_per_source, args.tolerance_s, args.seed)
    elif args.command == "train-events":
        result = train(ExperimentConfig.load(args.config), args.manifest, args.output, student_path=args.student,
                       teacher_path=args.teacher, teacher_kind=args.teacher_kind, resume=args.resume,
                       stop_after=args.stop_after, event_targets_path=args.event_targets)
    elif args.command == "evaluate-events":
        from .event_evaluation import evaluate_events
        result = evaluate_events(args.checkpoint, args.manifest, args.event_targets, args.output,
                                 args.split, args.device, args.threshold, args.tolerance_s, args.require_manual)
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
        from collections import Counter

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
                alignment = Counter(dataset.metadata(i)["timestamp_alignment_status"] for i in indices)
                counts[split] = {"samples": len(dataset), "checked_windows": len(indices),
                                 "missing_ecg_in_checked_windows": missing,
                                 "timestamp_alignment_in_checked_windows": dict(alignment),
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
    elif args.command == "audit-representation":
        from .representation_audit import audit_representations
        result = audit_representations(args.cache, args.calibration, args.output, args.labels,
                                       args.targets, args.seed, args.max_identity_groups)
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
                           args.protocol, args.target_source, args.test_manifest)
    else:
        result = extract_features(args.checkpoint, args.manifest, args.output,
                                  args.split, args.device, args.batch_size, args.noise_std, args.noise_seed)
    print(result, flush=True)


if __name__ == "__main__":
    main()
