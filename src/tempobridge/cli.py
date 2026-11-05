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
    p = sub.add_parser("audit-data", help="Validate manifest identities and signal shapes")
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    for name in ("pretrain", "distill", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument("--output", required=True)
        if name == "pretrain":
            p.add_argument("--modality", choices=["ppg", "ecg"], required=True)
        else:
            p.add_argument("--student", required=True)
            p.add_argument("--teacher", required=True)
            p.add_argument("--teacher-kind", choices=["native", "torchscript"], default="native")
        if name == "distill":
            p.add_argument("--calibration", required=True)
        if name != "calibrate":
            p.add_argument("--resume")
            p.add_argument("--stop-after", type=int, help="Stop after this epoch; keep the configured schedule for exact resume")
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
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        result = make_synthetic(args.output, args.subjects, args.windows, args.seconds, args.fs, args.seed)
    elif args.command == "audit-data":
        from .data import PairedDataset
        cfg = ExperimentConfig.load(args.config)
        records = read_manifest(args.manifest)
        counts = {}
        for split in ("train", "val", "test"):
            if any(r.split == split for r in records):
                dataset = PairedDataset(records, cfg.data, split=split, augment=False)
                missing = sum(not dataset[i]["ecg_valid"] for i in range(len(dataset)))
                counts[split] = {"samples": len(dataset), "missing_ecg": missing,
                                 "subjects": len({r.subject_id for r in dataset.records})}
        result = json.dumps(counts)
    elif args.command == "calibrate":
        artifact = calibrate(ExperimentConfig.load(args.config), args.manifest, args.student,
                             args.teacher, args.output, args.teacher_kind)
        result = json.dumps({"output": args.output, "weights": [s["weights"] for s in artifact["scales"]]})
    elif args.command in {"pretrain", "distill"}:
        result = train(ExperimentConfig.load(args.config), args.manifest, args.output,
                       modality=getattr(args, "modality", "ppg"),
                       student_path=getattr(args, "student", None),
                       teacher_path=getattr(args, "teacher", None),
                       calibration_path=getattr(args, "calibration", None),
                       teacher_kind=getattr(args, "teacher_kind", "native"),
                       resume=args.resume, stop_after=args.stop_after)
    elif args.command == "export":
        result = export_encoder(args.checkpoint, args.output)
    else:
        result = extract_features(args.checkpoint, args.manifest, args.output,
                                  args.split, args.device, args.batch_size)
    print(result, flush=True)


if __name__ == "__main__":
    main()
