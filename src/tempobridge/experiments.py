"""Reviewable experiment plans with resumable, content-verified Python stages.

Planning never trains. Execution is explicit and never invokes arbitrary shell code.
Runs share warm-up/teacher/cache only when their scientific input contracts match.
"""

import hashlib
import json
import math
import re
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from .config import ExperimentConfig
from .data import manifest_digest, read_manifest, source_ids, split_size
from .experiment_matrix import merge_config, variants_for
from .manifest_tools import filter_sources, relative_path
from .utils import device_for, file_sha256, load_checkpoint, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_digest():
    return digest({p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")})


def plan_experiments(config_path, manifest, output, seeds=(17, 29, 43), families=("core",),
                     labels=None, targets=(), label_fractions=(.01, .05, .1, .25, 1.),
                     teacher=None, teacher_kind="native", include_finetune=True, bootstrap_samples=200):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new experiment-plan directory")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("Independent seeds must be unique integers")
    if not label_fractions or any(not 0 < f <= 1 for f in label_fractions) or bootstrap_samples < 20:
        raise ValueError("Invalid evaluation fractions/bootstrap count")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", t["name"]) or t["task"] not in {"regression", "classification"} for t in targets):
        raise ValueError("Targets require safe names and regression/classification task types")
    if targets and not labels:
        raise ValueError("Evaluation targets require a label CSV")
    if teacher_kind not in {"native", "torchscript", "random_control"} or (teacher_kind != "native" and not teacher):
        raise ValueError("External/random teacher kind requires an explicit teacher artifact")
    config = ExperimentConfig.load(config_path)
    records = read_manifest(manifest, data_config=config.data)
    sources = sorted(source_ids(records, "train"))
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", s) for s in sources):
        raise ValueError("Experiment source names must be safe alphanumeric identifiers")
    base = config.to_dict()
    base["data"]["sampler"] = "patient"
    weights = config.data.source_weights or {s: 1 / len(sources) for s in sources}
    if set(weights) != set(sources):
        raise ValueError("Source weights must match manifest sources")
    base["data"]["source_weights"] = weights
    # All families share an exact optimizer-step/window budget, including source mixtures.
    ratios = [Fraction(str(w)).limit_denominator(10000) for w in weights.values()]
    total = sum(ratios)
    unit = math.lcm(config.train.batch_size * config.train.accumulation,
                    *((r / total).denominator for r in ratios), 10 if len(sources) == 2 else 1)
    budget = config.data.samples_per_epoch or split_size(records, "train") // unit * unit
    if budget < unit or budget % unit:
        raise ValueError(f"Plan samples_per_epoch must be a positive multiple of {unit}")
    base["data"]["samples_per_epoch"] = budget
    base["train"].update(checkpoint_selection="last", sample_subset="all")
    from .physio_targets import TargetTable
    has_references = bool(labels) and set(config.train.physio_targets).issubset(TargetTable(labels).units)
    from .teacher import FrozenTeacher
    frozen = FrozenTeacher(teacher, teacher_kind) if teacher else None
    dimension = frozen.target_dim if frozen else config.model.dim
    depth = frozen.encoder.config.depth if frozen and teacher_kind == "native" else config.model.depth
    variants, unavailable = variants_for(families, dimension, config.cosd.rank, sources, has_references,
                                         depth, teacher_kind == "native")
    for variant in variants:
        resolved = merge_config(base, variant["patch"])
        if variant.get("sources"):
            resolved["data"]["source_weights"] = {s: 1 / len(variant["sources"]) for s in variant["sources"]}
            resolved["data"]["source_roots"] = {}
        ExperimentConfig.from_dict(resolved)
        variant["config"] = resolved
        variant.pop("patch")
    root.mkdir(parents=True, exist_ok=True)
    plan = {"schema_version": 1, "code_sha256": code_digest(), "manifest": relative_path(manifest, root),
        "manifest_file_sha256": file_sha256(manifest), "manifest_sha256": manifest_digest(records),
        "base_config": base, "seeds": list(seeds),
        "families": list(families), "variants": variants, "unavailable": unavailable,
        "labels": relative_path(labels, root) if labels else None, "targets": list(targets),
        "labels_sha256": file_sha256(labels) if labels else None,
        "teacher": relative_path(teacher, root) if teacher else None, "teacher_kind": teacher_kind,
        "teacher_sha256": file_sha256(teacher) if teacher else None,
        "teacher_metadata_sha256": file_sha256(str(teacher) + ".json") if teacher and teacher_kind == "torchscript" else None,
        "evaluation": {"label_fractions": list(label_fractions), "finetune": include_finetune,
                       "bootstrap_samples": bootstrap_samples},
        "budget": {"warmup_epochs": config.train.epochs, "continuation_epochs": config.train.epochs,
                   "samples_per_epoch": budget, "effective_batch": config.train.batch_size * config.train.accumulation,
                   "continuation_optimizer_steps": config.train.epochs * budget // (config.train.batch_size * config.train.accumulation)},
        "scope": "engineering experiment plan; empirical Go criteria require real-data results"}
    write_json(root / "plan.json", plan)
    return root / "plan.json"


class ExperimentRunner:
    def __init__(self, plan_path):
        self.path = Path(plan_path).resolve()
        self.root = self.path.parent
        self.plan = json.loads(self.path.read_text(encoding="utf-8"))
        if self.plan.get("schema_version") != 1 or self.plan["code_sha256"] != code_digest():
            raise ValueError("Plan code version differs; generate a new plan for changed implementation")
        for field, hash_field in (("manifest", "manifest_file_sha256"), ("labels", "labels_sha256"), ("teacher", "teacher_sha256")):
            if self.plan.get(field) and file_sha256(self.input(self.plan[field])) != self.plan[hash_field]:
                raise ValueError(f"Plan input changed: {field}")
        records = read_manifest(self.input(self.plan["manifest"]),
                                data_config=ExperimentConfig.from_dict(self.plan["base_config"]).data)
        if manifest_digest(records) != self.plan["manifest_sha256"]:
            raise ValueError("Plan waveform content changed")
        if self.plan.get("teacher_metadata_sha256"):
            metadata = str(self.input(self.plan["teacher"])) + ".json"
            if file_sha256(metadata) != self.plan["teacher_metadata_sha256"]:
                raise ValueError("External teacher metadata changed")
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "plan_sha256": file_sha256(self.path), "jobs": {}, "variants": {}}
        if self.state["plan_sha256"] != file_sha256(self.path):
            raise ValueError("Plan changed after execution began")

    def input(self, value):
        return (self.root / value).resolve()

    def output(self, relative):
        path = (self.root / "artifacts" / relative).resolve()
        if not path.is_relative_to(self.root / "artifacts"):
            raise ValueError("Experiment output escapes artifact directory")
        return path

    def job(self, name, signature, action):
        identity = digest(signature)
        previous = self.state["jobs"].get(name)
        if previous and previous["signature"] != identity:
            raise ValueError(f"Job input changed: {name}")
        if previous and previous["status"] == "complete":
            if previous["signature"] != identity:
                raise ValueError(f"Completed job input changed: {name}")
            for path, checksum in previous["outputs"].items():
                if file_sha256(self.input(path)) != checksum:
                    raise ValueError(f"Completed job output changed: {name}")
            return [self.input(p) for p in previous["outputs"]]
        self.state["jobs"][name] = {"status": "running", "signature": identity}
        write_json(self.state_path, self.state)
        try:
            from .profiling import StageResources
            with StageResources() as resources:
                outputs = action()
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            fingerprints = {relative_path(p, self.root): file_sha256(p) for p in outputs}
            self.state["jobs"][name] = {"status": "complete", "signature": identity,
                                        "outputs": fingerprints, "resources": resources.report}
        except Exception as error:
            self.state["jobs"][name].update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(self.state_path, self.state)
            raise
        write_json(self.state_path, self.state)
        return outputs

    def fresh_job(self, name, signature, paths, action):
        """Keep interrupted non-resumable products before retrying a verified stage."""
        def run():
            from uuid import uuid4
            for path in paths:
                path = Path(path).resolve()
                if not path.is_relative_to(self.root / "artifacts"):
                    raise ValueError("Retry path escapes the experiment artifacts")
                if path.exists():
                    backup = self.output(f"interrupted/{uuid4().hex}/{path.name}")
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(backup)
            result = action()
            if len(paths) == 1 and Path(paths[0]).is_dir():
                return sorted(p for p in Path(paths[0]).rglob("*") if p.is_file())
            return result
        return self.job(name, signature, run)

    def training_job(self, name, config, manifest, **kwargs):
        from .training import train
        directory = self.output(name)
        signature = {"config": config.to_dict(), "manifest": file_sha256(manifest),
                     "arguments": {k: file_sha256(v) if k.endswith("_path") and v else v for k, v in kwargs.items()}}

        def action():
            checkpoint = directory / "last.pt"
            train(config, manifest, directory, resume=checkpoint if checkpoint.exists() else None, **kwargs)
            return [checkpoint, directory / "metrics.jsonl", directory / "provenance.json"]
        self.job(name, signature, action)
        return directory / "last.pt"

    def source_view(self, sources):
        original = self.input(self.plan["manifest"])
        name = "source_views/" + "_".join(sources) + original.suffix
        path = self.output(name)
        self.fresh_job(name, {"manifest": file_sha256(original), "sources": sources}, [path],
                 lambda: filter_sources(original, path, sources, ExperimentConfig.from_dict(self.plan["base_config"]).data))
        return path

    def run_variant(self, seed, variant):
        from .calibration import CalibrationWeights, calibrate_cached
        from .feature_cache import FeatureCache, cache_features
        from .inference import export_encoder
        from .representation_audit import audit_representations
        from .teacher import FrozenTeacher
        cfg = ExperimentConfig.from_dict(variant["config"])
        cfg.train.seed = seed
        manifest = self.source_view(variant["sources"]) if variant.get("sources") else self.input(self.plan["manifest"])
        initial = cfg.to_dict()
        initial["train"].update(transfer_method="none", transfer_mode="none", ssl_weight=1., pairing_control="none")
        # Calibration/objective knobs do not alter the common warm-up identity.
        initial["calibration"] = self.plan["base_config"]["calibration"]
        initial["cosd"] = self.plan["base_config"]["cosd"] | {"ranks_per_scale": []}
        warm_cfg = ExperimentConfig.from_dict(initial)
        warm_key = f"seed_{seed}/warmup_{digest({'model': initial['model'], 'data': initial['data'], 'subset': initial['train']['sample_subset']})[:12]}"
        # Use the SAME warm-up configuration for all members of the group.
        warm_cfg.train = ExperimentConfig.from_dict(self.plan["base_config"]).train
        warm_cfg.train.seed = seed
        warm_cfg.train.transfer_method, warm_cfg.train.transfer_mode = "none", "none"
        warm_cfg.train.sample_subset = cfg.train.sample_subset
        warm_cfg.train.pairing_control, warm_cfg.train.ssl_weight = "none", 1.
        warm_cfg.train.normalize_direction_weights = False
        student = self.training_job(warm_key, warm_cfg, manifest, modality="ppg")
        name = f"seed_{seed}/{variant['name']}"
        config_path = self.output(name + "/resolved_config.json")
        calibration, teacher, cache = None, None, None
        teacher_kind = self.plan["teacher_kind"]
        if cfg.train.transfer_method not in {"none", "physio"}:
            if self.plan["teacher"]:
                teacher = self.input(self.plan["teacher"])
            else:
                teacher_cfg = ExperimentConfig.from_dict(self.plan["base_config"])
                teacher_cfg.train.seed = seed
                teacher_cfg.train.transfer_method, teacher_cfg.train.transfer_mode = "none", "none"
                teacher_cfg.train.sample_subset, teacher_cfg.train.pairing_control = "all", "none"
                teacher_cfg.data = warm_cfg.data
                teacher_key = f"seed_{seed}/teacher_{digest({'data': asdict(teacher_cfg.data), 'manifest': file_sha256(manifest)})[:12]}"
                teacher = self.training_job(teacher_key, teacher_cfg, manifest, modality="ecg")
            if variant.get("teacher_target"):
                from .teacher_controls import teacher_variant
                original_teacher = teacher
                options = dict(variant["teacher_target"])
                if options.pop("random", False):
                    options["random_seed"] = seed + 100003
                    teacher_kind = "random_control"
                teacher = self.output(name + "/teacher.pt")
                self.fresh_job(name + "/teacher_variant", {"source": file_sha256(original_teacher), "options": options},
                               [teacher], lambda: teacher_variant(original_teacher, teacher, **options))
            cache_key = f"seed_{seed}/cache_{digest({'student': file_sha256(student), 'teacher': file_sha256(teacher), 'selection': asdict(cfg.calibration)})[:12]}"
            cache = self.output(cache_key)
            teacher_fingerprint = FrozenTeacher(teacher, teacher_kind).fingerprint

            def make_cache():
                if not (cache / "manifest.json").exists():
                    if cache.exists():
                        raise ValueError(f"Incomplete cache at {cache}; preserve it and choose a new plan output")
                    cache_features(cfg, manifest, student, teacher, cache, teacher_kind)
                verified = FeatureCache(cache)
                verified.validate(cfg, load_checkpoint(student)["manifest_sha256"],
                    teacher_fingerprint, file_sha256(student), teacher_kind)
                return [cache / "manifest.json", *(cache / p for p in verified.artifact["files"])]
            self.fresh_job(cache_key, {"student": file_sha256(student), "teacher": teacher_fingerprint,
                                "selection": asdict(cfg.calibration), "kind": teacher_kind}, [cache], make_cache)
            if variant.get("match_rank_to"):
                reference = self.output(f"seed_{seed}/{variant['match_rank_to']}/calibration.json")
                report = json.loads(reference.read_text())
                cfg.cosd.ranks_per_scale = [s["subspace"]["rank"] for s in report["scales"]]
            calibration = self.output(name + "/calibration.json")

            def calibrate_job():
                verified = FeatureCache(cache)
                if calibration.exists():
                    CalibrationWeights.load(calibration).validate(cfg, verified.artifact["manifest_sha256"],
                        verified.artifact["teacher_sha256"], file_sha256(student), teacher_kind)
                else:
                    calibrate_cached(cfg, verified, calibration)
                return calibration
            self.job(name + "/calibration", {"config": cfg.to_dict(), "cache": file_sha256(cache / "manifest.json")}, calibrate_job)
            if variant.get("match_rank_to") and CalibrationWeights.load(calibration).output_dims != cfg.cosd.ranks_per_scale:
                raise ValueError("Control cannot achieve the reference rank with nonconstant directions; revise the declared rank")
        write_json(config_path, cfg.to_dict())
        arguments = {"modality": "ppg", "student_path": student}
        if teacher:
            arguments.update(teacher_path=teacher, calibration_path=calibration, teacher_kind=teacher_kind)
        if cfg.train.transfer_method == "physio":
            arguments["reference_targets_path"] = self.input(self.plan["labels"])
        checkpoint = self.training_job(name + "/train", cfg, manifest, **arguments)
        exported = self.output(name + "/encoder.pt")
        self.fresh_job(name + "/export", {"checkpoint": file_sha256(checkpoint)}, [exported],
                       lambda: export_encoder(checkpoint, exported))
        if cfg.train.transfer_method == "cosd":
            audit = self.output(name + "/representation_audit.json")
            self.job(name + "/audit", {"calibration": file_sha256(calibration), "cache": file_sha256(cache / "manifest.json")},
                     lambda: audit_representations(cache, calibration, audit, seed=seed))
        metrics = [json.loads(line) for line in checkpoint.with_name("metrics.jsonl").read_text().splitlines()]
        if (len(metrics) != cfg.train.epochs or any(row["examples"] != self.plan["budget"]["samples_per_epoch"] for row in metrics)
                or metrics[-1]["global_step"] != self.plan["budget"]["continuation_optimizer_steps"]):
            raise ValueError(f"Training budget mismatch: {name}")
        summary = {"seed": seed, "variant": variant["name"], "family": variant["family"],
                   "diagnostic_only": variant.get("diagnostic_only", False) or teacher_kind == "random_control",
                   "encoder": relative_path(exported, self.root), "resolved_config": relative_path(config_path, self.root),
                   "optimizer_steps": metrics[-1]["global_step"], "windows": sum(r["examples"] for r in metrics),
                   "paired_windows": sum(r["paired_windows"] for r in metrics),
                   "active_transfer_windows": sum(r["active_transfer_windows"] for r in metrics),
                   "transfer_weight_sum": sum(r["transfer_weight_sum"] for r in metrics),
                   "actual_ranks": CalibrationWeights.load(calibration).output_dims if calibration else [],
                   "evaluations": []}
        if self.plan["labels"] and self.plan["targets"]:
            summary["evaluations"] = self.evaluate(name, cfg, manifest, exported, variant)
        self.state["variants"][name] = summary
        write_json(self.state_path, self.state)
        return summary

    def evaluate(self, name, config, manifest, encoder, variant):
        from .evaluation import evaluate_probe
        from .finetuning import fine_tune
        from .inference import extract_features
        labels = self.input(self.plan["labels"])
        options, results = self.plan["evaluation"], []
        for target_source in [None, *variant.get("source_held_out", [])]:
            test_manifest = self.source_view([target_source]) if target_source else manifest
            cohort = target_source or "within_source"
            features = {}
            for split in ("train", "val", "test"):
                source = test_manifest if split == "test" else manifest
                path = self.output(name + f"/features_{cohort}_{split}.npz")
                self.fresh_job(name + f"/features/{cohort}/{split}", {"encoder": file_sha256(encoder), "manifest": file_sha256(source), "split": split}, [path],
                         lambda source=source, split=split, path=path: extract_features(encoder, source, path, split,
                             device_for(config.train.device), config.train.batch_size))
                features[split] = path
            for target in self.plan["targets"]:
                for fraction in options["label_fractions"]:
                    prefix = name + f"/evaluation/{cohort}/{target['name']}/fraction_{fraction:g}"
                    protocol = "source_held_out" if target_source else "group_disjoint"
                    common = {"target": target, "fraction": fraction, "labels": file_sha256(labels),
                              "features": {s: file_sha256(p) for s, p in features.items()},
                              "encoder": file_sha256(encoder), "protocol": protocol, "seed": config.train.seed}
                    directory = self.output(prefix + "/probe")
                    self.fresh_job(prefix + "/probe", common, [directory],
                        lambda: evaluate_probe(features["train"], features["val"], features["test"], labels,
                            target["name"], directory, target["task"], fraction, config.train.seed,
                            bootstrap_samples=options["bootstrap_samples"], protocol=protocol, target_source=target_source))
                    results.append(relative_path(directory / "report.json", self.root))
                    if options["finetune"]:
                        for initialization in ("pretrained", "random"):
                            directory = self.output(prefix + f"/finetune_{initialization}")
                            initial = encoder if initialization == "pretrained" else None
                            self.fresh_job(prefix + f"/finetune_{initialization}", common | {"initialization": initialization}, [directory],
                                lambda: fine_tune(config, manifest, labels, target["name"], directory, initial,
                                    target["task"], fraction, options["bootstrap_samples"], protocol, target_source,
                                    test_manifest=test_manifest))
                            results.append(relative_path(directory / "report.json", self.root))
        return results


def run_experiments(plan_path, execute=False, only=()):
    runner = ExperimentRunner(plan_path)
    by_name = {v["name"]: v for v in runner.plan["variants"]}
    selected = set(only) if only else set(by_name)
    if not selected.issubset(by_name):
        raise ValueError("Unknown requested variant")
    for name in list(selected):
        reference = by_name[name].get("match_rank_to")
        if reference:
            selected.add(reference)
    if not execute:
        return {"execute": False, "seeds": runner.plan["seeds"], "variants": sorted(selected), "budget": runner.plan["budget"]}
    for seed in runner.plan["seeds"]:
        for variant in runner.plan["variants"]:
            if variant["name"] in selected:
                print(f"EXPERIMENT seed={seed} variant={variant['name']}", flush=True)
                runner.run_variant(seed, variant)
    summaries = list(runner.state["variants"].values())
    comparisons = []
    for seed in runner.plan["seeds"]:
        def get(name):
            return runner.state["variants"].get(f"seed_{seed}/{name}")
        a, b = get("pw_kd"), get("cosd_pw")
        if a and b:
            equal = (a["active_transfer_windows"] == b["active_transfer_windows"] and
                     abs(a["transfer_weight_sum"] - b["transfer_weight_sum"]) < 1e-5 * max(1, a["transfer_weight_sum"]))
            comparisons.append({"seed": seed, "first": "pw_kd", "second": "cosd_pw", "matched_coverage_and_weight": equal,
                                "reason": None if equal else "inspect rank-zero abstention or effective supervision"})
    report = {"schema_version": 1, "complete": len(summaries) == len(by_name) * len(runner.plan["seeds"]),
              "executed_variants": summaries, "comparisons": comparisons,
              "unavailable": runner.plan["unavailable"], "scope": "execution evidence only; no automatic scientific Go decision"}
    from .experiment_reporting import summarize_evaluations
    report.update(summarize_evaluations(runner))
    write_json(runner.root / "summary.json", report)
    return runner.root / "summary.json"
