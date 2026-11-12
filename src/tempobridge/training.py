"""Staged self-supervision and selective ECG transfer with epoch-exact resume."""

import contextlib
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .calibration import CalibrationWeights
from .config import DataConfig, ExperimentConfig, ModelConfig
from .data import PairedDataset, manifest_digest, read_manifest, source_ids, split_size, subject_ids
from .model import TempoBridge, TransferHeads, random_mask, reconstruction_loss
from .sampling import PatientSourceSampler, SourceBlockSampler
from .teacher import FrozenTeacher
from .transfer import distillation_loss, gradient_audit, parameter_group, physiological_loss
from .utils import (
    device_for,
    environment,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    seed_all,
    write_json,
)


def autocast_context(config, device):
    if config.train.precision == "fp32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed precision is supported on CUDA; use fp32 for CPU")
    dtype = torch.bfloat16 if config.train.precision == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Device does not support bf16")
    return torch.autocast("cuda", dtype=dtype)


def transfer_loss(model, heads, teacher, calibrator, ppg, ecg, valid,
                  ppg_quality, ecg_quality, scale_index, mode, protect_shallow):
    """Compatibility wrapper for the historical loss API."""
    loss, metrics = distillation_loss(model, heads, teacher, calibrator, ppg, ecg, valid,
                                     ppg_quality, ecg_quality, scale_index, mode, protect_shallow)
    return loss, ppg.new_tensor(metrics["mean_transfer_weight"])


def batch_loss(config, model, batch, device, modality, heads=None, teacher=None, calibrator=None,
               audit_gradients=False):
    ssl, kd, transfer_metrics = [], [], []
    if modality == "ecg":
        keep = batch["ecg_valid"]
    else:
        keep = torch.ones_like(batch["ecg_valid"])
    for index in range(len(config.data.scales_seconds)):
        wave = batch[modality][index][keep].to(device)
        mask = random_mask(wave, model.config.patch_size, config.train.mask_ratio)
        ssl.append(reconstruction_loss(model, wave, mask, config.train.morphology_loss_weight))
        if heads is not None and config.train.transfer_mode != "none" and config.train.transfer_method != "none":
            if config.train.transfer_method == "physio":
                loss, terms = physiological_loss(model, heads, calibrator, batch["ppg"][index].to(device),
                    batch["reference_values"].to(device), batch["reference_valid"].to(device),
                    index, config.train.protect_shallow)
            else:
                loss, terms = distillation_loss(
                    model, heads, teacher, calibrator,
                    batch["ppg"][index].to(device), batch["ecg"][index].to(device),
                    batch["ecg_valid"].to(device), batch["ppg_quality"].to(device),
                    batch["ecg_quality"].to(device), index,
                    config.train.transfer_mode, config.train.protect_shallow,
                    config.train.ppg_quality_weight, config.train.ecg_quality_weight,
                    config.train.normalize_direction_weights,
                    sources=batch["source"],
                )
            kd.append(loss)
            transfer_metrics.append(terms)
    self_loss = torch.stack(ssl).mean()
    distillation = torch.stack(kd).mean() if kd else self_loss.new_zeros(())
    total = config.train.ssl_weight * self_loss + config.train.transfer_weight * distillation
    names = ("mean_transfer_weight", "paired_fraction", "active_transfer_fraction", "target_rank")
    metrics = {name: sum(m[name] for m in transfer_metrics) / len(transfer_metrics)
               if transfer_metrics else 0. for name in names}
    for name in set().union(*(m.keys() for m in transfer_metrics)) - set(names):
        metrics[name] = sum(m.get(name, 0.) for m in transfer_metrics) / len(transfer_metrics)
    metrics["paired_fraction"] = float(batch["ecg_valid"][keep].float().mean())
    metrics["original_paired_fraction"] = float(batch.get("original_ecg_valid", batch["ecg_valid"])[keep].float().mean())
    if audit_gradients:
        metrics.update(gradient_audit(config.train.ssl_weight * self_loss, distillation, model, config.train.transfer_weight))
    return total, {"ssl": float(self_loss.detach()), "transfer": float(distillation.detach()),
                   **metrics}


@torch.inference_mode()
def evaluate_reconstruction(config, model, records, device, modality="ppg"):
    if not split_size(records, "val"):
        return None
    dataset = PairedDataset(records, config.data, split="val", seed=config.train.seed, augment=False)
    loader = DataLoader(dataset, batch_size=config.train.batch_size, num_workers=config.data.workers)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.train.seed + 10000)
    total, count = 0.0, 0
    for batch in loader:
        keep = batch["ecg_valid"] if modality == "ecg" else torch.ones_like(batch["ecg_valid"])
        if not keep.any():
            continue
        losses = []
        for index in range(len(config.data.scales_seconds)):
            wave = batch[modality][index][keep].to(device)
            mask = random_mask(wave, model.config.patch_size, config.train.mask_ratio, generator)
            losses.append(reconstruction_loss(model, wave, mask, config.train.morphology_loss_weight))
        total += float(torch.stack(losses).mean()) * int(keep.sum())
        count += int(keep.sum())
    dataset.close()
    return total / count if count else None


def train(config, manifest, output, modality="ppg", student_path=None, teacher_path=None,
          calibration_path=None, teacher_kind="native", resume=None, stop_after=None,
          reference_targets_path=None, event_targets_path=None):
    config.validate()
    seed_all(config.train.seed, config.train.threads)
    device = device_for(config.train.device)
    records = read_manifest(manifest, data_config=config.data)
    digest = manifest_digest(records)
    output = Path(output)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Run already has last.pt; use --resume or another output directory")
    output.mkdir(parents=True, exist_ok=True)
    stage = ("event" if event_targets_path else "physio" if reference_targets_path else
             "distill" if teacher_path else "pretrain")
    if stage == "event":
        latent = config.event.dynamic_weight > 0 and config.event.dynamic_target == "latent"
        if (modality != "ppg" or not student_path or reference_targets_path or calibration_path
                or config.train.transfer_method != "none" or config.train.pairing_control != "none"
                or config.data.window_mode != "full" or not config.train.ssl_weight):
            raise ValueError("V3 needs PPG warm-up, full windows, SSL and transfer_method=none")
        if bool(teacher_path) != latent:
            raise ValueError("V3 requires a frozen ECG teacher only for latent dynamic supervision")
    if stage == "pretrain" and config.train.ssl_weight == 0:
        raise ValueError("Pretraining without transfer requires positive SSL weight")
    if config.train.transfer_method == "physio" and stage != "physio":
        raise ValueError("Physio method requires --reference-targets")
    if stage == "physio" and (teacher_path or calibration_path or modality != "ppg" or
                              config.train.transfer_method != "physio"):
        raise ValueError("Reference supervision requires PPG/physio, without a neural teacher/calibration")
    if stage == "distill" and (modality != "ppg" or not student_path):
        raise ValueError("Distillation requires PPG warm-up")
    if stage == "distill" and config.train.transfer_method == "none":
        raise ValueError("SSL-only continuation should not load a teacher")
    if stage == "distill" and not calibration_path and config.train.transfer_method != "vanilla":
        raise ValueError("PW-KD/COSD requires a calibration artifact")
    model = TempoBridge(config.model).to(device)
    teacher, calibrator, heads = None, None, None
    initial, collate_fn, event_dynamic_dim = None, None, 0
    provenance = {"student_sha256": None, "teacher_sha256": None, "calibration_sha256": None,
                  "teacher_kind": teacher_kind if teacher_path else None, "reference_sha256": None}
    if student_path:
        initial = load_checkpoint(student_path)
        if asdict(ModelConfig.from_dict(initial["model_config"])) != config.to_dict()["model"] or initial["modality"] != "ppg":
            raise ValueError("Incompatible PPG warm-up checkpoint")
        if initial["manifest_sha256"] != digest:
            raise ValueError("Warm-up data/splits differ")
        if DataConfig.from_dict(initial["config"]["data"]).input_protocol() != config.data.input_protocol():
            raise ValueError("Warm-up input protocol differs; warm up and calibrate again")
        if initial.get("global_step", 0) < 1:
            raise ValueError("Warm-up checkpoint has not been trained")
        model.load_state_dict(initial["model"])
        provenance["student_sha256"] = file_sha256(student_path)
    if teacher_path and stage != "event":
        if calibration_path is None:
            from .calibration import calibrate
            calibration_path = output / "target_statistics.json"
            if not calibration_path.exists():
                calibrate(config, manifest, student_path, teacher_path, calibration_path, teacher_kind)
        teacher = FrozenTeacher(teacher_path, teacher_kind).to(device)
        teacher.verify_data(records, config.data.ecg_fs)
        calibrator = CalibrationWeights.load(calibration_path).to(device)
        calibrator.validate(config, digest, teacher.fingerprint, provenance["student_sha256"], teacher_kind)
        heads = TransferHeads(config.model.dim, calibrator.output_dims, config.data.scales_seconds).to(device)
        provenance.update(teacher_sha256=teacher.fingerprint,
                          calibration_sha256=file_sha256(calibration_path))
        if config.train.transfer_mode == "calibrated" and not (calibrator.weights > 0).any():
            print("NOTICE: all calibrated weights are zero; retaining PPG self-supervision only.", flush=True)
    dataset = PairedDataset(records, config.data, seed=config.train.seed)
    if config.train.sample_subset != "all":
        from .controls import WindowSubsetDataset
        dataset = WindowSubsetDataset(dataset, config.train.sample_subset)
        write_json(output / "sample_subset.json", dataset.report())
    if stage == "physio":
        from .physio_targets import (
            ReferenceDataset,
            ReferenceNormalizer,
            TargetTable,
            reference_statistics,
        )
        table = TargetTable(reference_targets_path)
        statistics = reference_statistics(dataset, table, config.train.physio_targets)
        write_json(output / "reference_statistics.json", statistics)
        calibrator = ReferenceNormalizer(statistics).to(device)
        heads = TransferHeads(config.model.dim, len(config.train.physio_targets), config.data.scales_seconds).to(device)
        provenance["reference_sha256"] = table.fingerprint
        dataset = ReferenceDataset(dataset, table, config.train.physio_targets)
    if stage == "event":
        from .dynamic_transfer import EventTransferHeads, compute_v3_loss
        from .event_targets import EventDataset, EventTable, collate_events

        table = EventTable(event_targets_path, digest, config.data.scales_seconds[0])
        provenance["event_targets_sha256"] = table.fingerprint
        if teacher_path:
            teacher = FrozenTeacher(teacher_path, teacher_kind).to(device)
            teacher.verify_data(records, config.data.ecg_fs)
            provenance["teacher_sha256"] = teacher.fingerprint
        if config.event.dynamic_weight:
            event_dynamic_dim = teacher.target_dim if teacher else len(config.event.dynamic_metrics)
        heads = EventTransferHeads(config, event_dynamic_dim).to(device)
        if initial.get("stage") == "event":
            heads.events.load_state_dict({key.removeprefix("events."): value
                                          for key, value in initial["heads"].items() if key.startswith("events.")})
        dataset = EventDataset(dataset, table, config.event)
        collate_fn = collate_events
        if config.event.rr_weight and not dataset.verified_covered:
            dataset.close()
            raise ValueError("RR training requires schema-2 clock evidence on training windows; rebuild event targets")
        pairs = int((dataset.neighbors >= 0).sum()) if dataset.neighbors is not None else 0
        if config.event.dynamic_weight and not pairs:
            raise ValueError("No adjacent, non-overlapping same-record training pairs with known timestamps")
        write_json(output / "event_coverage.json", {"windows": len(dataset), "reference_windows": dataset.covered,
                   "clock_verified_reference_windows": dataset.verified_covered,
                   "candidate_pairs": pairs, "event_targets_sha256": table.fingerprint})
    if stage == "distill" and config.train.pairing_control != "none":
        from .controls import PairingControlDataset
        dataset = PairingControlDataset(dataset, config.train, config.data.quality_edges)
    use_block_sampler = hasattr(dataset.records, "segments") or bool(config.data.source_weights)
    if config.data.sampler == "patient":
        sampler = PatientSourceSampler(dataset, config.data, config.train.batch_size,
                                        config.train.accumulation, config.train.seed)
    else:
        sampler = (SourceBlockSampler(dataset.records, config.data, config.train.batch_size,
                                      config.train.accumulation, config.train.seed)
                   if use_block_sampler else None)
    if config.data.samples_per_epoch and sampler is None:
        raise ValueError("samples_per_epoch requires explicit source_weights")
    parameters = list(model.parameters()) + (list(heads.parameters()) if heads else [])
    optimizer = torch.optim.AdamW(parameters, lr=config.train.lr, weight_decay=config.train.weight_decay)
    epoch_size = len(sampler) if sampler is not None else len(dataset)
    steps_per_epoch = math.ceil(math.ceil(epoch_size / config.train.batch_size) / config.train.accumulation)
    total_steps = config.train.epochs * steps_per_epoch
    warmup = round(total_steps * config.train.warmup_fraction)

    def lr_multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        fraction = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return config.train.min_lr_ratio + (1 - config.train.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.train.precision == "fp16")
    first_epoch, global_step, best = 0, 0, float("inf")
    if resume:
        saved = load_checkpoint(resume)
        saved["config"] = ExperimentConfig.from_dict(saved["config"]).to_dict()
        saved["provenance"].setdefault("reference_sha256", None)
        for key, expected in {"config": config.to_dict(), "manifest_sha256": digest,
                              "stage": stage, "modality": modality, "provenance": provenance}.items():
            if saved[key] != expected:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(saved["model"])
        if heads:
            heads.load_state_dict(saved["heads"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        first_epoch, global_step, best = saved["epoch"] + 1, saved["global_step"], saved["best_val"]
        torch.set_rng_state(saved["rng_cpu"])
        if device.type == "cuda" and saved["rng_cuda"]:
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
    write_json(output / "config.json", config.to_dict())
    write_json(output / "environment.json", environment())
    write_json(output / "provenance.json", {**provenance, "manifest_sha256": digest,
               "stage": stage, "modality": modality,
               "train_subjects": sorted(subject_ids(records, "train")),
               "train_sources": sorted(source_ids(records, "train")),
               "epoch_source_counts": sampler.counts if sampler is not None else None})
    for epoch in range(first_epoch, config.train.epochs):
        started = time.monotonic()
        dataset.set_epoch(epoch)
        if hasattr(dataset, "report"):
            write_json(output / f"pairing_epoch_{epoch:04}.json", dataset.report())
        if sampler is not None:
            sampler.epoch = epoch
        loader = DataLoader(dataset, batch_size=config.train.batch_size,
                            shuffle=sampler is None, sampler=sampler,
                            generator=torch.Generator().manual_seed(config.train.seed + epoch),
                            num_workers=config.data.workers, pin_memory=device.type == "cuda", collate_fn=collate_fn)
        model.train()
        if heads:
            heads.train()
        metrics, examples = {name: 0. for name in ("loss", "ssl", "transfer", "mean_transfer_weight",
                                                  "paired_fraction", "original_paired_fraction",
                                                  "active_transfer_fraction", "target_rank")}, 0
        update_norms, update_audits, audit_examples = {}, 0, {}
        audited_parameters = list(model.named_parameters())
        if stage == "event":
            audited_parameters += [(f"head/{name}", p) for name, p in heads.named_parameters()]
        iterator = iter(loader)
        while group := list(itertools.islice(iterator, config.train.accumulation)):
            counts = [int(b["ecg_valid"].sum()) if modality == "ecg" else len(b["ecg_valid"]) for b in group]
            if sum(counts) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            for batch, batch_count in zip(group, counts):
                if batch_count == 0:
                    continue
                with autocast_context(config, device):
                    audit_gradients = (config.train.gradient_audit_interval > 0 and
                                       global_step % config.train.gradient_audit_interval == 0)
                    if stage == "event":
                        loss, terms = compute_v3_loss(config, model, heads, batch, teacher, device, audit_gradients, epoch)
                    else:
                        loss, terms = batch_loss(config, model, batch, device, modality, heads, teacher,
                                                 calibrator, audit_gradients=audit_gradients)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss * batch_count / sum(counts)).backward()
                metrics["loss"] += float(loss.detach()) * batch_count
                for key, value in terms.items():
                    metrics[key] = metrics.get(key, 0.) + value * batch_count
                    if key.startswith(("v3_gradient_norm/", "v3_gradient_cosine/", "v3_loss/")):
                        audit_examples[key] = audit_examples.get(key, 0) + batch_count
                examples += batch_count
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.train.grad_clip, error_if_nonfinite=True)
            before = {name: p.detach().clone() for name, p in audited_parameters} if audit_gradients else {}
            scaler.step(optimizer)
            scaler.update()
            if before:
                squared = {}
                for name, p in audited_parameters:
                    key = parameter_group(name)
                    squared[key] = squared.get(key, 0.) + float((p.detach() - before[name]).float().square().sum())
                for key, value in squared.items():
                    update_norms[key] = update_norms.get(key, 0.) + value ** .5
                update_audits += 1
            scheduler.step()
            global_step += 1
        if not examples:
            raise ValueError("No usable training examples for selected modality")
        if stage == "event" and epoch == 0:
            expected = {"v3_rr_eligible_per_window": config.event.rr_weight,
                        "v3_event_fraction": max(config.event.foot_weight, config.event.peak_weight,
                                                 config.event.quality_proxy_weight),
                        "v3_dynamic_pair_fraction": config.event.dynamic_weight}
            missing = [key for key, weight in expected.items() if weight and not metrics.get(key)]
            if missing:
                raise ValueError(f"No valid V3 supervision in the first training epoch: {missing}")
        val = evaluate_reconstruction(config, model, records, device, modality)
        improved = config.train.checkpoint_selection == "last" or (val is not None and val < best)
        if improved:
            best = val if val is not None else best
        state = {
            "format_version": 1, "model": model.state_dict(), "model_config": model.model_config(),
            "heads": heads.state_dict() if heads else None, "config": config.to_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "stage": stage, "modality": modality,
            "epoch": epoch, "global_step": global_step, "best_val": best,
            "manifest_sha256": digest, "provenance": provenance,
            "train_subjects": sorted(subject_ids(records, "train")),
            "train_sources": sorted(source_ids(records, "train")),
            "teacher_training_subjects": teacher.training_subjects if teacher else [],
            "teacher_training_sources": teacher.training_sources if teacher else [],
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        if stage == "event":
            state["event_dynamic_dim"] = event_dynamic_dim
        save_checkpoint(output / "last.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        row = {"epoch": epoch + 1, "global_step": global_step, **{k: v / examples for k, v in metrics.items()},
               "val_reconstruction": val, "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.monotonic() - started, "examples": examples}
        for key in metrics:
            if key.startswith(("ssl_gradient_norm", "kd_gradient_norm")) or key == "gradient_cosine":
                row[key] = metrics[key] / max(metrics["gradient_audit_fraction"], 1.)
        row.update({key: metrics[key] / count for key, count in audit_examples.items()})
        row.update({f"parameter_update_norm/{key}": value / update_audits for key, value in update_norms.items()})
        row.update({key: value for key, value in metrics.items() if key.startswith("kd_coverage/")})
        row["parameter_update_audits"] = update_audits
        row["paired_windows"] = metrics["paired_fraction"]
        row["original_paired_windows"] = metrics["original_paired_fraction"]
        row["active_transfer_windows"] = metrics["active_transfer_fraction"]
        row["transfer_weight_sum"] = metrics["mean_transfer_weight"]
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        if isinstance(sampler, PatientSourceSampler):
            write_json(output / f"sampling_epoch_{epoch:04}.json", sampler.last_report)
        if stop_after is not None and epoch + 1 >= stop_after:
            break
    if teacher and (teacher.training or any(p.requires_grad for p in teacher.parameters())):
        raise RuntimeError("Teacher freeze invariant violated")
    dataset.close()
    return output / "last.pt"
