"""Staged self-supervision and selective ECG transfer with epoch-exact resume."""

import contextlib
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .calibration import CalibrationWeights
from .config import DataConfig, ExperimentConfig, ModelConfig
from .data import PairedDataset, manifest_digest, read_manifest, split_size, subject_ids
from .model import TempoBridge, TransferHeads, random_mask, reconstruction_loss
from .sampling import SourceBlockSampler
from .teacher import FrozenTeacher
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
    weights = calibrator.lookup(scale_index, ppg_quality, ecg_quality, valid, mode)
    selected = weights > 0
    if not selected.any():
        # No optimizer updates to transfer-only heads from absent supervision.
        return ppg.new_zeros(()), weights.mean()
    with torch.no_grad():
        target = calibrator.normalize(teacher(ecg[selected]), scale_index).detach()
    tokens = model.encode(ppg[selected], detach_shallow=protect_shallow)["context"]
    pred = heads(tokens, scale_index).float()
    error = F.smooth_l1_loss(pred, target.float(), reduction="none").mean(-1)
    # Fixed denominator: small calibrated weights must actually reduce supervision.
    loss = (weights[selected] * error).sum() / len(ppg)
    return loss, weights.mean()


def batch_loss(config, model, batch, device, modality, heads=None, teacher=None, calibrator=None):
    ssl, kd, coverage = [], [], []
    if modality == "ecg":
        keep = batch["ecg_valid"]
    else:
        keep = torch.ones_like(batch["ecg_valid"])
    for index in range(len(config.data.scales_seconds)):
        wave = batch[modality][index][keep].to(device)
        mask = random_mask(wave, model.config.patch_size, config.train.mask_ratio)
        ssl.append(reconstruction_loss(model, wave, mask, config.train.morphology_loss_weight))
        if heads is not None and config.train.transfer_mode != "none":
            loss, active = transfer_loss(
                model, heads, teacher, calibrator,
                batch["ppg"][index].to(device), batch["ecg"][index].to(device),
                batch["ecg_valid"].to(device), batch["ppg_quality"].to(device),
                batch["ecg_quality"].to(device), index,
                config.train.transfer_mode, config.train.protect_shallow,
            )
            kd.append(loss)
            coverage.append(active)
    self_loss = torch.stack(ssl).mean()
    distillation = torch.stack(kd).mean() if kd else self_loss.new_zeros(())
    weight = torch.stack(coverage).mean() if coverage else self_loss.new_zeros(())
    total = self_loss + config.train.transfer_weight * distillation
    return total, {"ssl": float(self_loss.detach()), "transfer": float(distillation.detach()),
                   "mean_transfer_weight": float(weight.detach())}


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
          calibration_path=None, teacher_kind="native", resume=None, stop_after=None):
    config.validate()
    seed_all(config.train.seed, config.train.threads)
    device = device_for(config.train.device)
    records = read_manifest(manifest, data_config=config.data)
    digest = manifest_digest(records)
    output = Path(output)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Run already has last.pt; use --resume or another output directory")
    output.mkdir(parents=True, exist_ok=True)
    stage = "distill" if teacher_path else "pretrain"
    if stage == "distill" and (modality != "ppg" or not student_path or not calibration_path):
        raise ValueError("Distillation requires PPG warm-up and calibration artifact")
    model = TempoBridge(config.model).to(device)
    teacher, calibrator, heads = None, None, None
    provenance = {"student_sha256": None, "teacher_sha256": None, "calibration_sha256": None,
                  "teacher_kind": teacher_kind if teacher_path else None}
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
    if teacher_path:
        teacher = FrozenTeacher(teacher_path, teacher_kind).to(device)
        teacher.verify_data(records, config.data.ecg_fs)
        calibrator = CalibrationWeights.load(calibration_path).to(device)
        calibrator.validate(config, digest, teacher.fingerprint, provenance["student_sha256"], teacher_kind)
        heads = TransferHeads(config.model.dim, calibrator.artifact["target_dim"], config.data.scales_seconds).to(device)
        provenance.update(teacher_sha256=teacher.fingerprint,
                          calibration_sha256=file_sha256(calibration_path))
        if config.train.transfer_mode == "calibrated" and not (calibrator.weights > 0).any():
            print("NOTICE: all calibrated weights are zero; retaining PPG self-supervision only.", flush=True)
    dataset = PairedDataset(records, config.data, seed=config.train.seed)
    use_block_sampler = hasattr(dataset.records, "segments") or bool(config.data.source_weights)
    sampler = (SourceBlockSampler(dataset.records, config.data, config.train.batch_size,
                                  config.train.accumulation, config.train.seed)
               if use_block_sampler else None)
    if config.data.samples_per_epoch and not use_block_sampler:
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
               "epoch_source_counts": sampler.counts if sampler is not None else None})
    for epoch in range(first_epoch, config.train.epochs):
        started = time.monotonic()
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.epoch = epoch
        loader = DataLoader(dataset, batch_size=config.train.batch_size,
                            shuffle=sampler is None, sampler=sampler,
                            generator=torch.Generator().manual_seed(config.train.seed + epoch),
                            num_workers=config.data.workers, pin_memory=device.type == "cuda")
        model.train()
        if heads:
            heads.train()
        metrics, examples = {"loss": 0.0, "ssl": 0.0, "transfer": 0.0, "mean_transfer_weight": 0.0}, 0
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
                    loss, terms = batch_loss(config, model, batch, device, modality, heads, teacher, calibrator)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss * batch_count / sum(counts)).backward()
                metrics["loss"] += float(loss.detach()) * batch_count
                for key, value in terms.items():
                    metrics[key] += value * batch_count
                examples += batch_count
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.train.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
        if not examples:
            raise ValueError("No usable training examples for selected modality")
        val = evaluate_reconstruction(config, model, records, device, modality)
        improved = val is not None and val < best
        if improved:
            best = val
        state = {
            "format_version": 1, "model": model.state_dict(), "model_config": model.model_config(),
            "heads": heads.state_dict() if heads else None, "config": config.to_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "stage": stage, "modality": modality,
            "epoch": epoch, "global_step": global_step, "best_val": best,
            "manifest_sha256": digest, "provenance": provenance,
            "train_subjects": sorted(subject_ids(records, "train")),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        save_checkpoint(output / "last.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        row = {"epoch": epoch + 1, "global_step": global_step, **{k: v / examples for k, v in metrics.items()},
               "val_reconstruction": val, "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.monotonic() - started, "examples": examples}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        if stop_after is not None and epoch + 1 >= stop_after:
            break
    if teacher and (teacher.training or any(p.requires_grad for p in teacher.parameters())):
        raise RuntimeError("Teacher freeze invariant violated")
    dataset.close()
    return output / "last.pt"
