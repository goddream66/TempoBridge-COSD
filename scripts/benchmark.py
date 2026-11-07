"""Measure real forward/backward cost for a configuration without claiming efficacy."""

import argparse
import json
import time

import torch

from tempobridge.config import ExperimentConfig
from tempobridge.model import TempoBridge, random_mask, reconstruction_loss
from tempobridge.teacher import FrozenTeacher
from tempobridge.training import autocast_context
from tempobridge.utils import device_for, environment, seed_all, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--output", default="runs/benchmark.json")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--teacher", help="Optional frozen teacher for forward overhead measurement")
    parser.add_argument("--teacher-kind", choices=["native", "torchscript", "random_control"], default="native")
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    seed_all(cfg.train.seed, cfg.train.threads)
    device = device_for(cfg.train.device)
    model = TempoBridge(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    length = round(max(cfg.data.scales_seconds) * cfg.data.ppg_fs)
    x = torch.randn(cfg.train.batch_size, 1, length, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    durations = []
    losses = []
    for step in range(args.steps + 2):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(cfg, device):
            loss = reconstruction_loss(model, x, random_mask(x, cfg.model.patch_size, cfg.train.mask_ratio))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= 2:
            durations.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
    model.eval()
    def latency(module, wave):
        times = []
        with torch.inference_mode():
            for i in range(args.steps + 2):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                module(wave)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 2:
                    times.append(time.perf_counter() - start)
        return sum(times) / len(times)
    inference_seconds = latency(model, x)
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
        model(x)
    counted_flops = sum(event.flops for event in profile.key_averages())
    teacher_seconds = None
    if args.teacher:
        teacher = FrozenTeacher(args.teacher, args.teacher_kind).to(device)
        if teacher.input_fs != cfg.data.ecg_fs:
            raise ValueError("Teacher rate differs from benchmark configuration")
        ecg = torch.randn(len(x), 1, round(max(cfg.data.scales_seconds) * cfg.data.ecg_fs), device=device)
        teacher_seconds = latency(teacher, ecg)
    result = {"environment": environment(), "parameters": sum(p.numel() for p in model.parameters()),
              "inference_parameters": sum(p.numel() for name, p in model.named_parameters()
                                           if not name.startswith(("reconstruction.", "morph_reconstruction."))),
              "batch_size": cfg.train.batch_size, "samples": length,
              "mean_step_seconds": sum(durations) / len(durations),
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
              "ppg_inference_batch_seconds": inference_seconds, "ppg_inference_windows_per_second": len(x) / inference_seconds,
              "teacher_forward_batch_seconds": teacher_seconds,
              "profiler_counted_forward_flops": counted_flops,
              "flops_scope": "partial operator count from PyTorch profiler; unsupported operators including some attention paths are omitted",
              "losses": losses, "scope": "synthetic PPG SSL steps and PPG-only forward; optional teacher forward; not total real-training GPU-hours"}
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
