"""Measure real forward/backward cost for a configuration without claiming efficacy."""

import argparse
import json
import time

import torch

from tempobridge.config import ExperimentConfig
from tempobridge.model import TempoBridge, random_mask, reconstruction_loss
from tempobridge.training import autocast_context
from tempobridge.utils import device_for, environment, seed_all, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--output", default="runs/benchmark.json")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
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
    result = {"environment": environment(), "parameters": sum(p.numel() for p in model.parameters()),
              "batch_size": cfg.train.batch_size, "samples": length,
              "mean_step_seconds": sum(durations) / len(durations),
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
              "losses": losses, "scope": "PPG reconstruction only; does not include teacher/calibration"}
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
