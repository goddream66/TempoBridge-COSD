"""TempoBridge multiview encoder and the checkpoint-compatible local baseline.

Only PPG is required for inference. Hidden samples are excluded before all
tokenizer computations; reconstruction targets are normalized separately.
"""

from dataclasses import asdict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .encoding import (
    TempoBridgeAttentionBlock,
    TempoBridgeSelectiveTokenizer,
    TempoBridgeTokenizer,
)


class TempoBridgeLocalRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def tempobridge_local_rotary(x: Tensor) -> Tensor:
    """Rotary position embedding on the time axis [B,H,N,D]."""
    n, d = x.shape[-2:]
    frequency = torch.exp(
        -torch.arange(0, d, 2, device=x.device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0, device=x.device)) / d)
    )
    phase = torch.arange(n, device=x.device, dtype=torch.float32)[:, None] * frequency
    a, b = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = phase.cos(), phase.sin()
    return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2).to(x.dtype)


class TempoBridgeLocalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dropout = cfg.heads, cfg.dropout
        self.norm1, self.norm2 = TempoBridgeLocalRMSNorm(cfg.dim), TempoBridgeLocalRMSNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        hidden = cfg.dim * cfg.ff_multiplier
        self.up = nn.Linear(cfg.dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        y = F.scaled_dot_product_attention(
            tempobridge_local_rotary(q), tempobridge_local_rotary(k), v,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).reshape(b, n, d)
        x = x + F.dropout(self.out(y), self.dropout, self.training)
        a, gate = self.up(self.norm2(x)).chunk(2, -1)
        return x + F.dropout(self.down(a * F.silu(gate)), self.dropout, self.training)


class TempoBridgeLocalTokenizer(nn.Module):
    """Every patch is convolved independently, including at patch boundaries."""
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, cfg.stem_width, 5, padding=2), nn.GELU(),
            nn.Conv1d(cfg.stem_width, cfg.stem_width, 3, padding=1), nn.GELU(),
        )
        self.projection = nn.Linear(cfg.stem_width + 2, cfg.dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches, mask):
        b, n, p = patches.shape
        safe = patches.masked_fill(mask[..., None], 0.0)
        local = self.conv(safe.reshape(b * n, 1, p)).mean(-1).reshape(b, n, -1)
        stats = torch.stack((safe.mean(-1), safe.std(-1, correction=0)), -1)
        tokens = self.projection(torch.cat((local, stats), -1))
        return torch.where(mask[..., None], self.mask_token, tokens)


class TempoBridge(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        enhanced = config.encoder != "local"
        tokenizer_types = {"local": TempoBridgeLocalTokenizer, "multiview": TempoBridgeTokenizer,
                           "selective": TempoBridgeSelectiveTokenizer}
        self.stem = tokenizer_types[config.encoder](config)
        block_type = TempoBridgeAttentionBlock if enhanced else TempoBridgeLocalBlock
        self.summary_token = nn.Parameter(torch.randn(config.dim)) if enhanced else None
        self.shallow = nn.ModuleList([block_type(config) for _ in range(config.protected_depth)])
        self.context = nn.ModuleList(
            [block_type(config) for _ in range(config.depth - config.protected_depth)]
        )
        self.context_norm = nn.Identity() if enhanced else TempoBridgeLocalRMSNorm(config.dim)
        self.morphology = nn.Sequential(
            nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim), nn.GELU(),
            nn.Linear(config.dim, config.dim),
        ) if config.morphology_branch else nn.Identity()
        self.morph_norm = TempoBridgeLocalRMSNorm(config.dim)
        self.reconstruction = nn.Linear(config.dim, config.patch_size)
        self.morph_reconstruction = nn.Linear(config.dim, config.patch_size)

    @property
    def feature_dim(self):
        if self.config.encoder != "local":
            if self.config.feature_pool == "combined":
                return 2 * self.config.dim
            if self.config.feature_pool != "morph_context":
                return self.config.dim
        return 2 * self.config.dim if self.config.morphology_branch else self.config.dim

    def prepare(self, x: Tensor, mask: Tensor | None = None):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected [batch, 1, time] single-lead waveform")
        p = self.config.patch_size
        if x.shape[-1] % p or x.shape[-1] < 2 * p:
            raise ValueError("Length must be a multiple of patch_size and contain >=2 patches")
        if not torch.isfinite(x).all():
            raise ValueError("Waveforms must be finite; mark missing ECG separately")
        raw = x.float().reshape(x.shape[0], -1, p)
        if self.config.num_tokens and raw.shape[1] != self.config.num_tokens:
            raise ValueError("Input length differs from fixed num_tokens * patch_size")
        if mask is None:
            mask = torch.zeros(raw.shape[:2], device=x.device, dtype=torch.bool)
        if mask.shape != raw.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean [batch, patches], True = hidden")
        visible = (~mask)[..., None].expand_as(raw)
        count = visible.sum((1, 2), keepdim=True)
        if (count == 0).any():
            raise ValueError("At least one patch must remain visible")
        mean = raw.masked_fill(~visible, 0).sum((1, 2), keepdim=True) / count
        variance = ((raw - mean).square().masked_fill(~visible, 0)).sum(
            (1, 2), keepdim=True
        ) / count
        scale = variance.sqrt().clamp_min(1e-4)
        target = (raw - mean) / scale
        stem_input = raw if self.config.encoder != "local" else target
        return self.stem(stem_input, mask), target, mask

    def encode(self, x, mask=None, detach_shallow=False, return_layer=-1):
        if not -1 <= return_layer < self.config.depth:
            raise ValueError("return_layer must be -1 or a valid zero-based layer")
        h, target, mask = self.prepare(x, mask)
        if self.summary_token is not None:
            h = torch.cat((h, self.summary_token[None, None].expand(len(h), 1, -1)), dim=1)
        selected = None
        for index, block in enumerate(self.shallow):
            h = block(h)
            if index == return_layer:
                selected = h
        patch_h = h[:, :-1] if self.summary_token is not None else h
        morph = self.morph_norm(patch_h + self.morphology(patch_h))
        context = h.detach() if detach_shallow else h
        for index, block in enumerate(self.context, start=len(self.shallow)):
            context = block(context)
            if index == return_layer:
                selected = context
        if selected is not None and detach_shallow and return_layer < len(self.shallow):
            # A requested shallow snapshot must obey the same gradient boundary
            # as the normal hand-off to the context blocks.
            selected = selected.detach()
        context = self.context_norm(context if selected is None else selected)
        summary = context[:, -1] if self.summary_token is not None else context.mean(1)
        patches = context[:, :-1] if self.summary_token is not None else context
        return {"morphology": morph, "context": patches, "summary": summary,
                "target": target, "mask": mask}

    def forward_features(self, x):
        result = self.encode(x, return_layer=self.config.feature_layer)
        c = result["context"].mean(1)
        m = result["morphology"].mean(1)
        if self.config.encoder != "local" and self.config.feature_pool != "morph_context":
            pool = self.config.feature_pool
            z = (torch.cat((result["summary"], c), -1) if pool == "combined" else
                 result["summary"] if pool == "cls" else c)
        else:
            z = torch.cat((m, c), -1) if self.config.morphology_branch else c
        return {"embedding": z, "context": c, "morphology": m,
                "summary": result["summary"], "tokens": result["context"]}

    def transfer_features(self, waveform, detach_shallow=False):
        """Shared calibration/KD contract, independent of exported feature pooling."""
        return self.encode(waveform, detach_shallow=detach_shallow)["context"].mean(1)

    def transfer_feature_spec(self):
        return {"version": 1, "layer": self.config.depth - 1,
                "pool": "context_mean", "dim": self.config.dim}

    def forward(self, ppg):
        return self.forward_features(ppg)["embedding"]

    def model_config(self):
        return asdict(self.config)


def random_mask(x, patch_size, ratio, generator=None):
    n = x.shape[-1] // patch_size
    count = min(n - 1, max(1, round(n * ratio)))
    ranks = torch.rand(x.shape[0], n, generator=generator, device=x.device).argsort(-1)
    mask = torch.zeros(x.shape[0], n, dtype=torch.bool, device=x.device)
    return mask.scatter_(1, ranks[:, :count], True)


def reconstruction_loss(model, waveform, mask, morphology_weight=0.25):
    result = model.encode(waveform, mask)
    target = result["target"].detach()
    if not mask.any():
        raise ValueError("Reconstruction requires at least one hidden patch")
    context = F.smooth_l1_loss(model.reconstruction(result["context"])[mask], target[mask])
    if model.config.morphology_branch:
        morph = F.smooth_l1_loss(model.morph_reconstruction(result["morphology"])[mask], target[mask])
    else:
        morph = context * 0
    return context + morphology_weight * morph


class TransferHeads(nn.Module):
    def __init__(self, dim, target_dim, scales):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                          nn.Linear(dim, target_dim)) for _ in scales
        ])

    def forward(self, tokens, scale_index):
        # Retain the historical token API for existing callers/checkpoints.
        features = tokens.mean(1) if tokens.ndim == 3 else tokens
        if features.ndim != 2:
            raise ValueError("Transfer head expects [batch, features] or [batch, tokens, features]")
        return self.heads[scale_index](features)
