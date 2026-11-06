"""TempoBridge waveform/statistics tokens and temporal attention components."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TempoBridgeScalarEncoder(nn.Module):
    """Embed a scalar at several magnitude scales before combining the views."""

    def __init__(self, scales, dim, epsilon=1.1):
        super().__init__()
        scales = torch.as_tensor(scales, dtype=torch.float32)
        if scales.ndim != 1 or not len(scales) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Scalar scales must be finite positive values")
        if dim < 1 or not math.isfinite(epsilon) or epsilon <= 1:
            raise ValueError("Scalar dimension must be positive and scalar_epsilon must exceed 1")
        self.register_buffer("scales", scales.clone())
        self.epsilon = epsilon
        self.slopes = nn.Parameter(torch.rand(len(scales), dim))
        self.offsets = nn.Parameter(torch.rand(len(scales), dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-15) for _ in scales])

    def forward(self, values):
        if values.shape[-1] != 1:
            raise ValueError("Scalar encoder expects a trailing singleton dimension")
        # Float32 arithmetic protects the extreme magnitude scales under AMP.
        values = values.float()
        magnitudes = values.abs() / self.scales
        weights = torch.log(magnitudes + self.epsilon).reciprocal().abs()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        views = [norm(values * self.slopes[i] + self.scales[i] * self.offsets[i])
                 for i, norm in enumerate(self.norms)]
        return (torch.stack(views, dim=-2) * weights.unsqueeze(-1)).sum(dim=-2)


def _visible_standardize(values, visible):
    """Sample-standardize finite visible positions, keeping hidden positions zero."""
    if bool(visible.all()) and values.shape[-1] > 1:
        return (values - values.mean(dim=-1, keepdim=True)) / (
            values.std(dim=-1, correction=1, keepdim=True) + 1e-5)
    count = visible.sum(dim=-1, keepdim=True)
    cleaned = values.masked_fill(~visible, 0.0)
    mean = cleaned.sum(dim=-1, keepdim=True) / count.clamp_min(1)
    centered = (cleaned - mean).masked_fill(~visible, 0.0)
    variance = centered.square().sum(dim=-1, keepdim=True) / (count - 1).clamp_min(1)
    # Zero/one valid positions have no measurable variance. Their standardized
    # values are zero; no fabricated observations enter the estimate.
    deviation = torch.where(variance > 0, variance.clamp_min(torch.finfo(variance.dtype).tiny).sqrt(),
                            torch.zeros_like(variance))
    return centered / (deviation + 1e-5)


class TempoBridgeTokenizer(nn.Module):
    """Fuse continuous waveform, difference, and local-statistics representations.

    Convolution runs across patch boundaries. A hidden sample is removed before
    either convolution, normalization, differences, or scalar statistics can use
    it. The original target is never an input to a visible token.
    """

    def __init__(self, cfg):
        super().__init__()
        kernel = cfg.token_kernel_size
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("token_kernel_size must be a positive odd integer")
        self.difference_conv = nn.Conv1d(1, cfg.dim, kernel, padding=kernel // 2)
        self.waveform_conv = nn.Conv1d(1, cfg.dim, kernel, padding=kernel // 2)
        self.difference_norm = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.waveform_norm = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.mean_encoder = TempoBridgeScalarEncoder(cfg.scalar_scales, cfg.scalar_dim, cfg.scalar_epsilon)
        self.std_encoder = TempoBridgeScalarEncoder(cfg.scalar_scales, cfg.scalar_dim, cfg.scalar_epsilon)
        self.projection = nn.Linear(2 * cfg.dim + 2 * cfg.scalar_dim, cfg.dim)
        self.output_norm = nn.LayerNorm(cfg.dim, eps=1e-15)
        self.mask_token = nn.Parameter(torch.empty(1, 1, cfg.dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches, mask):
        if patches.ndim != 3 or mask.shape != patches.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Expected patches [B,N,P] and boolean mask [B,N]")
        if patches.shape[-1] < 1 or patches.shape[1] < 1 or mask.all(dim=1).any():
            raise ValueError("At least one nonempty patch must remain visible")
        batch, groups, width = patches.shape
        observed = (~mask).unsqueeze(-1).expand_as(patches)
        clean = patches.float().masked_fill(~observed, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("Visible waveform samples must be finite")
        waveform = clean.reshape(batch, 1, groups * width)
        visible = observed.reshape_as(waveform)
        with torch.no_grad():
            local_mean = clean.mean(dim=-1, keepdim=True)
            local_std = (clean.std(dim=-1, correction=1, keepdim=True) if width > 1
                         else torch.zeros_like(local_mean))
            differences = F.pad(waveform[..., 1:] - waveform[..., :-1], (0, 1))
            valid_differences = torch.cat((visible[..., 1:] & visible[..., :-1], visible[..., -1:]), dim=-1)
            differences = differences.masked_fill(~valid_differences, 0.0)
        lanes = (
            (self.difference_conv, self.difference_norm, _visible_standardize(differences, valid_differences)),
            (self.waveform_conv, self.waveform_norm, _visible_standardize(waveform, visible)),
        )
        features = []
        for convolution, normalization, values in lanes:
            encoded = normalization(convolution(values).transpose(1, 2))
            features.append(encoded.reshape(batch, groups, width, -1).mean(dim=2))
        features.extend((self.mean_encoder(local_mean), self.std_encoder(local_std)))
        tokens = self.output_norm(self.projection(torch.cat(features, dim=-1)))
        return torch.where(mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)


class TempoBridgeRMSNorm(nn.Module):
    """Per-token RMS normalization with float32 accumulation."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, values):
        work = values.float()
        normalized = work / torch.sqrt(work.square().mean(dim=-1, keepdim=True) + 1e-8)
        return normalized.to(values.dtype) * self.weight


class TempoBridgeAttentionBlock(nn.Module):
    """Temporal pre-norm attention, extrapolatable rotation, and gated FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        if self.heads < 1 or cfg.dim % self.heads or (cfg.dim // self.heads) % 2:
            raise ValueError("Attention requires an even integral dimension per head")
        if cfg.ff_hidden_dim < 1 or not 0 <= cfg.dropout < 1:
            raise ValueError("Invalid feed-forward dimension or dropout")
        if not math.isfinite(cfg.xpos_scale_base) or cfg.xpos_scale_base <= 0:
            raise ValueError("xpos_scale_base must be finite and positive")
        self.head_dim = cfg.dim // self.heads
        self.dropout, self.scale_base = cfg.dropout, cfg.xpos_scale_base
        self.norm1, self.norm2 = TempoBridgeRMSNorm(cfg.dim), TempoBridgeRMSNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.dim)
        self.up = nn.Linear(cfg.dim, 2 * cfg.ff_hidden_dim)
        self.down = nn.Linear(cfg.ff_hidden_dim, cfg.dim)
        coordinates = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        self.register_buffer("position_frequencies", 10000.0 ** (-coordinates / self.head_dim), persistent=False)
        self.register_buffer("position_scales", (coordinates + 0.4 * self.head_dim) / (1.4 * self.head_dim), persistent=False)

    def rotate(self, queries, keys):
        length = queries.shape[-2]
        positions = torch.arange(length, dtype=torch.float32, device=queries.device)
        phases = (positions[:, None] * self.position_frequencies).repeat_interleave(2, dim=-1)
        exponent = (positions - (length - 1) // 2) / self.scale_base
        half_scale = self.position_scales ** exponent[:, None]
        scales = torch.cat((half_scale, half_scale), dim=-1)
        cosine, sine = phases.cos(), phases.sin()

        def apply(values, multiplier):
            work = values.float()
            pairs = work.reshape(*work.shape[:-1], self.head_dim // 2, 2)
            perpendicular = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
            return (work * cosine * multiplier + perpendicular * sine * multiplier).to(values.dtype)

        return apply(queries, scales), apply(keys, scales.reciprocal())

    def forward(self, tokens):
        batch, length, dim = tokens.shape
        # QKV packing interleaves heads within each coordinate; the output
        # projection receives the ordinary head-major concatenation.
        packed = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.head_dim, self.heads)
        query, key, value = packed.permute(2, 0, 4, 1, 3).unbind(dim=0)
        query, key = self.rotate(query, key)
        attended = F.scaled_dot_product_attention(
            query.contiguous(), key.contiguous(), value.contiguous(),
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        ).transpose(1, 2).reshape(batch, length, dim)
        residual = tokens + self.output(attended)
        gate, value = self.up(self.norm2(residual)).chunk(2, dim=-1)
        update = self.down(F.silu(gate) * value)
        return residual + F.dropout(update, p=self.dropout, training=self.training)
