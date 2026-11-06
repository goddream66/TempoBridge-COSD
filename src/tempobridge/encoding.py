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


class TempoBridgeSelectiveKernel(nn.Module):
    """Select a receptive field per patch and channel using visible evidence.

    Parallel depthwise/pointwise branches are summed to obtain a selection
    descriptor. A sample-weighted global descriptor, optionally supplemented by
    its patch-local counterpart, controls a softmax over branches. Coverage is
    measured against the support available inside the recording, so ordinary
    sequence boundaries are not confused with missing samples.
    """

    def __init__(self, cfg):
        super().__init__()
        self.width = cfg.selective_width
        self.kernel_sizes = tuple(cfg.selective_kernel_sizes)
        self.fusion = cfg.selective_fusion
        self.local_context = cfg.selective_local_context
        self.coverage_enabled = cfg.selective_coverage
        self.max_compensation = 4.0
        if any(type(v) is not int or v < 1 for v in (
            self.width, cfg.selective_reduction, cfg.selective_min_dim,
        )):
            raise ValueError("Selective dimensions and reduction must be positive integers")
        if (not self.kernel_sizes
                or any(type(k) is not int or k < 3 or k % 2 == 0 for k in self.kernel_sizes)
                or list(self.kernel_sizes) != sorted(set(self.kernel_sizes))):
            raise ValueError("Selective kernels must be increasing unique odd integers >=3")
        if self.fusion not in {"adaptive", "mean"}:
            raise ValueError("Selective fusion must be adaptive or mean")
        if type(self.local_context) is not bool or type(self.coverage_enabled) is not bool:
            raise ValueError("Selective switches must be boolean")
        self.input_projection = nn.Conv1d(1, self.width, 1)
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, k, padding=k // 2,
                      groups=self.width, bias=False) for k in self.kernel_sizes
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in self.kernel_sizes
        ])
        self.branch_norms = nn.ModuleList([
            nn.LayerNorm(self.width, eps=1e-5) for _ in self.kernel_sizes
        ])
        hidden = max(self.width // cfg.selective_reduction, cfg.selective_min_dim)
        # Both fusion modes retain this parameter structure for controlled
        # comparisons with the same branch weights and state dictionary.
        self.selector_reduce = nn.Linear(self.width, hidden)
        self.selector_norm = nn.LayerNorm(hidden, eps=1e-5)
        self.selector_expand = nn.Linear(hidden, len(self.kernel_sizes) * self.width)

    def forward(self, values, visible, patch_size):
        return self.forward_with_diagnostics(values, visible, patch_size)[0]

    def forward_with_diagnostics(self, values, visible, patch_size):
        if (values.ndim != 3 or values.shape[1] != 1
                or visible.shape != values.shape or visible.dtype != torch.bool):
            raise ValueError("Expected values [B,1,L] and a matching boolean visibility mask")
        if (type(patch_size) is not int or patch_size < 1
                or values.shape[-1] < 1 or values.shape[-1] % patch_size):
            raise ValueError("Length must be a positive multiple of patch_size")
        clean = values.float().masked_fill(~visible, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("Visible selective-kernel samples must be finite")
        batch, _, length = clean.shape
        groups = length // patch_size
        projected = self.input_projection(clean).masked_fill(~visible, 0.0)
        patch_visible = visible.reshape(batch, groups, patch_size, 1)
        sample_counts = patch_visible.sum(dim=2)
        features, coverages = [], []
        for kernel, depthwise, pointwise, norm in zip(
            self.kernel_sizes, self.depthwise_convs, self.pointwise_convs, self.branch_norms,
        ):
            # Counts use float32 even under AMP. The denominator describes only
            # genuine locations in the input; zero padding is not an observation.
            with torch.autocast(device_type=clean.device.type, enabled=False):
                support_filter = torch.ones(1, 1, kernel, device=clean.device, dtype=torch.float32)
                observed_support = F.conv1d(visible.float(), support_filter, padding=kernel // 2)
                possible_support = F.conv1d(torch.ones_like(clean), support_filter, padding=kernel // 2)
                point_coverage = observed_support / possible_support
                compensation = (possible_support / observed_support.clamp_min(1)).clamp_max(
                    self.max_compensation)
            branch = depthwise(projected)
            if self.coverage_enabled:
                branch = branch * compensation.to(branch.dtype)
            branch = pointwise(branch)
            branch = F.gelu(norm(branch.transpose(1, 2)))
            # Do not pool predictions centered on unobserved samples. This also
            # removes all convolution/normalization biases from empty patches.
            branch = branch.reshape(batch, groups, patch_size, self.width)
            branch = branch.masked_fill(~patch_visible, 0.0)
            features.append(branch.sum(dim=2) / sample_counts.clamp_min(1))
            point_coverage = point_coverage.reshape(batch, groups, patch_size, 1)
            coverage = (point_coverage * patch_visible).sum(dim=2) / sample_counts.clamp_min(1)
            coverages.append(coverage.squeeze(-1))
        branch_features = torch.stack(features, dim=2)
        coverage = torch.stack(coverages, dim=2)
        valid_branches = coverage > 0
        if self.fusion == "adaptive":
            local = branch_features.sum(dim=2)
            global_context = (local * sample_counts).sum(dim=1, keepdim=True) / (
                sample_counts.sum(dim=1, keepdim=True).clamp_min(1))
            descriptor = global_context + local if self.local_context else global_context.expand_as(local)
            latent = F.gelu(self.selector_norm(self.selector_reduce(descriptor)))
            logits = self.selector_expand(latent).reshape(batch, groups, len(self.kernel_sizes), self.width)
            logits = logits.float()
            if self.coverage_enabled:
                logits = logits + coverage.clamp_min(torch.finfo(coverage.dtype).tiny).log().unsqueeze(-1)
            logits = logits.masked_fill(~valid_branches.unsqueeze(-1), -torch.inf)
            # Softmax over all -inf is undefined. Empty patches instead receive
            # finite temporary logits and an exactly zero final allocation.
            empty = ~valid_branches.any(dim=2, keepdim=True)
            logits = torch.where(empty.unsqueeze(-1), torch.zeros_like(logits), logits)
            weights = logits.softmax(dim=2) * valid_branches.unsqueeze(-1)
        else:
            allocation = valid_branches.float() / valid_branches.sum(dim=2, keepdim=True).clamp_min(1)
            weights = allocation.unsqueeze(-1).expand(-1, -1, -1, self.width)
        combined = (branch_features * weights.to(branch_features.dtype)).sum(dim=2)
        return combined, {"weights": weights, "coverage": coverage}


class TempoBridgeSelectiveTokenizer(nn.Module):
    """Build waveform/difference/statistics tokens with adaptive receptive fields.

    Hidden target samples are removed before statistics, differences, branches,
    or selection descriptors are computed. Diagnostics are returned explicitly;
    the module does not retain input-dependent state or computation graphs.
    """

    def __init__(self, cfg):
        super().__init__()
        self.waveform_kernel = TempoBridgeSelectiveKernel(cfg)
        self.difference_kernel = TempoBridgeSelectiveKernel(cfg)
        self.mean_encoder = TempoBridgeScalarEncoder(cfg.scalar_scales, cfg.scalar_dim, cfg.scalar_epsilon)
        self.std_encoder = TempoBridgeScalarEncoder(cfg.scalar_scales, cfg.scalar_dim, cfg.scalar_epsilon)
        self.projection = nn.Linear(2 * cfg.selective_width + 2 * cfg.scalar_dim, cfg.dim)
        self.output_norm = nn.LayerNorm(cfg.dim, eps=1e-15)
        self.mask_token = nn.Parameter(torch.empty(1, 1, cfg.dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches, mask):
        return self.forward_with_diagnostics(patches, mask)[0]

    def forward_with_diagnostics(self, patches, mask):
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
        wave_features, wave_diagnostics = self.waveform_kernel.forward_with_diagnostics(
            _visible_standardize(waveform, visible), visible, width)
        difference_features, difference_diagnostics = self.difference_kernel.forward_with_diagnostics(
            _visible_standardize(differences, valid_differences), valid_differences, width)
        features = (difference_features, wave_features,
                    self.mean_encoder(local_mean), self.std_encoder(local_std))
        tokens = self.output_norm(self.projection(torch.cat(features, dim=-1)))
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)
        return tokens, {
            "weights_waveform": wave_diagnostics["weights"],
            "weights_difference": difference_diagnostics["weights"],
            "coverage_waveform": wave_diagnostics["coverage"],
            "coverage_difference": difference_diagnostics["coverage"],
        }
