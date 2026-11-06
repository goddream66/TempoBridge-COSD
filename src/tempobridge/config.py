"""Strict, serializable experiment settings. Durations are physical seconds."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    encoder: str = "multiview"
    dim: int = 256
    depth: int = 6
    protected_depth: int = 2
    heads: int = 8
    patch_size: int = 25
    stem_width: int = 32
    ff_multiplier: int = 3
    dropout: float = 0.0
    morphology_branch: bool = True
    token_kernel_size: int = 41
    scalar_dim: int = 32
    scalar_scales: list[float] = field(default_factory=lambda: [10.0 ** i for i in range(-4, 5)])
    scalar_epsilon: float = 1.1
    ff_hidden_dim: int = 512
    xpos_scale_base: float = 512.0
    num_tokens: int = 0
    feature_pool: str = "combined"
    feature_layer: int = -1

    def __post_init__(self):
        if self.encoder not in {"local", "multiview"}:
            raise ValueError("encoder must be local or multiview")
        if self.dim <= 0 or self.heads <= 0 or self.dim % self.heads:
            raise ValueError("dim must be positive and divisible by heads")
        if (self.dim // self.heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 < self.protected_depth < self.depth:
            raise ValueError("protected_depth must be between 1 and depth - 1")
        if self.patch_size < 4 or self.stem_width < 1 or self.ff_multiplier < 1:
            raise ValueError("Invalid patch/stem/FF dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.token_kernel_size < 1 or self.token_kernel_size % 2 == 0:
            raise ValueError("token_kernel_size must be a positive odd number")
        if self.scalar_dim < 1 or self.ff_hidden_dim < 1 or not self.xpos_scale_base > 0:
            raise ValueError("Invalid multiview encoder dimensions/position scale")
        if not self.scalar_scales or any(not 0 < s < float("inf") for s in self.scalar_scales):
            raise ValueError("scalar_scales must be nonempty finite positive values")
        if not 1 < self.scalar_epsilon < float("inf"):
            raise ValueError("scalar_epsilon must be finite and greater than one")
        if self.num_tokens < 0 or self.num_tokens == 1:
            raise ValueError("num_tokens must be zero (duration-preserving) or >=2")
        if self.feature_pool not in {"cls", "mean", "combined", "morph_context"}:
            raise ValueError("Invalid feature_pool")
        if not -1 <= self.feature_layer < self.depth:
            raise ValueError("feature_layer must be -1 or a valid zero-based layer")

    @classmethod
    def from_dict(cls, values):
        # Historical configs/checkpoints omitted this field and used the local stem.
        values = dict(values)
        values.setdefault("encoder", "local")
        return cls(**values)


@dataclass
class DataConfig:
    ppg_fs: int = 100
    ecg_fs: int = 100
    scales_seconds: list[float] = field(default_factory=lambda: [2.0, 8.0])
    quality_edges: list[float] = field(default_factory=lambda: [0.33, 0.67])
    workers: int = 0
    source_roots: dict[str, str] = field(default_factory=dict)
    integrity: str = "sha256"
    max_open_shards: int = 4
    source_weights: dict[str, float] = field(default_factory=dict)
    samples_per_epoch: int = 0
    shuffle_block_size: int = 256


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 32
    accumulation: int = 1
    lr: float = 0.0003
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    mask_ratio: float = 0.5
    morphology_loss_weight: float = 0.25
    transfer_weight: float = 1.0
    transfer_mode: str = "calibrated"
    protect_shallow: bool = True
    grad_clip: float = 1.0
    precision: str = "fp32"
    device: str = "auto"
    threads: int = 4


@dataclass
class CalibrationConfig:
    folds: int = 3
    ridge: float = 1.0
    min_subjects_per_bin: int = 3
    min_ecg_quality: float = 0.2
    probe: str = "consensus"
    mlp_hidden: int = 64
    mlp_steps: int = 200
    mlp_lr: float = 0.001
    max_windows_per_subject: int = 16
    probe_batch_size: int = 2048


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def validate(self):
        d, t, c = self.data, self.train, self.calibration
        if d.ppg_fs <= 0 or d.ecg_fs <= 0 or d.workers < 0:
            raise ValueError("Invalid sample rates/workers")
        if d.integrity not in {"sha256", "metadata"} or d.max_open_shards < 1 or d.shuffle_block_size < 1:
            raise ValueError("Invalid HDF5 integrity/cache/shuffle settings")
        if d.samples_per_epoch < 0 or any(not 0 < w < float("inf") for w in d.source_weights.values()):
            raise ValueError("Invalid epoch size or source weights")
        if not d.scales_seconds or len(set(d.scales_seconds)) != len(d.scales_seconds):
            raise ValueError("Scales must be nonempty and unique")
        for s in d.scales_seconds:
            for fs in (d.ppg_fs, d.ecg_fs):
                n = round(s * fs)
                if s <= 0 or abs(n - s * fs) > 1e-6 or n % self.model.patch_size:
                    raise ValueError("Each scale * sample rate must be an integer multiple of patch_size")
                if n < 2 * self.model.patch_size:
                    raise ValueError("Every crop needs at least two patches")
                if self.model.num_tokens and n != self.model.num_tokens * self.model.patch_size:
                    raise ValueError("Fixed num_tokens requires scale * fs = num_tokens * patch_size")
        if d.quality_edges != sorted(set(d.quality_edges)) or any(
            not 0 < e < 1 for e in d.quality_edges
        ):
            raise ValueError("quality_edges must be strictly increasing in (0,1)")
        if min(t.epochs, t.batch_size, t.accumulation, t.threads) < 1:
            raise ValueError("epochs, batch_size, accumulation, threads must be positive")
        if not 0 < t.mask_ratio < 1 or not 0 <= t.warmup_fraction < 1:
            raise ValueError("Invalid mask/warmup ratio")
        if t.lr <= 0 or t.weight_decay < 0 or t.grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= t.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0,1]")
        if t.transfer_weight < 0 or t.morphology_loss_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        if t.transfer_mode not in {"calibrated", "all", "quality", "random", "none"}:
            raise ValueError("Unknown transfer_mode")
        if t.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("Unknown precision")
        if c.folds < 2 or c.ridge <= 0 or c.min_subjects_per_bin < 1:
            raise ValueError("Invalid calibration settings")
        if c.probe not in {"ridge", "mlp", "consensus"}:
            raise ValueError("Unknown probe")
        if c.mlp_steps < 1 or c.mlp_hidden < 1 or c.mlp_lr <= 0:
            raise ValueError("Invalid MLP probe")
        if c.max_windows_per_subject < 1 or c.probe_batch_size < 1:
            raise ValueError("Calibration sample/batch bounds must be positive")
        if not 0 <= c.min_ecg_quality <= 1:
            raise ValueError("min_ecg_quality must be in [0,1]")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {"model", "data", "train", "calibration"}
        if unknown:
            raise ValueError(f"Unknown config sections: {unknown}")
        return cls(
            model=ModelConfig.from_dict(values["model"]) if "model" in values else ModelConfig(),
            data=DataConfig(**values.get("data", {})),
            train=TrainConfig(**values.get("train", {})),
            calibration=CalibrationConfig(**values.get("calibration", {})),
        ).validate()

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
