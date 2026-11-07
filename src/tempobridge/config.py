"""Strict, serializable experiment settings. Durations are physical seconds."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    encoder: str = "selective"
    dim: int = 256
    depth: int = 6
    protected_depth: int = 2
    heads: int = 8
    patch_size: int = 10
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
    selective_width: int = 32
    selective_kernel_sizes: list[int] = field(default_factory=lambda: [9, 21, 41])
    selective_reduction: int = 4
    selective_min_dim: int = 8
    selective_fusion: str = "adaptive"
    selective_local_context: bool = True
    selective_coverage: bool = True

    def __post_init__(self):
        if self.encoder not in {"local", "multiview", "selective"}:
            raise ValueError("encoder must be local, multiview or selective")
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
        if any(type(v) is not int or v < 1 for v in (
            self.selective_width, self.selective_reduction, self.selective_min_dim,
        )):
            raise ValueError("Selective widths/reduction must be positive integers")
        kernels = self.selective_kernel_sizes
        if (not kernels or any(type(k) is not int or k < 3 or k % 2 == 0 for k in kernels)
                or kernels != sorted(set(kernels))):
            raise ValueError("selective_kernel_sizes must be increasing unique odd integers >=3")
        if self.selective_fusion not in {"adaptive", "mean"}:
            raise ValueError("selective_fusion must be adaptive or mean")
        if type(self.selective_local_context) is not bool or type(self.selective_coverage) is not bool:
            raise ValueError("Selective switches must be boolean")

    @classmethod
    def from_dict(cls, values):
        # Historical configs/checkpoints omitted this field and used the local stem.
        values = dict(values)
        values.setdefault("encoder", "local")
        if values["encoder"] in {"local", "multiview"}:
            values.setdefault("patch_size", 25)
        return cls(**values)


@dataclass
class DataConfig:
    ppg_fs: int = 50
    ecg_fs: int = 50
    scales_seconds: list[float] = field(default_factory=lambda: [30.0])
    window_mode: str = "full"
    quality_edges: list[float] = field(default_factory=lambda: [0.33, 0.67])
    workers: int = 0
    source_roots: dict[str, str] = field(default_factory=dict)
    integrity: str = "sha256"
    max_open_shards: int = 4
    source_weights: dict[str, float] = field(default_factory=dict)
    samples_per_epoch: int = 0
    shuffle_block_size: int = 256

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "window_mode" not in values:
            # Historical JSON dictionaries described random/center cropping.
            values["window_mode"] = "crop"
            values.setdefault("scales_seconds", [2.0, 8.0])
            values.setdefault("ppg_fs", 100)
            values.setdefault("ecg_fs", 100)
        return cls(**values)

    def input_protocol(self):
        return {name: getattr(self, name) for name in
                ("ppg_fs", "ecg_fs", "scales_seconds", "window_mode")}


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
    transfer_method: str = "pw_kd"
    checkpoint_selection: str = "reconstruction"
    gradient_audit_interval: int = 0


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
class COSDConfig:
    basis: str = "standardized"
    selection: str = "topk_oof_skill"
    rank: int = 64
    min_skill: float = 0.0
    stability: bool = True
    bootstrap_samples: int = 200
    confidence: float = 0.95
    repeats: int = 1
    basis_fraction: float = 0.2
    audit_fraction: float = 0.2
    soft: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    cosd: COSDConfig = field(default_factory=COSDConfig)

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
        if d.window_mode not in {"full", "crop"}:
            raise ValueError("window_mode must be full or crop")
        if d.window_mode == "full" and len(d.scales_seconds) != 1:
            raise ValueError("full window_mode requires exactly one duration")
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
        if t.transfer_method not in {"none", "vanilla", "pw_kd", "cosd"}:
            raise ValueError("Unknown transfer_method")
        if t.transfer_method == "vanilla" and t.transfer_mode not in {"all", "quality"}:
            raise ValueError("Vanilla KD uses transfer_mode all or quality, without OOF weights")
        if t.checkpoint_selection not in {"reconstruction", "last"}:
            raise ValueError("checkpoint_selection must be reconstruction or last")
        if type(t.gradient_audit_interval) is not int or t.gradient_audit_interval < 0:
            raise ValueError("gradient_audit_interval must be a nonnegative number of optimizer steps")
        s = self.cosd
        if s.basis not in {"standardized", "pca", "random"}:
            raise ValueError("Unknown COSD basis")
        if s.selection not in {"topk_oof_skill", "random", "variance", "all"}:
            raise ValueError("Unknown COSD selection")
        if type(s.rank) is not int or s.rank < 0 or type(s.repeats) is not int or s.repeats < 1:
            raise ValueError("Invalid COSD rank/repeats")
        if not 0 <= s.min_skill < 1 or not 0.5 < s.confidence < 1:
            raise ValueError("Invalid COSD skill/confidence")
        if type(s.bootstrap_samples) is not int or s.bootstrap_samples < 20:
            raise ValueError("COSD bootstrap_samples must be >=20")
        if not 0 < s.basis_fraction < 1 or not 0 < s.audit_fraction < 1 or s.basis_fraction + s.audit_fraction >= 1:
            raise ValueError("COSD basis/audit fractions must leave discovery patients")
        if type(s.soft) is not bool or type(s.stability) is not bool:
            raise ValueError("COSD switches must be boolean")
        if s.selection == "variance" and s.basis != "pca":
            raise ValueError("Variance selection requires PCA basis")
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
        unknown = set(values) - {"model", "data", "train", "calibration", "cosd"}
        if unknown:
            raise ValueError(f"Unknown config sections: {unknown}")
        model = ModelConfig.from_dict(values["model"]) if "model" in values else ModelConfig()
        if "data" in values:
            data = DataConfig.from_dict(values["data"])
        elif "model" in values and model.encoder in {"local", "multiview"}:
            # Old abbreviated JSON could omit the entire data section.
            data = DataConfig.from_dict({})
        else:
            data = DataConfig()
        return cls(
            model=model,
            data=data,
            train=TrainConfig(**values.get("train", {})),
            calibration=CalibrationConfig(**values.get("calibration", {})),
            cosd=COSDConfig(**values.get("cosd", {})),
        ).validate()

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
