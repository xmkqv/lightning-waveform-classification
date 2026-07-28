"""Shared configuration for training and testing scripts.

This module provides:
- DataConfig: Data loading configuration
- TrainConfig: Training hyperparameters
- ModelConfig: Union of all model configurations (for tyro CLI)
- build_model: Factory function to instantiate models from config
- make_data: Factory function to instantiate data module
"""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import pytorch_lightning as pl
import tyro
from pydantic import BaseModel, Field, model_validator
from rich import print as rprint

from waveform_classification.data.lightning import Data
from waveform_classification.nets import bottleneck, dct, dppv, fcn, mlp, rproj, wavelet

if TYPE_CHECKING:
    from torch import Tensor


RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
LOG_DIR = Path(os.getenv("LOG_DIR", ".cache/logs"))
REPORT_HTML = RESULTS_DIR / "report.html"


class DataConfig(BaseModel):
    """Data loading configuration."""

    data_dir: str = Field(default="datasets/waveform/lightning", description="Path to waveform data")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Batch size")
    seed: int = Field(default=42, ge=0, description="Random seed")
    train_frac: float = Field(default=0.8, gt=0.0, le=1.0, description="Training fraction")
    val_frac: float = Field(default=0.1, ge=0.0, lt=1.0, description="Validation fraction")

    @model_validator(mode="after")
    def check_splits(self) -> "DataConfig":
        if self.train_frac + self.val_frac >= 1.0:
            msg = f"train_frac ({self.train_frac}) + val_frac ({self.val_frac}) must be < 1.0"
            raise ValueError(msg)
        return self


class TrainConfig(BaseModel):
    """Training configuration."""

    max_epochs: int = Field(default=100, ge=1, le=10000, description="Maximum epochs")
    lr: float = Field(default=1e-3, gt=0.0, le=1.0, description="Learning rate")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="Weight decay")
    patience: int = Field(default=20, ge=1, description="Early stopping patience")
    min_delta: float = Field(default=1e-3, ge=0.0, description="Minimum improvement for early stopping")
    grad_clip: float = Field(default=1.0, ge=0.0, description="Gradient clipping norm (0=off)")
    precision: Literal["32"] = Field(default="32", description="Training precision; MPS only supports 32")
    deterministic: bool = Field(default=False, description="Deterministic training for reproducibility")
    accelerator: Literal["auto", "cpu", "cuda", "mps", "gpu"] = Field(
        default="auto", description="Hardware accelerator"
    )


# Model configs as tyro subcommands
ModelConfig = (
    Annotated[mlp.Config, tyro.conf.subcommand("mlp")]
    | Annotated[fcn.Config, tyro.conf.subcommand("fcn")]
    | Annotated[bottleneck.Config, tyro.conf.subcommand("bottleneck")]
    | Annotated[dct.Config, tyro.conf.subcommand("dct")]
    | Annotated[dppv.Config, tyro.conf.subcommand("dppv")]
    | Annotated[rproj.Config, tyro.conf.subcommand("rproj")]
    | Annotated[wavelet.Config, tyro.conf.subcommand("wavelet")]
)


def make_data(cfg: DataConfig) -> Data:
    """Create and setup data module."""
    data = Data(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
    )
    data.setup(stage="fit")
    return data


def build_model(
    model_cfg: ModelConfig,
    data: Data,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> pl.LightningModule:
    """Build model from config."""
    input_dim = Data.input_dim
    n_class = data.n_class
    class_weights: Tensor | None = data.class_weights

    match model_cfg:
        case mlp.Config(hidden_dims=h, dropout=d):
            return mlp.Model(
                input_dim=input_dim,
                n_class=n_class,
                lr=lr,
                weight_decay=weight_decay,
                class_weights=class_weights,
                hidden_dims=h,
                dropout=d,
            )
        case fcn.Config(channels=c, kernel_sizes=k, dropout=d):
            return fcn.Model(
                input_dim=input_dim,
                n_class=n_class,
                lr=lr,
                weight_decay=weight_decay,
                class_weights=class_weights,
                channels=c,
                kernel_sizes=k,
                dropout=d,
            )
        case bottleneck.Config(rank=r, hidden_dims=h, dropout=d):
            return bottleneck.Model(
                input_dim=input_dim,
                n_class=n_class,
                lr=lr,
                weight_decay=weight_decay,
                class_weights=class_weights,
                rank=r,
                hidden_dims=h,
                dropout=d,
            )
        case dct.Config(n_freq=n, dual=dual, hidden_dims=h, dropout=d):
            return dct.Model(
                input_dim=input_dim,
                n_class=n_class,
                lr=lr,
                weight_decay=weight_decay,
                class_weights=class_weights,
                n_freq=n,
                dual=dual,
                hidden_dims=h,
                dropout=d,
            )
        case dppv.Config(
            dilations=dil,
            channels_per_dilation=cpd,
            kernel_size=k,
            n_thresholds=nt,
            hidden_dim=h,
            dropout=d,
        ):
            return dppv.Model(
                input_dim=input_dim,
                n_class=n_class,
                lr=lr,
                weight_decay=weight_decay,
                class_weights=class_weights,
                dilations=dil,
                channels_per_dilation=cpd,
                kernel_size=k,
                n_thresholds=nt,
                hidden_dim=h,
                dropout=d,
            )
        case rproj.Config(bottleneck=b, proj_dim=p, hidden_dim=h, dropout=d, seed=s):
            return rproj.Model(
                input_dim=input_dim,
                n_class=n_class,
                lr=lr,
                weight_decay=weight_decay,
                class_weights=class_weights,
                bottleneck=b,
                proj_dim=p,
                hidden_dim=h,
                dropout=d,
                seed=s,
            )
        case wavelet.Config(n_features=n, levels=lv, dual=dual, hidden_dims=hd, dropout=d):
            return wavelet.Model(
                input_dim=input_dim,
                n_class=n_class,
                lr=lr,
                weight_decay=weight_decay,
                class_weights=class_weights,
                n_features=n,
                levels=lv,
                dual=dual,
                hidden_dims=hd,
                dropout=d,
            )
        case _:
            msg = f"Unknown model config: {model_cfg}"
            raise ValueError(msg)


def model_name(model_cfg: ModelConfig) -> str:
    """Extract model name from config type."""
    # Config is in waveform_classification.nets.{name}, extract {name}
    parts = type(model_cfg).__module__.split(".")
    return parts[-1]  # e.g., "mlp", "fcn", "bottleneck"


def find_checkpoint(name: str) -> Path:
    """Find best checkpoint from latest training run.

    Excludes last.ckpt (final epoch) and returns the best checkpoint
    saved by ModelCheckpoint(save_top_k=1).
    """
    import re

    log_dir = LOG_DIR / name
    if not log_dir.exists():
        raise FileNotFoundError(f"No runs for {name}")

    # Match g{group}_s{seed} pattern
    run_pattern = re.compile(r"^g(\d+)_s(\d+)$")
    run_dirs = [p for p in log_dir.iterdir() if p.is_dir() and run_pattern.match(p.name)]
    if not run_dirs:
        raise FileNotFoundError(f"No runs for {name}")

    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    ckpts = [p for p in latest.glob("checkpoints/*.ckpt") if p.name != "last.ckpt"]
    if not ckpts:
        raise FileNotFoundError(f"No best checkpoint in {latest} (only last.ckpt found)")

    return max(ckpts, key=lambda p: p.stat().st_mtime)


def report(
    name: str,
    model: pl.LightningModule,
    results: Sequence[Mapping[str, object]],
    profile: bool = False,
) -> None:
    """Print test results in structured format."""
    from torchinfo import summary

    raw: Mapping[str, object] = results[0] if results else {}
    model_stats = summary(model, input_size=(1, Data.input_dim), verbose=0)

    def fmt(k: str, v: object) -> object:
        if not isinstance(v, float):
            return v
        if "t_per" in k:
            return round(v * 1000, 4)  # seconds → ms
        return round(v, 4)

    metrics: dict[str, object] = {
        "model": name,
        "params": model_stats.total_params,
        **{k.removeprefix("test/"): fmt(k, v) for k, v in raw.items()},
    }
    rprint(metrics)

    if profile:
        rprint("\n[bold]Model Summary:[/bold]")
        rprint(
            summary(
                model,
                input_size=(1, Data.input_dim),
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                depth=4,
                verbose=0,
            )
        )
