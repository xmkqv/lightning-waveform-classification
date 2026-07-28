import re

import numpy as np
import pytorch_lightning as pl
import torch
import tyro
from pydantic import BaseModel, Field
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from rich import print

from scripts.config import (
    LOG_DIR,
    DataConfig,
    ModelConfig,
    TrainConfig,
    build_model,
    make_data,
    model_name,
)
from waveform_classification.nets.utils import timed_forward

N_WARMUP_STEP = 10
N_TIMED_STEP = 50

# Pattern for run directories: g001_s42 (group 1, seed 42)
RUN_PATTERN = re.compile(r"^g(\d+)_s(\d+)$")


def next_group(name: str) -> int:
    """Get the next group ID for a model."""
    model_dir = LOG_DIR / name
    if not model_dir.exists():
        return 1

    max_group = 0
    for p in model_dir.iterdir():
        if p.is_dir() and (match := RUN_PATTERN.match(p.name)):
            max_group = max(max_group, int(match.group(1)))
    return max_group + 1


class RunConfig(BaseModel):
    """Configuration for a single training run."""

    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


class Config(RunConfig):
    """Train model and run test."""

    seeds: list[int] = Field(default=[42], description="Seeds for multi-run experiments")


def run(cfg: RunConfig, group: int, seed: int):
    seed_everything(seed, workers=True)

    data = make_data(cfg.data)

    model = build_model(cfg.model, data, cfg.train.lr, cfg.train.weight_decay)
    name = model_name(cfg.model)

    version = f"g{group:03d}_s{seed}"
    logger = CSVLogger(LOG_DIR, name=name, version=version)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        precision=cfg.train.precision,
        deterministic=cfg.train.deterministic,
        gradient_clip_val=cfg.train.grad_clip if cfg.train.grad_clip > 0 else None,
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="val/loss",
                patience=cfg.train.patience,
                min_delta=cfg.train.min_delta,
                mode="min",
            ),
            ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True),
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    logger.log_hyperparams(
        {
            "accelerator": cfg.train.accelerator,
            "precision": cfg.train.precision,
            "max_epochs": cfg.train.max_epochs,
            "batch_size": cfg.data.batch_size,
            "seed": seed,
            "n_weight": sum(p.numel() for p in model.parameters()),
        }
    )

    trainer.fit(model, data)

    # Get best model
    device = trainer.strategy.root_device
    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
    best_model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test
    trainer.test(best_model, data)

    # Get t_per_unit_batch
    best_model = best_model.to(device)
    best_model.eval()
    test_loader = data.test_dataloader()
    x_batch, _ = next(iter(test_loader))
    x_single = x_batch[0:1].to(device)

    with torch.no_grad():
        [best_model(x_single) for _ in range(N_WARMUP_STEP)]
        times = [timed_forward(lambda: best_model(x_single), device)[1] for _ in range(N_TIMED_STEP)]

    t_per_unit_batch = float(np.mean(times))
    logger.log_metrics({"test/t_per_unit_batch": t_per_unit_batch})


def run_model(cfg: Config) -> None:
    name = model_name(cfg.model)
    group = next_group(name)
    n_seed = len(cfg.seeds)

    print(f"[bold]Running {n_seed} seeds for {name} (group {group})[/bold]")
    for i, seed in enumerate(cfg.seeds):
        print(f"\n[cyan]━━━ g{group:03d}_s{seed} ({i + 1}/{n_seed}) ━━━[/cyan]")
        run(cfg, group, seed)


if __name__ == "__main__":
    cfg = tyro.cli(Config, config=(tyro.conf.OmitSubcommandPrefixes,))
    run_model(cfg)
