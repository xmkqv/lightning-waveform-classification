"""Smoke tests for model training."""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from waveform_classification.nets import mlp


def test_mlp_completes_a_training_and_validation_step() -> None:
    inputs = torch.randn(20, 1000)
    labels = torch.arange(20) % 10
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=10)
    model = mlp.Model(
        input_dim=1000,
        hidden_dims=(16,),
        n_class=10,
        lr=1e-3,
        dropout=0.0,
    )
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    assert trainer.global_step == 1
