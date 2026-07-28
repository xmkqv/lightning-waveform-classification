"""Base module for classifier networks.

Provides common Lightning functionality: metrics, training/validation/test steps,
optimizer configuration, and parameter counting.

All classifier architectures inherit from BaseClassifier to eliminate duplication.
"""

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class BaseClassifier(pl.LightningModule):
    """Base classifier with shared training logic.

    Subclasses must:
    1. Call super().__init__() and then init_classifier(n_class, lr, weight_decay, class_weights)
    2. Implement forward(x: Tensor) -> Tensor returning logits
    """

    # Type hints for buffers
    class_weights: Tensor | None

    def init_classifier(
        self,
        n_class: int,
        lr: float,
        weight_decay: float,
        class_weights: Tensor | None = None,
    ) -> None:
        """Initialize classifier components. Call after super().__init__()."""
        self.n_class = n_class
        self.lr = lr
        self.weight_decay = weight_decay

        self.register_buffer("class_weights", class_weights, persistent=False)

        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=n_class),
                "f1": MulticlassF1Score(num_classes=n_class, average="macro"),
                "precision": MulticlassPrecision(num_classes=n_class, average="macro"),
                "recall": MulticlassRecall(num_classes=n_class, average="macro"),
                "auroc": MulticlassAUROC(num_classes=n_class, average="macro"),
            },
            compute_groups=False,
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self._param_count = sum(p.numel() for p in self.parameters())

    def on_train_epoch_start(self) -> None:
        """Log parameter count at first epoch."""
        if self.current_epoch == 0:
            self.log("model/params", float(self._param_count))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y, weight=self.class_weights)
        self.train_metrics.update(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y, weight=self.class_weights)
        self.val_metrics.update(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        self.test_metrics.update(logits, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
