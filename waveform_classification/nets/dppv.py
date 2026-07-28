"""DPPV: Dilated PPV (Proportion of Positive Values) Network.

Architecture: Input (d) -> Parallel dilated Conv1d -> PPV pooling -> MLP -> Output

Design rationale:
- Multi-scale dilated convolutions capture multiple timescales
- PPV pooling counts pattern match proportion (more discriminative than GAP)
- Hierarchical channel allocation: more channels for larger dilations

PPV(z, b) = mean(z > b) measures what fraction of signal matches kernel pattern.

References:
    Dempster et al. (2020). ROCKET: Exceptionally fast and accurate time series
    classification using random convolutional kernels.
    https://arxiv.org/abs/1910.13051

    Dempster et al. (2021). MiniRocket: A Very Fast (Almost) Deterministic
    Transform for Time Series Classification.
    https://arxiv.org/abs/2012.08791
"""

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from .base import BaseClassifier


class Config(BaseModel):
    """DPPV configuration."""

    dilations: tuple[int, ...] = Field(
        default=(1, 4, 16, 64),
        min_length=1,
        description="Dilation factors for multi-scale convolutions",
    )
    channels_per_dilation: tuple[int, ...] = Field(
        default=(2, 4, 8, 16),
        min_length=1,
        description="Output channels per dilation (more for larger dilations)",
    )
    kernel_size: int = Field(default=9, ge=3, le=15)
    n_thresholds: int = Field(default=4, ge=2, le=8, description="PPV threshold count")
    hidden_dim: int = Field(default=32, ge=8)
    dropout: float = Field(default=0.2, ge=0.0, le=0.5)


class PPVPool(nn.Module):
    """Multi-threshold PPV pooling with global statistics.

    Computes PPV at fixed thresholds plus global max/mean.
    """

    thresholds: Tensor

    def __init__(self, n_thresholds: int = 4):
        super().__init__()
        thresholds = torch.linspace(-1.0, 1.0, n_thresholds)
        self.register_buffer("thresholds", thresholds)

    def forward(self, x: Tensor) -> Tensor:
        """Compute PPV features.

        Args:
            x: (B, C, T) convolution output

        Returns:
            (B, C * (n_thresholds + 2)) PPV + global max + global mean
        """
        # Vectorized PPV at all thresholds
        ppv = (x.unsqueeze(-1) > self.thresholds).float().mean(dim=2)
        ppv = ppv.reshape(x.shape[0], -1)

        gmax = x.max(dim=-1).values
        gmean = x.mean(dim=-1)

        return torch.cat([ppv, gmax, gmean], dim=-1)


class Model(BaseClassifier):
    """Dilated PPV Network for waveform classification.

    Architecture: Input (d) -> Parallel dilated Conv1d -> PPV pooling -> MLP -> Output
    """

    def __init__(
        self,
        input_dim: int,
        n_class: int,
        lr: float,
        dilations: tuple[int, ...] = (1, 4, 16, 64),
        channels_per_dilation: tuple[int, ...] = (2, 4, 8, 16),
        kernel_size: int = 9,
        n_thresholds: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        if len(dilations) != len(channels_per_dilation):
            msg = f"dilations ({len(dilations)}) and channels_per_dilation ({len(channels_per_dilation)}) must match"
            raise ValueError(msg)

        self.convs = nn.ModuleList()
        for d, c in zip(dilations, channels_per_dilation, strict=True):
            padding = (kernel_size - 1) * d // 2
            self.convs.append(nn.Conv1d(1, c, kernel_size, dilation=d, padding=padding))

        self.ppv = PPVPool(n_thresholds=n_thresholds)

        n_features = (n_thresholds + 2) * sum(channels_per_dilation)
        self.head = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_class),
        )

        self.init_classifier(n_class, lr, weight_decay, class_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # (B, d) -> (B, 1, d)

        features: list[Tensor] = []
        for conv in self.convs:
            z = conv(x)
            ppv = self.ppv(z)
            features.append(ppv)

        return self.head(torch.cat(features, dim=-1))
