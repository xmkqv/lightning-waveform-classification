"""FCN: Fully Convolutional Network for waveform classification.

Architecture: Input (1, d) -> [Conv1d -> BN -> ReLU]* -> GAP -> Linear -> Output

Design rationale:
- Multi-scale kernels capture both fast transients and slow decay
- BatchNorm stabilizes training, enables higher learning rates
- Global Average Pooling reduces parameters, enables interpretability
- Same padding preserves temporal alignment

References:
    Wang et al. (2017). Time Series Classification from Scratch with Deep Neural
    Networks: A Strong Baseline. IJCNN 2017.
    https://arxiv.org/abs/1611.06455
"""

from einops import rearrange
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor, nn

from .base import BaseClassifier


class Config(BaseModel):
    """FCN configuration."""

    channels: tuple[int, ...] = Field(default=(32, 64, 128), min_length=1)
    kernel_sizes: tuple[int, ...] = Field(default=(7, 5, 3), min_length=1)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_matching_lengths(self) -> "Config":
        if len(self.channels) != len(self.kernel_sizes):
            msg = f"channels ({len(self.channels)}) and kernel_sizes ({len(self.kernel_sizes)}) must match"
            raise ValueError(msg)
        return self

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(c <= 0 for c in v):
            msg = f"All channels must be positive, got: {v}"
            raise ValueError(msg)
        return v

    @field_validator("kernel_sizes")
    @classmethod
    def validate_kernel_sizes(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(k <= 0 or k % 2 == 0 for k in v):
            msg = f"All kernel sizes must be positive odd integers, got: {v}"
            raise ValueError(msg)
        return v


class Model(BaseClassifier):
    """Fully Convolutional Network for time series classification.

    Architecture: Input (1, d) -> Conv1d/BN/ReLU (x3) -> GAP -> FC -> Output
    """

    def __init__(
        self,
        input_dim: int,
        n_class: int,
        lr: float,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_channels = 1
        for out_channels, kernel_size in zip(channels, kernel_sizes, strict=True):
            self.convs.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )
            self.bns.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], n_class)

        self.init_classifier(n_class, lr, weight_decay, class_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "batch time -> batch 1 time")
        for conv, bn in zip(self.convs, self.bns, strict=True):
            x = nn.functional.relu(bn(conv(x)))
        x = rearrange(self.gap(x), "batch channels 1 -> batch channels")
        x = self.dropout(x)
        return self.fc(x)
