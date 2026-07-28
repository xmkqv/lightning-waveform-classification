"""Bottleneck: Compressed MLP with low-rank first layer.

Architecture: Input (d) -> Linear (rank) -> [Linear -> SiLU -> Dropout]* -> Output

Design rationale:
- Low-rank projection reduces compute from O(d*h) to O(d*r + r*h) where r << d
- Information bottleneck forces extraction of task-relevant features
- SiLU activation: smooth, self-gated, no dead neurons

For d=1000, r=48, h=64: 48K + 3K = 51K params vs 128K for standard first layer.

References:
    Tishby & Zaslavsky (2015). Deep Learning and the Information Bottleneck
    Principle. arXiv:1503.02406.
"""

from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn

from .base import BaseClassifier


class Config(BaseModel):
    """Bottleneck MLP configuration."""

    rank: int = Field(default=64, ge=8, description="Bottleneck dimension")
    hidden_dims: tuple[int, ...] = Field(default=(64,), min_length=0)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(d <= 0 for d in v):
            msg = f"All hidden dimensions must be positive, got: {v}"
            raise ValueError(msg)
        return v


class Model(BaseClassifier):
    """Bottleneck MLP with low-rank compression.

    Architecture: Input (d) -> proj (rank) -> encoder -> classifier (n_class)
    """

    def __init__(
        self,
        input_dim: int,
        n_class: int,
        lr: float,
        rank: int = 64,
        hidden_dims: tuple[int, ...] = (64,),
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Low-rank projection (no bias for efficiency)
        self.proj = nn.Linear(input_dim, rank, bias=False)

        # Encoder layers (optional depth)
        layers: list[nn.Module] = []
        prev_dim = rank
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()

        self.classifier = nn.Linear(prev_dim, n_class)

        self.init_classifier(n_class, lr, weight_decay, class_weights)

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x)
        h = self.encoder(h)
        return self.classifier(h)
