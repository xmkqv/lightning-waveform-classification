"""MLP: Multi-Layer Perceptron baseline for waveform classification.

Architecture: Input (d) -> [Linear -> ReLU -> Dropout]* -> Linear -> Output (n_class)

Design rationale:
- Universal function approximator with no domain-specific inductive bias
- Serves as baseline for comparison with structured architectures
- ReLU activation: computationally efficient, well-understood gradients
- Dropout regularization prevents overfitting

The MLP treats input as a flat feature vector, discarding temporal structure.
This establishes a lower bound for architectures that exploit waveform physics.

References:
    Wang et al. (2017). Time Series Classification from Scratch with Deep Neural
    Networks: A Strong Baseline. IJCNN 2017.
    https://arxiv.org/abs/1611.06455
"""

from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn

from .base import BaseClassifier


class Config(BaseModel):
    """MLP configuration."""

    hidden_dims: tuple[int, ...] = Field(default=(128, 64, 32), min_length=1)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(d <= 0 for d in v):
            msg = f"All hidden dimensions must be positive, got: {v}"
            raise ValueError(msg)
        return v


class Model(BaseClassifier):
    """Multi-Layer Perceptron baseline.

    Architecture: Input (d) -> Linear(h1) -> ReLU -> Dropout -> ... -> Output (n_class)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        n_class: int,
        lr: float,
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_class))
        self.net = nn.Sequential(*layers)

        self.init_classifier(n_class, lr, weight_decay, class_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
