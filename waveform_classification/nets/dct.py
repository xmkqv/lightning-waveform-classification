"""DCT: Discrete Cosine Transform projection for waveform classification.

Architecture: Input (d) -> DCT (n_freq) -> [dual repr] -> MLP -> Output

Design rationale:
- DCT-II basis provides frequency-domain inductive bias
- Energy compaction: most information in few low-frequency components
- Dual representation (signed + |abs|) preserves polarity information
- Single matmul projection is computationally efficient

DCT-II formula: X[k] = sum_n x[n] * cos(pi * k * (n + 0.5) / N)

References:
    Yi et al. (2023). Frequency-domain MLPs are More Effective Learners in Time
    Series Forecasting. NeurIPS 2023.

    Ahmed et al. (1974). Discrete Cosine Transform. IEEE Trans. Computers.
"""

# pyright: reportUnknownVariableType=false
import math

import torch
from einops import reduce, repeat
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn

from .base import BaseClassifier


def build_dct_basis(input_dim: int, n_freq: int) -> Tensor:
    """Build normalized DCT-II basis matrix.

    Args:
        input_dim: Signal length N
        n_freq: Number of frequency components to extract

    Returns:
        Tensor of shape (n_freq, input_dim) with unit-norm rows
    """
    n = torch.arange(input_dim, dtype=torch.float32)
    k = torch.arange(n_freq, dtype=torch.float32)
    k_expanded = repeat(k, "freq -> freq time", time=input_dim)
    dct_matrix = torch.cos(math.pi * k_expanded * (n + 0.5) / input_dim)
    norms: Tensor = reduce(dct_matrix**2, "freq time -> freq 1", "sum").sqrt().clamp(min=1e-8)
    return dct_matrix / norms


class Config(BaseModel):
    """DCT projection configuration."""

    n_freq: int = Field(default=64, ge=1, description="Number of DCT frequency components")
    dual: bool = Field(default=True, description="Use dual representation (signed + abs)")
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
    """DCT-based classifier with optional dual representation.

    Architecture: Input (d) -> DCT (n_freq) -> [signed, |abs|]? -> MLP -> Output

    With dual=True (default): 2*n_freq features capturing phase and magnitude.
    With dual=False: n_freq features (signed DCT coefficients only).
    """

    basis: Tensor

    def __init__(
        self,
        input_dim: int,
        n_class: int,
        lr: float,
        n_freq: int = 64,
        dual: bool = True,
        hidden_dims: tuple[int, ...] = (64,),
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.dual = dual

        # Fixed DCT-II basis (not learned)
        basis = build_dct_basis(input_dim, n_freq)
        self.register_buffer("basis", basis)

        # Feature dimension depends on dual representation
        feature_dim = 2 * n_freq if dual else n_freq

        # Encoder layers
        layers: list[nn.Module] = []
        prev_dim = feature_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()

        self.classifier = nn.Linear(prev_dim, n_class)

        self.init_classifier(n_class, lr, weight_decay, class_weights)

    def forward(self, x: Tensor) -> Tensor:
        # DCT projection via fixed basis
        dct = nn.functional.linear(x, self.basis)
        features = torch.cat([dct, dct.abs()], dim=1) if self.dual else dct
        h = self.encoder(features)
        return self.classifier(h)
