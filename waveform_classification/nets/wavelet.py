"""Wavelet: Multi-scale Haar wavelet basis for waveform classification.

Architecture: Input (d) -> Wavelet (n_features) -> [dual repr] -> MLP -> Output

Design rationale:
- Multi-scale Haar wavelets capture localized time-frequency features
- Fixed basis (not learned) - fast, interpretable
- Dual representation (signed + |abs|) preserves polarity
- Scales (1, 2, 4, ...) aligned to fast rise + slow decay timescales

Haar wavelet: psi(t) = +1 for t in [0, 0.5), -1 for t in [0.5, 1), 0 otherwise.
Localized in time (unlike DCT's global frequency modes).
"""

# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
import math

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from .base import BaseClassifier


def build_haar_basis(input_dim: int, n_features: int, levels: int) -> Tensor:
    """Build multi-scale Haar wavelet basis matrix.

    Args:
        input_dim: Signal length
        n_features: Number of basis functions
        levels: Number of wavelet scales (2^0 to 2^(levels-1))

    Returns:
        Tensor of shape (n_features, input_dim) with normalized basis vectors
    """
    bases: list[Tensor] = []
    features_per_level = n_features // levels

    for level in range(levels):
        scale = 2**level
        step = scale * 2
        n_wavelets = input_dim // step

        for i in range(min(n_wavelets, features_per_level)):
            vec = torch.zeros(input_dim)
            start = i * step
            end = min(start + step, input_dim)
            mid = start + scale

            if mid <= input_dim and end <= input_dim:
                vec[start:mid] = 1.0
                vec[mid:end] = -1.0
                vec = vec / math.sqrt(step)
                bases.append(vec)

    # Fill remaining with DCT-like bases
    while len(bases) < n_features:
        k = len(bases) - levels * features_per_level + 1
        n = torch.arange(input_dim, dtype=torch.float32)
        vec: Tensor = torch.cos(math.pi * k * (n + 0.5) / input_dim)
        vec = vec / vec.norm()
        bases.append(vec)

    return torch.stack(bases[:n_features])


class Config(BaseModel):
    """Wavelet configuration."""

    n_features: int = Field(default=128, ge=1, description="Number of wavelet basis features")
    levels: int = Field(default=6, ge=1, le=10, description="Number of wavelet scales")
    dual: bool = Field(default=True, description="Use dual representation (signed + abs)")
    hidden_dims: tuple[int, ...] = Field(default=(128, 64), description="Hidden layer dimensions")
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)


class Model(BaseClassifier):
    """Wavelet Network with multi-scale Haar basis.

    Architecture: Input (d) -> Wavelet (n_features) -> [signed, |abs|]? -> MLP -> Output
    """

    basis: Tensor

    def __init__(
        self,
        input_dim: int,
        n_class: int,
        lr: float,
        n_features: int = 128,
        levels: int = 6,
        dual: bool = True,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.dual = dual

        basis = build_haar_basis(input_dim, n_features, levels)
        self.register_buffer("basis", basis)

        feature_dim = 2 * n_features if dual else n_features

        # Build classifier with variable depth
        layers: list[nn.Module] = []
        prev_dim = feature_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_class))
        self.classifier = nn.Sequential(*layers)

        self.init_classifier(n_class, lr, weight_decay, class_weights)

    def forward(self, x: Tensor) -> Tensor:
        wavelet = nn.functional.linear(x, self.basis)
        features = torch.cat([wavelet, wavelet.abs()], dim=1) if self.dual else wavelet
        return self.classifier(features)
