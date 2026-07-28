"""RProj: Random Projection with Learned Bottleneck for waveform classification.

Architecture: Input (d) -> Learned bottleneck (b) -> Fixed random projection (k) -> MLP -> Output

Design rationale:
- Learned bottleneck extracts task-relevant features before projection
- Random projection preserves distances in the compressed space (JL lemma)
- Hybrid approach: learn what matters, then project cheaply

Key insight: Direct random projection from high-dim input loses task-relevant
structure. Learning a small bottleneck first preserves discriminative features,
then JL lemma applies to the compressed representation.

References:
    Johnson & Lindenstrauss (1984). Extensions of Lipschitz mappings
    into a Hilbert space.

    Rahimi & Recht (2007). Random Features for Large-Scale Kernel Machines.
    NIPS 2007.
"""

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from .base import BaseClassifier


class Config(BaseModel):
    """Random Projection with bottleneck configuration."""

    bottleneck: int = Field(default=64, ge=16, description="Learned compression dim")
    proj_dim: int = Field(default=128, ge=32, description="Random projection dim")
    hidden_dim: int = Field(default=64, ge=16)
    dropout: float = Field(default=0.2, ge=0.0, le=0.5)
    seed: int = Field(default=42, description="Random seed for reproducibility")


class Model(BaseClassifier):
    """Random Projection Network with Learned Bottleneck.

    Architecture: Input (d) -> Linear(b) -> RandomProj(k) -> MLP -> Output

    The bottleneck is learned; the random projection is fixed.
    """

    proj: Tensor

    def __init__(
        self,
        input_dim: int,
        n_class: int,
        lr: float,
        bottleneck: int = 64,
        proj_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        seed: int = 42,
        weight_decay: float = 1e-4,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Learned bottleneck: extract task-relevant features
        self.compress = nn.Linear(input_dim, bottleneck)

        # Fixed random projection from bottleneck
        # Scale by 1/sqrt(proj_dim) for variance preservation (JL lemma)
        gen = torch.Generator().manual_seed(seed)
        proj_matrix = torch.randn(bottleneck, proj_dim, generator=gen) / (proj_dim**0.5)
        self.register_buffer("proj", proj_matrix)

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(proj_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_class),
        )

        self.init_classifier(n_class, lr, weight_decay, class_weights)

    def forward(self, x: Tensor) -> Tensor:
        h = self.compress(x)  # input_dim -> bottleneck (learned)
        h = h @ self.proj  # bottleneck -> proj_dim (fixed random)
        return self.head(h)
