"""Architecture diagram generation using visualtorch.

Generates layered-style architecture diagrams as embeddable base64 PNGs.
"""

# pyright: reportUnknownVariableType=false

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING

import visualtorch

if TYPE_CHECKING:
    import torch.nn as nn

# Color map matching report theme
COLOR_MAP = {
    "Linear": "#2d8a7a",  # accent - learnable
    "Conv1d": "#2d8a7a",
    "BatchNorm1d": "#6a7a8a",  # slate
    "ReLU": "#c4854a",  # amber
    "SiLU": "#c4854a",
    "Dropout": "#dcdad6",  # border/light
    "AdaptiveAvgPool1d": "#2d7a4a",  # success
}


def model_to_base64_png(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 1000),
) -> str:
    """Generate architecture diagram as base64-encoded PNG.

    Args:
        model: PyTorch model
        input_shape: Model input shape (batch, features)

    Returns:
        Base64-encoded PNG string for embedding in HTML img src
    """
    img = visualtorch.layered_view(
        model,
        input_shape=input_shape,
        legend=True,
        scale_xy=3,
        scale_z=0.4,
        min_xy=15,
        min_z=15,
        color_map=COLOR_MAP,
        background_fill=(248, 247, 244),  # matches --surface
        spacing=8,
        padding=10,
    )

    # Convert PIL Image to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return f"data:image/png;base64,{b64}"
