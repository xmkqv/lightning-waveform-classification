"""Neural network architectures for waveform classification.

Flat module structure with precise naming:
- mlp: Multi-Layer Perceptron baseline
- fcn: Fully Convolutional Network
- bottleneck: Low-rank bottleneck MLP
- dct: Discrete Cosine Transform projection
- dppv: Dilated PPV (Proportion of Positive Values)
- rproj: Random Projection
- wavelet: Multi-scale Haar wavelet basis
"""

from . import bottleneck, dct, dppv, fcn, mlp, rproj, wavelet
from .base import BaseClassifier
from .utils import synchronize, timed_forward

__all__ = [
    "BaseClassifier",
    "bottleneck",
    "dct",
    "dppv",
    "fcn",
    "mlp",
    "rproj",
    "synchronize",
    "timed_forward",
    "wavelet",
]
