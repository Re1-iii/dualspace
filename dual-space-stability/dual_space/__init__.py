"""
Dual-Space Stability — plug-and-play components.

Three training-only stability axes for domain-generalized segmentation:
  * style  axis : Fourier low-frequency amplitude perturbation   (fourier_style)
  * scale  axis : multi-scale consistency under a confidence gate (scale_consistency)
  * weight axis : dense flat-minima weight averaging (SWAD)        (swad)
plus the uncertainty-gated consistency loss and segmentation loss (losses).

All components are backbone-agnostic and add nothing at inference.
"""
from .fourier_style import (
    fourier_style_perturbation,
    fourier_style_swap,
    FourierDomainAugmentor,
    DualSpaceStyleAugmentor,
)
from .scale_consistency import random_scale_size, scale_consistency_loss
from .swad import SWAD, update_bn
from .losses import (
    confidence_map,
    gated_consistency_loss,
    DiceBCELoss,
    feature_consistency_loss,
)

__all__ = [
    "fourier_style_perturbation", "fourier_style_swap",
    "FourierDomainAugmentor", "DualSpaceStyleAugmentor",
    "random_scale_size", "scale_consistency_loss",
    "SWAD", "update_bn",
    "confidence_map", "gated_consistency_loss",
    "DiceBCELoss", "feature_consistency_loss",
]
__version__ = "1.0.0"
