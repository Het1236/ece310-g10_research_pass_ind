"""Channel / path-loss / blockage model (shared by Phase-2 MC and Phase-3 transforms).

Extracted in P3-0.5 from the inline math in montecarlo.py (relocation only, no
algebra change). Both helpers take R2 = squared distance ||psi_i - psi_m^Pin||^2,
verbatim from the sampler.
"""
from __future__ import annotations

import numpy as np


def blockage_prob(R2: np.ndarray | float, phi: float) -> np.ndarray | float:
    """LoS-survival probability P(alpha=1) = exp(-phi * R2). Source: EDMA Eq. (3).

    R2 = ||psi_i - psi_m^Pin||^2 (already squared). Result in (0, 1].
    """
    return np.exp(-phi * R2)


def path_gain(R2: np.ndarray | float, eta: float) -> np.ndarray | float:
    """Free-space LoS power gain |h|^2 = eta / R2. Source: EDMA Eq. (2).

    R2 = ||psi_i - psi_m^Pin||^2 (already squared).
    """
    return eta / R2
