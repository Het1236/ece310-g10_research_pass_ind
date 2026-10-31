"""Unit tests for src/analytic.py - Phase-2 closed forms.

Closed-form-vs-numerical-quadrature regression guards. The EDMA Eq. 18 test
(``test_psi_y_equals_numerical_eq18``) is the gate that must pass before any
Monte-Carlo/figure is trusted. No randomness here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from src import analytic
from src.config import load_params


@pytest.fixture(scope="module")
def params() -> dict:
    """Loaded params.yaml (with eta computed)."""
    return load_params()


def _gauss(t: float, phi: float) -> float:
    return math.exp(-phi * t**2)


# ---------------------------------------------------------------------------
# 1. master integral vs quad
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a,b,phi",
    [(0.0, 5.0, 0.02), (-3.0, 3.0, 0.1), (2.0, 9.5, 0.02), (-7.0, -1.0, 0.5), (0.0, 20.0, 0.02)],
)
def test_gaussian_interval_vs_quad(a: float, b: float, phi: float) -> None:
    expected, _ = quad(_gauss, a, b, args=(phi,))
    assert analytic.gaussian_interval(a, b, phi) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. EDMA Eq. 18 GATE - psi_y == (1/D_W) int_{-D_W/2}^{D_W/2} e^{-phi y^2} dy
# ---------------------------------------------------------------------------
def test_psi_y_equals_numerical_eq18(params: dict) -> None:
    D_W = params["D_W"]
    phi = params["phi"]
    integral, _ = quad(_gauss, -D_W / 2.0, D_W / 2.0, args=(phi,))
    expected = integral / D_W
    assert analytic.psi_y(D_W, phi) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. psi_u vs (1/D_S) int_{a_ell}^{b_ell} e^{-phi u^2} du
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ell", [1, 2, 3, 4])
def test_psi_u_vs_numerical(params: dict, ell: int) -> None:
    D_S, phi, delta = params["D_S"], params["phi"], params["delta"]
    a = analytic.a_ell(ell, D_S, delta)
    b = analytic.b_ell(ell, D_S, delta)
    integral, _ = quad(_gauss, a, b, args=(phi,))
    expected = integral / D_S
    assert analytic.psi_u(ell, D_S, phi, delta) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 4. leakage weights decrease with ell; R < 1 for phi*D_S^2 >= 3
# ---------------------------------------------------------------------------
def test_ratio_decreasing_in_ell(params: dict) -> None:
    D_S, phi, d, delta = params["D_S"], params["phi"], params["d"], params["delta"]
    # Per-side weights decrease with ell (both sides). Far terms underflow to
    # *exactly* 0.0 once erf(sqrt(phi)*a) and erf(sqrt(phi)*b) both saturate to
    # 1.0 at machine precision (here by ell>=4, where leakage is already ~1e-14):
    # their difference is exact 0.0. So assert non-increasing everywhere, and
    # strict decrease only over the physically meaningful regime (terms > 0).
    for offset in (delta, D_S - delta):
        weights = [analytic._side_weight(ell, D_S, phi, d, offset) for ell in range(1, 6)]
        assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))
        positive = [w for w in weights if w > 0.0]
        assert weights[0] > 0.0  # adjacent (ell=1) term is non-negligible
        assert all(positive[i] > positive[i + 1] for i in range(len(positive) - 1))
    # adjacency dominates: R < 1 when the gate phi*D_S^2 >= 3 holds
    assert phi * D_S**2 >= 3.0
    assert analytic.ratio_R(params, delta, ell_max=4) < 1.0


# ---------------------------------------------------------------------------
# 5. honest two-sided ratio sits under its own envelope across a phi*D_S^2 sweep
# ---------------------------------------------------------------------------
def test_envelope_bound(params: dict) -> None:
    phi = params["phi"]
    p = dict(params)
    for phiDs2 in (3.0, 4.0, 5.0, 7.0, 10.0):
        D_S = math.sqrt(phiDs2 / phi)
        p["D_S"] = D_S
        env = analytic.R_envelope(phi, D_S)
        # worst over a delta grid (offset symmetry handled inside ratio_R)
        worst = max(
            analytic.ratio_R(p, delta=frac * D_S, ell_max=5)
            for frac in (0.1, 0.25, 0.5, 0.75, 1.0)
        )
        assert worst <= env, f"phi*D_S^2={phiDs2}: R={worst:.3e} > env={env:.3e}"


# ---------------------------------------------------------------------------
# 6. gate threshold: phi*D_S^2 = 3 => R_envelope ~= e^-3 ~= 0.0498
# ---------------------------------------------------------------------------
def test_gate_threshold() -> None:
    phi = 0.02
    D_S = math.sqrt(3.0 / phi)  # phi*D_S^2 = 3
    assert analytic.R_envelope(phi, D_S) == pytest.approx(math.exp(-3.0), rel=1e-12)
    assert analytic.R_envelope(phi, D_S) == pytest.approx(0.0498, abs=1e-4)


# ---------------------------------------------------------------------------
# Extra sanity: K_star is finite, positive, and small at EDMA's high SNR.
# ---------------------------------------------------------------------------
def test_kstar_lossless_small_at_high_snr(params: dict) -> None:
    kstar = analytic.K_star(params)  # wg_loss_factor defaults to 1.0 (lossless)
    assert math.isfinite(kstar) and kstar > 0.0
    assert kstar < 1.0  # rho = 120 dB => deeply interference-limited => small K*
    # lossier waveguide (factor < 1) raises K*
    assert analytic.K_star(params, wg_loss_factor=0.5) > kstar
