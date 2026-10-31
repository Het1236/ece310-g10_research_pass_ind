"""Position-dependent inter-segment interference floor F(x), and the geometry-only
bound on its logarithmic derivative used by the certified placement search.

WHY THIS MODULE EXISTS
----------------------
Phase 4 optimised the antenna position against a floor held CONSTANT in x:

    R_m^sum(x) = log2(1 + sum_k P_k g_k(x) / F),      F = I_m + N0 fixed.

That is wrong, and the error is not small. The interfering users sit in the adjacent
segments and do not move; the antenna does, and the leakage those users deliver depends
on how far the antenna is from them. ``analytic.L1_exact`` has always been a function of
the within-segment offset, which is exactly why Section III-A takes a worst case over
that offset. Measured on the committed parameters, F(x)/F(centre) runs from 1.00 at the
segment centre to 9.71 at either edge.

The correct floor is therefore

    F(x) = N0 + K P [ L1(x) + L1(D_S - x) ],                                    (F1)

where L1(delta) is the per-side exact mean adjacent leakage of ``analytic.L1_exact`` at
within-segment offset delta. The right neighbour is seen at offset x, the left neighbour
at offset D_S - x.

THE LOGARITHMIC DERIVATIVE, IN CLOSED FORM
------------------------------------------
L1(delta) is an average over the adjacent region whose INTEGRAND does not depend on
delta; only the u-limits do. With

    L1(delta) = (1/(D_S D_W)) int_{-D_W/2}^{D_W/2} int_{D_S-delta}^{2 D_S-delta}
                Lambda(u, y) du dy,     Lambda = eta e^{-phi r^2} / r^2,

Leibniz gives both limits moving at -1, so

    dL1/ddelta = (1/(D_S D_W)) int [ Lambda(D_S-delta, y) - Lambda(2 D_S-delta, y) ] dy.  (F2)

Lambda decreases in u, so this is positive: sliding the antenna toward a neighbour raises
the leakage from it. Differentiating (F1),

    F'(x) = K P [ L1'(x) - L1'(D_S - x) ].                                       (F3)

WHY THE BOUND IS FREE OF THE LOAD
---------------------------------
K multiplies both leakage terms and appears nowhere else, so dropping N0 from the
denominator gives a bound that holds for every K and is approached from below as K grows:

    |F'(x)/F(x)|  <=  B_F  :=  max_x  |L1'(x) - L1'(D_S-x)| / [L1(x) + L1(D_S-x)].     (F4)

The ratio is taken POINTWISE before the maximum. Taking the largest numerator over the
largest denominator separately would also bound |F'/F|, but by a factor of about 6.5 too
much on these parameters, because |F'| and F peak at the same place: both are largest at
the segment edge, where the antenna sits closest to one neighbour.

B_F depends on the deployment geometry alone. It is free of the load, of the transmit
powers, of the SNR and of the blockage-weighted level itself, which are the four
invariances the certified search relies on. On the committed parameters B_F = 0.431 per
metre, and a direct finite-difference measurement of |F'/F| returns 0.431 at K = 1, 2 and
4 alike, as (F4) predicts.

CONSEQUENCE FOR THE LIPSCHITZ CONSTANT
--------------------------------------
``maxmin_closed`` derives |dV/dx| <= max_k |g_k'/g_k| / ln2 from the Version-A wall
gamma (1+gamma)^{e_j} = P_max g_j(x) / F and the Version-B budget a_j = F / g_j(x). With
F a function of x, the quantity whose logarithmic derivative appears is g_j/F, so the two
logarithmic derivatives add:

    |dV/dx|  <=  [ max_k |g_k'/g_k| + B_F ] / ln2
             =  [ 1/min_k sqrt(y_k^2 + d^2) + B_F ] / ln2.                       (F5)

Both terms are geometry-only, so (F5) is still closed form and still costs no evaluation
of V. See ``maxmin_closed.lipschitz_V(..., floor_logderiv=B_F)``.

STATUS OF THE NUMERICS
----------------------
(F2) is exact. The maximum and minimum in (F4) are taken over a fine grid in x, which is
a scan of the deployment geometry and not a probe of V; L1 and its derivative are smooth
in x, so the grid resolves them. This is the same numerical status the paper already
gives L1 itself, which is non-elementary in u and evaluated by Gauss-Legendre quadrature.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import CubicSpline

from src.analytic import _phys, _position_grid

# Grid used for the max and min in (F4). L1 and L1' are smooth on [0, D_S], so a few
# hundred points resolve them; this is a one-off geometry scan, not an evaluation of V.
_BF_GRID_N: int = 401

# Guard band kept away from the two segment edges when scanning. At delta = 0 the right
# neighbour starts at u = D_S and at delta = D_S it starts at u = 0, where Lambda is
# largest; both are legitimate, but a PA is never placed exactly on a segment boundary and
# the quadrature grid degenerates there.
_EDGE_EPS: float = 1e-6


def _lambda_line(u: float, D_W: float, d: float, eta: float, phi: float) -> float:
    """Average of Lambda(u, y) = eta e^{-phi r^2}/r^2 over y in [-D_W/2, D_W/2].

    This is the y-integral of (F2) at one u, normalised by D_W. Uses the same
    Gauss-Legendre grid as ``analytic``, so it inherits that module's quadrature.
    """
    r2, w = _position_grid(u, u, D_W, d)          # degenerate in u: all nodes at this u
    return float(eta * np.sum(w * np.exp(-phi * r2) / r2))


def L1_at(delta: float, params: Mapping[str, Any]) -> float:
    """Per-side exact mean adjacent leakage L1(delta), for an arbitrary offset.

    Same quantity and same quadrature as ``analytic.L1_exact``, but taking the offset as
    an argument instead of reading ``params['delta']``. The bound cross-check that
    ``L1_exact`` asserts is omitted here because this is called inside the placement
    search; ``analytic.L1_exact`` remains the checked entry point.
    """
    eta, phi, d, D_S, D_W, _delta, _P, _N0 = _phys(params)
    lo = min(max(D_S - delta, 0.0), 2.0 * D_S)
    r2, w = _position_grid(lo, lo + D_S, D_W, d)
    return float(eta * np.sum(w * np.exp(-phi * r2) / r2))


def dL1_at(delta: float, params: Mapping[str, Any]) -> float:
    """dL1/ddelta by (F2). Exact: only the u-limits carry the offset."""
    eta, phi, d, D_S, D_W, _delta, _P, _N0 = _phys(params)
    u_near = D_S - delta
    u_far = 2.0 * D_S - delta
    return (_lambda_line(u_near, D_W, d, eta, phi)
            - _lambda_line(u_far, D_W, d, eta, phi)) / D_S


def floor_at(x: float, K: int, params: Mapping[str, Any]) -> float:
    """Interference floor F(x) = N0 + K P [L1(x) + L1(D_S - x)], equation (F1).

    ``x`` is the antenna offset WITHIN its own segment, in [0, D_S]; the right neighbour
    is seen at offset x and the left neighbour at offset D_S - x.
    """
    _eta, _phi, _d, D_S, _D_W, _delta, P, N0 = _phys(params)
    xc = min(max(float(x), _EDGE_EPS), D_S - _EDGE_EPS)
    return N0 + K * P * (L1_at(xc, params) + L1_at(D_S - xc, params))


def floor_fn_absolute(K: int, params: Mapping[str, Any], seg_lo: float,
                      exact: bool = False):
    """Return F as a function of the ABSOLUTE antenna coordinate.

    The placement code works on the absolute grid x in [seg_lo, seg_lo + D_S] (the P4-1
    locked convention), whereas the leakage offset is measured from the segment's own left
    edge. This adapter carries that mapping in one place.

    F is a property of the DEPLOYMENT, not of the user drop: it depends on the segment
    geometry, the blockage parameter and the load, and on nothing that changes from one
    drop to the next. It is therefore built once as a cubic interpolant over the segment
    and evaluated in microseconds thereafter, instead of running two Gauss-Legendre
    quadratures inside every evaluation of the value function. ``exact=True`` bypasses the
    interpolant and is what ``_INTERP_N`` is validated against.
    """
    _eta, _phi, _d, D_S, _D_W, _delta, _P, _N0 = _phys(params)

    if exact:
        def F_exact(x_abs: float) -> float:
            return floor_at(float(x_abs) - seg_lo, K, params)
        return F_exact

    spline = _floor_spline(int(K), _phys_key(params))

    def F(x_abs: float) -> float:
        u = min(max(float(x_abs) - seg_lo, 0.0), D_S)
        return float(spline(u))

    return F


# Nodes for the interpolant. L1 is smooth on the segment, so a few hundred cubic nodes
# resolve it far below the 1e-3 bps/Hz certified gap; ``max_interp_error`` measures the
# residual against direct evaluation and the test gate pins it.
_INTERP_N: int = 513


def _phys_key(params: Mapping[str, Any]) -> tuple:
    """Hashable identity of everything F depends on, for the interpolant cache."""
    return _phys(params)


@lru_cache(maxsize=32)
def _floor_spline(K: int, key: tuple):
    """Cubic interpolant of F over the offset range [0, D_S]. Cached per (K, geometry)."""
    eta, phi, d, D_S, D_W, delta, P, N0 = key
    params = {"eta": eta, "phi": phi, "d": d, "D_S": D_S, "D_W": D_W, "delta": delta,
              "phase3": {"P": P, "N0": N0}}
    us = np.linspace(0.0, D_S, _INTERP_N)
    vals = np.array([floor_at(float(u), K, params) for u in us])
    return CubicSpline(us, vals)


def max_interp_error(K: int, params: Mapping[str, Any], n_check: int = 400) -> float:
    """Largest relative error of the interpolant against direct evaluation of F.

    The check points are deliberately offset from the interpolation nodes. A grid that
    shares nodes with the spline reports zero error and measures nothing.
    """
    _eta, _phi, _d, D_S, _D_W, _delta, _P, _N0 = _phys(params)
    approx = floor_fn_absolute(K, params, 0.0)
    exact = floor_fn_absolute(K, params, 0.0, exact=True)
    step = D_S / (_INTERP_N - 1)
    xs = np.linspace(0.5 * step, D_S - 0.5 * step, n_check)
    return float(max(abs(approx(x) - exact(x)) / exact(x) for x in xs))


def floor_logderiv_bound(params: Mapping[str, Any]) -> float:
    """B_F of (F4): a bound on |F'(x)/F(x)| valid for every load K.

    Geometry only. Independent of K, of the transmit powers, of the SNR and of the
    blockage level. Costs no evaluation of the value function.
    """
    _eta, _phi, _d, D_S, _D_W, _delta, _P, _N0 = _phys(params)
    xs = np.linspace(_EDGE_EPS, D_S - _EDGE_EPS, _BF_GRID_N)
    num = np.array([abs(dL1_at(float(x), params) - dL1_at(float(D_S - x), params))
                    for x in xs])
    den = np.array([L1_at(float(x), params) + L1_at(float(D_S - x), params) for x in xs])
    if float(den.min()) <= 0.0:
        raise ValueError("adjacent leakage vanished; check the geometry parameters")
    return float((num / den).max())


def floor_swing(params: Mapping[str, Any], K: int = 1) -> tuple[float, float]:
    """(min, max) of F(x)/F(D_S/2) across the segment. Reporting helper for the paper."""
    _eta, _phi, _d, D_S, _D_W, _delta, _P, _N0 = _phys(params)
    xs = np.linspace(_EDGE_EPS, D_S - _EDGE_EPS, _BF_GRID_N)
    vals = np.array([floor_at(float(x), K, params) for x in xs])
    centre = floor_at(D_S / 2.0, K, params)
    return float(vals.min() / centre), float(vals.max() / centre)
