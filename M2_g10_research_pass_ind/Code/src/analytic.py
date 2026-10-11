"""Phase-2 analytic closed forms for the NOMA-EDMA feasibility gate.

Every function names its source equation. Sources:
  * EDMA  = Ding, Schober, Poor, "Environment Division Multiple Access (EDMA):
            A Feasibility Study via Pinching Antennas," IEEE TWC 25:15675-15691,
            2026  (context/BasePaper/...Feasibility_Study...).
  * GT     = Phase-2 ground truth in CLAUDE.md Sec. 3.
  * SM     = SystemModel/ (segmented-waveguide NOMA layout).

Notation matches EDMA where symbols exist. The "probability integral" Phi of
EDMA Eq. (18) is ref. [31] = Gradshteyn & Ryzhik (8.250.1), defined
Phi(x) = (2/sqrt(pi)) * int_0^x e^{-t^2} dt == erf(x); so the constant relating
EDMA's Phi to scipy.special.erf is exactly 1.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import dblquad, quad
from scipy.special import erf, exp1

# ----------------------------------------------------------------------------
# Master integral
# ----------------------------------------------------------------------------
def gaussian_interval(a: float, b: float, phi: float) -> float:
    """int_a^b e^{-phi t^2} dt = 1/2 sqrt(pi/phi) [erf(sqrt(phi) b) - erf(sqrt(phi) a)].

    Source: GT master Gaussian integral (G&R [31] 3.321). Requires phi > 0.
    """
    if phi <= 0.0:
        raise ValueError("phi must be > 0")
    sp = math.sqrt(phi)
    return 0.5 * math.sqrt(math.pi / phi) * (erf(sp * b) - erf(sp * a))


# ----------------------------------------------------------------------------
# Gaussian factors
# ----------------------------------------------------------------------------
def psi_y(D_W: float, phi: float) -> float:
    """Psi_y = (1/D_W) sqrt(pi/phi) erf(sqrt(phi) D_W/2).

    Equals EDMA Eq. (18) exactly (Phi == erf). This is the y-axis average of the
    blockage factor e^{-phi y^2} over y in [-D_W/2, D_W/2].
    """
    if phi <= 0.0:
        raise ValueError("phi must be > 0")
    return (1.0 / D_W) * math.sqrt(math.pi / phi) * erf(math.sqrt(phi) * D_W / 2.0)


def a_ell(ell: int, D_S: float, delta: float) -> float:
    """Lower x-offset of the ell-th adjacent segment: a_ell = ell*D_S - delta. Source: GT."""
    return ell * D_S - delta


def b_ell(ell: int, D_S: float, delta: float) -> float:
    """Upper x-offset of the ell-th adjacent segment: b_ell = (ell+1)*D_S - delta. Source: GT."""
    return (ell + 1) * D_S - delta


def u_min(ell: int, D_S: float, delta: float) -> float:
    """Nearest x-offset to the ell-th adjacent segment: u_min,ell = ell*D_S - delta (= a_ell). Source: GT."""
    return ell * D_S - delta


def psi_u(ell: int, D_S: float, phi: float, delta: float) -> float:
    """Psi_u^(ell) = (1/(2 D_S)) sqrt(pi/phi) [erf(sqrt(phi) b_ell) - erf(sqrt(phi) a_ell)].

    Equivalently (1/D_S) int_{a_ell}^{b_ell} e^{-phi u^2} du. Source: GT.

    `delta` here is the *effective* x-offset: pass `delta` for right-side
    neighbours and `D_S - delta` for left-side neighbours (CLAUDE.md Sec. 3).
    """
    a = a_ell(ell, D_S, delta)
    b = b_ell(ell, D_S, delta)
    return (1.0 / D_S) * gaussian_interval(a, b, phi)


# ----------------------------------------------------------------------------
# Per-region leakage bound
# ----------------------------------------------------------------------------
def L_bar(ell: int, params: Mapping[str, Any], delta: float) -> float:
    """Per-region leakage upper bound (right side, offset delta):

        L_bar^(ell) <= [eta * e^{-phi d^2} / (u_min,ell^2 + d^2)] * Psi_u^(ell) * Psi_y.

    Source: GT per-region leakage bound (EDMA Eq. 2/3 + Eq. 18).
    """
    eta = float(params["eta"])
    phi = float(params["phi"])
    d = float(params["d"])
    D_S = float(params["D_S"])
    D_W = float(params["D_W"])

    umin = u_min(ell, D_S, delta)
    prefactor = eta * math.exp(-phi * d**2) / (umin**2 + d**2)
    return prefactor * psi_u(ell, D_S, phi, delta) * psi_y(D_W, phi)


# ----------------------------------------------------------------------------
# Far / adjacent leakage ratio (two-sided, honest)
# ----------------------------------------------------------------------------
def _side_weight(ell: int, D_S: float, phi: float, d: float, offset: float) -> float:
    """Per-side, per-ell leakage weight W(ell) = Psi_u^(ell) / (u_min,ell^2 + d^2).

    eta * e^{-phi d^2} * Psi_y is common to every term and cancels in ratio_R, so
    it is omitted here. `offset` is delta (right) or D_S - delta (left).
    Source: GT far/adjacent ratio.
    """
    umin = u_min(ell, D_S, offset)
    return psi_u(ell, D_S, phi, offset) / (umin**2 + d**2)


def ratio_R(params: Mapping[str, Any], delta: float, ell_max: int) -> float:
    """Two-sided far/adjacent leakage ratio, truncated at ell_max:

        R = ( sum_{ell=2}^{ell_max} [W_right(ell) + W_left(ell)] )
            / ( W_right(1) + W_left(1) ),

    where right-side neighbours use offset delta and left-side neighbours use
    offset (D_S - delta). Source: GT (CLAUDE.md Sec. 3 two-sided form). This is
    the like-for-like quantity the Monte-Carlo computes; the clean per-side bound
    e^{-phi D_S^2} (see ``R_envelope``) is the Lemma statement only.
    """
    if ell_max < 2:
        raise ValueError("ell_max must be >= 2 (need at least one far segment)")
    D_S = float(params["D_S"])
    phi = float(params["phi"])
    d = float(params["d"])
    off_r = delta
    off_l = D_S - delta

    denom = _side_weight(1, D_S, phi, d, off_r) + _side_weight(1, D_S, phi, d, off_l)
    numer = sum(
        _side_weight(ell, D_S, phi, d, off_r) + _side_weight(ell, D_S, phi, d, off_l)
        for ell in range(2, ell_max + 1)
    )
    return numer / denom


def R_envelope(phi: float, D_S: float) -> float:
    """Analytic envelope e^{-phi D_S^2} (per-side worst case delta = D_S).

    Source: GT envelope (Lemma statement). Gate: phi*D_S^2 >= 3 => R_envelope <= 0.05.
    """
    return math.exp(-phi * D_S**2)


# ----------------------------------------------------------------------------
# Saturation count
# ----------------------------------------------------------------------------
def K_star(params: Mapping[str, Any], wg_loss_factor: float = 1.0) -> float:
    """Saturation user count K* ~= 1 / (2 * rho * wg_loss_factor * L_bar^(1)).

    Source: GT (corrected). `wg_loss_factor = e^{-2 kappa x_tilde_m}` is the
    in-waveguide propagation-loss factor (general PASS literature, NOT EDMA).
    Default 1.0 = lossless low-loss-dielectric assumption, since SystemModel/ and
    EDMA model no in-waveguide loss. High SNR (rho large) => K* is expected small.

    L_bar^(1) is evaluated at the adjacent segment (ell=1) with the configured
    `delta`.
    """
    rho = float(params["rho"])
    delta = float(params["delta"])
    return 1.0 / (2.0 * rho * wg_loss_factor * L_bar(1, params, delta))


# ============================================================================
# PHASE 3 (ergodic rate, crossover, scaling).  Sources: GT3 = Phase-3 ground
# truth, CLAUDE.md Sec. 7. Position averages use a fixed Gauss-Legendre grid
# (the 1/r^2 / e^{-phi r^2} integrands are smooth: r^2 >= d^2 > 0). Numerical
# knob (OURS), not physical:
_POS_QUAD_N: int = 64  # Gauss-Legendre nodes per axis for the (u, y) position average.


def _phys(params: Mapping[str, Any]) -> tuple[float, ...]:
    """Pull the Phase-3 physical scalars (eta, phi, d, D_S, D_W, delta, P, N0)."""
    p3 = params["phase3"]
    return (
        float(params["eta"]), float(params["phi"]), float(params["d"]),
        float(params["D_S"]), float(params["D_W"]), float(params["delta"]),
        float(p3["P"]), float(p3["N0"]),
    )


def _require_symmetric_delta(delta: float, D_S: float, tol: float = 1e-9) -> None:
    """Guard the E[I_m]=2K P L1 / [h]^{2K} symmetric-adjacent convention (CLAUDE.md 7.7, C2).

    The 2x (both adjacent sides via one per-side L1) is exact ONLY at delta = D_S/2.
    """
    if abs(delta - D_S / 2.0) > tol:
        raise ValueError(
            f"2x symmetric-adjacent convention requires delta=D_S/2 "
            f"(got delta={delta}, D_S={D_S}). Either set delta=D_S/2 or implement "
            f"the two-sided sum K*P*(L1(delta)+L1(D_S-delta))."
        )


@lru_cache(maxsize=64)
def _position_grid(u_lo: float, u_hi: float, D_W: float, d: float, n: int = _POS_QUAD_N):
    """Gauss-Legendre (u, y) grid over [u_lo,u_hi] x [-D_W/2, D_W/2]; weights sum to 1.

    Returns (r2, w) flattened: r2 = u^2 + y^2 + d^2, w the normalized average weights
    (so sum(w)=1, i.e. sum(w * f) approximates the position average of f). Cached.
    """
    x, wq = leggauss(n)
    u = u_lo + 0.5 * (x + 1.0) * (u_hi - u_lo)
    y = -0.5 * D_W + 0.5 * (x + 1.0) * D_W
    wu, wy = 0.5 * wq, 0.5 * wq                       # each sums to 1 over its axis
    U, Y = np.meshgrid(u, y, indexing="ij")
    WU, WY = np.meshgrid(wu, wy, indexing="ij")
    r2 = (U * U + Y * Y + d * d).ravel()
    w = (WU * WY).ravel()
    return r2, w


# ----------------------------------------------------------------------------
# Mean constants
# ----------------------------------------------------------------------------
def G_bar(params: Mapping[str, Any]) -> float:
    """G_bar = E_pos[eta/r^2] over the HOME segment, u in [-delta, D_S-delta].

    Source: GT3 (CLAUDE.md 7.4); identity -g'(0)=P*G_bar. Non-elementary u-integral
    => numerical (Gauss-Legendre), cross-checked against adaptive 2D quadrature.
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    r2, w = _position_grid(-delta, D_S - delta, D_W, d)
    val = eta * float(np.sum(w / r2))
    num, _ = dblquad(lambda y, u: eta / (u * u + y * y + d * d),
                     -delta, D_S - delta, -D_W / 2.0, D_W / 2.0)
    num /= (D_S * D_W)
    assert abs(val - num) <= 1e-6 * abs(num) + 1e-18, f"G_bar GL vs dblquad: {val} vs {num}"
    return val


def G_bar_full(params: Mapping[str, Any]) -> float:
    """G_bar_full = E_pos[eta/r^2] over the WHOLE region (no segmentation).

    Single conventional antenna at the region centre (D_L/2, 0, d) => u in
    [-D_L/2, D_L/2]. Source: GT3 (CLAUDE.md 7.7); used by R_conv. Asserts
    G_bar > G_bar_full (home segment is closer on average).
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    D_L = float(params["D_L"])
    r2, w = _position_grid(-D_L / 2.0, D_L / 2.0, D_W, d)
    val = eta * float(np.sum(w / r2))
    assert val < G_bar(params), "expected G_bar > G_bar_full"
    return val


def G_bar_inf(params: Mapping[str, Any]) -> float:
    """Finite signal ceiling G_bar_inf = (2 eta)/(d D_W) * arctan(D_W/(2d)).

    Elementary (u=0, y-only average). Source: GT3 (CLAUDE.md 7.4).
    Limits: D_W->0 => eta/d^2 ; D_W->inf => 0.
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    return (2.0 * eta) / (d * D_W) * math.atan(D_W / (2.0 * d))


def L1_exact(params: Mapping[str, Any]) -> float:
    """L1_exact = E_pos[eta e^{-phi r^2}/r^2] over ONE ADJACENT segment (per-side, offset delta).

    EXACT form (numerical). Source: GT3 (CLAUDE.md 7.4); identity -h'(0)=P*L1_exact.
    Cross-phase invariant (asserted): L1_exact <= L_bar(1, params, delta) (the Phase-2
    bounded value; the 1/r^2 freeze inflates the bound).
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    r2, w = _position_grid(D_S - delta, 2.0 * D_S - delta, D_W, d)
    val = eta * float(np.sum(w * np.exp(-phi * r2) / r2))
    bound = L_bar(1, params, delta)
    assert val <= bound * (1.0 + 1e-9), f"L1_exact {val} exceeds Phase-2 bound {bound}"
    return val


# ----------------------------------------------------------------------------
# Per-user Laplace transforms (PER-USER, K-free; powers live in ergodic_rate_exact)
# ----------------------------------------------------------------------------
def g_transform(z: float, params: Mapping[str, Any]) -> float:
    """g(z) = E_home[exp(-z P eta / r^2)], home segment u in [-delta, D_S-delta].

    Per-user (no K). Source: GT3 (CLAUDE.md 7.2). Invariant 0 < g <= 1; g(0)=1.
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    r2, w = _position_grid(-delta, D_S - delta, D_W, d)
    g = float(np.sum(w * np.exp(-z * P * eta / r2)))
    # g in (0,1] for z>=0 (underflows to 0 only at very large z); g>1 for z<0, used
    # by the finite-difference derivative test. Bound-check only for z>=0.
    if z >= 0.0:
        assert 0.0 <= g <= 1.0 + 1e-12, f"g(z>=0)={g} out of [0,1]"
    return g


def h_transform(z: float, params: Mapping[str, Any]) -> float:
    """h(z) = E_adj[1 - e^{-phi r^2}(1 - e^{-z P eta / r^2})], adjacent (per-side, offset delta).

    Per-user (no K). Source: GT3 (CLAUDE.md 7.2). Invariant 0 < h <= 1; h(0)=1;
    h decreasing in z. Combines blockage (EDMA Eq.3) and path gain (EDMA Eq.2).
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    r2, w = _position_grid(D_S - delta, 2.0 * D_S - delta, D_W, d)
    h = float(np.sum(w * (1.0 - np.exp(-phi * r2) * (1.0 - np.exp(-z * P * eta / r2)))))
    # h in (0,1] for z>=0; h>1 for z<0 (finite-difference test). Bound-check z>=0 only.
    if z >= 0.0:
        assert 0.0 < h <= 1.0 + 1e-12, f"h(z>=0)={h} out of (0,1]"
    return h


# ----------------------------------------------------------------------------
# Exact ergodic per-segment rate
# ----------------------------------------------------------------------------
def ergodic_rate_exact(K: int, params: Mapping[str, Any]) -> tuple[float, float]:
    """Exact ergodic per-segment rate (rate, trunc_err). Source: GT3 (CLAUDE.md 7.3).

        R_bar_m = (1/ln2) int_0^inf (1/z) e^{-z N0} [h(z)]^{2K} (1 - [g(z)]^K) dz.

    The z=0 singularity is REMOVABLE (limit = K*P*G_bar), special-cased and
    self-verified at runtime. z_max is NOT fixed: z_hi doubles until the rigorous
    tail bound trunc_err = [h(z_hi)]^{2K} * E_1(z_hi N0) / ln2 < z_quad_tol; that
    bound (which assumes h decreasing -- asserted) is the returned diagnostic.
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    _require_symmetric_delta(delta, D_S)
    tol = float(params["phase3"]["z_quad_tol"])
    n_sub = int(params["phase3"]["z_grid_init"])
    ln2 = math.log(2.0)
    Gbar = G_bar(params)

    # removable-singularity threshold: z * P eta / r^2 <= ~1e-6 (linear regime).
    z_thresh = 1e-6 * d * d / (P * eta)

    def F(z: float) -> float:
        if z <= z_thresh:
            return K * P * Gbar
        g = g_transform(z, params)
        h = h_transform(z, params)
        # stable 1 - g^K; g underflows to 0 at large z -> 1 - 0^K = 1.
        one_minus_gK = 1.0 if g <= 0.0 else -math.expm1(K * math.log(g))
        return (1.0 / z) * math.exp(-z * N0) * (h ** (2 * K)) * one_minus_gK

    # A1 self-verify: full form at z_thresh matches the limit K*P*G_bar.
    g_t, h_t = g_transform(z_thresh, params), h_transform(z_thresh, params)
    F_full = (1.0 / z_thresh) * math.exp(-z_thresh * N0) * (h_t ** (2 * K)) * (
        -math.expm1(K * math.log(g_t)))
    assert abs(F_full - K * P * Gbar) < tol, (
        f"removable-singularity guard off at z_thresh: {F_full} vs {K * P * Gbar}")

    def tail_err(zhi: float) -> float:
        return (h_transform(zhi, params) ** (2 * K)) * float(exp1(zhi * N0)) / ln2

    # extend z_hi until the rigorous tail bound < tol.
    h_start = h_transform(z_thresh, params)
    z_hi = 1.0 / N0
    while tail_err(z_hi) >= tol:
        z_hi *= 2.0
    # A5: tail bound assumes h decreasing -- observe it.
    assert h_transform(z_hi, params) <= h_start + 1e-12, "h(z) not decreasing; tail bound invalid"

    # integrate: [0, z_thresh] ~ K*P*G_bar (flat); [z_thresh, z_hi] via log-substitution.
    val0 = K * P * Gbar * z_thresh
    main, _ = quad(lambda w: F(math.exp(w)) * math.exp(w),
                   math.log(z_thresh), math.log(z_hi), limit=n_sub)
    rate = (val0 + main) / ln2
    return rate, tail_err(z_hi)


# ----------------------------------------------------------------------------
# Edge-aware M-system rate (Fig.4 scaling; far interference for M > M_gate)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _y_grid(D_W: float, d: float, n: int = _POS_QUAD_N):
    """y-only Gauss-Legendre grid at u=0 (PA at user's x): r2 = y^2 + d^2, weights sum to 1."""
    x, wq = leggauss(n)
    y = -0.5 * D_W + 0.5 * (x + 1.0) * D_W
    return (y * y + d * d), (0.5 * wq)


def h_ell(z: float, ell: int, params: Mapping[str, Any]) -> float:
    """Per-interferer Laplace transform for a segment at distance ell (offset delta).

    h_ell(z) = E_pos[1 - e^{-phi r^2}(1 - e^{-z P eta / r^2})] over u in
    [ell*D_S-delta, (ell+1)*D_S-delta]. Generalises h_transform (ell=1). Source: GT3.
    Invariant 0 < h_ell <= 1 (z>=0); h_ell(0)=1; decreasing in z.
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    r2, w = _position_grid(ell * D_S - delta, (ell + 1) * D_S - delta, D_W, d)
    h = float(np.sum(w * (1.0 - np.exp(-phi * r2) * (1.0 - np.exp(-z * P * eta / r2)))))
    if z >= 0.0:
        assert 0.0 < h <= 1.0 + 1e-12, f"h_ell(ell={ell})={h} out of (0,1]"
    return h


def system_sum_rate_exact(
    K: int, M: int, params: Mapping[str, Any], home_placement: str = "u0"
) -> tuple[float, float]:
    """Edge-aware ergodic M-system sum rate (returns (rate, max_trunc_err)). Source: GT3.

    Sum over the M segments; each segment m's interference is the PRODUCT over the
    neighbours that ACTUALLY exist given its position (edge-aware):
        R_m = (1/ln2) int (1/z) e^{-zN0} (1-[g]^K) * Prod_{ell>=1} [h_ell(z)]^{K*n_sides(m,ell)} dz,
    n_sides(m,ell) = [m-ell>=0] + [m+ell<=M-1] in {0,1,2}. For M=3 the far terms (ell>=2)
    are present only for edge segments and are negligible => reduces to the 2-neighbour
    result. `home_placement`: "u0" (PA at user x, G=Gbar_inf, EDMA-faithful) or "uavg".

    Re-derivations (verified, not assumed):
      * z->0 limit of every per-segment integrand = K*P*Gbar (each h_ell(0)=1 => product
        ->1; 1-[g]^K ~ K*P*Gbar*z), self-verified at runtime.
      * tail bound uses the PRODUCT at z_hi (each h_ell decreasing => product decreasing);
        monotonicity asserted.
    """
    if home_placement not in ("u0", "uavg"):
        raise ValueError(f"home_placement must be 'u0' or 'uavg', got {home_placement!r}")
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    _require_symmetric_delta(delta, D_S)
    tol = float(params["phase3"]["z_quad_tol"])
    n_sub = int(params["phase3"]["z_grid_init"])
    ln2 = math.log(2.0)
    Gbar = G_bar_inf(params) if home_placement == "u0" else G_bar(params)
    if home_placement == "u0":
        r2h, wh = _y_grid(D_W, d)
    else:
        r2h, wh = _position_grid(-delta, D_S - delta, D_W, d)

    def g_of(z: float) -> float:
        return float(np.sum(wh * np.exp(-z * P * eta / r2h)))

    z_thresh = 1e-6 * d * d / (P * eta)
    total, max_te = 0.0, 0.0
    for m in range(M):
        counts = {}
        for ell in range(1, M):
            n = (1 if m - ell >= 0 else 0) + (1 if m + ell <= M - 1 else 0)
            if n > 0:
                counts[ell] = n

        def intf_factor(z: float) -> float:   # product of [h_ell]^{K*n_sides}
            p = 1.0
            for ell, n in counts.items():
                p *= h_ell(z, ell, params) ** (K * n)
            return p

        def F(z: float) -> float:
            if z <= z_thresh:
                return K * P * Gbar
            g = g_of(z)
            omg = 1.0 if g <= 0.0 else -math.expm1(K * math.log(g))
            return (1.0 / z) * math.exp(-z * N0) * intf_factor(z) * omg

        # self-verify the removable-singularity limit for the PRODUCT
        F_full = (1.0 / z_thresh) * math.exp(-z_thresh * N0) * intf_factor(z_thresh) * (
            -math.expm1(K * math.log(g_of(z_thresh))))
        assert abs(F_full - K * P * Gbar) < tol, f"m={m}: product limit off {F_full} vs {K*P*Gbar}"

        intf_start = intf_factor(z_thresh)
        z_hi = 1.0 / N0
        tail = lambda zhi: intf_factor(zhi) * float(exp1(zhi * N0)) / ln2
        while tail(z_hi) >= tol:
            z_hi *= 2.0
        assert intf_factor(z_hi) <= intf_start + 1e-12, f"m={m}: product not decreasing"

        val0 = K * P * Gbar * z_thresh
        main, _ = quad(lambda w: F(math.exp(w)) * math.exp(w),
                       math.log(z_thresh), math.log(z_hi), limit=n_sub)
        total += (val0 + main) / ln2
        max_te = max(max_te, tail(z_hi))
    return total, max_te


# ----------------------------------------------------------------------------
# Comparators / crossover
# ----------------------------------------------------------------------------
def R_det(K: int, params: Mapping[str, Any]) -> float:
    """Deterministic (mean-SINR) rate log2(1 + K P G_bar / (N0 + 2K P L1_exact)).

    Source: GT3 (CLAUDE.md 7.5). At K=1 reduces to the EDMA single-user form
    log2(1 + G_bar/(1/rho + 2 L1)). Backstop R_tilde_m <= R_bar_m (Jensen).
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    _require_symmetric_delta(delta, D_S)
    Gbar, L1 = G_bar(params), L1_exact(params)
    return math.log2(1.0 + (K * P * Gbar) / (N0 + 2.0 * K * P * L1))


def R_tilde(K: int, params: Mapping[str, Any]) -> float:
    """Jensen (mean-interference) ergodic rate R_tilde = E_S[log2(1+S/(N0+2K P L1))].

    Signal position-averaged (via the home transform g), interference REPLACED by its
    mean 2K P L1_exact -- i.e. the blockage windfall is stripped while position diversity
    is kept (contrast R_det, which strips position too). Backstop R_tilde <= R_bar
    (Jensen); this is the fair windfall-stripped NOMA rate vs interference-free TDMA.
    Source: GT3 (CLAUDE.md 7.5). Lifted from tests/test_phase3_analytic.py (P3-4) so the
    windfall-decomposition driver and the Jensen test share ONE source of truth.
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    _require_symmetric_delta(delta, D_S)
    Gbar, L1 = G_bar(params), L1_exact(params)
    c = N0 + 2.0 * K * P * L1
    z0 = 1e-6 * d * d / (P * eta)            # removable-singularity threshold (matches R_bar)

    def integ(w: float) -> float:
        z = math.exp(w)
        g = g_transform(z, params)
        omg = 1.0 if g <= 0.0 else -math.expm1(K * math.log(g))
        return (1.0 / z) * math.exp(-z * c) * omg * z   # *z from the log-substitution dz

    # integrand ~ e^{-zc}; 50/c covers ~50 e-foldings of the tail.
    main, _ = quad(integ, math.log(z0), math.log(50.0 / c), limit=256)
    return (K * P * Gbar * z0 + main) / math.log(2.0)


def gain_factor(K: int, params: Mapping[str, Any]) -> float:
    """NOMA-INTERNAL overloading SINR gain Gamma(K) = K(N0+2P L1)/(N0+2K P L1).

    Equals SINR_NOMA(K)/SINR_NOMA(1) -- how the K-user NOMA sum-SINR grows with K.
    Source: GT3 (CLAUDE.md 7.7). Invariant Gamma(K) in [1, K] (asserted). -> K when
    noise-limited (N0 >> leakage), -> 1 when interference-limited.
    NOTE: this is NOT the NOMA-vs-TDMA crossover -- EDMA's TDMA (Eq. 9) is
    interference-free, so the crossover is rho/power-model-governed (see system_rates),
    not Gamma(K).
    """
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    _require_symmetric_delta(delta, D_S)
    L1 = L1_exact(params)
    gamma = K * (N0 + 2.0 * P * L1) / (N0 + 2.0 * K * P * L1)
    assert 1.0 - 1e-9 <= gamma <= K + 1e-9, f"Gamma(K)={gamma} not in [1,{K}]"
    return gamma


def tdma_rate_eq9(
    power_model: str, K: int, M: float, params: Mapping[str, Any],
    placement: str = "u0",
) -> float:
    """EDMA TDMA sum rate (Eq. 9), VALIDATED to 2 dp against EDMA Table I (P-/C-TDMA).

        R_TDMA = (1/(MK)) * E_pos[ log2(1 + boost * rho * eta / r^2) ]   (interference-free)

    Orthogonal (one user active system-wide per slot) => INTERFERENCE-FREE; the MK total
    users are TIME-SHARED (the 1/(MK) factor -- the piece both prior wrong TDMA forms
    omitted), with avg-power boost MK (Version "B" = EDMA Eq.9 as published) or peak
    (Version "A" variant, no boost). Placement: "u0" => PA at active user's x (P-TDMA,
    r^2=y^2+d^2); "uavg" => fixed antenna at segment centre, user uniform over the segment
    (C-TDMA, segment width D_S=D_L/M). rho = P/N0. Source: EDMA Eq.(9) + Table I.
    """
    if power_model not in ("A", "B"):
        raise ValueError(f"power_model must be 'A' or 'B', got {power_model!r}")
    if placement not in ("u0", "uavg"):
        raise ValueError(f"placement must be 'u0' or 'uavg', got {placement!r}")
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    rho = P / N0
    boost = (M * K) if power_model == "B" else 1.0
    if placement == "u0":                                  # PA at user x: u=0, r^2=y^2+d^2
        val, _ = quad(lambda y: math.log2(1.0 + boost * rho * eta / (y * y + d * d)),
                      -D_W / 2.0, D_W / 2.0)
        mean = val / D_W
    else:                                                  # fixed antenna at segment centre
        D_Sm = float(params["D_L"]) / M                    # physical segment width for M segments
        val, _ = dblquad(
            lambda y, u: math.log2(1.0 + boost * rho * eta / (u * u + y * y + d * d)),
            -D_Sm / 2.0, D_Sm / 2.0, -D_W / 2.0, D_W / 2.0)
        mean = val / (D_Sm * D_W)
    return mean / (M * K)


def tdma_system_rate(
    power_model: str, K: int, M: float, params: Mapping[str, Any],
    placement: str = "u0",
) -> float:
    """SYSTEM sum rate of orthogonal (time-shared, interference-free) access, v8.

        R_TDMA^sys = sum_{i=1}^{MK} (1/(MK)) E_pos[ log2(1 + b rho eta / r^2) ]
                   = E_pos[ log2(1 + b rho eta / r^2) ]          (b = MK or 1)

    Each of the MK users holds a 1/(MK) slot; the system rate is the SUM of the users'
    time-share-weighted rates, the same convention as the equal-total-power time-sharing
    identity of Remark `rem:mactie` (sum_k tau_k log2(1 + P_k g_k/(tau_k F)) with the tau_k
    summing to one). It equals ``tdma_rate_eq9(...) * M * K``; it is computed here from
    the integrals directly so that identity is a TEST and not a definition.

    ``tdma_rate_eq9`` is left unchanged: it carries the extra 1/(MK) that reproduces the
    numbers recorded for EDMA's Table I, and its regression test still asserts them.
    EDMA's printed eq. (9), (1/M) sum_m log2(...), has an expectation WITHOUT that factor.
    Placement as in ``tdma_rate_eq9``.
    """
    if power_model not in ("A", "B"):
        raise ValueError(f"power_model must be 'A' or 'B', got {power_model!r}")
    if placement not in ("u0", "uavg"):
        raise ValueError(f"placement must be 'u0' or 'uavg', got {placement!r}")
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    rho = P / N0
    boost = (M * K) if power_model == "B" else 1.0
    if placement == "u0":
        val, _ = quad(lambda y: math.log2(1.0 + boost * rho * eta / (y * y + d * d)),
                      -D_W / 2.0, D_W / 2.0)
        return val / D_W
    D_Sm = float(params["D_L"]) / M
    val, _ = dblquad(
        lambda y, u: math.log2(1.0 + boost * rho * eta / (u * u + y * y + d * d)),
        -D_Sm / 2.0, D_Sm / 2.0, -D_W / 2.0, D_W / 2.0)
    return val / (D_Sm * D_W)


def system_rates(
    power_model: str, K: int, M: float, params: Mapping[str, Any]
) -> dict[str, float]:
    """System sum-rates {NOMA, TDMA, conv}. Source: GT3 (CLAUDE.md 7.7), EDMA Eq.5/Eq.9.

    All ERGODIC (to match EDMA Table I and be type-consistent):
      NOMA = M * ergodic_rate_exact(K)            (M segments simultaneous; EDMA Eq.5 side).
      TDMA = tdma_rate_eq9(power_model, K, M)      (EDMA Eq.9; interference-free, TIME-SHARED).
      conv = log2(1 + P Gbar_full / N0)            (single antenna, no segments).
    `power_model` REQUIRED ("A"/"B"). NOMA's advantage GROWS with K because TDMA's
    orthogonal time-share (1/(MK)) is crushed by overloading -- NOT because NOMA improves
    (NOMA's own rate falls with K; TDMA's falls faster). `M` is explicit (Sec.8 TODO).
    """
    if power_model not in ("A", "B"):
        raise ValueError(f"power_model must be 'A' or 'B', got {power_model!r}")
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    _require_symmetric_delta(delta, D_S)
    R_noma = M * ergodic_rate_exact(K, params)[0]
    R_tdma = tdma_rate_eq9(power_model, K, M, params)
    R_conv = math.log2(1.0 + (P * G_bar_full(params)) / N0)
    return {"NOMA": R_noma, "TDMA": R_tdma, "conv": R_conv}


def access_latency(scheme: str, M: float, K: int) -> float:
    """Access delay [slots until served] for orthogonal TDMA vs simultaneous NOMA.

    EDMA's TDMA (Eq. 9) time-shares all M*K users one-per-slot, so a serving round is
    M*K slots long (a user waits up to M*K-1 slots and is served once per round); we
    report the round length M*K as the access-delay scale. NOMA serves the K co-segment
    users simultaneously (all M segments in parallel) => delay 1. POWER-MODEL-
    INDEPENDENT: the 1/(MK) time-share is structural, identical under Version A and B
    (only the in-slot power boost differs). Source: EDMA Eq. (9) time-share factor.
    """
    if scheme not in ("TDMA", "NOMA"):
        raise ValueError(f"scheme must be 'TDMA' or 'NOMA', got {scheme!r}")
    return float(M * K) if scheme == "TDMA" else 1.0
