"""Closed-form max-min power allocation and a provable Lipschitz constant for the
max-min value function (Phase 4 speed/certification track, V4 plan items A3 + F-2).

WHY THIS MODULE EXISTS
----------------------
Two separate defects in the previous max-min placement route, both fixed here:

  (1) SPEED. ``power_alloc.maxmin_bisection`` answers "what is the largest common rate
      floor achievable at this position" by bisecting on t and calling a general-purpose
      LP (scipy/HiGHS) as the feasibility oracle at every step: ~23 LP solves per call,
      ~36 ms. The certified placement search calls it ~230 times per segment, so a single
      segment costs ~8 s. The question has a CLOSED FORM (below), so the LP is unnecessary.

  (2) CERTIFICATION. ``maxmin_global.cell_lipschitz_V`` ESTIMATES the per-cell Lipschitz
      constant by probing 60 points and inflating the largest observed slope by 1.5. A
      sampled slope is a LOWER bound on the Lipschitz constant, and inflating a lower
      bound does not produce an upper bound, so the Piyavskii-Shubert "certificate" built
      on it was heuristic. ``lipschitz_V`` below is a PROVEN upper bound.

NEITHER ``power_alloc`` NOR ``maxmin_global`` IS MODIFIED BY THIS FILE. This module is
additive; ``power_alloc.maxmin_bisection`` is retained unchanged and is used as the
reference oracle in tests/test_maxmin_closed.py Gate A.

THE CLOSED FORM
---------------
Fix a PA position, hence the gains g_k, and the interference floor F = I_m + N0 > 0. Fix
the SIC decoding order (``power_alloc.sic_order``, strongest-gain-first); index users by
DECODE POSITION j = 0..K-1, so j = 0 is decoded first and sees every other user as
interference, and j = K-1 is decoded last and sees only F.

Let gamma = 2^t - 1 and S_j = sum_{l >= j} P_l g_l (in decode order). Requiring rate
exactly t for the user at position j gives P_j g_j = gamma (F + S_{j+1}), hence the
backward recursion

    S_j = (1 + gamma) S_{j+1} + gamma F,      S_K = 0,

whose solution is S_j = F[(1+gamma)^{K-j} - 1], and therefore

    P_j g_j = gamma (1 + gamma)^{K-1-j} F.                             (*)

(*) is the COMPONENTWISE-MINIMAL power vector achieving R_k >= t for every k under this
order: each stage's requirement is forced once the later stages are fixed, and the
recursion is solved from the last stage backwards, so no feasible vector can be smaller in
any coordinate. Cross-check: sum_j P_j g_j = F[(1+gamma)^K - 1], so the telescoping SIC
identity gives sum rate log2(1 + sum_j P_j g_j / F) = K t, as it must.

The max-min floor is then the largest t for which (*) respects the power constraints:

  Version "A" (per-user peak P_max):  gamma (1+gamma)^{e_j} <= P_max g_j / F for every j,
      with e_j = K-1-j. Each is a scalar monotone equation in gamma, and
      t* = min_j t_j. (Componentwise minimality is what makes "min over users" exact.)

  Version "B" (shared budget P_tot):  Omega(gamma) = F sum_j gamma (1+gamma)^{e_j}/g_j
      <= P_tot. Omega is strictly increasing in gamma, so t* solves Omega(gamma) = P_tot.

Both are solved by safeguarded Newton in w = ln(gamma), which keeps gamma > 0 and keeps
the (1+gamma)^e factors in log space (no overflow at large K).

THE LIPSCHITZ BOUND
-------------------
Within a smooth cell the decode order is fixed, so the exponents e_j are constants.

  Version A. V(x) = min_j V_j(x) with V_j = log2(1 + gamma_j) and Q(gamma_j) = r_j(x),
  where Q(gamma) = gamma (1+gamma)^{e} and r_j(x) = P_max g_j(x)/F. (Q is named Q, not F,
  so it does not collide with the floor F.) Since

      (1+gamma) Q'(gamma) = (1+gamma)^e (1 + gamma + e gamma)
                         >= (1+gamma)^e gamma = Q(gamma) = r,

  implicit differentiation gives

      |dV_j/dx| = |r_j'| / ((1+gamma_j) Q'(gamma_j) ln2)
               <= |r_j'| / (r_j ln2) = |g_j'(x)| / (g_j(x) ln2),

  and with g(x) = eta/((x-x_k)^2 + C_k), C_k = y_k^2 + d^2,

      |g'/g| = 2|x - x_k| / ((x - x_k)^2 + C_k) <= 1/sqrt(C_k)

  (maximised at |x - x_k| = sqrt(C_k)). A pointwise minimum of functions has
  |d min_j/dx| <= max_j |dV_j/dx| wherever it is differentiable, so

      L = 1 / (ln2 * min_k sqrt(y_k^2 + d^2))  <=  1 / (d ln2).        (**)

  Version B. Omega(gamma) = sum_j a_j gamma (1+gamma)^{e_j} with a_j = F/g_j(x). The same
  termwise inequality gives (1+gamma) Omega'(gamma) >= Omega, and differentiating
  Omega(gamma(x)) = P_tot implicitly yields |dV/dx| <= max_j |g_j'/g_j| / ln2 again, so
  (**) holds for BOTH power models.

(**) is geometry-only: independent of K, of the powers, of the SNR, and of the blockage.
Gate D of tests/test_maxmin_closed.py checks it against a dense finite-difference
reference; the bound held on every instance tested and is 1.6x-5x loose.

A MOVING FLOOR
--------------
(**) was derived with F held constant in x. It is not. The interferers sit in the adjacent
segments and do not move, but the antenna does, so the leakage reaching it depends on the
offset; see ``floor_position`` for the measurement and the correction. Both walls above
contain the gains only through the ratio g_j(x)/F(x), so allowing F to vary replaces
|g_j'/g_j| by |g_j'/g_j - F'/F| everywhere, and the triangle inequality gives

    |dV/dx| <= [ max_k |g_k'/g_k| + max_x |F'/F| ] / ln2.

``lipschitz_V`` therefore takes the floor's logarithmic-derivative bound as an argument.
``floor_position.floor_logderiv_bound`` supplies it in the same geometry-only, load-free
form, so the constant keeps every invariance the certified search relies on.
"""
from __future__ import annotations

import math

import numpy as np

from src.power_alloc import sic_order

# Newton settings for the scalar root solves. The iteration is on w = ln(gamma), where
# every residual below is monotone increasing with derivative bounded away from zero, so
# convergence is fast and the bisection safeguard is a formality (it exists so that a
# pathological instance degrades gracefully instead of returning a wrong root).
_NEWTON_MAX_ITER: int = 100
_NEWTON_TOL: float = 1e-13
_W_LO: float = -80.0        # gamma ~ 1.8e-35, far below any physical rate floor
_W_HI: float = 80.0         # gamma ~ 5.5e34,  far above any physical rate floor


def _solve_versionA_w(e: float, log_r: float) -> float:
    """Root w = ln(gamma) of  ln(gamma) + e*ln(1+gamma) = log_r  (Version-A wall).

    The residual is strictly increasing in w with derivative in [1, 1+e], so Newton from
    w = 0 converges; the bracket [_W_LO, _W_HI] safeguards it.
    """
    lo, hi = _W_LO, _W_HI
    w = 0.0
    for _ in range(_NEWTON_MAX_ITER):
        g = math.exp(w)
        res = w + e * math.log1p(g) - log_r
        if res > 0.0:
            hi = min(hi, w)
        else:
            lo = max(lo, w)
        if abs(res) < _NEWTON_TOL:
            return w
        der = 1.0 + e * g / (1.0 + g)
        w_new = w - res / der
        if not (lo < w_new < hi):
            w_new = 0.5 * (lo + hi)
        if abs(w_new - w) < _NEWTON_TOL:
            return w_new
        w = w_new
    return w


def _log_omega(w: float, e: np.ndarray, log_a: np.ndarray) -> tuple[float, float]:
    """log(Omega(gamma)) and d log(Omega)/dw at gamma = exp(w), Version-B budget function.

    Omega(gamma) = sum_j a_j * gamma * (1+gamma)^{e_j}; every term is evaluated in log
    space so that (1+gamma)^{e_j} cannot overflow at large K.
    """
    g = math.exp(w)
    l1p = math.log1p(g)
    terms = log_a + w + e * l1p                       # log of each term
    tmax = float(terms.max())
    wts = np.exp(terms - tmax)
    ssum = float(wts.sum())
    log_om = tmax + math.log(ssum)
    # d/dw of (w + e*log1p(g)) is 1 + e*g/(1+g); weight by each term's share.
    dlog = float((wts * (1.0 + e * g / (1.0 + g))).sum() / ssum)
    return log_om, dlog


def _solve_versionB_w(e: np.ndarray, log_a: np.ndarray, log_Ptot: float) -> float:
    """Root w = ln(gamma) of  log Omega(exp(w)) = log_Ptot  (Version-B budget)."""
    lo, hi = _W_LO, _W_HI
    w = 0.0
    for _ in range(_NEWTON_MAX_ITER):
        log_om, dlog = _log_omega(w, e, log_a)
        res = log_om - log_Ptot
        if res > 0.0:
            hi = min(hi, w)
        else:
            lo = max(lo, w)
        if abs(res) < _NEWTON_TOL:
            return w
        w_new = w - res / dlog
        if not (lo < w_new < hi):
            w_new = 0.5 * (lo + hi)
        if abs(w_new - w) < _NEWTON_TOL:
            return w_new
        w = w_new
    return w


def minpower_sic(
    gamma: float, gains: np.ndarray, floor: float, order: np.ndarray | None = None
) -> np.ndarray:
    """Componentwise-minimal powers achieving R_k >= t for every k, gamma = 2^t - 1.

    Implements (*) from the module docstring: the user at decode position j needs
    P_j g_j = gamma (1+gamma)^{K-1-j} floor. Returned in ORIGINAL user indexing, matching
    ``power_alloc.power_lp`` / ``power_alloc.per_user_rates``.

    ``order`` defaults to ``power_alloc.sic_order`` (strongest-gain-first), which is
    constraint (C5) of (P1)/(P2); the recursion is valid ONLY for a fixed order.
    """
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 1 or gains.size == 0 or np.any(gains <= 0.0):
        raise ValueError("gains must be a non-empty 1-D array of positive values")
    if floor <= 0.0:
        raise ValueError(f"floor = I_m + N0 must be > 0, got {floor}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    order = sic_order(gains) if order is None else np.asarray(order, dtype=int)
    K = gains.size
    P = np.empty(K, dtype=float)
    for j, k in enumerate(order):
        P[k] = gamma * (1.0 + gamma) ** (K - 1 - j) * floor / gains[k]
    return P


def maxmin_closed(
    gains: np.ndarray,
    floor: float,
    power_model: str,
    P_max: float | None = None,
    P_tot: float | None = None,
) -> tuple[float, np.ndarray]:
    """Exact max-min fair rate floor t* and the powers achieving it, in closed form.

    Drop-in replacement for the (t_star, P_star) pair returned by
    ``power_alloc.maxmin_bisection``, with two differences that are both improvements:
    t* is the EXACT optimum rather than the last certified-feasible bisection iterate
    (so this value is >= the bisection's by up to its t_tol), and no LP is solved.

    Version "A" (per-user peak P_max) and "B" (shared budget P_tot) follow the module
    docstring. Always feasible for some t > 0 whenever every gain is positive, so unlike
    the QoS LP there is no infeasible branch.
    """
    if power_model not in ("A", "B"):
        raise ValueError(f"power_model must be 'A' or 'B', got {power_model!r}")
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 1 or gains.size == 0 or np.any(gains <= 0.0):
        raise ValueError("gains must be a non-empty 1-D array of positive values")
    if floor <= 0.0:
        raise ValueError(f"floor = I_m + N0 must be > 0, got {floor}")

    order = sic_order(gains)
    gs = gains[order]                       # gains in decode order
    K = gs.size
    e = (K - 1 - np.arange(K)).astype(float)

    if power_model == "A":
        if P_max is None:
            raise ValueError("Version A requires P_max (per-user peak)")
        if P_max <= 0.0:
            raise ValueError(f"P_max must be > 0, got {P_max}")
        # gamma_j solves gamma (1+gamma)^{e_j} = P_max g_j / floor; t* = min_j t_j.
        log_r = np.log(P_max) + np.log(gs) - math.log(floor)
        w_star = min(_solve_versionA_w(float(e[j]), float(log_r[j])) for j in range(K))
    else:
        if P_tot is None:
            raise ValueError("Version B requires P_tot (shared budget)")
        if P_tot <= 0.0:
            raise ValueError(f"P_tot must be > 0, got {P_tot}")
        log_a = math.log(floor) - np.log(gs)         # a_j = floor / g_j
        w_star = _solve_versionB_w(e, log_a, math.log(P_tot))

    gamma = math.exp(w_star)
    t_star = math.log1p(gamma) / math.log(2.0)
    return t_star, minpower_sic(gamma, gains, floor, order=order)


def lipschitz_V(users_y: np.ndarray, d: float, floor_logderiv: float = 0.0) -> float:
    """PROVEN Lipschitz upper bound for the max-min value function V(x).

        L = [ 1/min_k sqrt(y_k^2 + d^2) + floor_logderiv ] / ln2.

    The first term is (**) of the module docstring. The second admits a floor that moves
    with the antenna: every wall in the derivation contains the gains only through the
    ratio g_j(x)/F(x), so with F a function of x the two logarithmic derivatives add and
    ``floor_logderiv`` is any bound on |F'(x)/F(x)|. Supply
    ``floor_position.floor_logderiv_bound(params)``, which is geometry-only and holds at
    every load. The default 0.0 recovers the constant-floor bound.

    Holds for BOTH power models. Independent of K, of the powers, of the SNR and of the
    blockage level, and costs zero evaluations of V.

    This REPLACES ``maxmin_global.cell_lipschitz_V``, which probed 60 points and inflated
    the largest observed slope; a sampled slope is a lower bound on the Lipschitz
    constant, so that estimate could not certify anything.
    """
    C = np.asarray(users_y, dtype=float) ** 2 + float(d) ** 2
    if C.size == 0:
        raise ValueError("users_y must be non-empty")
    if floor_logderiv < 0.0:
        raise ValueError(f"floor_logderiv must be >= 0, got {floor_logderiv}")
    return (1.0 / math.sqrt(float(C.min())) + float(floor_logderiv)) / math.log(2.0)
