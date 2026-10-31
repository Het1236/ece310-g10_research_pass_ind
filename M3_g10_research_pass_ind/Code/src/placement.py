"""Per-segment pinching-antenna placement core (Phase 4, P4-1).

Sources: GT4 = Phase-4 ground truth, CLAUDE.md Sec. 9 (each function cites its item);
EDMA = Ding, Schober, Poor, IEEE TWC 25:15675-15691, 2026 (context/BasePaper/...).

Coordinate convention (LOCKED, P4-1): ABSOLUTE x-grid, segment m = 0..M-1 spans
[m*D_S, (m+1)*D_S] with D_S = D_L/M -- the same absolute layout as the system-rate MC
samplers (montecarlo.system_sum_rate_mc_*). geometry.py's helpers are CENTRE-RELATIVE
(centre-segment left edge at x=0, signed offsets s): the mapping is
x_abs = m_c*D_S + x_rel, with m_c the centre segment's absolute index. geometry.py is
NOT modified; placement takes user coordinates already in absolute form.

Model scope (GT4, P4-1 locked): the placement objective is the DETERMINISTIC
INSTANTANEOUS rate at a FIXED user drop with a CONSTANT interference-plus-noise floor
(I_m + N0) passed in as a parameter -- NOT the position-averaged ergodic quantities of
analytic.py (those are guarded by _require_symmetric_delta and are not functions of x).
Home/desired links are PURE LoS (no blockage coin), matching
montecarlo.ergodic_rate_mc's home convention. In-waveguide attenuation is NEGLECTED
(lossless low-loss-dielectric assumption, standard for EDMA/PASS -- GT4 channel gain);
in-waveguide phase drops out because only |h|^2 enters. Powers are FIXED/uniform in
P4-1 (per-user P); power optimization is P4-2.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from src import channel

# Golden-ratio conjugate tau = (sqrt(5)-1)/2 (EDMA Algorithm 1).
_TAU: float = (math.sqrt(5.0) - 1.0) / 2.0


def collected_power(
    x: float | np.ndarray,
    users_x: np.ndarray,
    users_y: np.ndarray,
    eta: float,
    d: float,
    P: float,
) -> float | np.ndarray:
    """Collected power G_m(x) = sum_k P * g_{m,k}(x). Source: GT4 collected-power proxy.

    g_{m,k}(x) = eta / ((x - x_{m,k})^2 + C_{m,k}), C_{m,k} = y_{m,k}^2 + d^2 (GT4
    channel gain = EDMA Eq. 2, squared distance in, repo convention: no sqrt anywhere;
    computed via channel.path_gain). Uniform per-user power P (P4-1).

    ARGMAX-EQUIVALENT proxy to the true rate (log2(1+c*G) strictly increasing in G) --
    used for diagnostics/unimodality checks ONLY, never as the production objective.

    `x` may be a scalar (returns float) or 1-D array (returns array of G values) --
    the vectorized form keeps ``exhaustive_max`` cheap.
    """
    x_arr = np.asarray(x, dtype=float)
    C = users_y * users_y + d * d                       # (K,)
    R2 = (x_arr[..., None] - users_x) ** 2 + C          # (..., K) squared distance
    G = P * channel.path_gain(R2, eta).sum(axis=-1)     # EDMA Eq. (2) via shared helper
    return float(G) if np.isscalar(x) or x_arr.ndim == 0 else G


def rate_at_position(
    x: float | np.ndarray,
    users_x: np.ndarray,
    users_y: np.ndarray,
    eta: float,
    d: float,
    P: float,
    floor: float | Callable[[float], float],
) -> float | np.ndarray:
    """TRUE instantaneous per-segment sum rate. Source: GT4 true instantaneous rate.

        R_m^sum(x) = log2(1 + G_m(x) / F(x)),  F = I_m + N0.

    ``floor`` may be a constant or a callable F(x). It MOVES WITH THE ANTENNA whenever the
    leakage is modelled honestly: the interfering users sit in the adjacent segments and
    do not move, but the antenna does, so the distance between them is a function of the
    placement variable. See ``floor_position``, which supplies F(x) and measures the swing
    (a factor of 9.7 across a segment on the committed parameters). A constant is accepted
    so that the frozen-floor case stays reachable for comparison.

    This -- not the collected-power proxy -- is what golden_section and exhaustive_max
    optimise. With a moving floor the two are no longer argmax-equivalent, so the proxy is
    a diagnostic only. Home links pure LoS (P4-1 locked). Scalar x -> float; array x ->
    array.
    """
    G = collected_power(x, users_x, users_y, eta, d, P)
    if callable(floor):
        F = np.array([floor(float(xi)) for xi in np.atleast_1d(x)])
        F = F.reshape(np.shape(G)) if isinstance(G, np.ndarray) else float(F[0])
    else:
        F = float(floor)
    if np.any(np.asarray(F) <= 0.0):
        raise ValueError("floor = I_m + N0 must be > 0")
    return np.log2(1.0 + G / F) if isinstance(G, np.ndarray) else math.log2(1.0 + G / F)


def golden_section(
    objective: Callable[[float], float],
    x_lo: float,
    x_hi: float,
    tol: float,
) -> tuple[float, float, int]:
    """Golden-section MAXIMIZATION of a unimodal objective. Source: EDMA Algorithm 1 (GT4).

    tau = (sqrt5-1)/2; two probes initialise the bracket, then each iteration reuses one
    probe and makes exactly ONE new objective evaluation (asserted by Gate F), shrinking
    the interval by tau until x_hi - x_lo <= tol. Returns (x_star, f_star, n_iter) with
    x_star the final midpoint and f_star = objective(x_star) (one final evaluation).
    Total evaluations = n_iter + 3. n_iter = O(log(1/tol)) (Gate F bound).

    Correct ONLY under unimodality (EDMA Lemma 4 sufficient condition delta^2*phi >= 1,
    K=1); outside that regime it converges to A local optimum -- certify against
    ``exhaustive_max`` (GT4 unimodality item).
    """
    if not x_hi > x_lo:
        raise ValueError(f"need x_hi > x_lo, got [{x_lo}, {x_hi}]")
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")

    x1 = x_hi - _TAU * (x_hi - x_lo)
    x2 = x_lo + _TAU * (x_hi - x_lo)
    f1 = objective(x1)
    f2 = objective(x2)
    n_iter = 0
    while (x_hi - x_lo) > tol:
        if f1 >= f2:            # max in [x_lo, x2]: drop the right end, reuse x1 as new x2
            x_hi, x2, f2 = x2, x1, f1
            x1 = x_hi - _TAU * (x_hi - x_lo)
            f1 = objective(x1)  # the ONE new evaluation this iteration
        else:                   # max in [x1, x_hi]: drop the left end, reuse x2 as new x1
            x_lo, x1, f1 = x1, x2, f2
            x2 = x_lo + _TAU * (x_hi - x_lo)
            f2 = objective(x2)  # the ONE new evaluation this iteration
        n_iter += 1
    x_star = 0.5 * (x_lo + x_hi)
    return x_star, objective(x_star), n_iter


def golden_section_kstart(
    objective: Callable[[float], float],
    x_lo: float,
    x_hi: float,
    user_xs: np.ndarray | Sequence[float],
    tol: float,
) -> tuple[float, float, int]:
    """K-start golden-section: basin decomposition at the users. Source: GT4 K-start
    algorithm (CLAUDE.md Sec. 9, P4-1.5; OURS - the K-general placement algorithm).

    Rationale: the TRUE rate is a sum of K unimodal (Lorentzian) bumps centred at the
    user x-positions, so every local maximum - hence the global one - lies in a
    user-anchored basin. EDMA Lemma 4's unimodality (delta^2*phi >= 1) governs K=1
    ONLY (phi is absent from the pure-LoS home objective), so for K >= 2 plain
    golden-section may lock onto the wrong hump (P4-1 finding). Remedy: decompose and
    conquer, then certify against ``exhaustive_max`` (Gate A/D).

    Bracket rule (LOCKED): sort user_xs, clip to [x_lo, x_hi]; split the interval at the
    MIDPOINTS between consecutive distinct users, giving <= K brackets, each containing
    exactly one user's basin; duplicate-user (zero-width) brackets are skipped. Run the
    unchanged ``golden_section`` (EDMA Algorithm 1) on each bracket over the SAME TRUE
    rate callable and return the best. At K=1 there is ONE bracket = [x_lo, x_hi], so
    this degenerates EXACTLY to a single golden_section call (EDMA collapse, Gate C).

    Complexity: <= K brackets x O(log(1/tol)) each => O(K log(1/tol)) per segment,
    O(M K log(1/tol)) over all segments (Gate F). Returns (x_star, f_star, total_iters)
    with total_iters summed over brackets.
    """
    xs = np.sort(np.clip(np.asarray(user_xs, dtype=float).ravel(), x_lo, x_hi))
    if xs.size == 0:
        raise ValueError("user_xs must contain at least one user position")
    mids = 0.5 * (xs[:-1] + xs[1:])
    edges = np.concatenate(([x_lo], mids, [x_hi]))

    best_x, best_f, total_iters = math.nan, -math.inf, 0
    for a, b in zip(edges[:-1], edges[1:]):
        if not b > a:                      # duplicate users => zero-width bracket: skip
            continue
        x_b, f_b, n_b = golden_section(objective, float(a), float(b), tol)
        total_iters += n_b
        if f_b > best_f:
            best_x, best_f = x_b, f_b
    return best_x, best_f, total_iters


def scan_refine_max(
    objective: Callable[[float], float],
    x_lo: float,
    x_hi: float,
    n_scan: int,
    tol: float,
) -> tuple[float, float, int]:
    """Coarse-scan + golden-section refine - the MAX-MIN x-step solver (P4-4.5).
    Source: GT4 scan-refine item (CLAUDE.md Sec. 9); fixes the ⚠️ P4-4 finding.

    Rationale: the max-min block objective min_k R_k(x) is the LOWER ENVELOPE of K
    rate curves - non-smooth (kinked at argmin crossings) and with maxima at CROSSING/
    BALANCE points that are NOT user-anchored, so the K-start basin decomposition is
    INVALID for it (P4-4 pinned counterexample). A uniform n_scan-point scan is robust
    to kinks and locates the correct basin (the grid includes both segment endpoints,
    where envelope maxima can sit); the unchanged ``golden_section`` then refines
    inside the +-1-grid-step bracket around the best scan point, which is locally
    unimodal for any scan dense enough - certified against ``exhaustive_max`` in Gate
    A-maxmin (config: phase4.maxmin_scan_points). The overall best of (scan grid,
    refined point) is returned, so refinement can never lose to the scan.

    Complexity: O(n_scan + log(1/tol)) evaluations - the HONEST cost of the non-smooth
    fairness sub-problem; the log-only claim remains sum-rate/K-start ONLY.
    Returns (x_star, f_star, n_evals) with n_evals the total objective evaluations.

    NOT used for the sum-rate objective (that stays on ``golden_section_kstart``,
    which is exact for the weighted sum-of-bumps and cheaper).
    """
    if not x_hi > x_lo:
        raise ValueError(f"need x_hi > x_lo, got [{x_lo}, {x_hi}]")
    if n_scan < 2:
        raise ValueError(f"n_scan must be >= 2, got {n_scan}")
    grid = np.linspace(x_lo, x_hi, n_scan)               # includes both endpoints
    f = np.array([float(objective(float(x))) for x in grid])
    i = int(np.argmax(f))
    best_x, best_f = float(grid[i]), float(f[i])
    n_evals = n_scan

    lo, hi = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, n_scan - 1)])
    x_ref, f_ref, n_iter = golden_section(objective, lo, hi, tol)
    n_evals += n_iter + 3
    if f_ref > best_f:                                   # refinement can only add
        best_x, best_f = x_ref, f_ref
    return best_x, best_f, n_evals


def exhaustive_max(
    objective: Callable[[float | np.ndarray], float | np.ndarray],
    x_lo: float,
    x_hi: float,
    n_points: int,
) -> tuple[float, float]:
    """Dense-grid maximization -- CERTIFICATION REFERENCE ONLY, never production placement.

    Source: GT4 unimodality item (empirical validation where EDMA Lemma 4's sufficient
    condition fails or K > 1). Evaluates `objective` on a uniform n_points grid over
    [x_lo, x_hi] (vectorized when the callable supports array x, else pointwise) and
    returns (x_star, f_star) at the grid argmax.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    grid = np.linspace(x_lo, x_hi, n_points)
    try:
        f = np.asarray(objective(grid), dtype=float)
        if f.shape != grid.shape:
            raise TypeError("objective is not vectorized")
    except TypeError:
        f = np.array([float(objective(float(x))) for x in grid])
    i = int(np.argmax(f))
    return float(grid[i]), float(f[i])


def midpoint_placement(
    x_left_anchor: float,
    x_right_anchor: float,
    x_lo: float,
    x_hi: float,
) -> float:
    """Closed-form placement x* = (x_left_anchor + x_right_anchor)/2, clamped to [x_lo, x_hi].

    Source: EDMA Remark 8 (GT4): in the symmetric high-blockage y=0 regime the optimum
    is the anchor midpoint when interior; otherwise the nearer segment boundary (the
    clamp). Low-complexity rule + verification benchmark for golden_section (Gate B).
    Anchors are the neighbouring segments' reference points.
    """
    if not x_hi > x_lo:
        raise ValueError(f"need x_hi > x_lo, got [{x_lo}, {x_hi}]")
    mid = 0.5 * (x_left_anchor + x_right_anchor)
    return min(max(mid, x_lo), x_hi)


def delta_from_anchors(x_left_anchor: float, x_right_anchor: float) -> float:
    """Half anchor spacing delta = (x_{m+1} - x_{m-1})/2 for EDMA Lemma 4. Source: GT4."""
    return 0.5 * (x_right_anchor - x_left_anchor)


def unimodality_flag(delta: float, phi: float) -> bool:
    """EDMA Lemma 4 SUFFICIENT condition for unimodality: delta^2 * phi >= 1. Source: GT4.

    SUFFICIENT, NOT NECESSARY -- and EDMA-proved at K=1. Where False (or K > 1), do NOT
    claim unimodality; fall back to empirical validation vs ``exhaustive_max`` (Gate D
    logs the empirical modality instead of asserting it).
    """
    return delta * delta * phi >= 1.0


def optimize_all_segments(
    users_x: np.ndarray,
    users_y: np.ndarray,
    eta: float,
    d: float,
    P: float,
    floor: float | Sequence[float],
    D_L: float,
    M: int,
    tol: float,
) -> list[dict[str, float | int]]:
    """Independent per-segment placement: M golden_section_kstart calls. Source: GT4
    separability + K-start algorithm (P4-1.5).

    UPLINK separability (licensed by Phase-2's near-constant I_m floor): segment m's rate
    depends only on its OWN antenna and OWN users, so the M-segment problem separates
    into M independent 1-D searches -- segments are NOT coupled (Gate E). Each segment
    runs the K-start basin decomposition on its own users' x-positions. Total work =
    M x O(K log(1/tol)) (Gate F bound).

    Args:
        users_x, users_y: ABSOLUTE user coordinates, shape (M, K) (segment m's users in
            row m; x within [m*D_S, (m+1)*D_S], D_S = D_L/M).
        floor: constant I_m + N0; scalar (shared) or per-segment sequence of length M.
        Remaining args as in ``rate_at_position`` / ``golden_section``.

    Returns:
        list of M dicts {'m', 'x_star', 'f_star', 'n_iter'}.
    """
    users_x = np.asarray(users_x, dtype=float)
    users_y = np.asarray(users_y, dtype=float)
    if users_x.shape != users_y.shape or users_x.shape[0] != M:
        raise ValueError(f"users_x/users_y must both be (M, K) with M={M}, "
                         f"got {users_x.shape} and {users_y.shape}")
    floors = [float(floor)] * M if np.isscalar(floor) else [float(f) for f in floor]
    if len(floors) != M:
        raise ValueError(f"floor must be scalar or length-{M}, got length {len(floors)}")
    D_S = D_L / M

    results: list[dict[str, float | int]] = []
    for m in range(M):
        ux, uy, fl = users_x[m], users_y[m], floors[m]

        def obj(x: float, ux=ux, uy=uy, fl=fl) -> float:
            return rate_at_position(x, ux, uy, eta, d, P, fl)

        x_star, f_star, n_iter = golden_section_kstart(
            obj, m * D_S, (m + 1) * D_S, ux, tol)
        results.append({"m": m, "x_star": x_star, "f_star": f_star, "n_iter": n_iter})
    return results
