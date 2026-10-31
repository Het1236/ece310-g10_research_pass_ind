"""Certified-global max-min PA placement (Proposition 6 / app:ao). Sources: this-session
convergence track (CLAUDE.md status log, dated entry); proof in paper/sections/appendix.tex
Appendix F.

Replaces the AO x-step (``altopt.ao_multistart`` -> ``placement.scan_refine_max``) as the
PRODUCTION solver for the max-min placement problem (P2). ``altopt.ao_multistart`` is
retained unchanged and is now used ONLY to produce the explicit "naive frozen-power
alternation" comparison baseline (the crosses in Fig. 6(b)); it is no longer the source
of the plotted max-min curve.

WHY a closed-form cell decomposition works here (mirrors app:ao (i)-(iii)):
  (i)   For fixed x the per-segment SIC decoding order is the order of the gains
        g_{m,k}(x) = eta / ((x - x_k)^2 + C_k). Two gains cross where the order flips;
        cancelling the common eta and clearing denominators leaves a LINEAR equation in x
        (the x^2 terms cancel), giving a single closed-form root per user pair --
        ``sic_breakpoints``. Between consecutive breakpoints the decoding order -- hence
        the inner problem's active-constraint structure -- is fixed, so the max-min value
        V(x) is continuous and piecewise smooth on each cell. Within a cell V(x) is
        available in CLOSED FORM (``maxmin_closed.maxmin_closed``): the componentwise-
        minimal SIC powers are P_j g_j = gamma (1+gamma)^{K-1-j} (I_m+N0), so the floor is
        the root of one monotone scalar equation and no LP is solved.
  (ii)  On a smooth cell, V is Lipschitz with a constant available IN CLOSED FORM:
        L = 1/(ln2 * min_k sqrt(y_k^2 + d^2)) <= 1/(d ln2), geometry-only and independent
        of K, of the powers, of the SNR and of the blockage. Derived and validated in
        ``maxmin_closed.lipschitz_V`` / tests/test_maxmin_closed.py Gate D, and used by
        default here (``deterministic_global(..., lipschitz="analytic")``).
        ⚠️ The former route, ``cell_lipschitz_V``, ESTIMATED the constant from a 60-point
        probe inflated by 1.5. A sampled slope is a LOWER bound on the Lipschitz constant,
        so that estimate could not certify anything; it is retained as
        ``lipschitz="probe"`` for comparison only.
  (iii) With a valid Lipschitz constant, the Piyavskii-Shubert method
        (piyavskii1972algorithm, shubert1972sequential) maintains a piecewise-linear upper
        envelope of V and always samples next at the envelope's current maximiser. The
        gap between the envelope max and the best sample is a valid, monotonically
        decreasing upper bound on the remaining optimality gap, so the search certifies
        V(x*) >= max V - tol in finitely many DETERMINISTIC evaluations -- no random
        restarts, so repeated runs return the identical solution (variance 0).

Taking the best incumbent over the finitely many cells gives the certified GLOBAL max-min
placement for the whole segment. This is unlike ``altopt.ao_multistart``, whose x-step
(``placement.scan_refine_max``) optimises a FIXED-POWER surrogate of the min-rate lower
envelope and only reaches a coordinate-wise/KKT-stationary point of the true joint (position,
power) problem -- hence its init-dependent spread (Fig. 6(b) crosses).

Validated reference numbers (this session, reused here rather than re-derived):
  pinned instance   (ux=[12.416,10.171], uy=[-1.790,-3.107], K=2): V*=2.579661 at x*=12.415,
                     gap=9.63e-4, n_eval=228, variance 0 over 25 runs, matches an 8000-point
                     exhaustive reference to |DV|=2.34e-9.
  disparity instance(ux=[4.0,9.0],      uy=[0.3,4.6],       K=2): V*=2.77412  at x*=3.999,
                     gap=9.95e-4, variance 0 over 25 runs; V(x) genuinely multimodal
                     (global at x=4.0, secondary local max at x=9.0, breakpoint at x=8.607).
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src import channel, maxmin_closed, power_alloc

# Piyavskii-Shubert default certified-gap tolerance (bps/Hz). Matches the tolerance used
# to produce the validated reference numbers above.
_DEFAULT_TOL: float = 1e-3


def sic_breakpoints(
    users_x: np.ndarray, users_y: np.ndarray, d: float, lo: float, hi: float
) -> list[float]:
    """Closed-form SIC decoding-order breakpoints in (lo, hi) (app:ao (i)).

    Two users' gains g_i(x) = eta/((x-x_i)^2 + C_i), g_j(x) = eta/((x-x_j)^2 + C_j) cross
    where g_i(x) = g_j(x); cancelling eta and clearing denominators cancels the x^2 terms,
    leaving a single linear root per pair. At most K(K-1)/2 breakpoints per segment.
    """
    users_x = np.asarray(users_x, dtype=float)
    C = np.asarray(users_y, dtype=float) ** 2 + float(d) ** 2
    bps: list[float] = []
    K = users_x.size
    for i in range(K):
        for j in range(i + 1, K):
            den = 2.0 * (users_x[j] - users_x[i])
            if abs(den) < 1e-15:
                continue  # coincident x: no crossing root (degenerate, excluded)
            x = ((users_x[j] ** 2 + C[j]) - (users_x[i] ** 2 + C[i])) / den
            if lo < x < hi:
                bps.append(float(x))
    return sorted(bps)


def cell_lipschitz_V(
    V: Callable[[float], float], a: float, b: float, n_probe: int = 60, inflate: float = 1.5
) -> float:
    """LEGACY, NOT A BOUND. Finite-difference Lipschitz ESTIMATE for V on [a, b]: probes
    ``n_probe`` points and inflates the max observed slope.

    ⚠️ A sampled slope is a LOWER bound on the Lipschitz constant, and inflating a lower
    bound does not produce an upper bound, so a Piyavskii-Shubert search run with this
    constant does NOT certify anything. Superseded by
    ``maxmin_closed.lipschitz_V``, which is a PROVEN geometry-only upper bound
    (1/(ln2 * min_k sqrt(y_k^2+d^2)) <= 1/(d ln2)); see that module's docstring for the
    derivation and tests/test_maxmin_closed.py Gate D for its validation.

    Retained only so ``deterministic_global(..., lipschitz="probe")`` can reproduce the
    pre-fix behaviour for comparison. Do not use it on a path that claims a certificate.
    """
    xs = np.linspace(a, b, n_probe)
    vs = np.array([V(x) for x in xs])
    slopes = np.abs(np.diff(vs)) / np.diff(xs)
    return float(slopes.max() * inflate)


def piyavskii_max(
    V: Callable[[float], float], a: float, b: float, L: float,
    tol: float = _DEFAULT_TOL, max_eval: int = 4000,
) -> tuple[float, float, float, int]:
    """Piyavskii-Shubert certified global maximiser of V on [a, b] given Lipschitz
    constant L (app:ao (iii)). Deterministic: maintains a saw-tooth upper envelope built
    from sampled values and slope L, always samples next at the envelope's current
    maximiser, and stops once the envelope-max-to-incumbent gap is <= tol.

    Returns (x_best, f_best, upper_bound, n_eval).
    """
    fa, fb = V(a), V(b)
    n_eval = 2

    def ub(xi: float, xj: float, fi: float, fj: float) -> float:
        return 0.5 * (fi + fj) + 0.5 * L * (xj - xi)

    heap: list[tuple[float, float, float, float, float]] = [(-ub(a, b, fa, fb), a, b, fa, fb)]
    x_best, f_best = (a, fa) if fa >= fb else (b, fb)
    while heap and n_eval < max_eval:
        neg_u, xi, xj, fi, fj = heapq.heappop(heap)
        u = -neg_u
        if u - f_best <= tol:
            heapq.heappush(heap, (neg_u, xi, xj, fi, fj))
            break
        # Envelope maximiser on [xi, xj]: the rising line fi + L(x - xi) meets the falling
        # line fj + L(xj - x) at x = (xi + xj)/2 + (fj - fi)/(2L), where the envelope
        # equals ub(). The sign was previously reversed, which sampled the MIRROR point:
        # the certificate stayed valid (each child bound ub() is correct and cannot exceed
        # its parent's), but the search did not sample the envelope maximiser that the
        # paper's Algorithm 1 and Appendix F describe.
        xm = 0.5 * (xi + xj) + (fj - fi) / (2.0 * L)
        xm = min(max(xm, xi), xj)
        fm = V(xm)
        n_eval += 1
        if fm > f_best:
            x_best, f_best = xm, fm
        for xl, xr, fl, fr in ((xi, xm, fi, fm), (xm, xj, fm, fj)):
            if xr - xl > 1e-12:
                heapq.heappush(heap, (-ub(xl, xr, fl, fr), xl, xr, fl, fr))
    upper = max((-h[0] for h in heap), default=f_best)
    return x_best, f_best, upper, n_eval


def deterministic_global(
    V: Callable[[float], float], users_x: np.ndarray, users_y: np.ndarray, d: float,
    lo: float, hi: float, tol: float = _DEFAULT_TOL, inflate: float = 1.5,
    lipschitz: str = "analytic", floor_logderiv: float = 0.0, use_cells: bool = False,
) -> dict[str, Any]:
    """Certified global maximiser of the max-min value function V(x) over [lo, hi]
    (Proposition 6 / app:ao). Runs a Piyavskii-Shubert search with a proven Lipschitz
    constant and returns the best incumbent together with its certified optimality gap.
    Deterministic sample points => identical result on every call (variance 0).

    ``lipschitz`` selects where the constant comes from:
      "analytic" (DEFAULT, the only certifying option) -- ``maxmin_closed.lipschitz_V``,
          a PROVEN geometry-only upper bound. Costs zero evaluations of V.
      "probe" (LEGACY, does NOT certify) -- ``cell_lipschitz_V``'s 60-point estimate,
          retained only to reproduce the pre-fix behaviour for comparison.

    ``floor_logderiv`` is the bound on |F'(x)/F(x)| passed to ``lipschitz_V`` when the
    interference floor moves with the antenna; 0.0 recovers the constant-floor constant.

    ``use_cells`` selects whether the segment is first split at the SIC decoding-order
    breakpoints:
      False (DEFAULT) -- ONE search over [lo, hi]. Piyavskii-Shubert needs Lipschitz
          continuity, not differentiability, and the analytic constant bounds the slope on
          every piece, so V is globally L-Lipschitz and a single search carries the
          identical certificate at a lower evaluation count. The breakpoints remain the
          reason the piecewise slope bound exists; they are a step in the proof, not a
          step in the algorithm.
      True (LEGACY) -- the per-cell search. Required for ``lipschitz="probe"``, whose
          estimate is per-cell by construction, and retained so the two can be compared.

    Returns {x, f, gap, n_eval, breakpoints, cells, L_source}.
    """
    if lipschitz not in ("analytic", "probe"):
        raise ValueError(f"lipschitz must be 'analytic' or 'probe', got {lipschitz!r}")
    bps = sic_breakpoints(users_x, users_y, d, lo, hi)
    if lipschitz == "probe" and not use_cells:
        use_cells = True  # a probed constant is only ever defined per cell
    edges = [lo] + bps + [hi] if use_cells else [lo, hi]
    n_eval_total = 0
    x_best, f_best, worst_gap = None, -np.inf, 0.0
    cells: list[dict[str, Any]] = []
    # The analytic bound is geometry-only, so it is valid on every cell and is computed
    # once outside the loop; the probe estimate is per-cell by construction.
    L_analytic = (maxmin_closed.lipschitz_V(users_y, d, floor_logderiv=floor_logderiv)
                  if lipschitz == "analytic" else None)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1e-9:
            continue
        if lipschitz == "analytic":
            L = L_analytic
        else:
            L = cell_lipschitz_V(V, a, b, inflate=inflate)
            n_eval_total += 60  # cell_lipschitz_V's probe budget (n_probe default)
        xc, fc, uc, ne = piyavskii_max(V, a, b, L, tol=tol)
        n_eval_total += ne
        cells.append(dict(a=a, b=b, L=L, x=xc, f=fc, upper=uc, gap=uc - fc, n_eval=ne))
        if fc > f_best:
            x_best, f_best = xc, fc
        worst_gap = max(worst_gap, uc - f_best)
    return dict(x=x_best, f=f_best, gap=worst_gap, n_eval=n_eval_total,
                breakpoints=bps, cells=cells, L_source=lipschitz)


def maxmin_value_function(
    users_x: np.ndarray, users_y: np.ndarray, d: float,
    eta: float, floor: float | Callable[[float], float],
    power_model: str, P_max: float | None = None, P_tot: float | None = None,
    t_tol: float = 1e-6, solver: str = "closed",
) -> Callable[[float], float]:
    """Build V(x) = the exact max-min power-allocation optimum at PA position x. The
    inner problem is solved exactly for every x, so only the outer 1-D search over x
    needs a global-optimisation certificate.

    ``floor`` is either a constant or a callable F(x). It is a function of x whenever the
    inter-segment leakage is modelled honestly, because the interferers stay put while the
    antenna moves; see ``floor_position``. Pass
    ``floor_position.floor_fn_absolute(K, params, seg_lo)`` for the correct model, and
    remember to pass the matching ``floor_logderiv`` to ``deterministic_global`` so the
    Lipschitz constant covers the extra variation.

    ``solver`` selects the inner route:
      "closed" (DEFAULT) -- ``maxmin_closed.maxmin_closed``, the closed-form SIC
          minimum-power recursion plus one scalar Newton solve. Exact, and ~540x faster
          than the LP route (measured); ``t_tol`` is then unused.
      "lp" (LEGACY) -- ``power_alloc.maxmin_bisection``, bisection on t with an LP
          feasibility oracle, truncated at ``t_tol``.

    The two agree to within ``t_tol`` (tests/test_maxmin_closed.py Gate A). Where they
    disagree the closed form is the correct one: the LP occasionally accepts a marginally
    infeasible floor inside HiGHS's own primal tolerance, which that test pins down.
    """
    users_x = np.asarray(users_x, dtype=float)
    users_y = np.asarray(users_y, dtype=float)
    if solver not in ("closed", "lp"):
        raise ValueError(f"solver must be 'closed' or 'lp', got {solver!r}")

    floor_is_fn = callable(floor)

    def V(x: float) -> float:
        gains = channel.path_gain((x - users_x) ** 2 + users_y ** 2 + float(d) ** 2, eta)
        F = float(floor(x)) if floor_is_fn else float(floor)
        if solver == "closed":
            t, _ = maxmin_closed.maxmin_closed(
                gains, F, power_model, P_max=P_max, P_tot=P_tot)
            return t
        t, _, _ = power_alloc.maxmin_bisection(
            gains, F, power_model, P_max=P_max, P_tot=P_tot, t_tol=t_tol)
        return t

    return V


def maxmin_global_placement(
    users_x: np.ndarray, users_y: np.ndarray, seg_lo: float, seg_hi: float, d: float,
    eta: float, floor: float | Callable[[float], float], power_model: str,
    P_max: float | None = None, P_tot: float | None = None, tol: float = _DEFAULT_TOL,
    floor_logderiv: float = 0.0, use_cells: bool = False,
) -> dict[str, Any]:
    """Top-level entry point: certified-global max-min PA placement for one segment.
    Thin wrapper composing ``maxmin_value_function`` + ``deterministic_global`` -- this is
    the function ``altopt``/the figure drivers should call for the production max-min
    x-step (see altopt.ao_segment's "maxmin" route and experiments/fig_placement_gain.py).

    ``floor`` may be a callable F(x); pass the matching ``floor_logderiv`` so the
    certificate covers the floor's own variation.
    """
    V = maxmin_value_function(users_x, users_y, d, eta, floor, power_model,
                               P_max=P_max, P_tot=P_tot)
    return deterministic_global(V, users_x, users_y, d, seg_lo, seg_hi, tol=tol,
                                floor_logderiv=floor_logderiv, use_cells=use_cells)
