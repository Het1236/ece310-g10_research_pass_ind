"""Monte-Carlo leakage sampler - ground-truth far/adjacent ratio (exact 1/r^2).

Validates the analytic ``ratio_R`` like-for-like. Sources:
  * EDMA  = Ding, Schober, Poor, "EDMA: A Feasibility Study via Pinching
            Antennas," IEEE TWC 25:15675-15691, 2026 (context/BasePaper/...).
  * GT     = Phase-2 ground truth in CLAUDE.md Sec. 3.

Per-user leakage law (EDMA Eq. 1/2/3, exact, no freeze):
    R^2 = (x - x_m^Pin)^2 + y^2 + d^2                 # ||psi_i - psi_m^Pin||^2
    alpha ~ Bernoulli( e^{-phi * R^2} )                # EDMA Eq. (3), coeff phi (not 2*phi)
    leakage = alpha * eta / R^2                         # EDMA Eq. (2): |h|^2 = eta/R^2

Geometry matches analytic exactly (CLAUDE.md Sec. 3, two-sided):
  * centre PA in the middle segment; p_m = (centre-segment left edge) + delta.
  * segment at signed offset s spans x in [p_m - delta + s*D_S, p_m - delta + (s+1)*D_S];
    s = 0 is the SERVING segment, excluded from leakage.
  * |s| = 1 -> adjacent (ell=1), |s| >= 2 -> far (ell>=2).
  * (x - p_m)^2 is symmetric, so right neighbours realise offset delta and left
    neighbours offset (D_S - delta) automatically (the analytic two-sided split).

The far/adjacent ratio is K-independent in expectation (K cancels); K only sets the
per-trial sample count (variance). This K-independence is itself a check of the
K-uniform adjacency claim.
"""
from __future__ import annotations

from typing import Any, Mapping

import math

import numpy as np

from src import channel, geometry

# How many users to drop per segment per trial. K cancels in the ratio (see module
# docstring); a value > 1 lowers variance and avoids empty adjacent buckets. This is
# a sampler/numerical knob (OURS), not a physical parameter.
_USERS_PER_SEGMENT: int = 20


def sample_leakage_buckets(
    params: Mapping[str, Any],
    delta: float,
    rng: np.random.Generator,
    users_per_segment: int | None = None,
) -> tuple[float, float]:
    """One Monte-Carlo trial: exact leakage at the centre PA, bucketed.

    Drops ``users_per_segment`` users in every non-centre segment (s in
    {-H..H}\\{0}, H = n_segments//2), computes exact per-user leakage
    ``alpha * eta / R^2`` (EDMA Eq. 1/2/3), and returns ``(sum_far, sum_adjacent)``
    where adjacent = |s|==1 and far = |s|>=2.

    Args:
        params: parameter mapping (phi, d, eta, D_S, D_W, n_segments).
        delta:  within-segment PA x-offset, delta in [0, D_S].
        rng:    seeded NumPy Generator (reproducibility).
        users_per_segment: users dropped per segment per trial. None (default)
            falls back to the module default ``_USERS_PER_SEGMENT`` -- the ratio
            is K-independent in expectation, so this only sets the sample count.
            The figure passes K from K_list to render one series per K.

    Returns:
        (sum_far, sum_adjacent) leakage for this trial.
    """
    phi = float(params["phi"])
    d = float(params["d"])
    eta = float(params["eta"])
    D_S = float(params["D_S"])
    D_W = float(params["D_W"])
    n_segments = int(params["phase2_far_probe_segments"])   # far-decay probe (NOT physical M)
    K = _USERS_PER_SEGMENT if users_per_segment is None else int(users_per_segment)

    # Geometry + uniform user drop (P3-0.5: shared helpers; identical draw order).
    p_m = geometry.pa_position(delta)                       # = delta (centre-left at x=0)
    offsets = geometry.neighbour_offsets(n_segments)        # s in {-H..H}\{0}
    x, y = geometry.drop_users(offsets, D_S, D_W, K, rng)   # (n_seg, K) each; x-call then y-call

    R2 = (x - p_m) ** 2 + y**2 + d**2
    p_los = channel.blockage_prob(R2, phi)                               # EDMA Eq. (3)
    alpha = rng.random(size=R2.shape) < p_los                           # Bernoulli
    leakage = np.where(alpha, channel.path_gain(R2, eta), 0.0)          # EDMA Eq. (2)

    per_seg = leakage.sum(axis=1)                                        # (n_seg,)
    abs_s = np.abs(offsets)
    sum_adj = float(per_seg[abs_s == 1].sum())
    sum_far = float(per_seg[abs_s >= 2].sum())
    return sum_far, sum_adj


def ratio_mc(
    params: Mapping[str, Any],
    delta: float,
    n_trials: int,
    seed: int,
    users_per_segment: int | None = None,
) -> dict[str, float | int]:
    """Estimate the true far/adjacent leakage ratio (exact 1/r^2 ground truth).

    ratio = mean_t(sum_far) / mean_t(sum_adjacent). Standard error via batching:
    the trials are split into batches, a ratio is formed per batch, and stderr is
    the std of the batch ratios divided by sqrt(n_batches).

    Args:
        params:   parameter mapping (see ``sample_leakage_buckets``).
        delta:    within-segment PA x-offset.
        n_trials: number of Monte-Carlo trials.
        seed:     RNG seed (reproducibility).
        users_per_segment: users per segment (K). None (default) uses
            ``_USERS_PER_SEGMENT``; pass K from K_list for the K-collapse figure.

    Returns:
        {'ratio', 'stderr', 'mean_far', 'mean_adj', 'n_trials'}.
    """
    rng = np.random.default_rng(seed)
    far = np.empty(n_trials, dtype=float)
    adj = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        far[t], adj[t] = sample_leakage_buckets(params, delta, rng, users_per_segment)

    mean_far = float(far.mean())
    mean_adj = float(adj.mean())
    ratio = mean_far / mean_adj

    # Batched standard error on the ratio of means.
    n_batches = min(20, n_trials)
    bounds = np.linspace(0, n_trials, n_batches + 1, dtype=int)
    batch_ratios = []
    for i in range(n_batches):
        lo, hi = bounds[i], bounds[i + 1]
        a = adj[lo:hi].mean()
        if a > 0.0:
            batch_ratios.append(far[lo:hi].mean() / a)
    stderr = float(np.std(batch_ratios, ddof=1) / np.sqrt(len(batch_ratios)))

    return {
        "ratio": ratio,
        "stderr": stderr,
        "mean_far": mean_far,
        "mean_adj": mean_adj,
        "n_trials": n_trials,
    }


# ============================================================================
# PHASE 3  --  Monte-Carlo ergodic per-segment rate (Exit Gate 3). Source: GT3.
# ============================================================================
def ergodic_rate_mc(
    K: int,
    params: Mapping[str, Any],
    power_model: str,
    rng: np.random.Generator,
    n_trials: int | None = None,
) -> dict[str, float | int]:
    """Monte-Carlo ergodic per-segment rate: E[log2(1 + S/(N0 + I))].

    Per trial: K own-segment users (S, LoS, no blockage) at offset delta, plus 2K
    adjacent interferers (I, both sides, with blockage coins). Reuses the SHARED
    physics (channel.blockage_prob / channel.path_gain) and geometry (pa_position,
    drop_users). Ground truth for ``analytic.ergodic_rate_exact`` (Exit Gate 3).
    Source: GT3 (CLAUDE.md 7.3); home = LoS, adjacent-only leakage.

    `power_model` is REQUIRED ("A"/"B"), no default -- raises otherwise. A == B for
    the per-segment NOMA rate ONLY (both models put K*P into the SIC receiver
    simultaneously); the models DIVERGE in ``system_rates`` (the TDMA slot
    power-boost / MAC tie), which is why power_model is required and must not be read
    as collapsing the distinction the crossover theorem depends on. delta is read
    from params (must be D_S/2 for the symmetric 2-side model).

    Returns {'mean_rate', 'stderr', 'n_trials'} (stderr via batching).
    """
    if power_model not in ("A", "B"):
        raise ValueError(f"power_model must be 'A' or 'B', got {power_model!r}")
    phi = float(params["phi"]); d = float(params["d"]); eta = float(params["eta"])
    D_S = float(params["D_S"]); D_W = float(params["D_W"]); delta = float(params["delta"])
    P = float(params["phase3"]["P"]); N0 = float(params["phase3"]["N0"])
    n_trials = int(params["N_trials"]) if n_trials is None else int(n_trials)

    p_m = geometry.pa_position(delta)
    home_off = np.array([0.0])              # own segment s=0
    adj_off = np.array([-1.0, 1.0])         # both adjacent segments
    ln2 = math.log(2.0)

    rates = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        xh, yh = geometry.drop_users(home_off, D_S, D_W, K, rng)        # (1, K)
        xa, ya = geometry.drop_users(adj_off, D_S, D_W, K, rng)         # (2, K) -> 2K
        R2h = (xh - p_m) ** 2 + yh**2 + d**2
        R2a = (xa - p_m) ** 2 + ya**2 + d**2
        S = P * float(np.sum(channel.path_gain(R2h, eta)))             # K home signals
        coins = rng.random(size=R2a.shape) < channel.blockage_prob(R2a, phi)
        I = P * float(np.sum(np.where(coins, channel.path_gain(R2a, eta), 0.0)))
        rates[t] = math.log2(1.0 + S / (N0 + I))

    mean_rate = float(rates.mean())
    n_batches = min(20, n_trials)
    bounds = np.linspace(0, n_trials, n_batches + 1, dtype=int)
    batch_means = [rates[bounds[i]:bounds[i + 1]].mean() for i in range(n_batches)]
    stderr = float(np.std(batch_means, ddof=1) / math.sqrt(n_batches))
    return {"mean_rate": mean_rate, "stderr": stderr, "n_trials": n_trials}


# ----------------------------------------------------------------------------
# Full-interference SYSTEM sum-rate MC (Fig.4 / Table-I validation).
#
# These two samplers lay out an ABSOLUTE M-segment grid (m = 0 .. M-1) with
# edge-aware FULL interference (each segment sees the users in every OTHER
# segment, so end segments naturally have fewer interferers) -- distinct from the
# centre-segment ratio sampler above. Both were validated against EDMA Table I
# (lifted here from tests/test_phase3_analytic.py so the regression tests and the
# Fig.4 driver share ONE source of truth). Physics via channel.* (EDMA Eq. 2/3).
# Source: EDMA Eq. 5 / Table I; GT3 (CLAUDE.md 7.7).
# ----------------------------------------------------------------------------
def _batched_stats(per_trial: np.ndarray, n_trials: int) -> dict[str, float | int]:
    """Mean and batched standard error of a per-trial sum-rate array.

    Same batching estimator as ``ergodic_rate_mc`` (<=20 contiguous batches).
    """
    mean_rate = float(per_trial.mean())
    n_batches = min(20, n_trials)
    bounds = np.linspace(0, n_trials, n_batches + 1, dtype=int)
    batch_means = [per_trial[bounds[i]:bounds[i + 1]].mean() for i in range(n_batches)]
    stderr = float(np.std(batch_means, ddof=1) / math.sqrt(n_batches))
    return {"mean_rate": mean_rate, "stderr": stderr, "n_trials": n_trials}


def system_sum_rate_mc_triangular(
    M: int,
    params: Mapping[str, Any],
    *,
    n_trials: int = 100_000,
    seed: int = 2,
) -> dict[str, float | int]:
    """EDMA-faithful system sum-rate MC: u=0 PA-tracks-user, K=1, full interference.

    Each of the M segments holds ONE user; its PA sits at that user's x (u=0 =>
    r^2 = y^2 + d^2, the EDMA placement). Interference at segment m is the sum over
    the single user of every OTHER segment (edge-aware: end segments have fewer
    neighbours), each gated by an independent blockage coin. Reproduces EDMA Table I
    (the NOMA/EDMA row). This is "Pair A" -- the machinery-validation sampler; the
    Fig.4 driver and ``test_edma_full_interference_matches_table_I`` both call it.
    Source: EDMA Eq. 2/3/5, Table I.

    Returns {'mean_rate', 'stderr', 'n_trials'} (stderr via batching).
    """
    eta = float(params["eta"]); phi = float(params["phi"]); d = float(params["d"])
    D_W = float(params["D_W"]); D_L = float(params["D_L"])
    P = float(params["phase3"]["P"]); N0 = float(params["phase3"]["N0"])
    D_Sm = D_L / M                                    # physical segment width for M segments
    rng = np.random.default_rng(seed)
    x = np.stack([rng.uniform(m * D_Sm, (m + 1) * D_Sm, n_trials) for m in range(M)])
    y = rng.uniform(-D_W / 2.0, D_W / 2.0, (M, n_trials))
    per_trial = np.zeros(n_trials)
    for m in range(M):
        S = P * channel.path_gain(y[m] ** 2 + d**2, eta)        # u=0: r^2 = y^2 + d^2
        I = np.zeros(n_trials)
        for j in range(M):
            if j != m:
                R2 = (x[j] - x[m]) ** 2 + y[j] ** 2 + d**2
                coins = rng.random(n_trials) < channel.blockage_prob(R2, phi)
                I += np.where(coins, P * channel.path_gain(R2, eta), 0.0)
        per_trial += np.log2(1.0 + S / (N0 + I))
    return _batched_stats(per_trial, n_trials)


def system_sum_rate_mc_fixed_delta(
    M: int,
    params: Mapping[str, Any],
    *,
    K: int = 1,
    n_trials: int = 120_000,
    seed: int = 3,
) -> dict[str, float | int]:
    """OUR-system sum-rate MC: PA fixed at each segment centre (u-avg home), K users/seg.

    Per segment: PA at the segment centre x = m*D_Sm + D_Sm/2; K users dropped
    uniformly over the segment (signal = SIC sum of the K home gains). Interference
    at segment m sums over the K users of every OTHER segment (edge-aware), each
    blockage-gated. This is the "Pair B" sampler -- it certifies
    ``analytic.system_sum_rate_exact(home_placement='uavg')`` in OUR fixed-delta
    placement model; ``test_full_interference_analytic_matches_mc_fixed_delta``
    calls it at K=1. Source: GT3 (CLAUDE.md 7.7), EDMA Eq. 2/3.

    Returns {'mean_rate', 'stderr', 'n_trials'} (stderr via batching).
    """
    eta = float(params["eta"]); phi = float(params["phi"]); d = float(params["d"])
    D_W = float(params["D_W"]); D_L = float(params["D_L"])
    P = float(params["phase3"]["P"]); N0 = float(params["phase3"]["N0"])
    D_Sm = D_L / M
    pa = np.array([m * D_Sm + D_Sm / 2.0 for m in range(M)])     # fixed PA x per segment
    rng = np.random.default_rng(seed)
    # K users per segment: shape (M, K, n_trials). At K=1 this reduces to the
    # original fixed-delta sampler (one user/segment).
    x = np.stack([
        np.stack([rng.uniform(m * D_Sm, (m + 1) * D_Sm, n_trials) for _ in range(K)])
        for m in range(M)
    ])
    y = rng.uniform(-D_W / 2.0, D_W / 2.0, (M, K, n_trials))
    per_trial = np.zeros(n_trials)
    for m in range(M):
        R2h = (x[m] - pa[m]) ** 2 + y[m] ** 2 + d**2            # (K, n_trials)
        S = P * channel.path_gain(R2h, eta).sum(axis=0)
        I = np.zeros(n_trials)
        for j in range(M):
            if j != m:
                R2 = (x[j] - pa[m]) ** 2 + y[j] ** 2 + d**2     # (K, n_trials)
                coins = rng.random((K, n_trials)) < channel.blockage_prob(R2, phi)
                I += np.where(coins, P * channel.path_gain(R2, eta), 0.0).sum(axis=0)
        per_trial += np.log2(1.0 + S / (N0 + I))
    return _batched_stats(per_trial, n_trials)
