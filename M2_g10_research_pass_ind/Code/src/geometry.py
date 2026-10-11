"""Geometry helpers for the segmented-waveguide NOMA-EDMA model.

Extracted in P3-0.5 from the inline layout/user-drop in montecarlo.py (relocation
only). Convention (SystemModel/ + CLAUDE.md 3): centre-segment left edge at x=0, PA
fixed at within-segment offset delta (p_m = delta); users uniform => uniform
u-separation on [a_ell, b_ell]. NOT EDMA's PA-tracks-user model.

`drop_users` is vectorised over all segments in a single pair of RNG calls (x then
y), preserving the exact draw order of the original inline sampler so results stay
byte-identical for a fixed seed.
"""
from __future__ import annotations

import numpy as np


def pa_position(delta: float) -> float:
    """Centre-segment PA x-coordinate p_m = delta (centre-segment left edge at x=0). Source: SM."""
    return delta


def neighbour_offsets(n_segments: int) -> np.ndarray:
    """Signed segment offsets s in {-H..H}\\{0}, H = n_segments//2 (centre s=0 excluded).

    Float array, matching the original inline order. Source: SM / CLAUDE.md 3.
    """
    H = n_segments // 2
    return np.array([s for s in range(-H, H + 1) if s != 0], dtype=float)


def segment_x_span(s: int, D_S: float) -> tuple[float, float]:
    """x-span [s*D_S, (s+1)*D_S] of the segment at signed offset s (centre-left at x=0).

    Independent of delta: the offset enters via p_m = delta (see ``pa_position``), so
    the user-PA separation u = x - p_m ranges over [s*D_S - delta, (s+1)*D_S - delta].
    Source: SM / CLAUDE.md 3.
    """
    return s * D_S, (s + 1) * D_S


def drop_users(
    offsets: np.ndarray, D_S: float, D_W: float, n_users: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly drop `n_users` per segment over all `offsets`.

    Returns (x, y), each shape (len(offsets), n_users), with
    x ~ U[s*D_S, s*D_S + D_S] and y ~ U[-D_W/2, D_W/2]. Drawn as one uniform call
    for x then one for y -- the exact order of the original inline sampler (so a
    fixed seed reproduces identical numbers). Source: SM (uniform user distribution).
    """
    seg_left = offsets[:, None] * D_S                       # centre_left = 0
    x = seg_left + rng.uniform(0.0, D_S, size=(offsets.size, n_users))
    y = rng.uniform(-D_W / 2.0, D_W / 2.0, size=(offsets.size, n_users))
    return x, y
