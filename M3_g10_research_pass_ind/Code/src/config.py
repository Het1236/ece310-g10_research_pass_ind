"""Tiny config loader: read config/params.yaml into a plain dict.

Keeps all physical parameters out of code (guardrail: no hard-coded physical
params). Computes derived constants (eta) when left null in the YAML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Speed of light [m/s] - physical constant, used only to derive eta from f_c.
_SPEED_OF_LIGHT: float = 299_792_458.0

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def compute_eta(f_c: float) -> float:
    """Free-space constant eta = c^2 / (16 pi^2 f_c^2).  Source: EDMA Eq. (2)."""
    import math

    return _SPEED_OF_LIGHT**2 / (16.0 * math.pi**2 * f_c**2)


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load params.yaml and fill derived values (eta) if absent.

    Args:
        path: optional path to a YAML file; defaults to config/params.yaml.

    Returns:
        dict of parameters with ``eta`` guaranteed non-null.
    """
    path = Path(path) if path is not None else _DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)

    if params.get("eta") is None:
        params["eta"] = compute_eta(float(params["f_c"]))

    # Geometry is single-sourced from M: D_S = D_L/M, delta = D_S/2 (symmetric), both DERIVED.
    if params.get("D_S") is None:
        params["D_S"] = float(params["D_L"]) / float(params["M"])
    if params.get("delta") is None:
        params["delta"] = float(params["D_S"]) / 2.0
    return params
