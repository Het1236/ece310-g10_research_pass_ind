"""Small helpers shared by the M2 experiment scripts."""
from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

RESULTS_DIR = CODE_DIR.parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAROON = "#7D1916"
GREY = "#6B7280"
BLUE = "#1F4E79"
GREEN = "#2E7D32"
ORANGE = "#C77700"


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 120,
    })
    return plt


def save_figure(fig, name: str) -> Path:
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved {path}")
    return path
