"""Publication-style matplotlib setup and save helpers."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Silence noisy-but-harmless fontTools PDF timestamp chatter
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")
warnings.filterwarnings("ignore", message=".*regarding as unix timestamp.*")

_ft = logging.getLogger("fontTools")
_ft.setLevel(logging.ERROR)
_ft.propagate = False
if not _ft.handlers:
    _ft.addHandler(logging.NullHandler())
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Shared colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "GT": "#000000",
    "ID": "#D62728",  # red
    "FD": "#1F77B4",  # blue
    "ENV": "#000000",
}

# ---------------------------------------------------------------------------
# Figure size constants (inches, single-/double-column)
# ---------------------------------------------------------------------------
COL_W = 3.37
FULL_W = 6.95
H_COL = 2.8


def set_pub_style(grid: bool = False, base_size: int = 8):
    """Apply a Nature/NPJ-ish RC style for publication-ready figures."""
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
            "font.size": base_size,
            "axes.labelsize": base_size + 1,
            "legend.fontsize": base_size - 1,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "lines.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": grid,
            "grid.alpha": 0.15,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.transparent": False,
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _savefig(fig: plt.Figure, path: Path):
    """Save a figure with mkdir and high DPI for raster/PDF formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"):
        fig.savefig(path, dpi=600)
    else:
        fig.savefig(path)
