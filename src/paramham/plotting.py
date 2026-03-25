"""Publication-style matplotlib setup and save helpers."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
    "ID": "#EE6677",  # paper red
    "FD": "#4477AA",  # paper blue
    "ENV": "#000000",
    "VQE": "#EE6677",
    "QAOA": "#4477AA",
    "NEUTRAL": "#BBBBBB",
    "MUTED": "#666666",
    "REFERENCE": "#888888",
}

COLOR_CYCLE = [
    COLORS["FD"],
    COLORS["ID"],
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    COLORS["NEUTRAL"],
]

METHOD_CMAPS = {
    "ID": LinearSegmentedColormap.from_list("paramham_id", ["#FFFFFF", COLORS["ID"]]),
    "FD": LinearSegmentedColormap.from_list("paramham_fd", ["#FFFFFF", COLORS["FD"]]),
}

ADVANTAGE_CMAP = LinearSegmentedColormap.from_list(
    "paramham_advantage",
    [COLORS["FD"], "#F7F7F7", COLORS["ID"]],
)

HEATMAP_CMAP = "RdBu_r"

# ---------------------------------------------------------------------------
# Figure size constants (inches, single-/double-column)
# ---------------------------------------------------------------------------
COL_W = 3.4
FULL_W = 7.0
H_COL = 2.6


def set_pub_style(grid: bool = False, base_size: int = 8, theme: str = "paper"):
    """Apply the repo-local paper style used for experiment figures."""
    mpl.rcdefaults()
    if theme not in {"paper", "reference"}:
        raise ValueError(f"Unknown plotting theme: {theme}")

    params = {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "font.size": base_size,
        "axes.labelsize": base_size + 1,
        "legend.fontsize": base_size - 1,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "axes.formatter.use_mathtext": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.0,
        "axes.prop_cycle": plt.cycler("color", COLOR_CYCLE),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.grid": grid,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "errorbar.capsize": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.transparent": False,
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
    if theme == "reference":
        params.update(
            {
                "font.serif": ["STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
                "axes.labelsize": base_size + 2,
                "axes.titlesize": base_size + 3,
                "legend.fontsize": base_size,
                "xtick.labelsize": base_size,
                "ytick.labelsize": base_size,
                "lines.linewidth": 2.2,
                "lines.markersize": 7,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "axes.spines.top": True,
                "axes.spines.right": True,
                "axes.linewidth": 1.15,
                "axes.edgecolor": "#1F1F1F",
                "axes.grid": grid,
                "grid.alpha": 0.7,
                "grid.linestyle": "-",
                "grid.linewidth": 0.75,
                "grid.color": "#D7D7D7",
                "xtick.major.size": 4.5,
                "ytick.major.size": 4.5,
                "xtick.minor.size": 0.0,
                "ytick.minor.size": 0.0,
                "legend.frameon": False,
                "legend.framealpha": 0.0,
                "legend.borderpad": 0.2,
                "legend.handlelength": 1.4,
                "legend.handletextpad": 0.45,
                "legend.columnspacing": 0.9,
                "axes.facecolor": "#FBFBF8",
                "savefig.facecolor": "white",
            }
        )
    mpl.rcParams.update(params)


def apply_reference_axes_style(ax):
    """Apply per-axis cosmetics matching the softer reference look."""

    ax.set_axisbelow(True)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", which="major", direction="out", colors="#222222", pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("#1F1F1F")


def _savefig(fig: plt.Figure, path: Path):
    """Save a figure with mkdir and high DPI for raster/PDF formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    bbox = fig.bbox_inches
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"):
        fig.savefig(path, dpi=600, bbox_inches=bbox, pad_inches=0.0)
    else:
        fig.savefig(path, bbox_inches=bbox, pad_inches=0.0)


def add_panel_legend(ax, *, placement: str = "below", ncol: int = 1, **kwargs):
    """Add a consistent legend placed away from the plotted data."""

    placements = {
        "above": {"loc": "lower center", "bbox_to_anchor": (0.5, 1.02)},
        "above_left": {"loc": "lower left", "bbox_to_anchor": (0.0, 1.02)},
        "above_right": {"loc": "lower right", "bbox_to_anchor": (1.0, 1.02)},
        "below": {"loc": "upper center", "bbox_to_anchor": (0.5, -0.17)},
        "below_left": {"loc": "upper left", "bbox_to_anchor": (0.0, -0.17)},
        "below_right": {"loc": "upper right", "bbox_to_anchor": (1.0, -0.17)},
        "right": {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)},
    }
    if placement not in placements:
        raise ValueError(f"Unknown legend placement: {placement}")

    legend_kwargs = {
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.94,
        "facecolor": "white",
        "edgecolor": "0.85",
        "borderaxespad": 0.2,
        "ncol": int(ncol),
    }
    legend_kwargs.update(placements[placement])
    legend_kwargs.update(kwargs)
    return ax.legend(**legend_kwargs)


def add_figure_legend(fig: plt.Figure, handles, labels, *, ncol: int = 1, **kwargs):
    """Add a shared legend below the full figure."""

    legend_kwargs = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.02),
        "bbox_transform": fig.transFigure,
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.94,
        "facecolor": "white",
        "edgecolor": "0.85",
        "borderaxespad": 0.2,
        "ncol": int(ncol),
    }
    legend_kwargs.update(kwargs)
    return fig.legend(handles, labels, **legend_kwargs)
