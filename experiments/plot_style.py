"""Shared thesis-sized plot helpers for experiment scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# 0.98 * 422.52348 pt / 72.27 = 5.7295 inch
FIG_W = 5.7295
FIG_H = 2.45
_EXP2_LEGACY_FULL_W = 7.0
_EXP2_DUAL_LEGACY_H = 2.6 + 0.55
_EXP2_GRID_ROW_LEGACY_H = 2.6 + 0.14
_EXP2_GRID_FOOTER_LEGACY_H = 0.62


def use_thesis_style():
    """Apply fixed-width typography that matches the thesis body text."""

    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "text.latex.preamble": r"""
            \usepackage[T1]{fontenc}
            \usepackage{lmodern}
        """,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "cm",
        }
    )


def newfig(height: float = FIG_H):
    """Create a standard-width thesis figure."""

    fig, ax = plt.subplots(figsize=(FIG_W, float(height)), constrained_layout=True)
    return fig, ax


def dual_panel_size():
    """Return the exact Exp2 1x2 figure size."""

    return (FIG_W, FIG_W * _EXP2_DUAL_LEGACY_H / _EXP2_LEGACY_FULL_W)


def grid_size(n_rows: int, *, footer: float = _EXP2_GRID_FOOTER_LEGACY_H):
    """Return the exact Exp2-style stacked-grid size for a fixed thesis width."""

    return (
        FIG_W,
        FIG_W * (float(n_rows) * _EXP2_GRID_ROW_LEGACY_H + float(footer)) / _EXP2_LEGACY_FULL_W,
    )


def scaled_height(old_width: float, old_height: float) -> float:
    """Preserve an old aspect ratio while switching to the fixed thesis width."""

    return FIG_W * float(old_height) / float(old_width)


def scaled_size(old_width: float, old_height: float):
    """Return a figsize tuple with fixed thesis width and preserved aspect ratio."""

    return (FIG_W, scaled_height(old_width, old_height))


def apply_thesis_axes_style(ax, *, grid: bool = True):
    """Apply crisp thesis-style axis cosmetics."""

    ax.set_axisbelow(True)
    ax.grid(bool(grid), which="major")
    ax.tick_params(axis="both", which="major", direction="out", width=0.8, length=3.0, pad=4, colors="#222222")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#1F1F1F")


def savepdf(fig, path):
    """Save PDF without tight-bbox resizing so the physical size stays exact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")


def save_figure(fig, path):
    """Save a figure while preserving exact PDF dimensions."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        savepdf(fig, path)
    else:
        fig.savefig(path, dpi=600)
