#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp03_readout_realism_best_mode.py
================================

Experiment 3: Readout realism --- Best-of-S and Mode
-------------------------------------------------------------------

Reviewer question:
  "Expectation value is better... but do I get better *solutions* on hardware?"

What we simulate:
  At each outer iteration t:
    1) run an inner VQE solve at the current lambda_t   (approximate value query F(lambda_t))
    2) perform a FIXED readout budget of S bitstring samples from the resulting state
       and compute two practical readout metrics:
         - Best-of-S:  max cut value among the S samples
         - Mode cut:   cut value of the most frequent sampled bitstring

We compare:
  - ID        : correlator-reuse implicit differentiation (CR-ImpDiff)
  - BD        : black-box finite difference on the VALUE function F(lambda) (requires re-solves at lambda+-c)
                (Legend label requested: "VQE + BD")

Minimal paper output:
  - One 2-panel figure (Best-of-S | Mode), for the periodic family (default), aggregated over instances
    as mean +/- stderr.

Key fairness choice (matches "fixed readout shots per outer step"):
  - We apply the readout budget S ONCE PER OUTER STEP for BOTH methods, at the *center* candidate state.
    BD performs extra perturbed inner re-solves internally; we do not allocate extra readout shots to those
    perturbed solves, because the goal here is to compare "what solution do I get if I read out my current candidate
    each iteration with a fixed readout budget".

Normalization:
  - We normalize cut values by a classical diagnostic upper bound
        J* = max_{lambda in grid} max_{z in {0,1}^n} J(z;lambda),
    computed by enumerating all 2^n bitstrings (feasible for n<=12).

Plots:
  - --xaxis iters  : x-axis = outer iteration t (cleanest for "fixed readout shots per outer step")
  - --xaxis budget : x-axis = energy evaluations, using step-function interpolation onto a shared budget grid
                     (aligns with the budget-efficiency story in Fig. 0A)

Outputs (saved in --out):
  - fig3_readout_best_mode_<suffix>_xIters.<fmt>   or   _xBudget.<fmt>
  - runs3_readout_metrics.csv
  - table3_readout_summary.csv / .tex
  - SUMMARY.txt

Example:
  python exp03_readout_realism_best_mode.py \
    --family periodic --periodic_K 6 \
    --n 12 --p_edge 0.45 \
    --outer 30 --inner 28 --L 2 \
    --readout_shots 256 \
    --num_instances 20 \
    --xaxis iters \
    --fmt pdf --out output/exp03/iters

"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_thesis_axes_style, dual_panel_size, grid_size, save_figure, use_thesis_style

from paramham.experiment_defaults import (
    CANONICAL_SETUP,
    can_run_step,
    generate_er_family1d_instance,
    publication_cache_dir,
    publication_output_dir,
    vqe_fd_value_step_cost,
    vqe_id_step_cost,
)
from paramham.io import parse_str_list, write_csv
from paramham.maxcut import (
    build_cut_mask,
    classical_Jstar,
)
from paramham.maxcut import (
    precompute_z as precompute_z_big_endian,
)
from paramham.metrics import mean_stderr, step_interp
from paramham.plotting import (
    COLORS,
)

# ---------------------------------------------------------------------------
# Shared library imports
# ---------------------------------------------------------------------------
from paramham.seeds import to_uint_seed
from paramham.simulator import vqe_state
from paramham.spsa import spsa_minimize

# ==============================================================================
# Experiment-specific constants
# ==============================================================================

H_TWO = 2.6


# ==============================================================================
# Helpers
# ==============================================================================


def classical_Jstar_max(fam, cut_mask, grid_points):
    """Wrapper: return only the J* float from the shared classical_Jstar."""
    J, _ = classical_Jstar(fam, cut_mask, grid_points)
    return J


def _family_label(kind: str) -> str:
    return {
        "linear": "Linear",
        "quadratic": "Quadratic",
        "periodic": "Periodic",
    }.get(str(kind), str(kind).title())


def _set_exp03_plot_style(grid: bool = True):
    use_thesis_style()
    plt.rcParams["axes.grid"] = bool(grid)
    plt.rcParams["grid.alpha"] = 0.7
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.75
    plt.rcParams["grid.color"] = "#D7D7D7"


def _metric_ylabel(metric: str) -> str:
    if metric == "bestS":
        return r"Best-of-$S$ / $J^*$"
    if metric == "mode":
        return r"Mode cut / $J^*$"
    raise ValueError(f"Unknown metric: {metric}")


def _cache_default_dir(out: Path, xaxis: str) -> Path:
    if str(xaxis) == "iters":
        return publication_cache_dir("exp03", "iters")
    return publication_cache_dir("exp03", str(xaxis))


def _cache_meta(args, families) -> dict:
    return {
        "seed0": int(args.seed0),
        "num_instances": int(args.num_instances),
        "families": [str(f) for f in families],
        "periodic_K": int(args.periodic_K),
        "n": int(args.n),
        "p_edge": float(args.p_edge),
        "graph_seed": int(args.graph_seed),
        "lam_min": float(args.lam_min),
        "lam_max": float(args.lam_max),
        "lam0": float(args.lam0),
        "grid": int(args.grid),
        "outer": int(args.outer),
        "inner": int(args.inner),
        "L": int(args.L),
        "eta0": float(args.eta0),
        "eta_pow": float(args.eta_pow),
        "step_clip": None if args.step_clip is None else float(args.step_clip),
        "c_frac": float(args.c_frac),
        "budget_evals": float(args.budget_evals),
        "readout_shots": int(args.readout_shots),
        "xaxis": str(args.xaxis),
        "budget_points": int(args.budget_points),
    }


def _metric_arrays(payload: dict, metric: str):
    if metric == "bestS":
        return payload["metric_id_bestS"], payload["metric_fd_bestS"]
    if metric == "mode":
        return payload["metric_id_mode"], payload["metric_fd_mode"]
    raise ValueError(f"Unknown metric: {metric}")


def _metric_t20_markers(payload: dict, metric: str):
    if metric == "bestS":
        return payload.get("marker_id_bestS_t20"), payload.get("marker_fd_bestS_t20")
    if metric == "mode":
        return payload.get("marker_id_mode_t20"), payload.get("marker_fd_mode_t20")
    raise ValueError(f"Unknown metric: {metric}")


def _point_to_array(point) -> np.ndarray:
    if point is None:
        return np.array([np.nan, np.nan], dtype=float)
    return np.asarray(point, dtype=float)


def _array_to_point(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        return None
    return (float(arr[0]), float(arr[1]))


def _avg_eval_metric_coord(eval_list: List[np.ndarray], metric_list: List[np.ndarray], target_idx: int = 19):
    xs, ys = [], []
    for ev, metric in zip(eval_list, metric_list):
        if target_idx < ev.size and target_idx < metric.size:
            xs.append(float(ev[target_idx]))
            ys.append(float(metric[target_idx]))
    if not xs:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))


def _metric_summary_rows(rows: List[Dict[str, float]], family: str) -> list[dict]:
    def _summ(col: str):
        vals = np.array([row[col] for row in rows], float)
        mu = float(np.nanmean(vals))
        se = float(np.nanstd(vals, ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(vals)))))
        return mu, se

    out = []
    for label, id_col, fd_col in [
        ("Best-of-S final / J*", "bestS_final_ID", "bestS_final_FD"),
        ("Mode final / J*", "mode_final_ID", "mode_final_FD"),
        ("Best-of-S AUC (steps)", "bestS_auc_ID", "bestS_auc_FD"),
        ("Mode AUC (steps)", "mode_auc_ID", "mode_auc_FD"),
    ]:
        idm, ids = _summ(id_col)
        fdm, fds = _summ(fd_col)
        out.append(
            {
                "family": family,
                "metric": label,
                "ID_mean": f"{idm:.6f}",
                "ID_stderr": f"{ids:.6f}",
                "BD_mean": f"{fdm:.6f}",
                "BD_stderr": f"{fds:.6f}",
            }
        )
    return out


def save_exp03_cache(cache_dir: Path, meta: dict, payloads: dict, rows_by_family: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for family, payload in payloads.items():
        prefix = f"{family}__"
        for key, value in payload.items():
            arrays[prefix + key] = np.asarray(value, dtype=float)
    np.savez_compressed(cache_dir / "curves_cache.npz", **arrays)
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (cache_dir / "rows_by_family.json").write_text(json.dumps(rows_by_family, indent=2), encoding="utf-8")


def load_exp03_cache(cache_dir: Path, meta_expected: dict):
    meta_path = cache_dir / "cache_meta.json"
    npz_path = cache_dir / "curves_cache.npz"
    rows_path = cache_dir / "rows_by_family.json"
    if not meta_path.exists() or not npz_path.exists() or not rows_path.exists():
        return None
    try:
        meta_found = json.loads(meta_path.read_text(encoding="utf-8"))
        rows_by_family = json.loads(rows_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta_found != meta_expected:
        return None

    payloads = {}
    with np.load(npz_path) as data:
        for family in meta_expected["families"]:
            prefix = f"{family}__"
            required = [
                prefix + "x",
                prefix + "metric_id_bestS",
                prefix + "metric_fd_bestS",
                prefix + "metric_id_mode",
                prefix + "metric_fd_mode",
            ]
            if any(key not in data for key in required):
                return None
            payload = {}
            for key in required:
                short = key[len(prefix) :]
                payload[short] = np.asarray(data[key], dtype=float)
            for short in [
                "marker_id_bestS_t20",
                "marker_fd_bestS_t20",
                "marker_id_mode_t20",
                "marker_fd_mode_t20",
            ]:
                full = prefix + short
                payload[short] = _array_to_point(np.asarray(data[full], dtype=float)) if full in data else None
            payloads[family] = payload
    return payloads, rows_by_family


def interesting_readout_ylim(
    payloads: dict, families: list[str], metric: str, *, ymax: float = 1.01
) -> tuple[float, float]:
    mins = []
    maxs = []
    for family in families:
        y_id, y_fd = _metric_arrays(payloads[family], metric)
        mu_id, se_id = mean_stderr(y_id, axis=0)
        mu_fd, se_fd = mean_stderr(y_fd, axis=0)
        for arr in (mu_id - se_id, mu_fd - se_fd, mu_id, mu_fd, mu_id + se_id, mu_fd + se_fd):
            vals = np.asarray(arr, dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0.05)]
            if vals.size:
                mins.append(float(np.min(vals)))
                maxs.append(float(np.max(vals)))
    if not mins:
        return (0.0, float(ymax))
    ymin = min(mins) - 0.02
    ymax_data = max(maxs) + 0.015
    ymin = math.floor(ymin / 0.02) * 0.02
    ymax_plot = math.ceil(ymax_data / 0.02) * 0.02
    ymin = float(np.clip(ymin, 0.72, 0.94))
    ymax_plot = float(np.clip(max(ymax_plot, ymin + 0.10), ymin + 0.10, ymax))
    return (ymin, ymax_plot)


def interesting_readout_ylim_multi(
    payloads: dict, families: list[str], metrics: list[str], *, ymax: float = 1.01
) -> tuple[float, float]:
    lows = []
    highs = []
    for metric in metrics:
        y0, y1 = interesting_readout_ylim(payloads, families, metric, ymax=ymax)
        lows.append(float(y0))
        highs.append(float(y1))
    return (min(lows), max(highs))


# ==============================================================================
# Plotting
# ==============================================================================


def plot_2panel_iters(path: Path, best_id: np.ndarray, best_fd: np.ndarray, mode_id: np.ndarray, mode_fd: np.ndarray):
    """
    best_*: (N, T) cumulative best-of-S ratios (monotone)
    mode_*: (N, T) per-step mode ratios
    """
    _set_exp03_plot_style(grid=True)
    T = best_id.shape[1]
    t = np.arange(1, T + 1)

    mu_b_id, se_b_id = mean_stderr(best_id, axis=0)
    mu_b_fd, se_b_fd = mean_stderr(best_fd, axis=0)
    mu_m_id, se_m_id = mean_stderr(mode_id, axis=0)
    mu_m_fd, se_m_fd = mean_stderr(mode_fd, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=dual_panel_size(), constrained_layout=True, sharey=True)

    ax = axs[0]
    ax.plot(t, mu_b_id, color=COLORS["ID"], lw=2.2, label="VQE + ID", solid_capstyle="round")
    ax.fill_between(t, mu_b_id - se_b_id, mu_b_id + se_b_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(t, mu_b_fd, color=COLORS["FD"], lw=2.0, ls=(0, (4, 2)), label="VQE + BD", solid_capstyle="round")
    ax.fill_between(t, mu_b_fd - se_b_fd, mu_b_fd + se_b_fd, color=COLORS["FD"], alpha=0.14, linewidth=0)
    ax.set_xlabel(r"Outer iteration $t$")
    ax.set_ylabel(r"Best-of-$S$ / $J^*$")
    ax.set_xlim(1, T)

    ax = axs[1]
    ax.plot(t, mu_m_id, color=COLORS["ID"], lw=2.2, label="VQE + ID", solid_capstyle="round")
    ax.fill_between(t, mu_m_id - se_m_id, mu_m_id + se_m_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(t, mu_m_fd, color=COLORS["FD"], lw=2.0, ls=(0, (4, 2)), label="VQE + BD", solid_capstyle="round")
    ax.fill_between(t, mu_m_fd - se_m_fd, mu_m_fd + se_m_fd, color=COLORS["FD"], alpha=0.14, linewidth=0)
    ax.set_xlabel(r"Outer iteration $t$")
    ax.set_ylabel(r"Mode cut / $J^*$")
    ax.set_xlim(1, T)

    y_all = np.concatenate([mu_b_id, mu_b_fd, mu_m_id, mu_m_fd])
    y0 = max(0.0, float(np.nanmin(y_all) - 0.04))
    y1 = min(1.05, float(np.nanmax(y_all) + 0.04))
    for ax in axs:
        apply_thesis_axes_style(ax, grid=True)
        ax.set_ylim(y0, y1)
        ax.legend(loc="upper left", frameon=False)

    save_figure(fig, path)
    plt.close(fig)


def plot_2panel_budget(
    path: Path,
    budget_grid: np.ndarray,
    best_id_grid: np.ndarray,
    best_fd_grid: np.ndarray,
    mode_id_grid: np.ndarray,
    mode_fd_grid: np.ndarray,
):
    """
    *_grid: (N, G) traces already interpolated onto a shared budget grid.
    """
    _set_exp03_plot_style(grid=True)
    b = np.asarray(budget_grid, float)

    mu_b_id, se_b_id = mean_stderr(best_id_grid, axis=0)
    mu_b_fd, se_b_fd = mean_stderr(best_fd_grid, axis=0)
    mu_m_id, se_m_id = mean_stderr(mode_id_grid, axis=0)
    mu_m_fd, se_m_fd = mean_stderr(mode_fd_grid, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=dual_panel_size(), constrained_layout=True, sharey=True)

    ax = axs[0]
    ax.plot(b, mu_b_id, color=COLORS["ID"], lw=2.2, label="VQE + ID", solid_capstyle="round")
    ax.fill_between(b, mu_b_id - se_b_id, mu_b_id + se_b_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(b, mu_b_fd, color=COLORS["FD"], lw=2.0, ls=(0, (4, 2)), label="VQE + BD", solid_capstyle="round")
    ax.fill_between(b, mu_b_fd - se_b_fd, mu_b_fd + se_b_fd, color=COLORS["FD"], alpha=0.14, linewidth=0)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Best-of-$S$ / $J^*$")
    ax.set_xlim(float(b[0]), float(b[-1]))

    ax = axs[1]
    ax.plot(b, mu_m_id, color=COLORS["ID"], lw=2.2, label="VQE + ID", solid_capstyle="round")
    ax.fill_between(b, mu_m_id - se_m_id, mu_m_id + se_m_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(b, mu_m_fd, color=COLORS["FD"], lw=2.0, ls=(0, (4, 2)), label="VQE + BD", solid_capstyle="round")
    ax.fill_between(b, mu_m_fd - se_m_fd, mu_m_fd + se_m_fd, color=COLORS["FD"], alpha=0.14, linewidth=0)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Mode cut / $J^*$")
    ax.set_xlim(float(b[0]), float(b[-1]))

    y_all = np.concatenate([mu_b_id, mu_b_fd, mu_m_id, mu_m_fd])
    y0 = max(0.0, float(np.nanmin(y_all) - 0.04))
    y1 = min(1.05, float(np.nanmax(y_all) + 0.04))
    for ax in axs:
        apply_thesis_axes_style(ax, grid=True)
        ax.set_ylim(y0, y1)
        ax.legend(loc="upper left", frameon=False)

    save_figure(fig, path)
    plt.close(fig)


def _metric_title(metric: str) -> str:
    if metric == "bestS":
        return r"Best-of-$S$"
    if metric == "mode":
        return "Mode cut"
    raise ValueError(f"Unknown metric: {metric}")


def _draw_family_metric_panel(
    ax,
    *,
    x: np.ndarray,
    y_id: np.ndarray,
    y_fd: np.ndarray,
    family_label: str,
    xaxis: str,
    metric: str,
    y_limits: tuple[float, float],
    show_xlabel: bool,
    marker_id_t20=None,
    marker_fd_t20=None,
):
    mu_id, se_id = mean_stderr(y_id, axis=0)
    mu_fd, se_fd = mean_stderr(y_fd, axis=0)

    ax.axhline(1.0, color=COLORS["REFERENCE"], lw=1.0, ls=":", alpha=0.85, zorder=1, label="_nolegend_")
    ax.fill_between(x, mu_fd - se_fd, mu_fd + se_fd, color=COLORS["FD"], alpha=0.14, linewidth=0, zorder=1)
    ax.fill_between(x, mu_id - se_id, mu_id + se_id, color=COLORS["ID"], alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, mu_fd, color=COLORS["FD"], lw=2.0, ls=(0, (4, 2)), label="VQE + BD", zorder=3, solid_capstyle="round")
    ax.plot(x, mu_id, color=COLORS["ID"], lw=2.2, ls="-", label="VQE + ID", zorder=4, solid_capstyle="round")

    if xaxis == "budget":
        if marker_id_t20 is not None:
            x_id, y_id_marker = marker_id_t20
            if np.isfinite(x_id) and np.isfinite(y_id_marker):
                ax.plot(
                    x_id,
                    y_id_marker,
                    marker="o",
                    color=COLORS["ID"],
                    markersize=5.5,
                    zorder=10,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                )
                ax.annotate(
                    "t=20",
                    (x_id, y_id_marker),
                    xytext=(0, 10),
                    textcoords="offset points",
                    color=COLORS["ID"],
                    fontsize=10,
                    ha="center",
                )
        if marker_fd_t20 is not None:
            x_fd, y_fd_marker = marker_fd_t20
            if np.isfinite(x_fd) and np.isfinite(y_fd_marker):
                ax.plot(
                    x_fd,
                    y_fd_marker,
                    marker="s",
                    color=COLORS["FD"],
                    markersize=5.2,
                    zorder=10,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                )
                ax.annotate(
                    "t=20",
                    (x_fd, y_fd_marker),
                    xytext=(0, -12),
                    textcoords="offset points",
                    color=COLORS["FD"],
                    fontsize=10,
                    ha="center",
                )

    apply_thesis_axes_style(ax, grid=True)
    ax.set_ylim(*y_limits)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylabel(_metric_ylabel(metric))
    if show_xlabel:
        ax.set_xlabel(r"Outer iteration $t$" if xaxis == "iters" else "Energy evaluations")

    ax.text(
        0.03,
        0.93,
        family_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=COLORS["MUTED"],
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#D9D5CB", alpha=0.98),
    )


def plot_family_metric_grid(
    path: Path,
    payloads: dict,
    families: list[str],
    *,
    metric: str,
    xaxis: str,
    y_limits: tuple[float, float],
    layout: str = "stack",
):
    _set_exp03_plot_style(grid=True)

    handles = [
        mlines.Line2D([], [], color=COLORS["ID"], lw=2.2, label="VQE + ID"),
        mlines.Line2D([], [], color=COLORS["FD"], lw=2.0, ls=(0, (4, 2)), label="VQE + BD"),
    ]
    ncol = 2
    if xaxis == "budget":
        handles.extend(
            [
                mlines.Line2D([], [], color=COLORS["REFERENCE"], lw=1.0, ls=":", label=r"Reference $J^*/J^* = 1$"),
                mlines.Line2D([], [], color=COLORS["ID"], marker="o", ls="None", ms=5, label=r"ID at $t=20$"),
                mlines.Line2D([], [], color=COLORS["FD"], marker="s", ls="None", ms=5, label=r"BD at $t=20$"),
            ]
        )
        ncol = 3

    if layout == "square" and len(families) > 1:
        n_cols = 2
        n_plot = len(families)
        n_slots = n_plot + 1  # reserve one panel-sized slot for the legend
        n_rows = int(math.ceil(n_slots / n_cols))
        fig = plt.figure(figsize=(dual_panel_size()[0], dual_panel_size()[0]), constrained_layout=True)
        gs = fig.add_gridspec(n_rows, n_cols)

        axs = []
        share_ax = None
        for idx, family in enumerate(families):
            r = idx // n_cols
            c = idx % n_cols
            ax = fig.add_subplot(gs[r, c], sharex=share_ax, sharey=share_ax)
            if share_ax is None:
                share_ax = ax
            axs.append(ax)
            ax.set_box_aspect(1.0)
            payload = payloads[family]
            y_id, y_fd = _metric_arrays(payload, metric)
            marker_id_t20, marker_fd_t20 = _metric_t20_markers(payload, metric)
            _draw_family_metric_panel(
                ax,
                x=np.asarray(payload["x"], dtype=float),
                y_id=y_id,
                y_fd=y_fd,
                family_label=_family_label(family),
                xaxis=xaxis,
                metric=metric,
                y_limits=y_limits,
                show_xlabel=(r == n_rows - 1),
                marker_id_t20=marker_id_t20,
                marker_fd_t20=marker_fd_t20,
            )
            if c > 0:
                ax.set_ylabel("")

        legend_idx = len(families)
        legend_ax = fig.add_subplot(gs[legend_idx // n_cols, legend_idx % n_cols])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles,
            loc="center",
            ncol=1 if xaxis == "budget" else ncol,
            frameon=False,
            fancybox=False,
            borderpad=0.35,
            columnspacing=1.2,
            handlelength=1.8,
            handletextpad=0.6,
        )

        for empty_idx in range(legend_idx + 1, n_rows * n_cols):
            empty_ax = fig.add_subplot(gs[empty_idx // n_cols, empty_idx % n_cols])
            empty_ax.axis("off")
    else:
        n_rows = len(families)
        figsize = dual_panel_size() if n_rows == 1 else grid_size(n_rows)
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(n_rows + 1, 1, height_ratios=[1.0] * n_rows + [0.16])
        axs = []
        for r, family in enumerate(families):
            sharex = axs[0] if axs else None
            ax = fig.add_subplot(gs[r, 0], sharex=sharex)
            axs.append(ax)
            payload = payloads[family]
            y_id, y_fd = _metric_arrays(payload, metric)
            marker_id_t20, marker_fd_t20 = _metric_t20_markers(payload, metric)
            _draw_family_metric_panel(
                ax,
                x=np.asarray(payload["x"], dtype=float),
                y_id=y_id,
                y_fd=y_fd,
                family_label=_family_label(family),
                xaxis=xaxis,
                metric=metric,
                y_limits=y_limits,
                show_xlabel=(r == n_rows - 1),
                marker_id_t20=marker_id_t20,
                marker_fd_t20=marker_fd_t20,
            )

        legend_ax = fig.add_subplot(gs[-1, 0])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles,
            loc="center",
            ncol=ncol,
            frameon=False,
            fancybox=False,
            borderpad=0.35,
            columnspacing=1.2,
            handlelength=1.8,
            handletextpad=0.6,
        )

    save_figure(fig, path)
    plt.close(fig)


def plot_family_dual_metric_grid(
    path: Path,
    payloads: dict,
    families: list[str],
    *,
    xaxis: str,
    y_limits: tuple[float, float],
):
    _set_exp03_plot_style(grid=True)

    metrics = ["bestS", "mode"]
    n_rows = len(families)
    n_cols = len(metrics)
    fig = plt.figure(figsize=grid_size(n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1.0] * n_rows + [0.17])
    axs = np.empty((n_rows, n_cols), dtype=object)

    for r in range(n_rows):
        for c in range(n_cols):
            sharex = axs[0, c] if r > 0 else None
            sharey = axs[r, 0] if c > 0 else None
            axs[r, c] = fig.add_subplot(gs[r, c], sharex=sharex, sharey=sharey)

    for r, family in enumerate(families):
        payload = payloads[family]
        for c, metric in enumerate(metrics):
            ax = axs[r, c]
            y_id, y_fd = _metric_arrays(payload, metric)
            marker_id_t20, marker_fd_t20 = _metric_t20_markers(payload, metric)
            _draw_family_metric_panel(
                ax,
                x=np.asarray(payload["x"], dtype=float),
                y_id=y_id,
                y_fd=y_fd,
                family_label=_family_label(family),
                xaxis=xaxis,
                metric=metric,
                y_limits=y_limits,
                show_xlabel=(r == n_rows - 1),
                marker_id_t20=marker_id_t20,
                marker_fd_t20=marker_fd_t20,
            )
            if c > 0:
                ax.set_ylabel("")
            if r == 0:
                ax.set_title(_metric_title(metric), pad=8)

    handles = [
        mlines.Line2D([], [], color=COLORS["ID"], lw=2.2, label="VQE + ID"),
        mlines.Line2D([], [], color=COLORS["FD"], lw=2.0, ls=(0, (4, 2)), label="VQE + BD"),
    ]
    ncol = 2
    if xaxis == "budget":
        handles.extend(
            [
                mlines.Line2D([], [], color=COLORS["REFERENCE"], lw=1.0, ls=":", label=r"Reference $J^*/J^* = 1$"),
                mlines.Line2D([], [], color=COLORS["ID"], marker="o", ls="None", ms=5, label=r"ID at $t=20$"),
                mlines.Line2D([], [], color=COLORS["FD"], marker="s", ls="None", ms=5, label=r"BD at $t=20$"),
            ]
        )
        ncol = 3
    legend_ax = fig.add_subplot(gs[-1, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        fancybox=False,
        borderpad=0.35,
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.6,
        fontsize=10,
    )

    save_figure(fig, path)
    plt.close(fig)


# ==============================================================================
# VQE helpers (experiment-specific signatures)
# ==============================================================================


def zexp_edges(probs: np.ndarray, edges, Z: np.ndarray) -> np.ndarray:
    z = np.empty(len(edges), dtype=float)
    zz = Z.astype(np.float64)
    with np.errstate(all="ignore"):
        for e, (i, j) in enumerate(edges):
            z[e] = float(probs @ (zz[i] * zz[j]))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(z, -1.0, 1.0)


def vqe_expect(n: int, edges, Z: np.ndarray, w: np.ndarray, params: np.ndarray, L: int):
    psi = vqe_state(n, params, L)
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    zexp = zexp_edges(probs, edges, Z)
    p_cut = 0.5 * (1.0 - zexp)
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return J, psi, zexp


def vqe_energy(n: int, edges, Z: np.ndarray, w: np.ndarray, params: np.ndarray, L: int) -> float:
    J, _, _ = vqe_expect(n, edges, Z, w, params, L)
    return -J


# ==============================================================================
# Readout metrics + outer loops
# ==============================================================================


def readout_best_and_mode(rng: np.random.Generator, psi: np.ndarray, cut_vals: np.ndarray, shots: int):
    """
    Sample bitstrings from |psi|^2, return:
      best sampled cut value among shots,
      mode cut value (most frequent bitstring among shots).
    """
    if shots <= 0:
        return float("nan"), float("nan")
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    idx = rng.choice(np.arange(probs.size), size=shots, replace=True, p=probs)
    best = float(np.max(cut_vals[idx]))
    counts = np.bincount(idx, minlength=probs.size)
    mode = int(np.argmax(counts))
    return best, float(cut_vals[mode])


def run_outer_with_readout(
    n: int,
    edges,
    Z: np.ndarray,
    fam,
    cut_mask: np.ndarray,
    lam0: float,
    outer: int,
    inner: int,
    L: int,
    seed: int,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    mode: str,
    c_frac: float,
    readout_shots: int,
    budget_evals: float,
):
    """
    Returns arrays per outer step:
      - evals_cum: cumulative energy evaluations
      - best_of_S: best sampled cut (S shots) at each step
      - mode_cut:  mode cut (S shots) at each step
    Note: readout is computed only at the CENTER candidate (lambda_t) once per outer step.
    """
    rng_read = np.random.default_rng(to_uint_seed(seed + 888888))
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    c = float(c_frac * (lam_max - lam_min))
    evals = 0.0

    out = {k: [] for k in ["lam_pre", "lam", "evals_cum", "best_of_S", "mode_cut"]}

    step_cost = vqe_id_step_cost(inner) if mode == "ID" else vqe_fd_value_step_cost(inner)

    for t in range(1, outer + 1):
        if not can_run_step(evals_used=evals, budget_evals=budget_evals, step_cost=step_cost):
            break
        out["lam_pre"].append(lam)

        # center inner solve
        w = fam.w(lam)

        def Efun(pvec):
            return vqe_energy(n, edges, Z, w, pvec, L)

        params, _, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals += ev_in

        # center value and state
        _Jc, psi, zexp = vqe_expect(n, edges, Z, w, params, L)
        evals += 1.0

        # readout sampling at center
        with np.errstate(all="ignore"):
            cut_vals = cut_mask @ w
        cut_vals = np.nan_to_num(cut_vals, nan=-1e30, posinf=-1e30, neginf=-1e30)
        bS, mC = readout_best_and_mode(rng_read, psi, cut_vals, readout_shots)
        out["best_of_S"].append(float(bS))
        out["mode_cut"].append(float(mC))

        # outer signal + update
        if mode == "ID":
            p_cut = 0.5 * (1.0 - zexp)
            g = float(fam.dw_dlam(lam) @ p_cut)
        elif mode == "FD_VALUE":
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))

            # +c solve
            w_p = fam.w(lp)

            def Efun_p(pvec):
                return vqe_energy(n, edges, Z, w_p, pvec, L)

            p_p, _, evp = spsa_minimize(Efun_p, params, bounds, iters=inner, seed=seed + 1000 * t + 17)
            evals += evp
            _Jp, _, _ = vqe_expect(n, edges, Z, w_p, p_p, L)
            evals += 1.0

            # -c solve
            w_m = fam.w(lm)

            def Efun_m(pvec):
                return vqe_energy(n, edges, Z, w_m, pvec, L)

            p_m, _, evm = spsa_minimize(Efun_m, params, bounds, iters=inner, seed=seed + 1000 * t + 29)
            evals += evm
            _Jm, _, _ = vqe_expect(n, edges, Z, w_m, p_m, L)
            evals += 1.0

            g = (_Jp - _Jm) / (2.0 * c) if c > 0 else 0.0
        else:
            raise ValueError("mode must be ID or FD_VALUE")

        eta = float(eta0 / (t**eta_pow))
        step = float(eta * g)
        if step_clip is not None:
            step = float(np.clip(step, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

        out["lam"].append(lam)
        out["evals_cum"].append(float(evals))

    for k in out:
        out[k] = np.array(out[k], dtype=float)
    return out


def auc_over_steps(values: np.ndarray) -> float:
    """
    Simple outer-step AUC: average over t (trapezoid rule).
    """
    v = np.asarray(values, float)
    if v.size < 2:
        return float(v[0]) if v.size else float("nan")
    x = np.arange(v.size, dtype=float)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(v, x)
    else:
        area = np.trapz(v, x)
    return float(area / (x[-1] - x[0] + 1e-12))


def collect_family_payload(a, family: str, Z: np.ndarray):
    best_id_list: List[np.ndarray] = []
    best_fd_list: List[np.ndarray] = []
    mode_id_list: List[np.ndarray] = []
    mode_fd_list: List[np.ndarray] = []
    eval_id_list: List[np.ndarray] = []
    eval_fd_list: List[np.ndarray] = []
    rows: List[Dict[str, float]] = []

    for r in range(a.num_instances):
        seed = a.seed0 + r
        edges, fam = generate_er_family1d_instance(
            a.n,
            a.p_edge,
            family,
            (a.lam_min, a.lam_max),
            graph_seed=a.graph_seed,
            periodic_K=a.periodic_K,
            instance_id=seed,
            safety_bounds=False,
        )
        if not edges or fam is None:
            continue
        cut_mask = build_cut_mask(edges, Z)

        J_star = classical_Jstar_max(fam, cut_mask, a.grid)
        if not np.isfinite(J_star) or J_star <= 0:
            J_star = 1.0

        hist_id = run_outer_with_readout(
            a.n,
            edges,
            Z,
            fam,
            cut_mask,
            a.lam0,
            a.outer,
            a.inner,
            a.L,
            seed,
            a.eta0,
            a.eta_pow,
            a.step_clip,
            "ID",
            a.c_frac,
            a.readout_shots,
            a.budget_evals,
        )
        hist_fd = run_outer_with_readout(
            a.n,
            edges,
            Z,
            fam,
            cut_mask,
            a.lam0,
            a.outer,
            a.inner,
            a.L,
            seed,
            a.eta0,
            a.eta_pow,
            a.step_clip,
            "FD_VALUE",
            a.c_frac,
            a.readout_shots,
            a.budget_evals,
        )

        best_id = np.maximum.accumulate(hist_id["best_of_S"]) / J_star
        best_fd = np.maximum.accumulate(hist_fd["best_of_S"]) / J_star
        mode_id = hist_id["mode_cut"] / J_star
        mode_fd = hist_fd["mode_cut"] / J_star

        best_id_list.append(best_id)
        best_fd_list.append(best_fd)
        mode_id_list.append(mode_id)
        mode_fd_list.append(mode_fd)
        eval_id_list.append(hist_id["evals_cum"])
        eval_fd_list.append(hist_fd["evals_cum"])

        rows.append(
            {
                "instance": r,
                "seed": seed,
                "family": family,
                "K": float(a.periodic_K) if family == "periodic" else float("nan"),
                "n": float(a.n),
                "p_edge": float(a.p_edge),
                "graph_seed": float(a.graph_seed),
                "outer": float(a.outer),
                "inner": float(a.inner),
                "L": float(a.L),
                "budget_evals": float(a.budget_evals),
                "readout_shots": float(a.readout_shots),
                "J_star": float(J_star),
                "evals_final_ID": float(hist_id["evals_cum"][-1]),
                "evals_final_FD": float(hist_fd["evals_cum"][-1]),
                "bestS_final_ID": float(best_id[-1]),
                "bestS_final_FD": float(best_fd[-1]),
                "mode_final_ID": float(mode_id[-1]),
                "mode_final_FD": float(mode_fd[-1]),
                "bestS_auc_ID": auc_over_steps(best_id),
                "bestS_auc_FD": auc_over_steps(best_fd),
                "mode_auc_ID": auc_over_steps(mode_id),
                "mode_auc_FD": auc_over_steps(mode_fd),
            }
        )

    if not best_id_list:
        raise RuntimeError(f"No instances generated for family={family}. Try increasing p_edge or changing seed0.")

    if a.xaxis == "iters":
        T_common = min(
            min(arr.size for arr in best_id_list),
            min(arr.size for arr in best_fd_list),
            min(arr.size for arr in mode_id_list),
            min(arr.size for arr in mode_fd_list),
        )
        payload = {
            "x": np.arange(1, T_common + 1, dtype=float),
            "metric_id_bestS": np.vstack([arr[:T_common] for arr in best_id_list]),
            "metric_fd_bestS": np.vstack([arr[:T_common] for arr in best_fd_list]),
            "metric_id_mode": np.vstack([arr[:T_common] for arr in mode_id_list]),
            "metric_fd_mode": np.vstack([arr[:T_common] for arr in mode_fd_list]),
            "marker_id_bestS_t20": _point_to_array(_avg_eval_metric_coord(eval_id_list, best_id_list)),
            "marker_fd_bestS_t20": _point_to_array(_avg_eval_metric_coord(eval_fd_list, best_fd_list)),
            "marker_id_mode_t20": _point_to_array(_avg_eval_metric_coord(eval_id_list, mode_id_list)),
            "marker_fd_mode_t20": _point_to_array(_avg_eval_metric_coord(eval_fd_list, mode_fd_list)),
        }
    else:
        budget_grid = np.linspace(0.0, float(a.budget_evals), int(a.budget_points))
        payload = {
            "x": budget_grid,
            "metric_id_bestS": np.vstack(
                [step_interp(ev, y, budget_grid) for ev, y in zip(eval_id_list, best_id_list)]
            ),
            "metric_fd_bestS": np.vstack(
                [step_interp(ev, y, budget_grid) for ev, y in zip(eval_fd_list, best_fd_list)]
            ),
            "metric_id_mode": np.vstack([step_interp(ev, y, budget_grid) for ev, y in zip(eval_id_list, mode_id_list)]),
            "metric_fd_mode": np.vstack([step_interp(ev, y, budget_grid) for ev, y in zip(eval_fd_list, mode_fd_list)]),
            "marker_id_bestS_t20": _point_to_array(_avg_eval_metric_coord(eval_id_list, best_id_list)),
            "marker_fd_bestS_t20": _point_to_array(_avg_eval_metric_coord(eval_fd_list, best_fd_list)),
            "marker_id_mode_t20": _point_to_array(_avg_eval_metric_coord(eval_id_list, mode_id_list)),
            "marker_fd_mode_t20": _point_to_array(_avg_eval_metric_coord(eval_fd_list, mode_fd_list)),
        }
    return payload, rows


# ==============================================================================
# CLI + experiment driver
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--render_only", action="store_true")

    # instances / seeds
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--num_instances", type=int, default=20)

    # problem
    p.add_argument("--family", type=str, default=None, choices=["linear", "quadratic", "periodic"])
    p.add_argument("--families", type=str, default="linear,quadratic,periodic")
    p.add_argument("--periodic_K", type=int, default=CANONICAL_SETUP.periodic_K)
    p.add_argument("--n", type=int, default=CANONICAL_SETUP.n)
    p.add_argument("--p_edge", type=float, default=CANONICAL_SETUP.p_edge)
    p.add_argument("--graph_seed", type=int, default=CANONICAL_SETUP.graph_seed)
    p.add_argument("--lam_min", type=float, default=CANONICAL_SETUP.lam_min)
    p.add_argument("--lam_max", type=float, default=CANONICAL_SETUP.lam_max)
    p.add_argument("--lam0", type=float, default=CANONICAL_SETUP.lam0)
    p.add_argument("--grid", type=int, default=401)

    # optimization
    p.add_argument("--outer", type=int, default=100)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)
    p.add_argument("--budget_evals", type=float, default=CANONICAL_SETUP.budget_evals)

    # readout
    p.add_argument("--readout_shots", type=int, default=128)

    # plotting axis choice
    p.add_argument(
        "--xaxis",
        type=str,
        default="iters",
        choices=["iters", "budget"],
        help="Plot x-axis: 'iters' (outer iteration) or 'budget' (energy evaluations).",
    )
    p.add_argument(
        "--budget_points", type=int, default=220, help="Number of points for shared budget grid when --xaxis=budget."
    )
    p.add_argument("--metric", type=str, default="bestS", choices=["bestS", "mode", "both"])
    p.add_argument(
        "--family_grid_layout",
        type=str,
        default="stack",
        choices=["stack", "square"],
        help="Layout for multi-family grid plots.",
    )
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=1.01)

    return p.parse_args()


def main():
    a = parse_args()
    outdir = (
        Path(a.out)
        if a.out is not None
        else publication_output_dir("exp03", "budget" if a.xaxis == "budget" else "iters")
    )
    outdir.mkdir(parents=True, exist_ok=True)
    families = [a.family] if a.family is not None else parse_str_list(a.families)
    cache_dir = Path(a.cache_dir) if a.cache_dir is not None else _cache_default_dir(outdir, a.xaxis)
    meta = _cache_meta(a, families)

    cached = None if a.recompute else load_exp03_cache(cache_dir, meta)
    if cached is not None:
        payloads, rows_by_family = cached
        print(f"[cache] Loaded readout payloads from {cache_dir.resolve()}")
    elif a.render_only:
        raise SystemExit(f"No matching cache found in {cache_dir}")
    else:
        Z = precompute_z_big_endian(a.n)
        payloads = {}
        rows_by_family = {}
        for family in families:
            payloads[family], rows_by_family[family] = collect_family_payload(a, family, Z)
            N_family = int(payloads[family]["metric_id_bestS"].shape[0])
            print(f"[done] {family:9s} | N={N_family}")
        save_exp03_cache(cache_dir, meta, payloads, rows_by_family)
        print(f"[cache] Saved readout payloads to {cache_dir.resolve()}")

    rows = [row for family in families for row in rows_by_family[family]]
    summary_rows = [row for family in families for row in _metric_summary_rows(rows_by_family[family], family)]

    suffix_families = "-".join(families)
    N_total = sum(int(payloads[family]["metric_id_bestS"].shape[0]) for family in families)
    suf = f"{suffix_families}_n{a.n}_S{a.readout_shots}_seed0{a.seed0}_N{N_total}"
    x_suffix = "xIters" if a.xaxis == "iters" else "xBudget"
    layout_suffix = "_square" if a.family_grid_layout == "square" and len(families) > 1 else ""

    if a.metric == "both" and len(families) == 1:
        family = families[0]
        payload = payloads[family]
        fig_path = outdir / f"fig3_readout_best_mode_{suf}_{x_suffix}.{a.fmt}"
        if a.xaxis == "iters":
            plot_2panel_iters(
                fig_path,
                payload["metric_id_bestS"],
                payload["metric_fd_bestS"],
                payload["metric_id_mode"],
                payload["metric_fd_mode"],
            )
        else:
            plot_2panel_budget(
                fig_path,
                payload["x"],
                payload["metric_id_bestS"],
                payload["metric_fd_bestS"],
                payload["metric_id_mode"],
                payload["metric_fd_mode"],
            )
    elif a.metric == "both":
        y_limits = (
            interesting_readout_ylim_multi(payloads, families, ["bestS", "mode"], ymax=a.ymax)
            if a.ymin is None
            else (float(a.ymin), float(a.ymax))
        )
        fig_path = outdir / f"fig3_readout_best_mode_family_grid_{suf}_{x_suffix}.{a.fmt}"
        plot_family_dual_metric_grid(fig_path, payloads, families, xaxis=a.xaxis, y_limits=y_limits)
    else:
        metric = "bestS" if a.metric == "both" else a.metric
        y_limits = (
            interesting_readout_ylim(payloads, families, metric, ymax=a.ymax)
            if a.ymin is None
            else (float(a.ymin), float(a.ymax))
        )
        fig_path = outdir / f"fig3_readout_{metric}_family_grid_{suf}_{x_suffix}{layout_suffix}.{a.fmt}"
        plot_family_metric_grid(
            fig_path,
            payloads,
            families,
            metric=metric,
            xaxis=a.xaxis,
            y_limits=y_limits,
            layout=a.family_grid_layout,
        )

        if len(families) > 1:
            for family in families:
                family_limits = (
                    interesting_readout_ylim(payloads, [family], metric, ymax=a.ymax)
                    if a.ymin is None
                    else (float(a.ymin), float(a.ymax))
                )
                family_fig_path = (
                    outdir / f"fig3_readout_{metric}_{family}_n{a.n}_S{a.readout_shots}_{x_suffix}.{a.fmt}"
                )
                plot_family_metric_grid(
                    family_fig_path,
                    payloads,
                    [family],
                    metric=metric,
                    xaxis=a.xaxis,
                    y_limits=family_limits,
                )

    csv_path = outdir / "runs3_readout_metrics.csv"
    write_csv(csv_path, rows)

    table_csv = outdir / "table3_readout_summary.csv"
    write_csv(table_csv, summary_rows)

    table_tex = outdir / "table3_readout_summary.tex"
    with open(table_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by exp03_readout_realism_best_mode.py\n")
        f.write("\\begin{tabular}{l l c c}\n")
        f.write("\\toprule\n")
        f.write("Family & Metric & VQE+ID & VQE+BD\\\\\n")
        f.write("\\midrule\n")
        for row in summary_rows:
            f.write(
                f"{row['family']} & {row['metric']} & "
                f"{float(row['ID_mean']):.3f}$\\pm${float(row['ID_stderr']):.3f} & "
                f"{float(row['BD_mean']):.3f}$\\pm${float(row['BD_stderr']):.3f}\\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    txt_path = outdir / "SUMMARY.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            f"Experiment 3 (Readout realism) | families={','.join(families)} | n={a.n} | "
            f"N_total={N_total} | S_readout={a.readout_shots} | xaxis={a.xaxis} | metric={a.metric}\n"
        )
        f.write(f"budget_evals={a.budget_evals} | graph_seed={a.graph_seed}\n")
        for row in summary_rows:
            f.write(
                f"{row['family']} | {row['metric']}: "
                f"ID={float(row['ID_mean']):.4f}+/-{float(row['ID_stderr']):.4f} | "
                f"BD={float(row['BD_mean']):.4f}+/-{float(row['BD_stderr']):.4f}\n"
            )
        f.write(f"Figure: {fig_path.name}\n")
        f.write(f"Runs: {csv_path.name}\n")

    print("Saved to:", outdir.resolve())
    print("Figure:", fig_path.name)
    print("Runs CSV:", csv_path.name)
    print("Summary tables:", table_csv.name, "/", table_tex.name)


if __name__ == "__main__":
    main()
