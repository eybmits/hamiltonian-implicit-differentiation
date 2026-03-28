#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal Experiment 2 implementation with the final matched-budget plots."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from plot_style import FIG_H, FIG_W, apply_thesis_axes_style, newfig, save_figure, scaled_height, use_thesis_style

from paramham.experiment_defaults import (
    CANONICAL_SETUP,
    can_run_step,
    generate_er_family1d_instance,
    publication_cache_dir,
    publication_output_dir,
    vqe_fd_value_step_cost,
    vqe_id_step_cost,
)
from paramham.io import parse_int_list, parse_str_list, write_csv
from paramham.maxcut import build_cut_mask, build_ZZ_edges, classical_Jstar
from paramham.maxcut import precompute_z as precompute_z_big_endian
from paramham.metrics import mean_stderr, step_interp
from paramham.plotting import (
    COLORS,
    FULL_W,
    H_COL,
    add_panel_legend,
)
from paramham.seeds import to_uint_seed
from paramham.simulator import vqe_state
from paramham.spsa import spsa_minimize


def set_exp02_plot_style():
    """Typography/size profile for Experiment 2, matched to thesis text size."""

    use_thesis_style()
    mpl.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.7,
            "grid.linestyle": "-",
            "grid.linewidth": 0.75,
            "grid.color": "#D7D7D7",
        }
    )


def stderr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def auc_trapz(budgets: np.ndarray, values: np.ndarray) -> float:
    budgets = np.asarray(budgets, dtype=float)
    values = np.asarray(values, dtype=float)
    if budgets.size < 2:
        return float("nan")
    return float(np.trapezoid(values, budgets))


def estimate_J_and_zexp(psi, w, ZZ, shots, rng):
    w = np.asarray(w, dtype=np.float64)
    ZZ = np.asarray(ZZ, dtype=np.int8)
    m, K = ZZ.shape
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    if shots is None or shots <= 0:
        zexp = ZZ.astype(np.float64) @ probs
    else:
        idx = rng.choice(np.arange(K), size=int(shots), replace=True, p=probs)
        zexp = np.mean(ZZ[:, idx].astype(np.float64), axis=1)
    zexp = np.clip(np.nan_to_num(zexp, nan=0.0), -1.0, 1.0)
    p_cut = 0.5 * (1.0 - zexp)
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return J, zexp


def vqe_eval(n, params, L, w, ZZ, shots, rng_meas):
    psi = vqe_state(n, params, L)
    return estimate_J_and_zexp(psi, w, ZZ, shots, rng_meas)


def classical_Jstar_max(fam, cut_mask, grid_points):
    J_star, _ = classical_Jstar(fam, cut_mask, grid_points)
    return J_star


@dataclass
class RunHist:
    evals: np.ndarray
    best: np.ndarray
    best_norm: np.ndarray
    final_best_at_budget: float
    auc_at_budget: float


def run_outer(
    *, mode, n, edges, ZZ, fam, lam0, outer, inner, L, shots, seed, eta0, eta_pow, step_clip, c_frac, J_star_max, budget
):
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))
    D = 2 * n * L
    params = np.zeros(D, dtype=np.float64)
    bounds = [(-math.pi, math.pi)] * D
    rng_meas = np.random.default_rng(to_uint_seed(seed + 99991))
    evals_cum = 0.0
    best = -1e18
    evals_trace, best_trace = [], []
    c = float(c_frac * (lam_max - lam_min))

    step_cost = vqe_id_step_cost(inner) if mode == "ID" else vqe_fd_value_step_cost(inner)

    for t in range(1, outer + 1):
        if not can_run_step(evals_used=evals_cum, budget_evals=budget, step_cost=step_cost):
            break
        w = fam.w(lam)

        def energy_fun(pvec):
            J_hat, _ = vqe_eval(n, pvec, L, w, ZZ, shots, rng_meas)
            return -J_hat

        params, _, ev_in = spsa_minimize(energy_fun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals_cum += ev_in
        J, zexp = vqe_eval(n, params, L, w, ZZ, shots, rng_meas)
        evals_cum += 1.0
        best = max(best, float(J))

        if mode == "ID":
            p_cut = 0.5 * (1.0 - zexp)
            g = float(fam.dw_dlam(lam) @ p_cut)
        elif mode == "FD_VALUE":
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))
            w_p = fam.w(lp)

            def efp(pvec):
                J_hat, _ = vqe_eval(n, pvec, L, w_p, ZZ, shots, rng_meas)
                return -J_hat

            p_p, _, evp = spsa_minimize(efp, params, bounds, iters=inner, seed=seed + 1000 * t + 17)
            evals_cum += evp
            Jp, _ = vqe_eval(n, p_p, L, w_p, ZZ, shots, rng_meas)
            evals_cum += 1.0
            w_m = fam.w(lm)

            def efm(pvec):
                J_hat, _ = vqe_eval(n, pvec, L, w_m, ZZ, shots, rng_meas)
                return -J_hat

            p_m, _, evm = spsa_minimize(efm, params, bounds, iters=inner, seed=seed + 1000 * t + 29)
            evals_cum += evm
            Jm, _ = vqe_eval(n, p_m, L, w_m, ZZ, shots, rng_meas)
            evals_cum += 1.0
            best = max(best, float(Jp), float(Jm))
            g = float((Jp - Jm) / (2.0 * c)) if c > 0 else 0.0
            try:
                lam_new_cand = float(np.clip(lam + (eta0 / (t**eta_pow)) * g, lam_min, lam_max))
                cand = [(lam, params), (lp, p_p), (lm, p_m)]
                _, p_closest = min(cand, key=lambda pr: abs(pr[0] - lam_new_cand))
                params = np.asarray(p_closest, dtype=np.float64).copy()
            except Exception:
                pass
        else:
            raise ValueError("mode?")

        eta_t = eta0 / (t**eta_pow)
        step = float(eta_t * g)
        if step_clip is not None:
            step = float(np.clip(step, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))
        evals_trace.append(float(evals_cum))
        best_trace.append(float(best))

    evals_arr = np.asarray(evals_trace, dtype=float)
    best_arr = np.asarray(best_trace, dtype=float)
    best_norm = best_arr / max(1e-12, float(J_star_max))
    B = float(budget)
    budgets = np.linspace(0.0, B, 200)
    best_on_grid = step_interp(evals_arr, best_norm, budgets)
    final_best = float(best_on_grid[-1])
    auc = auc_trapz(budgets, best_on_grid) / max(1e-12, B)
    return RunHist(
        evals=evals_arr, best=best_arr, best_norm=best_norm, final_best_at_budget=final_best, auc_at_budget=float(auc)
    )


@dataclass
class AggCurves:
    budgets: np.ndarray
    mean_id: np.ndarray
    se_id: np.ndarray
    mean_fd: np.ndarray
    se_fd: np.ndarray
    final_id_mean: float
    final_id_se: float
    final_fd_mean: float
    final_fd_se: float
    auc_id_mean: float
    auc_id_se: float
    auc_fd_mean: float
    auc_fd_se: float
    budget_used: float
    id_p20: Optional[Tuple[float, float]]
    fd_p20: Optional[Tuple[float, float]]


def _curve_cache_prefix(kind: str, shots: int) -> str:
    return f"{kind}__shots{int(shots)}"


def _cache_default_dir(out: Path) -> Path:
    return publication_cache_dir("exp02")


def _cache_meta(args, families, shots_list) -> dict:
    return {
        "seed0": int(args.seed0),
        "num_seeds": int(args.num_seeds),
        "n": int(args.n),
        "p_edge": float(args.p_edge),
        "graph_seed": int(args.graph_seed),
        "lam_min": float(args.lam_min),
        "lam_max": float(args.lam_max),
        "lam0": float(args.lam0),
        "families": [str(x) for x in families],
        "periodic_K": int(args.periodic_K),
        "shots_list": [int(x) for x in shots_list],
        "outer": int(args.outer),
        "inner": int(args.inner),
        "L": int(args.L),
        "eta0": float(args.eta0),
        "eta_pow": float(args.eta_pow),
        "step_clip": None if args.step_clip is None else float(args.step_clip),
        "c_frac": float(args.c_frac),
        "grid": int(args.grid),
        "budget": float(args.budget),
        "budget_points": int(args.budget_points),
    }


def _point_to_array(point: Optional[Tuple[float, float]]) -> np.ndarray:
    if point is None:
        return np.array([np.nan, np.nan], dtype=float)
    return np.asarray(point, dtype=float)


def _array_to_point(arr: np.ndarray) -> Optional[Tuple[float, float]]:
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        return None
    return (float(arr[0]), float(arr[1]))


def save_curves_cache(cache_dir: Path, meta: dict, curves_by_family) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for kind, curves_by_shots in curves_by_family.items():
        for shots, C in curves_by_shots.items():
            prefix = _curve_cache_prefix(kind, shots)
            arrays[f"{prefix}__budgets"] = np.asarray(C.budgets, dtype=float)
            arrays[f"{prefix}__mean_id"] = np.asarray(C.mean_id, dtype=float)
            arrays[f"{prefix}__se_id"] = np.asarray(C.se_id, dtype=float)
            arrays[f"{prefix}__mean_fd"] = np.asarray(C.mean_fd, dtype=float)
            arrays[f"{prefix}__se_fd"] = np.asarray(C.se_fd, dtype=float)
            arrays[f"{prefix}__scalars"] = np.asarray(
                [
                    C.final_id_mean,
                    C.final_id_se,
                    C.final_fd_mean,
                    C.final_fd_se,
                    C.auc_id_mean,
                    C.auc_id_se,
                    C.auc_fd_mean,
                    C.auc_fd_se,
                    C.budget_used,
                ],
                dtype=float,
            )
            arrays[f"{prefix}__id_p20"] = _point_to_array(C.id_p20)
            arrays[f"{prefix}__fd_p20"] = _point_to_array(C.fd_p20)
    np.savez_compressed(cache_dir / "curves_cache.npz", **arrays)
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def load_curves_cache(cache_dir: Path, meta_expected: dict):
    meta_path = cache_dir / "cache_meta.json"
    npz_path = cache_dir / "curves_cache.npz"
    if not meta_path.exists() or not npz_path.exists():
        return None
    try:
        meta_found = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    if meta_found != meta_expected:
        return None

    curves_by_family = {}
    with np.load(npz_path) as data:
        for kind in meta_expected["families"]:
            curves_by_shots = {}
            for shots in meta_expected["shots_list"]:
                prefix = _curve_cache_prefix(kind, shots)
                required = [
                    f"{prefix}__budgets",
                    f"{prefix}__mean_id",
                    f"{prefix}__se_id",
                    f"{prefix}__mean_fd",
                    f"{prefix}__se_fd",
                    f"{prefix}__scalars",
                    f"{prefix}__id_p20",
                    f"{prefix}__fd_p20",
                ]
                if any(key not in data for key in required):
                    return None
                scalars = np.asarray(data[f"{prefix}__scalars"], dtype=float)
                curves_by_shots[int(shots)] = AggCurves(
                    budgets=np.asarray(data[f"{prefix}__budgets"], dtype=float),
                    mean_id=np.asarray(data[f"{prefix}__mean_id"], dtype=float),
                    se_id=np.asarray(data[f"{prefix}__se_id"], dtype=float),
                    mean_fd=np.asarray(data[f"{prefix}__mean_fd"], dtype=float),
                    se_fd=np.asarray(data[f"{prefix}__se_fd"], dtype=float),
                    final_id_mean=float(scalars[0]),
                    final_id_se=float(scalars[1]),
                    final_fd_mean=float(scalars[2]),
                    final_fd_se=float(scalars[3]),
                    auc_id_mean=float(scalars[4]),
                    auc_id_se=float(scalars[5]),
                    auc_fd_mean=float(scalars[6]),
                    auc_fd_se=float(scalars[7]),
                    budget_used=float(scalars[8]),
                    id_p20=_array_to_point(np.asarray(data[f"{prefix}__id_p20"], dtype=float)),
                    fd_p20=_array_to_point(np.asarray(data[f"{prefix}__fd_p20"], dtype=float)),
                )
            curves_by_family[kind] = curves_by_shots
    return curves_by_family


def build_summary_tables(curves_by_family, families, shots_list):
    table_rows = []
    latex_table = {}
    for kind in families:
        for shots in shots_list:
            curves = curves_by_family[kind][shots]
            shots_lbl = "exact" if shots <= 0 else str(shots)
            table_rows.append(
                {
                    "family": kind,
                    "shots": shots_lbl,
                    "budget": f"{curves.budget_used:.0f}",
                    "ID_mean": f"{curves.final_id_mean:.4f}",
                    "ID_stderr": f"{curves.final_id_se:.4f}",
                    "FD_mean": f"{curves.final_fd_mean:.4f}",
                    "FD_stderr": f"{curves.final_fd_se:.4f}",
                }
            )
            latex_table[(kind, shots_lbl)] = {
                "ID": (curves.final_id_mean, curves.final_id_se),
                "FD": (curves.final_fd_mean, curves.final_fd_se),
            }
    return table_rows, latex_table


def interesting_y_limits(curves_iter, *, ymax: float = 1.01) -> tuple[float, float]:
    ymin_candidates = []
    for C in curves_iter:
        mask = np.asarray(C.budgets, dtype=float) > 0.0
        for arr in (C.mean_id - C.se_id, C.mean_fd - C.se_fd, C.mean_id, C.mean_fd):
            vals = np.asarray(arr, dtype=float)[mask]
            vals = vals[np.isfinite(vals) & (vals > 0.05)]
            if vals.size:
                ymin_candidates.append(float(np.min(vals)))
        for point in (C.id_p20, C.fd_p20):
            if point is not None and np.isfinite(point[1]) and point[1] > 0.05:
                ymin_candidates.append(float(point[1]))
    if not ymin_candidates:
        return (0.0, float(ymax))
    ymin = min(ymin_candidates) - 0.05
    ymin = math.floor(ymin / 0.05) * 0.05
    ymin = float(np.clip(ymin, 0.45, 0.80))
    ymax = float(max(ymax, ymin + 0.15))
    return (ymin, ymax)


def aggregate_runs(runs_id, runs_fd, budget, budget_points):
    B = float(budget)
    budgets = np.linspace(0.0, B, int(budget_points))
    Y_id = np.stack([step_interp(r.evals, r.best_norm, budgets) for r in runs_id], axis=0)
    Y_fd = np.stack([step_interp(r.evals, r.best_norm, budgets) for r in runs_fd], axis=0)
    m_id, se_id_arr = mean_stderr(Y_id, axis=0)
    m_fd, se_fd_arr = mean_stderr(Y_fd, axis=0)
    final_id = np.array([r.final_best_at_budget for r in runs_id], dtype=float)
    final_fd = np.array([r.final_best_at_budget for r in runs_fd], dtype=float)
    auc_id = np.array([r.auc_at_budget for r in runs_id], dtype=float)
    auc_fd = np.array([r.auc_at_budget for r in runs_fd], dtype=float)

    target_idx = 19

    def get_avg_coord(runs, idx):
        xs, ys = [], []
        for r in runs:
            if idx < r.evals.size:
                xs.append(r.evals[idx])
                ys.append(r.best_norm[idx])
        if not xs:
            return None
        return (float(np.mean(xs)), float(np.mean(ys)))

    return AggCurves(
        budgets=budgets,
        mean_id=m_id,
        se_id=se_id_arr,
        mean_fd=m_fd,
        se_fd=se_fd_arr,
        final_id_mean=float(np.mean(final_id)),
        final_id_se=stderr(final_id),
        final_fd_mean=float(np.mean(final_fd)),
        final_fd_se=stderr(final_fd),
        auc_id_mean=float(np.mean(auc_id)),
        auc_id_se=stderr(auc_id),
        auc_fd_mean=float(np.mean(auc_fd)),
        auc_fd_se=stderr(auc_fd),
        budget_used=B,
        id_p20=get_avg_coord(runs_id, target_idx),
        fd_p20=get_avg_coord(runs_fd, target_idx),
    )


def _family_label(kind: str) -> str:
    return {
        "linear": "Linear",
        "quadratic": "Quadratic",
        "periodic": "Periodic",
    }.get(str(kind), str(kind).title())


def _draw_budget_panel(
    ax,
    C: AggCurves,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    y_limits: tuple[float, float],
    family_label: str | None = None,
    column_title: str | None = None,
    panel_legend: bool = False,
    label_markers: bool = True,
):
    x = C.budgets
    ax.axhline(1.0, color=COLORS["REFERENCE"], lw=1.0, ls=":", alpha=0.85, zorder=1, label="_nolegend_")

    ax.fill_between(x, C.mean_fd - C.se_fd, C.mean_fd + C.se_fd, color=COLORS["FD"], alpha=0.14, linewidth=0, zorder=1)
    ax.fill_between(x, C.mean_id - C.se_id, C.mean_id + C.se_id, color=COLORS["ID"], alpha=0.18, linewidth=0, zorder=2)
    ax.plot(
        x,
        C.mean_fd,
        color=COLORS["FD"],
        lw=2.0,
        ls=(0, (4, 2)),
        label="VQE + FD",
        zorder=3,
        solid_capstyle="round",
    )
    ax.plot(x, C.mean_id, color=COLORS["ID"], lw=2.2, ls="-", label="VQE + CR-ID", zorder=4, solid_capstyle="round")

    if C.id_p20:
        x_id, y_id = C.id_p20
        ax.plot(
            x_id,
            y_id,
            marker="o",
            color=COLORS["ID"],
            markersize=5.5,
            zorder=10,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )
        if label_markers:
            ax.annotate(
                "t=20",
                (x_id, y_id),
                xytext=(0, 10),
                textcoords="offset points",
                color=COLORS["ID"],
                fontsize=10,
                ha="center",
            )
    if C.fd_p20:
        x_fd, y_fd = C.fd_p20
        if x_fd <= C.budget_used:
            ax.plot(
                x_fd,
                y_fd,
                marker="s",
                color=COLORS["FD"],
                markersize=5.2,
                zorder=10,
                markeredgecolor="white",
                markeredgewidth=0.9,
            )
            if label_markers:
                ax.annotate(
                    "t=20",
                    (x_fd, y_fd),
                    xytext=(0, -12),
                    textcoords="offset points",
                    color=COLORS["FD"],
                    fontsize=10,
                    ha="center",
                )

    apply_thesis_axes_style(ax)
    ax.set_xlim(0, C.budget_used)
    ax.set_ylim(*y_limits)
    if show_xlabel:
        ax.set_xlabel("Energy evaluations")
    if show_ylabel:
        ax.set_ylabel(r"Approximation ratio $\widehat{F}/J^\star$")

    if family_label is not None:
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

    if column_title is not None:
        ax.set_title(column_title, pad=8)

    if panel_legend:
        add_panel_legend(ax, placement="below", ncol=2, fontsize=10, frameon=False)


def plot_budget_curves(path, curves_by_shots, shots_list, family_name: str | None = None, y_limits=None):
    set_exp02_plot_style()

    all_curves = [curves_by_shots[shots] for shots in shots_list]
    if y_limits is None:
        y_limits = interesting_y_limits(all_curves)

    if len(shots_list) == 2:
        fig_h = scaled_height(FULL_W, H_COL + 0.55)
        fig, axs = plt.subplots(1, 2, figsize=(FIG_W, fig_h), constrained_layout=True, sharey=True)
        axes = list(axs)
    else:
        fig, ax = newfig(height=FIG_H)
        axes = [ax]

    for ax, shots in zip(axes, shots_list):
        C = curves_by_shots[shots]
        _draw_budget_panel(
            ax,
            C,
            show_ylabel=ax is axes[0],
            show_xlabel=True,
            y_limits=y_limits,
            family_label=family_name if ax is axes[0] and family_name is not None else None,
            column_title=None,
            panel_legend=True,
        )

    save_figure(fig, path)
    plt.close(fig)


def plot_budget_family_grid(path: Path, curves_by_family, families, shots_list, *, y_limits: tuple[float, float]):
    set_exp02_plot_style()

    n_rows = len(families)
    n_cols = len(shots_list)
    legacy_fig_h = n_rows * (H_COL + 0.14) + 0.62
    fig_h = scaled_height(FULL_W, legacy_fig_h)
    fig = plt.figure(figsize=(FIG_W, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1.0] * n_rows + [0.17])
    axs = np.empty((n_rows, n_cols), dtype=object)

    for r in range(n_rows):
        for c in range(n_cols):
            sharex = axs[0, c] if r > 0 else None
            sharey = axs[r, 0] if c > 0 else None
            axs[r, c] = fig.add_subplot(gs[r, c], sharex=sharex, sharey=sharey)

    for r, kind in enumerate(families):
        for c, shots in enumerate(shots_list):
            ax = axs[r, c]
            _draw_budget_panel(
                ax,
                curves_by_family[kind][shots],
                show_ylabel=(c == 0),
                show_xlabel=(r == n_rows - 1),
                y_limits=y_limits,
                family_label=_family_label(kind),
                column_title=None,
                panel_legend=False,
            )

    handles = [
        mlines.Line2D([], [], color=COLORS["ID"], lw=1.9, label="VQE + CR-ID"),
        mlines.Line2D([], [], color=COLORS["FD"], lw=1.7, ls="--", label="VQE + FD"),
        mlines.Line2D([], [], color=COLORS["REFERENCE"], lw=1.0, ls=":", label=r"Reference $J^*/J^* = 1$"),
        mlines.Line2D([], [], color=COLORS["ID"], marker="o", ls="None", ms=5, label=r"CR-ID at $t=20$"),
        mlines.Line2D([], [], color=COLORS["FD"], marker="s", ls="None", ms=5, label=r"FD at $t=20$"),
    ]
    legend_ax = fig.add_subplot(gs[-1, :])
    legend_ax.axis("off")
    legend = legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fancybox=False,
        borderpad=0.35,
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.6,
        fontsize=10,
    )
    legend.set_zorder(20)

    save_figure(fig, path)
    plt.close(fig)


def write_table_latex(path, table):
    lines = [r"\begin{tabular}{l c c c}", r"\hline", r"Family & Shots & VQE--CR-ID & VQE--FD \\", r"\hline"]
    for (kind, shots_lbl), d in table.items():
        mid, seid = d["ID"]
        mfd, sefd = d["FD"]
        lines.append(f"{kind} & {shots_lbl} & {mid:.3f} $\\pm$ {seid:.3f} & {mfd:.3f} $\\pm$ {sefd:.3f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--render_only", action="store_true")
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--n", type=int, default=CANONICAL_SETUP.n)
    p.add_argument("--p_edge", type=float, default=CANONICAL_SETUP.p_edge)
    p.add_argument("--graph_seed", type=int, default=CANONICAL_SETUP.graph_seed)
    p.add_argument("--lam_min", type=float, default=CANONICAL_SETUP.lam_min)
    p.add_argument("--lam_max", type=float, default=CANONICAL_SETUP.lam_max)
    p.add_argument("--lam0", type=float, default=CANONICAL_SETUP.lam0)
    p.add_argument("--families", type=str, default="linear,quadratic,periodic")
    p.add_argument("--periodic_K", type=int, default=CANONICAL_SETUP.periodic_K)
    p.add_argument("--shots_list", type=str, default="0,256")
    p.add_argument("--outer", type=int, default=100)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)
    p.add_argument("--grid", type=int, default=401)
    p.add_argument("--budget", "--budget_evals", dest="budget", type=float, default=CANONICAL_SETUP.budget_evals)
    p.add_argument("--budget_points", type=int, default=500)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=1.01)
    args = p.parse_args()

    out = Path(args.out) if args.out is not None else publication_output_dir("exp02")
    out.mkdir(parents=True, exist_ok=True)
    families = parse_str_list(args.families)
    shots_list = parse_int_list(args.shots_list)
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else _cache_default_dir(out)
    meta = _cache_meta(args, families, shots_list)

    curves_by_family = None if args.recompute else load_curves_cache(cache_dir, meta)
    if curves_by_family is not None:
        print(f"[cache] Loaded aggregate curves from {cache_dir.resolve()}")
    elif args.render_only:
        raise SystemExit(f"No matching cache found in {cache_dir}")
    else:
        Z = precompute_z_big_endian(args.n)
        curves_by_family = {}
        for kind in families:
            curves_by_shots = {}
            for shots in shots_list:
                runs_id, runs_fd = [], []
                for i in range(args.num_seeds):
                    seed = args.seed0 + i
                    edges, fam = generate_er_family1d_instance(
                        args.n,
                        args.p_edge,
                        kind,
                        (args.lam_min, args.lam_max),
                        graph_seed=args.graph_seed,
                        periodic_K=args.periodic_K,
                        instance_id=seed,
                    )
                    if not edges or fam is None:
                        continue
                    ZZ = build_ZZ_edges(edges, Z)
                    cut_mask = build_cut_mask(edges, Z)
                    J_star_max = classical_Jstar_max(fam, cut_mask, args.grid)
                    budget = float(args.budget)
                    runs_id.append(
                        run_outer(
                            mode="ID",
                            n=args.n,
                            edges=edges,
                            ZZ=ZZ,
                            fam=fam,
                            lam0=args.lam0,
                            outer=args.outer,
                            inner=args.inner,
                            L=args.L,
                            shots=shots,
                            seed=to_uint_seed(seed + 111),
                            eta0=args.eta0,
                            eta_pow=args.eta_pow,
                            step_clip=args.step_clip,
                            c_frac=args.c_frac,
                            J_star_max=J_star_max,
                            budget=budget,
                        )
                    )
                    runs_fd.append(
                        run_outer(
                            mode="FD_VALUE",
                            n=args.n,
                            edges=edges,
                            ZZ=ZZ,
                            fam=fam,
                            lam0=args.lam0,
                            outer=args.outer,
                            inner=args.inner,
                            L=args.L,
                            shots=shots,
                            seed=to_uint_seed(seed + 222),
                            eta0=args.eta0,
                            eta_pow=args.eta_pow,
                            step_clip=args.step_clip,
                            c_frac=args.c_frac,
                            J_star_max=J_star_max,
                            budget=budget,
                        )
                    )

                curves = aggregate_runs(runs_id, runs_fd, budget=float(args.budget), budget_points=args.budget_points)
                curves_by_shots[shots] = curves
                shots_lbl = "exact" if shots <= 0 else str(shots)
                print(
                    f"[done] {kind:9s} {shots_lbl:>5s} | ID {curves.final_id_mean:.3f} | FD {curves.final_fd_mean:.3f}"
                )
            curves_by_family[kind] = curves_by_shots
        save_curves_cache(cache_dir, meta, curves_by_family)
        print(f"[cache] Saved aggregate curves to {cache_dir.resolve()}")

    table_rows, latex_table = build_summary_tables(curves_by_family, families, shots_list)
    all_curves = [curves_by_family[kind][shots] for kind in families for shots in shots_list]
    y_limits = (
        interesting_y_limits(all_curves, ymax=args.ymax) if args.ymin is None else (float(args.ymin), float(args.ymax))
    )

    for kind in families:
        curves_by_shots = curves_by_family[kind]
        if len(shots_list) == 2:
            plot_budget_curves(
                out / f"fig2_budget_{kind}.{args.fmt}",
                curves_by_shots,
                shots_list,
                family_name=_family_label(kind),
                y_limits=y_limits,
            )
        else:
            for shots in shots_list:
                plot_budget_curves(
                    out / f"fig2_budget_{kind}_shots{shots}.{args.fmt}",
                    {shots: curves_by_shots[shots]},
                    [shots],
                    family_name=_family_label(kind),
                    y_limits=y_limits,
                )

    if len(families) > 1 and len(shots_list) == 2:
        plot_budget_family_grid(
            out / f"fig2_budget_family_grid.{args.fmt}",
            curves_by_family,
            families,
            shots_list,
            y_limits=y_limits,
        )

    write_csv(out / "table2_summary.csv", table_rows)
    write_table_latex(out / "table2_summary.tex", latex_table)

    summary_lines = [
        "Experiment 2 — Systematic Cost Advantage",
        f"families={','.join(families)} | shots_list={','.join(str(int(s)) for s in shots_list)}",
        f"n={args.n} | p_edge={args.p_edge} | graph_seed={args.graph_seed} | periodic_K={args.periodic_K}",
        f"budget_evals={args.budget:.1f} | seed0={args.seed0} | num_seeds={args.num_seeds}",
        "",
        "Matched-budget summary rows:",
    ]
    for row in table_rows:
        shots_raw = str(row["shots"]).strip().lower()
        if shots_raw in {"exact", "0", "0.0"}:
            shots_label = "exact"
        else:
            shots_label = str(int(float(row["shots"])))
        summary_lines.append(
            f"{row['family']} | shots={shots_label}: "
            f"CR-ID={float(row['ID_mean']):.4f}+/-{float(row['ID_stderr']):.4f} | "
            f"FD={float(row['FD_mean']):.4f}+/-{float(row['FD_stderr']):.4f}"
        )
    (out / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\nSaved to:", out.resolve())


if __name__ == "__main__":
    main()
