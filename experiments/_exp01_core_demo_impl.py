#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal Experiment 1 implementation.

This module renders the final paper-style Exp-1 plots for one family and can
also assemble the three-family spectrum-compare collage used in the publication
artifact set.
"""

import argparse
import math
from pathlib import Path
from typing import Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from paramham.experiment_defaults import (
    CANONICAL_SETUP,
    can_run_step,
    generate_er_family1d_instance,
    publication_output_dir,
    vqe_fd_value_step_cost,
    vqe_id_step_cost,
)
from paramham.maxcut import build_cut_mask
from paramham.maxcut import precompute_z as precompute_z_big_endian
from paramham.plotting import (
    COL_W,
    COLORS,
    FULL_W,
    H_COL,
    METHOD_CMAPS,
    _savefig,
    add_figure_legend,
    add_panel_legend,
    set_pub_style,
)
from paramham.seeds import to_uint_seed
from paramham.simulator import vqe_state
from paramham.spsa import spsa_minimize


def fig_size() -> Tuple[float, float]:
    return (COL_W, H_COL)


def _truncate_step_to_budget(evals: np.ndarray, y: np.ndarray, budget: float):
    evals = np.asarray(evals, float)
    y = np.asarray(y, float)
    budget = float(budget)
    m = evals <= budget
    ev = evals[m]
    yy = y[m]
    if ev.size == 0:
        return np.array([0.0, budget]), np.array([y[0], y[0]])
    if ev[-1] < budget:
        ev = np.append(ev, budget)
        yy = np.append(yy, yy[-1])
    return ev, yy


# ==============================================================================
# THE 5 ADJUSTED PLOTTING FUNCTIONS
# ==============================================================================


def plot_envelope_improved(path: Path, lam_grid, J_cl_star, hist_id, hist_fd):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    lam_grid = np.asarray(lam_grid, dtype=float)
    J_cl_star = np.asarray(J_cl_star, dtype=float)

    ax.plot(lam_grid, J_cl_star, color=COLORS["ENV"], lw=1.5, alpha=0.8, zorder=1)

    t_id = np.arange(len(hist_id["J"]))
    t_fd = np.arange(len(hist_fd["J"]))

    ax.scatter(
        hist_id["lam_pre"],
        hist_id["J"],
        s=15,
        c=t_id,
        cmap=METHOD_CMAPS["ID"],
        vmin=0,
        vmax=len(t_id) * 1.2,
        marker="o",
        edgecolors="black",
        linewidth=0.3,
        alpha=0.6,
        zorder=3,
    )
    ax.scatter(
        hist_fd["lam_pre"],
        hist_fd["J"],
        s=15,
        c=t_fd,
        cmap=METHOD_CMAPS["FD"],
        vmin=0,
        vmax=len(t_fd) * 1.2,
        marker="s",
        edgecolors="black",
        linewidth=0.3,
        alpha=0.6,
        zorder=2,
    )

    ax.set_xlabel(r"Control parameter $\lambda$")
    ax.set_ylabel(r"Value estimate $\hat F(\lambda)$")
    ax.set_xlim(float(lam_grid[0]), float(lam_grid[-1]))

    line_env = mlines.Line2D([], [], color=COLORS["ENV"], lw=1.5, label="Envelope")
    dot_id = mlines.Line2D([], [], color=COLORS["ID"], marker="o", ls="None", ms=5, label="VQE + ID")
    dot_fd = mlines.Line2D([], [], color=COLORS["FD"], marker="s", ls="None", ms=5, label="VQE + FD")
    add_panel_legend(
        ax,
        handles=[line_env, dot_id, dot_fd],
        placement="below",
        ncol=3,
        fontsize=7,
    )

    _savefig(fig, path)
    plt.close(fig)


def plot_efficiency_improved(path: Path, hist_id, hist_fd, J_cl_max=None, budget: float = None):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    if budget is None:
        budget = float(min(hist_id["evals_cum"][-1], hist_fd["evals_cum"][-1]))
    budget = float(budget)

    ev_id, y_id = _truncate_step_to_budget(hist_id["evals_cum"], hist_id["J_best"], budget)
    ev_fd, y_fd = _truncate_step_to_budget(hist_fd["evals_cum"], hist_fd["J_best"], budget)

    y_fd_interp = np.interp(ev_id, ev_fd, y_fd)
    ax.fill_between(
        ev_id,
        y_id,
        y_fd_interp,
        where=(y_id > y_fd_interp),
        color=COLORS["ID"],
        alpha=0.1,
        interpolate=True,
        label="Advantage Zone",
    )

    ax.plot(ev_id, y_id, color=COLORS["ID"], lw=1.8, label="VQE + ID")
    ax.plot(ev_fd, y_fd, color=COLORS["FD"], lw=1.8, ls="--", label=r"VQE + FD")

    if J_cl_max is not None:
        ax.axhline(J_cl_max, color=COLORS["REFERENCE"], lw=1.0, ls=":", label=r"Grid Max $J^*$")
        thresh = 0.99 * J_cl_max
        idx = np.argmax(y_id >= thresh)
        if y_id[idx] >= thresh:
            cx = ev_id[idx]
            ax.vlines(cx, 0, thresh, color=COLORS["ID"], lw=1.0, alpha=0.5, linestyles="-.")
            ax.text(cx, thresh * 0.85, f"99% @ {int(cx)}", color=COLORS["ID"], fontsize=7, ha="right", rotation=90)

    target_step = 20
    for name, h, col, mk in [("ID", hist_id, COLORS["ID"], "o"), ("FD", hist_fd, COLORS["FD"], "s")]:
        if len(h["evals_cum"]) > target_step:
            e, j = h["evals_cum"][target_step], h["J_best"][target_step]
            if e <= budget:
                ax.scatter(e, j, s=20, color="white", edgecolors=col, marker=mk, zorder=10, lw=0.8)
                ax.annotate(
                    f"t={target_step}",
                    (e, j),
                    xytext=(0, -15 if name == "FD" else 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color=col,
                )

    ax.set_xlabel("Cumulative Energy Evaluations")
    ax.set_ylabel(r"Best-so-far Value $\hat F$")
    ax.set_xlim(0.0, budget)
    add_panel_legend(ax, placement="below", ncol=2, fontsize=7)
    _savefig(fig, path)
    plt.close(fig)


def plot_xray_improved(path: Path, lams, all_J, J_cl_star, active_ids, switch_lams, switch_vals):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    K = all_J.shape[0]
    sample = np.linspace(0, K - 1, min(K, 320), dtype=int)
    ax.plot(lams, all_J[sample].T, color=COLORS["REFERENCE"], alpha=0.10, lw=0.42, rasterized=True, zorder=0)

    unique_ids = np.unique(active_ids)
    first_appearance = [(np.where(active_ids == uid)[0][0], uid) for uid in unique_ids]
    first_appearance.sort()
    sorted_uids = [u for _, u in first_appearance]
    curve_colors = [COLORS["ID"], "black"]

    for i, uid in enumerate(sorted_uids):
        col = curve_colors[i % 2]
        ax.plot(lams, all_J[uid], color=col, lw=1.2, ls="--", alpha=0.60, zorder=2, label="_nolegend_")

    changes = np.where(active_ids[1:] != active_ids[:-1])[0] + 1
    bounds = np.concatenate(([0], changes, [len(lams)]))

    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        uid = active_ids[s]
        try:
            c_idx = sorted_uids.index(uid)
            col = curve_colors[c_idx % 2]
        except ValueError:
            continue
        s_p = max(0, s)
        e_p = min(len(lams), e + 1)
        ax.plot(lams[s_p:e_p], all_J[uid, s_p:e_p], color=col, lw=2.2, alpha=1.0, zorder=4)

    if switch_lams.size > 0:
        ax.scatter(switch_lams, switch_vals, color="white", s=34, edgecolors=COLORS["ENV"], linewidths=0.9, zorder=10)

    ax.set_xlabel(r"Control parameter $\lambda$")
    ax.set_ylabel(r"Energy landscape $J(z;\lambda)$")

    y_max = np.max(all_J)
    y_env_min = np.min(J_cl_star)
    range_y = y_max - y_env_min
    ax.set_ylim(y_env_min - 0.4 * range_y, y_max + 0.05 * range_y)
    ax.set_xlim(lams[0], lams[-1])

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=COLORS["ENV"], lw=2, label="Active Branch"),
        Line2D([0], [0], color=COLORS["ENV"], lw=1, ls="--", alpha=0.5, label="Full Curve"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor=COLORS["ENV"], markersize=6, label="Switch Point"),
    ]
    add_panel_legend(
        ax,
        handles=legend_elements,
        placement="below",
        ncol=3,
        fontsize=7,
    )

    _savefig(fig, path)
    plt.close(fig)


def plot_cost_gap_improved(path: Path, hist_id, hist_fd, budget: float):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    t_id, t_fd = np.arange(1, hist_id["evals_cum"].size + 1), np.arange(1, hist_fd["evals_cum"].size + 1)
    e_id, e_fd = hist_id["evals_cum"], hist_fd["evals_cum"]

    slope_id = e_id[-1] / t_id[-1]
    slope_fd = e_fd[-1] / t_fd[-1]

    common_t = min(t_id[-1], t_fd[-1])
    ax.fill_between(
        t_id[:common_t],
        e_id[:common_t],
        e_fd[:common_t],
        color=COLORS["REFERENCE"],
        alpha=0.2,
        linewidth=0.0,
        label="Cost Overhead",
    )

    ax.plot(t_id, e_id, color=COLORS["ID"], lw=2.0, label=f"VQE + ID (Slope $\\approx$ {slope_id:.1f})")
    ax.plot(t_fd, e_fd, color=COLORS["FD"], lw=2.0, ls="--", label=f"VQE + FD (Slope $\\approx$ {slope_fd:.1f})")

    idx_id = np.searchsorted(e_id, budget)
    idx_fd = np.searchsorted(e_fd, budget)

    ax.axhline(budget, color=COLORS["ENV"], ls=":", lw=1.0)
    if idx_id < len(t_id):
        ax.vlines(t_id[idx_id], 0, budget, color=COLORS["ID"], lw=1.5, alpha=0.8)
        ax.scatter([t_id[idx_id]], [budget], color=COLORS["ID"], s=25, zorder=5)
    if idx_fd < len(t_fd):
        ax.vlines(t_fd[idx_fd], 0, budget, color=COLORS["FD"], lw=1.5, alpha=0.8)
        ax.scatter([t_fd[idx_fd]], [budget], color=COLORS["FD"], s=25, zorder=5)

    ax.set_xlabel(r"Outer iteration $t$")
    ax.set_ylabel("Cumulative Energy Evaluations")
    ax.set_xlim(0, max(t_id[-1], t_fd[-1]))
    ax.set_ylim(0, max(e_fd[-1], budget) * 1.1)
    add_panel_legend(ax, placement="below", ncol=2)
    _savefig(fig, path)
    plt.close(fig)


def plot_trajectory_improved(path: Path, hist_id, hist_fd, lam_true, lam_bounds):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
    t_id = np.arange(len(hist_id["lam_pre"]))
    t_fd = np.arange(len(hist_fd["lam_pre"]))

    ax.axhline(lam_true, color=COLORS["ENV"], lw=1.0, alpha=0.8, ls="-", label=r"Optimum $\lambda^*$")
    ax.plot(t_id, hist_id["lam_pre"], color=COLORS["ID"], lw=1.5, label="VQE + ID")
    ax.plot(t_fd, hist_fd["lam_pre"], color=COLORS["FD"], lw=1.5, ls="--", label="VQE + FD")
    ax.scatter([0], [hist_id["lam_pre"][0]], color=COLORS["ENV"], s=20, marker="x", zorder=5, label="Start")
    ax.scatter([t_id[-1]], [hist_id["lam_pre"][-1]], color=COLORS["ID"], s=20, edgecolors="white", lw=0.5, zorder=5)
    ax.scatter(
        [t_fd[-1]],
        [hist_fd["lam_pre"][-1]],
        color=COLORS["FD"],
        s=20,
        marker="s",
        edgecolors="white",
        lw=0.5,
        zorder=5,
    )

    ax.set_ylim(lam_bounds)
    ax.set_xlabel("Outer iteration $t$")
    ax.set_ylabel(r"Parameter $\lambda_t$")
    ax.set_xlim(0, max(t_id[-1], t_fd[-1]))
    add_panel_legend(ax, placement="below", ncol=2)
    _savefig(fig, path)
    plt.close(fig)


def _draw_xray_panel(ax, family_label: str, lams, all_J, J_cl_star, active_ids, switch_lams, switch_vals):
    K = all_J.shape[0]
    sample = np.linspace(0, K - 1, min(K, 320), dtype=int)
    ax.plot(lams, all_J[sample].T, color=COLORS["REFERENCE"], alpha=0.10, lw=0.42, rasterized=True, zorder=0)

    unique_ids = np.unique(active_ids)
    first_appearance = [(np.where(active_ids == uid)[0][0], uid) for uid in unique_ids]
    first_appearance.sort()
    sorted_uids = [u for _, u in first_appearance]
    curve_colors = [COLORS["ID"], "black"]

    for i, uid in enumerate(sorted_uids):
        col = curve_colors[i % 2]
        ax.plot(lams, all_J[uid], color=col, lw=1.0, ls="--", alpha=0.45, zorder=1)

    changes = np.where(active_ids[1:] != active_ids[:-1])[0] + 1
    bounds = np.concatenate(([0], changes, [len(lams)]))
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        uid = active_ids[s]
        try:
            c_idx = sorted_uids.index(uid)
            col = curve_colors[c_idx % 2]
        except ValueError:
            continue
        s_p = max(0, s)
        e_p = min(len(lams), e + 1)
        ax.plot(lams[s_p:e_p], all_J[uid, s_p:e_p], color=col, lw=1.8, alpha=1.0, zorder=3)

    if switch_lams.size > 0:
        ax.scatter(switch_lams, switch_vals, color="white", s=22, edgecolors=COLORS["ENV"], linewidths=0.7, zorder=6)

    y_max = np.max(all_J)
    y_env_min = np.min(J_cl_star)
    range_y = y_max - y_env_min
    ax.set_ylim(y_env_min - 0.38 * range_y, y_max + 0.05 * range_y)
    ax.set_xlim(lams[0], lams[-1])
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$J(z;\lambda)$")
    ax.text(0.5, 1.01, family_label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


def plot_spectrum_compare_collage(path: Path, panels):
    set_pub_style(grid=False)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, H_COL), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.26, top=0.92, wspace=0.28)

    for ax, panel in zip(axes, panels):
        _draw_xray_panel(
            ax,
            panel["label"],
            panel["lams"],
            panel["all_J"],
            panel["J_star"],
            panel["active"],
            panel["sw_l"],
            panel["sw_v"],
        )

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLORS["REFERENCE"], lw=1.0, alpha=0.5, label="Full spectrum"),
        Line2D([0], [0], color=COLORS["ID"], lw=1.8, label="Active branch"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor=COLORS["ENV"], markersize=5, label="Switch points"),
    ]
    add_figure_legend(fig, handles, [h.get_label() for h in handles], ncol=3)
    _savefig(fig, path)
    plt.close(fig)


# ==============================================================================
# Simulation Logic (uses shared primitives)
# ==============================================================================


def zexp_edges(probs: np.ndarray, edges, Z: np.ndarray) -> np.ndarray:
    z = np.empty(len(edges), dtype=float)
    zz = Z.astype(float)
    with np.errstate(all="ignore"):
        for e, (i, j) in enumerate(edges):
            z[e] = float(probs @ (zz[i] * zz[j]))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(z, -1.0, 1.0)


def vqe_expect(n, edges, Z, w, params, L):
    psi = vqe_state(n, params, L)
    probs = (psi.conj() * psi).real.astype(float)
    s = np.sum(probs)
    probs = probs / s if (np.isfinite(s) and s > 0) else np.full_like(probs, 1.0 / probs.size)
    z = zexp_edges(probs, edges, Z)
    J = float(0.5 * (1.0 - z) @ w)
    return (0.0 if not np.isfinite(J) else J), psi, z


def vqe_energy(n, edges, Z, w, params, L):
    return -vqe_expect(n, edges, Z, w, params, L)[0]


def envelope_spectrum(fam, cut_mask, grid_points):
    lams = np.linspace(fam.lam_min, fam.lam_max, grid_points)
    all_J = np.empty((cut_mask.shape[0], grid_points), np.float32)
    J_star = np.empty(grid_points, np.float32)
    active = np.empty(grid_points, np.int32)
    for t, lam in enumerate(lams):
        w = fam.w(lam)
        vals = np.nan_to_num(cut_mask @ w, nan=-1e30)
        all_J[:, t] = vals
        idx = int(np.argmax(vals))
        active[t], J_star[t] = idx, vals[idx]
    sw = np.where(active[1:] != active[:-1])[0] + 1
    return lams, all_J, J_star, active, lams[sw], J_star[sw]


def readout_metrics(rng, psi, cut_vals, shots):
    probs = (psi.conj() * psi).real.astype(float)
    s = np.sum(probs)
    probs = probs / s if (np.isfinite(s) and s > 0) else np.full_like(probs, 1.0 / probs.size)
    idx = rng.choice(np.arange(probs.size), size=shots, p=probs)
    return float(np.max(cut_vals[idx])), float(cut_vals[np.argmax(np.bincount(idx, minlength=probs.size))])


def run_outer(
    n,
    edges,
    Z,
    fam,
    cut_mask,
    lam0,
    outer,
    inner,
    L,
    seed,
    eta0,
    eta_pow,
    step_clip,
    mode,
    readout_shots,
    c_frac,
    budget_evals,
):
    rng_read = np.random.default_rng(to_uint_seed(seed + 1234567))
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))
    params = np.zeros(2 * n * L, float)
    bounds = [(-math.pi, math.pi)] * params.size
    hist = {k: [] for k in ["lam_pre", "lam", "J", "J_best", "evals_cum", "J_best_cut", "J_mode_cut"]}
    evals, best = 0.0, -1e18
    c = c_frac * (lam_max - lam_min)

    step_cost = vqe_id_step_cost(inner) if mode == "ID" else vqe_fd_value_step_cost(inner)

    for t in range(1, outer + 1):
        if not can_run_step(evals_used=evals, budget_evals=budget_evals, step_cost=step_cost):
            break
        hist["lam_pre"].append(lam)
        w = fam.w(lam)

        def Efun(p):
            return vqe_energy(n, edges, Z, w, p, L)

        params, _, ev = spsa_minimize(Efun, params, bounds, inner, seed + 1000 * t)
        evals += ev
        J, psi, zexp = vqe_expect(n, edges, Z, w, params, L)
        evals += 1.0
        best = max(best, J)

        jb, jm = (
            readout_metrics(rng_read, psi, (cut_mask @ w).astype(float), readout_shots)
            if readout_shots > 0
            else (float("nan"), float("nan"))
        )

        if mode == "ID":
            g = float(fam.dw_dlam(lam) @ (0.5 * (1.0 - zexp)))
        elif mode == "FD_VALUE":
            lp, lm = np.clip(lam + c, lam_min, lam_max), np.clip(lam - c, lam_min, lam_max)
            wp = fam.w(lp)
            pp, _, evp = spsa_minimize(
                lambda p: vqe_energy(n, edges, Z, wp, p, L), params, bounds, inner, seed + 1000 * t + 17
            )
            evals += evp + 1.0
            Jp = vqe_expect(n, edges, Z, wp, pp, L)[0]
            wm = fam.w(lm)
            pm, _, evm = spsa_minimize(
                lambda p: vqe_energy(n, edges, Z, wm, p, L), params, bounds, inner, seed + 1000 * t + 29
            )
            evals += evm + 1.0
            Jm = vqe_expect(n, edges, Z, wm, pm, L)[0]
            best = max(best, Jp, Jm)
            g = (Jp - Jm) / (2.0 * c) if c > 0 else 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        step = float(np.clip((eta0 / (t**eta_pow)) * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))
        hist["lam"].append(lam)
        hist["J"].append(J)
        hist["J_best"].append(best)
        hist["evals_cum"].append(evals)
        hist["J_best_cut"].append(jb)
        hist["J_mode_cut"].append(jm)

    for k in hist:
        hist[k] = np.array(hist[k], float)
    return hist


def _run_single_family(args, kind: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    edges, fam = generate_er_family1d_instance(
        args.n,
        args.p_edge,
        kind,
        (args.lam_min, args.lam_max),
        graph_seed=args.graph_seed,
        periodic_K=args.periodic_K,
        safety_bounds=False,
    )
    if not edges or fam is None:
        raise RuntimeError("Graph has 0 edges; increase p_edge or change graph_seed.")
    Z = precompute_z_big_endian(args.n)
    mask = build_cut_mask(edges, Z)

    print(f"Precomputing envelope for {kind}...")
    lams, all_J, J_star, active, sw_l, sw_v = envelope_spectrum(fam, mask, args.grid)
    lam_true = float(lams[np.argmax(J_star)])
    J_max = float(np.max(J_star))

    print(f"Running ID vs FD for {kind}...")
    h_id = run_outer(
        args.n,
        edges,
        Z,
        fam,
        mask,
        args.lam0,
        args.outer,
        args.inner,
        args.L,
        args.seed,
        args.eta0,
        args.eta_pow,
        args.step_clip,
        "ID",
        args.readout_shots,
        args.c_frac,
        args.budget_evals,
    )
    h_fd = run_outer(
        args.n,
        edges,
        Z,
        fam,
        mask,
        args.lam0,
        args.outer,
        args.inner,
        args.L,
        args.seed,
        args.eta0,
        args.eta_pow,
        args.step_clip,
        "FD_VALUE",
        args.readout_shots,
        args.c_frac,
        args.budget_evals,
    )

    B = float(args.budget_evals)
    suf = f"{kind}_n{args.n}_seed{args.seed}"
    plot_envelope_improved(out / f"1_envelope_zoom_{suf}.{args.fmt}", lams, J_star, h_id, h_fd)
    plot_xray_improved(out / f"2_xray_segments_bw_{suf}.{args.fmt}", lams, all_J, J_star, active, sw_l, sw_v)
    plot_efficiency_improved(out / f"3_efficiency_zone_{suf}.{args.fmt}", h_id, h_fd, J_cl_max=J_max, budget=B)
    plot_cost_gap_improved(out / f"4_cost_gap_waste_{suf}.{args.fmt}", h_id, h_fd, budget=B)
    plot_trajectory_improved(
        out / f"5_trajectory_target_{suf}.{args.fmt}", h_id, h_fd, lam_true, (args.lam_min, args.lam_max)
    )

    summary_lines = [
        "Experiment 1 — Core ID vs FD Demo",
        f"family={kind} | n={args.n} | seed={args.seed} | p_edge={args.p_edge} | graph_seed={args.graph_seed}",
        f"periodic_K={args.periodic_K} | budget_evals={args.budget_evals} | lam0={args.lam0}",
        f"final best ID={float(h_id['J_best'][-1]):.4f} | final best FD={float(h_fd['J_best'][-1]):.4f}",
        f"final evals ID={float(h_id['evals_cum'][-1]):.1f} | final evals FD={float(h_fd['evals_cum'][-1]):.1f}",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "label": kind.title(),
        "lams": lams,
        "all_J": all_J,
        "J_star": J_star,
        "active": active,
        "sw_l": sw_l,
        "sw_v": sw_v,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--fmt", type=str, default="pdf")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--kind", type=str, default=CANONICAL_SETUP.family, choices=["linear", "quadratic", "periodic"])
    p.add_argument(
        "--suite",
        action="store_true",
        help="Render the full paper suite for linear, quadratic, and periodic plus the spectrum-compare collage.",
    )
    p.add_argument("--periodic_K", type=int, default=CANONICAL_SETUP.periodic_K)
    p.add_argument("--n", type=int, default=CANONICAL_SETUP.n)
    p.add_argument("--p_edge", type=float, default=CANONICAL_SETUP.p_edge)
    p.add_argument("--graph_seed", type=int, default=CANONICAL_SETUP.graph_seed)
    p.add_argument("--lam_min", type=float, default=CANONICAL_SETUP.lam_min)
    p.add_argument("--lam_max", type=float, default=CANONICAL_SETUP.lam_max)
    p.add_argument("--lam0", type=float, default=CANONICAL_SETUP.lam0)
    p.add_argument("--grid", type=int, default=401)
    p.add_argument("--outer", type=int, default=100)
    p.add_argument("--inner", type=int, default=30)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)
    p.add_argument("--readout_shots", type=int, default=0)
    p.add_argument("--budget_evals", type=float, default=CANONICAL_SETUP.budget_evals)
    args = p.parse_args()

    if args.suite:
        root = Path(args.out) if args.out is not None else publication_output_dir("exp01")
        panels = []
        for kind in ["linear", "quadratic", "periodic"]:
            panels.append(_run_single_family(args, kind, root / kind))
        compare_dir = root / "spectrum_compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        compare_path = (
            compare_dir / f"exp01_spectrum_compare_linear_quadratic_periodic_n{args.n}_seed{args.seed}.{args.fmt}"
        )
        plot_spectrum_compare_collage(compare_path, panels)
        compare_summary = [
            "Experiment 1 — Spectrum compare",
            f"families=linear,quadratic,periodic | n={args.n} | seed={args.seed}",
            f"graph_seed={args.graph_seed} | p_edge={args.p_edge} | periodic_K={args.periodic_K}",
            compare_path.name,
        ]
        (compare_dir / "SUMMARY.txt").write_text("\n".join(compare_summary) + "\n", encoding="utf-8")
        print(f"Done! Results in {root.resolve()}")
        return

    out = Path(args.out) if args.out is not None else publication_output_dir("exp01", args.kind)
    _run_single_family(args, args.kind, out)
    print(f"Done! Results in {out.resolve()}")


if __name__ == "__main__":
    main()
