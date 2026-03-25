#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp05_inner_budget_ablation.py
==============================

Experiment 5: Inner-budget ablation (iters x restarts)
--------------------------------------------------------------------

Reviewer question:
  "Is the CR-ImpDiff advantage just an artifact of inner-solver convergence?"

What we do:
  Sweep the inner VQE SPSA budget across a grid:
    - inner_iters  (SPSA iterations per restart)
    - restarts     (number of independent SPSA runs; best-of-restarts chosen)
  For each grid cell, run paired outer optimization with:
    - ID        : correlator-reuse implicit differentiation (CR-ImpDiff)
    - FD_VALUE  : black-box finite differences on the VALUE function F(lambda)
                 (requires re-solving the inner problem at lambda+-c per outer step)

Fairness / cost model:
  - We compare at a FIXED total evaluation budget B (energy evaluations).
  - Best-so-far for FD_VALUE is defined over all value queries it actually makes
    (center and perturbed), so we do NOT discard evaluated candidates.

Metrics:
  - AUC_B of best-so-far curve y(b)/J* over b in [0,B], with step-function integral
  - AUC_gain = AUC_ID - AUC_FD_VALUE
  - win_rate = fraction of instances with AUC_gain > 0

Outputs (paper-ready):
  - fig5_inner_budget_heatmap.<fmt>
      2 panels: (a) mean dAUC heatmap, (b) win-rate heatmap
  - runs5_inner_budget_metrics.csv
      per-instance per-cell metrics
  - table5_summary.csv / .tex
      per-cell mean +- s.e.m. summary
  - SUMMARY.txt
      compact text summary (copyable into supplement notes)

Recommended minimal run (not too heavy):
  python exp05_inner_budget_ablation.py \\
    --out output/exp05 \\
    --kind periodic --periodic_K 6 \\
    --inner_iters_list 14,28,42 \\
    --restarts_list 1,2,4 \\
    --num_seeds 8 \\
    --budget_evals 5100 \\
    --shots 0 \\
    --fmt pdf

Notes:
  - n=12 keeps the classical diagnostic J* feasible (2^n = 4096).
  - shots=0 means exact expectation evaluation (statevector).
  - shots>0 simulates shot-noisy energy evaluation with that many bitstring samples
    per energy evaluation.

"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plot_style import FIG_H, FIG_W, apply_thesis_axes_style, dual_panel_size, save_figure, use_thesis_style

from paramham.experiment_defaults import (
    CANONICAL_SETUP,
    can_run_step,
    generate_er_family1d_instance,
    publication_cache_dir,
    publication_output_dir,
    restarted_fd_step_cost,
    restarted_id_step_cost,
)
from paramham.families import Family1D
from paramham.io import parse_int_list
from paramham.maxcut import build_cut_mask, build_ZZ_edges
from paramham.maxcut import precompute_z as precompute_z_big_endian
from paramham.seeds import to_uint_seed
from paramham.simulator import vqe_state
from paramham.spsa import spsa_minimize

# ==============================================================================
# 1) Experiment-specific plotting
# ==============================================================================


def _set_exp05_plot_style(grid: bool = False):
    use_thesis_style()
    plt.rcParams["axes.grid"] = bool(grid)


def _cache_default_dir(out: Path) -> Path:
    return publication_cache_dir("exp05")


def _cache_meta(args, inner_iters_list: List[int], restarts_list: List[int]) -> dict:
    return {
        "seed0": int(args.seed0),
        "num_seeds": int(args.num_seeds),
        "n": int(args.n),
        "p_edge": float(args.p_edge),
        "graph_seed": int(args.graph_seed),
        "kind": str(args.kind),
        "periodic_K": int(args.periodic_K),
        "lam_min": float(args.lam_min),
        "lam_max": float(args.lam_max),
        "lam0": float(args.lam0),
        "L": int(args.L),
        "shots": int(args.shots),
        "inner_iters_list": [int(x) for x in inner_iters_list],
        "restarts_list": [int(x) for x in restarts_list],
        "outer_max": int(args.outer_max),
        "eta0": float(args.eta0),
        "eta_pow": float(args.eta_pow),
        "step_clip": float(args.step_clip),
        "c_frac": float(args.c_frac),
        "budget_evals": float(args.budget_evals),
        "Jstar_grid": int(args.Jstar_grid),
    }


def save_exp05_cache(cache_dir: Path, meta: dict, run_rows: List[Dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (cache_dir / "run_rows.json").write_text(json.dumps(run_rows, indent=2), encoding="utf-8")


def load_exp05_cache(cache_dir: Path, meta_expected: dict):
    meta_path = cache_dir / "cache_meta.json"
    rows_path = cache_dir / "run_rows.json"
    if not meta_path.exists() or not rows_path.exists():
        return None
    try:
        meta_found = json.loads(meta_path.read_text(encoding="utf-8"))
        run_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta_found != meta_expected:
        return None
    return run_rows


def plot_inner_budget_heatmap(
    path: Path,
    inner_iters_list: List[int],
    restarts_list: List[int],
    auc_gain_mean: np.ndarray,
    win_rate: np.ndarray,
):
    """
    Paper-ready: 2-panel figure
      (a) mean dAUC heatmap, (b) win-rate heatmap
    No titles; includes axis labels and colorbars.
    Fix: panel (b) no longer unreadable (auto text color + better colormap).
    """
    _set_exp05_plot_style(grid=False)
    fig, axs = plt.subplots(1, 2, figsize=dual_panel_size(), constrained_layout=True)

    im0 = _draw_auc_gain_heatmap(axs[0], inner_iters_list, restarts_list, auc_gain_mean)
    cb0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    cb0.set_label(r"$\Delta \mathrm{AUC}_B$  (ID $-$ BD)")

    im1 = _draw_win_rate_heatmap(axs[1], inner_iters_list, restarts_list, win_rate)
    cb1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cb1.set_label("Win rate (dAUC>0)")

    save_figure(fig, path)
    plt.close(fig)


def _draw_auc_gain_heatmap(ax, inner_iters_list: List[int], restarts_list: List[int], auc_gain_mean: np.ndarray):
    vmax = float(np.max(np.abs(auc_gain_mean))) if np.isfinite(auc_gain_mean).any() else 1.0
    vmax = max(vmax, 1e-6)
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(
        auc_gain_mean,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=-vmax,
        vmax=+vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("Inner SPSA iterations")
    ax.set_ylabel("Restarts")
    ax.set_xticks(np.arange(len(inner_iters_list)))
    ax.set_xticklabels([str(x) for x in inner_iters_list])
    ax.set_yticks(np.arange(len(restarts_list)))
    ax.set_yticklabels([str(r) for r in restarts_list])
    apply_thesis_axes_style(ax, grid=False)

    norm = mpl.colors.Normalize(vmin=-vmax, vmax=+vmax)
    for iy in range(len(restarts_list)):
        for ix in range(len(inner_iters_list)):
            val = float(auc_gain_mean[iy, ix])
            if np.isfinite(val):
                rgba = cmap(norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "black" if lum > 0.6 else "white"
                ax.text(ix, iy, f"{val:+.3f}", ha="center", va="center", fontsize=7, color=txt_color)
    return im


def _draw_win_rate_heatmap(ax, inner_iters_list: List[int], restarts_list: List[int], win_rate: np.ndarray):
    cmap = plt.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(
        win_rate,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("Inner SPSA iterations")
    ax.set_ylabel("Restarts")
    ax.set_xticks(np.arange(len(inner_iters_list)))
    ax.set_xticklabels([str(x) for x in inner_iters_list])
    ax.set_yticks(np.arange(len(restarts_list)))
    ax.set_yticklabels([str(r) for r in restarts_list])
    apply_thesis_axes_style(ax, grid=False)

    for iy in range(len(restarts_list)):
        for ix in range(len(inner_iters_list)):
            val = float(win_rate[iy, ix])
            if np.isfinite(val):
                rgba = cmap(norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "black" if lum > 0.6 else "white"
                ax.text(ix, iy, f"{100 * val:.0f}%", ha="center", va="center", fontsize=7, color=txt_color)
    return im


def plot_auc_gain_heatmap(path: Path, inner_iters_list: List[int], restarts_list: List[int], auc_gain_mean: np.ndarray):
    _set_exp05_plot_style(grid=False)
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), constrained_layout=True)
    im = _draw_auc_gain_heatmap(ax, inner_iters_list, restarts_list, auc_gain_mean)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta \mathrm{AUC}_B$  (ID $-$ BD)")
    save_figure(fig, path)
    plt.close(fig)


def plot_win_rate_heatmap(path: Path, inner_iters_list: List[int], restarts_list: List[int], win_rate: np.ndarray):
    _set_exp05_plot_style(grid=False)
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), constrained_layout=True)
    im = _draw_win_rate_heatmap(ax, inner_iters_list, restarts_list, win_rate)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Win rate (dAUC>0)")
    save_figure(fig, path)
    plt.close(fig)


# ==============================================================================
# 2) VQE expectation (experiment-specific: shot-noise path + ZZ_edges arg)
# ==============================================================================


def vqe_expect(
    n: int,
    params: np.ndarray,
    L: int,
    w: np.ndarray,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    shots: int,
    rng: np.random.Generator,
):
    """
    Returns:
      J (float), p_cut (m,), psi (statevector)
    If shots>0, p_cut is estimated from bitstring samples.
    """
    psi = vqe_state(n, params, L)
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    np.nan_to_num(probs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if shots > 0:
        # sample computational basis indices
        idx = rng.choice(np.arange(probs.size), size=shots, replace=True, p=probs)
        # cut_mask[idx] -> (shots, m), average -> p_cut
        with np.errstate(all="ignore"):
            p_cut = np.mean(cut_mask[idx, :], axis=0).astype(np.float64)
        np.nan_to_num(p_cut, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # exact correlators: zexp = ZZ @ probs
        with np.errstate(all="ignore"):
            zexp = (ZZ_edges.astype(np.float64) @ probs).astype(np.float64)
        np.nan_to_num(zexp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        zexp = np.clip(zexp, -1.0, 1.0)
        p_cut = 0.5 * (1.0 - zexp)

    with np.errstate(all="ignore"):
        J = float(np.dot(p_cut, w))
    if not np.isfinite(J):
        J = 0.0

    return J, p_cut, psi


def vqe_energy(
    n: int,
    params: np.ndarray,
    L: int,
    w: np.ndarray,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> float:
    J, _, _ = vqe_expect(n, params, L, w, ZZ_edges, cut_mask, shots, rng)
    return -J


# ==============================================================================
# 3) Inner solver (experiment-specific restart logic)
# ==============================================================================


def inner_solve(
    n: int,
    L: int,
    w: np.ndarray,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    init_params: np.ndarray,
    inner_iters: int,
    restarts: int,
    seed_base: int,
    shots: int,
):
    """
    Run restarts SPSA inner solves and pick best.
    Returns:
      best_params, best_J, best_p_cut, evals_used
    """
    D = init_params.size
    bounds = [(-math.pi, math.pi)] * D

    best_params = init_params.copy()
    best_E = float("inf")
    evals_total = 0

    for r in range(restarts):
        # restart init
        if r == 0:
            p0 = init_params.copy()
        else:
            rrng = np.random.default_rng(to_uint_seed(seed_base + 999 * (r + 1)))
            p0 = rrng.uniform(-math.pi, math.pi, size=D).astype(float)

        # shot RNG for this inner run (only used if shots>0)
        eval_rng = np.random.default_rng(to_uint_seed(seed_base + 1337 * (r + 1)))

        def Efun(pvec):
            return vqe_energy(n, pvec, L, w, ZZ_edges, cut_mask, shots, eval_rng)

        p_star, E_star, ev = spsa_minimize(Efun, p0, bounds, iters=inner_iters, seed=seed_base + 17 * (r + 1))
        evals_total += ev

        if np.isfinite(E_star) and E_star < best_E:
            best_E = E_star
            best_params = p_star.copy()

    # evaluate final best to obtain J and p_cut (counts as ONE energy evaluation)
    eval_rng_final = np.random.default_rng(to_uint_seed(seed_base + 424242))
    J, p_cut, _ = vqe_expect(n, best_params, L, w, ZZ_edges, cut_mask, shots, eval_rng_final)
    evals_total += 1

    return best_params, float(J), p_cut.astype(np.float64), int(evals_total)


# ==============================================================================
# 4) Classical diagnostic scale J* and outer loops with budget
# ==============================================================================


def classical_Jstar_max(fam: Family1D, cut_mask: np.ndarray, grid_points: int) -> float:
    """
    J* = max_{lambda in grid} max_z J(z;lambda)
    (Used only for normalization / diagnostic.)
    """
    lams = np.linspace(fam.lam_min, fam.lam_max, int(grid_points))
    J_star = -1e30
    for lam in lams:
        w = fam.w(float(lam)).astype(np.float64)
        with np.errstate(all="ignore"):
            vals = cut_mask @ w
        vals = np.nan_to_num(vals, nan=-1e30, posinf=-1e30, neginf=-1e30)
        J_star = max(J_star, float(np.max(vals)))
    if not np.isfinite(J_star):
        J_star = 1.0
    return float(J_star)


def run_outer_budget(
    mode: str,
    n: int,
    edges,
    fam: Family1D,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    lam0: float,
    outer_max: int,
    inner_iters: int,
    restarts: int,
    L: int,
    seed: int,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    c_frac: float,
    shots: int,
    budget_evals: float,
    Jstar: float,
) -> Dict[str, np.ndarray]:
    """
    Runs outer optimization until budget or outer_max.
    Returns a trace of (evals, best_norm) at each value-query completion.
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L
    params = np.zeros(D, float)

    evals = 0.0
    best = -1e30

    events_evals = []
    events_best = []

    c = float(c_frac * (lam_max - lam_min))

    step_cost = (
        restarted_id_step_cost(inner_iters, restarts) if mode == "ID" else restarted_fd_step_cost(inner_iters, restarts)
    )

    for t in range(1, int(outer_max) + 1):
        if not can_run_step(evals_used=evals, budget_evals=budget_evals, step_cost=step_cost):
            break
        # center value query F(lam) via inner solve
        w = fam.w(lam)

        params, J, p_cut, ev_in = inner_solve(
            n=n,
            L=L,
            w=w,
            ZZ_edges=ZZ_edges,
            cut_mask=cut_mask,
            init_params=params,
            inner_iters=inner_iters,
            restarts=restarts,
            seed_base=seed + 100000 * t + 111,
            shots=shots,
        )
        evals += float(ev_in)
        best = max(best, float(J))
        events_evals.append(evals)
        events_best.append(best / Jstar)

        # outer signal
        if mode == "ID":
            g = float(np.dot(fam.dw_dlam(lam), p_cut))
        elif mode == "FD_VALUE":
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))

            # +c value query
            w_p = fam.w(lp)
            p_p, Jp, _, evp = inner_solve(
                n=n,
                L=L,
                w=w_p,
                ZZ_edges=ZZ_edges,
                cut_mask=cut_mask,
                init_params=params,
                inner_iters=inner_iters,
                restarts=restarts,
                seed_base=seed + 100000 * t + 777,
                shots=shots,
            )
            evals += float(evp)
            best = max(best, float(Jp))
            events_evals.append(evals)
            events_best.append(best / Jstar)

            # -c value query
            w_m = fam.w(lm)
            p_m, Jm, _, evm = inner_solve(
                n=n,
                L=L,
                w=w_m,
                ZZ_edges=ZZ_edges,
                cut_mask=cut_mask,
                init_params=params,
                inner_iters=inner_iters,
                restarts=restarts,
                seed_base=seed + 100000 * t + 999,
                shots=shots,
            )
            evals += float(evm)
            best = max(best, float(Jm))
            events_evals.append(evals)
            events_best.append(best / Jstar)

            # gradient
            g = (float(Jp) - float(Jm)) / (2.0 * c) if c > 0 else 0.0

            # optionally: warm-start next center at the better perturbed params
            # (keeps baseline strong; but also increases coupling.)
            # We'll keep params as the last center solution for simplicity.
            _ = p_p, p_m
        else:
            raise ValueError("mode must be ID or FD_VALUE")

        # outer update
        eta = float(eta0 / (t**eta_pow))
        step = float(eta * g)
        step = float(np.clip(step, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

    return {
        "evals": np.asarray(events_evals, dtype=float),
        "best_norm": np.asarray(events_best, dtype=float),
        "evals_end": np.array([evals], dtype=float),
    }


def auc_step(evals: np.ndarray, values: np.ndarray, budget: float) -> float:
    """
    Step-function AUC on [0, budget].
    Assumes best-so-far is held constant between recorded events.
    """
    evals = np.asarray(evals, float)
    values = np.asarray(values, float)
    if evals.size == 0:
        return 0.0

    # prepend at 0 with first value
    if evals[0] > 0.0:
        evals = np.insert(evals, 0, 0.0)
        values = np.insert(values, 0, float(values[0]))

    # truncate to <= budget and append exactly budget
    m = evals <= budget
    evals = evals[m]
    values = values[m]
    if evals.size == 0:
        # nothing before budget -> constant at first value
        return float(values[0])

    if evals[-1] < budget:
        evals = np.append(evals, budget)
        values = np.append(values, values[-1])

    # step integral: sum (dx * y_i)
    dx = np.diff(evals)
    area = float(np.sum(dx * values[:-1]))
    return area / float(budget)


# ==============================================================================
# 5) I/O helpers: CSV + LaTeX table
# ==============================================================================


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_table_tex(path: Path, rows: List[Dict], caption: str, label: str):
    """
    Minimal booktabs LaTeX table:
      restarts, inner_iters, mean_auc_gain +- sem, win_rate
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{ccc c}\n")
        f.write("\\toprule\n")
        f.write("Restarts & Inner iters & $\\Delta\\mathrm{AUC}_B$ (mean $\\pm$ s.e.m.) & Win rate \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r['restarts']} & {r['inner_iters']} & "
                f"{r['auc_gain_mean']:+.4f} $\\pm$ {r['auc_gain_sem']:.4f} & "
                f"{100.0 * r['win_rate']:.0f}\\% \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


# ==============================================================================
# 6) Main
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--render_only", action="store_true")

    # instance / family
    p.add_argument("--seed0", type=int, default=7)
    p.add_argument("--num_seeds", type=int, default=7)
    p.add_argument("--n", type=int, default=CANONICAL_SETUP.n)
    p.add_argument("--p_edge", type=float, default=CANONICAL_SETUP.p_edge)
    p.add_argument("--graph_seed", type=int, default=CANONICAL_SETUP.graph_seed)
    p.add_argument("--kind", type=str, default=CANONICAL_SETUP.family, choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=CANONICAL_SETUP.periodic_K)
    p.add_argument("--lam_min", type=float, default=CANONICAL_SETUP.lam_min)
    p.add_argument("--lam_max", type=float, default=CANONICAL_SETUP.lam_max)
    p.add_argument("--lam0", type=float, default=CANONICAL_SETUP.lam0)

    # VQE / inner
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--shots", type=int, default=0, help="shots per energy evaluation (0 = exact)")

    # ablation grid
    p.add_argument("--inner_iters_list", type=str, default="14,28,42")
    p.add_argument("--restarts_list", type=str, default="1,2,4")

    # outer
    p.add_argument("--outer_max", type=int, default=200)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)

    # budget + J* grid
    p.add_argument("--budget_evals", type=float, default=CANONICAL_SETUP.budget_evals)
    p.add_argument("--Jstar_grid", type=int, default=401)

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out) if a.out is not None else publication_output_dir("exp05")
    out.mkdir(parents=True, exist_ok=True)

    inner_iters_list = parse_int_list(a.inner_iters_list)
    restarts_list = parse_int_list(a.restarts_list)
    if not inner_iters_list or not restarts_list:
        raise ValueError("inner_iters_list and restarts_list must be non-empty.")

    cache_dir = Path(a.cache_dir) if a.cache_dir is not None else _cache_default_dir(out)
    meta = _cache_meta(a, inner_iters_list, restarts_list)

    # collect per-run rows
    run_rows = None if a.recompute else load_exp05_cache(cache_dir, meta)
    if run_rows is not None:
        print(f"[cache] Loaded ablation rows from {cache_dir.resolve()}")
    elif a.render_only:
        raise SystemExit(f"No matching cache found in {cache_dir}")
    else:
        run_rows = []
        Rn = len(restarts_list)
        In = len(inner_iters_list)
        Sn = int(a.num_seeds)
        auc_gain = np.full((Rn, In, Sn), np.nan, dtype=float)
        win = np.zeros((Rn, In, Sn), dtype=float)

        for s_idx in range(Sn):
            seed = int(a.seed0 + s_idx)
            edges, fam = generate_er_family1d_instance(
                a.n,
                a.p_edge,
                a.kind,
                (a.lam_min, a.lam_max),
                graph_seed=a.graph_seed,
                periodic_K=a.periodic_K,
                instance_id=seed,
                safety_bounds=False,
            )
            if not edges or fam is None:
                raise RuntimeError("Graph has 0 edges; increase p_edge or change graph_seed.")

            Z = precompute_z_big_endian(a.n)
            cut_mask = build_cut_mask(edges, Z)
            ZZ_edges = build_ZZ_edges(edges, Z)
            Jstar = classical_Jstar_max(fam, cut_mask, a.Jstar_grid)
            if Jstar <= 0 or not np.isfinite(Jstar):
                Jstar = 1.0

            for iy, restarts in enumerate(restarts_list):
                for ix, inner_iters in enumerate(inner_iters_list):
                    hist_id = run_outer_budget(
                        mode="ID",
                        n=a.n,
                        edges=edges,
                        fam=fam,
                        ZZ_edges=ZZ_edges,
                        cut_mask=cut_mask,
                        lam0=a.lam0,
                        outer_max=a.outer_max,
                        inner_iters=inner_iters,
                        restarts=restarts,
                        L=a.L,
                        seed=seed,
                        eta0=a.eta0,
                        eta_pow=a.eta_pow,
                        step_clip=a.step_clip,
                        c_frac=a.c_frac,
                        shots=a.shots,
                        budget_evals=a.budget_evals,
                        Jstar=Jstar,
                    )

                    hist_fd = run_outer_budget(
                        mode="FD_VALUE",
                        n=a.n,
                        edges=edges,
                        fam=fam,
                        ZZ_edges=ZZ_edges,
                        cut_mask=cut_mask,
                        lam0=a.lam0,
                        outer_max=a.outer_max,
                        inner_iters=inner_iters,
                        restarts=restarts,
                        L=a.L,
                        seed=seed,
                        eta0=a.eta0,
                        eta_pow=a.eta_pow,
                        step_clip=a.step_clip,
                        c_frac=a.c_frac,
                        shots=a.shots,
                        budget_evals=a.budget_evals,
                        Jstar=Jstar,
                    )

                    auc_id = auc_step(hist_id["evals"], hist_id["best_norm"], a.budget_evals)
                    auc_fd = auc_step(hist_fd["evals"], hist_fd["best_norm"], a.budget_evals)
                    gain = float(auc_id - auc_fd)

                    auc_gain[iy, ix, s_idx] = gain
                    win[iy, ix, s_idx] = 1.0 if gain > 0 else 0.0

                    run_rows.append(
                        {
                            "seed": seed,
                            "kind": a.kind,
                            "periodic_K": a.periodic_K,
                            "graph_seed": a.graph_seed,
                            "n": a.n,
                            "p_edge": a.p_edge,
                            "shots": a.shots,
                            "budget_evals": a.budget_evals,
                            "inner_iters": inner_iters,
                            "restarts": restarts,
                            "Jstar": Jstar,
                            "auc_id": auc_id,
                            "auc_fd": auc_fd,
                            "auc_gain": gain,
                            "evals_end_id": float(hist_id["evals_end"][0]),
                            "evals_end_fd": float(hist_fd["evals_end"][0]),
                        }
                    )

                    print(
                        f"[seed={seed:3d}] iters={inner_iters:3d} restarts={restarts:2d} "
                        f"AUC_ID={auc_id:.4f} AUC_FD={auc_fd:.4f} gain={gain:+.4f}"
                    )

        save_exp05_cache(cache_dir, meta, run_rows)
        print(f"[cache] Saved ablation rows to {cache_dir.resolve()}")

    # save per-run CSV
    runs_csv = out / "runs5_inner_budget_metrics.csv"
    write_csv(runs_csv, run_rows, fieldnames=list(run_rows[0].keys()) if run_rows else [])

    # aggregate per-cell
    Rn = len(restarts_list)
    In = len(inner_iters_list)
    Sn = int(a.num_seeds)
    summary_rows = []
    auc_gain_mean = np.full((Rn, In), np.nan, dtype=float)
    win_rate = np.full((Rn, In), np.nan, dtype=float)

    for iy, restarts in enumerate(restarts_list):
        for ix, inner_iters in enumerate(inner_iters_list):
            vals = np.array(
                [
                    r["auc_gain"]
                    for r in run_rows
                    if int(r["restarts"]) == int(restarts) and int(r["inner_iters"]) == int(inner_iters)
                ],
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            N = int(vals.size)
            if N == 0:
                mean = float("nan")
                sem = float("nan")
                wr = float("nan")
            else:
                mean = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / math.sqrt(N)) if N > 1 else float("nan")
                wr = float(np.mean(vals > 0.0))

            auc_gain_mean[iy, ix] = mean
            win_rate[iy, ix] = wr

            summary_rows.append(
                {
                    "restarts": int(restarts),
                    "inner_iters": int(inner_iters),
                    "N": int(N),
                    "auc_gain_mean": mean,
                    "auc_gain_sem": sem,
                    "win_rate": wr,
                }
            )

    # save summary table
    summary_csv = out / "table5_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=["restarts", "inner_iters", "N", "auc_gain_mean", "auc_gain_sem", "win_rate"],
    )
    summary_tex = out / "table5_summary.tex"
    write_table_tex(
        summary_tex,
        summary_rows,
        caption=(
            "Experiment 5 inner-budget ablation (periodic family by default). "
            "Each cell reports mean+-s.e.m. of dAUC_B = AUC_ID - AUC_BB-FD "
            "at fixed evaluation budget B, over paired random instances."
        ),
        label="tab:exp5_inner_budget",
    )

    # figure
    fig_path = out / f"fig5_inner_budget_heatmap.{a.fmt}"
    plot_inner_budget_heatmap(
        fig_path,
        inner_iters_list=inner_iters_list,
        restarts_list=restarts_list,
        auc_gain_mean=auc_gain_mean,
        win_rate=win_rate,
    )
    plot_auc_gain_heatmap(
        out / f"fig5_inner_budget_auc_heatmap.{a.fmt}",
        inner_iters_list=inner_iters_list,
        restarts_list=restarts_list,
        auc_gain_mean=auc_gain_mean,
    )
    plot_win_rate_heatmap(
        out / f"fig5_inner_budget_winrate_heatmap.{a.fmt}",
        inner_iters_list=inner_iters_list,
        restarts_list=restarts_list,
        win_rate=win_rate,
    )

    # compact text summary
    txt = out / "SUMMARY.txt"
    with txt.open("w", encoding="utf-8") as f:
        f.write("Experiment 5 -- Inner-budget ablation\n")
        f.write(
            f"kind={a.kind} | periodic_K={a.periodic_K} | n={a.n} | p_edge={a.p_edge} | graph_seed={a.graph_seed}\n"
        )
        f.write(f"shots={a.shots} | budget_evals={a.budget_evals} | seeds={a.num_seeds}\n")
        f.write(f"inner_iters_list={inner_iters_list}\n")
        f.write(f"restarts_list={restarts_list}\n\n")
        # global summary
        flat = np.array([r["auc_gain"] for r in run_rows], dtype=float)
        flat = flat[np.isfinite(flat)]
        if flat.size:
            f.write(f"Overall mean dAUC over all cells: {float(np.mean(flat)):+.4f}\n")
            f.write(f"Overall win rate over all cells: {float(np.mean(flat > 0)) * 100:.1f}%\n")
        f.write("\nPer-cell summary (restarts, iters, mean+-sem, win_rate):\n")
        for r in summary_rows:
            f.write(
                f"  r={r['restarts']}, it={r['inner_iters']}: "
                f"{r['auc_gain_mean']:+.4f} +- {r['auc_gain_sem']:.4f}, "
                f"{100.0 * r['win_rate']:.0f}%\n"
            )

    print("\nSaved to:", out.resolve())
    print("Figure:", fig_path.name)
    print("Runs:", runs_csv.name)
    print("Table:", summary_tex.name)


if __name__ == "__main__":
    main()
