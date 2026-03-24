#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp06_graphclass_regime_heatmap.py
====================================

Experiment 6: (n, p) heatmaps showing where Implicit Differentiation (ID)
beats Black-Box Finite Differences (BB-FD) under a matched *energy-evaluation*
budget, across multiple graph classes.

What this is meant to show (ICML-friendly story)
------------------------------------------------
- In bilevel Max-Cut we optimize F(λ) = max_ϑ J(ϑ, λ).
- A naive outer hypergradient estimate via value probing / finite differences
  (BB-FD) needs *two additional* evaluations of F(λ±c). In a true bilevel
  setting, that implies *two additional inner solves* per outer step.
- ID (correlator reuse / envelope) avoids those extra probes and therefore
  makes better progress for the same evaluation budget.

This script quantifies that effect on a grid of Erdős–Rényi parameters G(n,p)
(and optionally other graph generators mapped to the same expected degree), and
summarizes the advantage in a heatmap.

Primary metric
--------------
ΔAUC_B = AUC_B(ID) - AUC_B(BB-FD)
where AUC_B is the normalized area under the best-so-far curve of J/J* vs.
energy-evaluation budget B.

Each heatmap cell also annotates the win-rate (fraction of seeds with ΔAUC_B>0).

Outputs (in --out)
------------------
- runs6_graphclass_metrics.csv                 (per-run raw results)
- table6_graphclass_summary.csv               (per-cell aggregates)
- fig6_graphclass_heatmap_deltaAUC.<fmt>      (multi-panel heatmap)
- SUMMARY.txt

Example
-------
python exp06_graphclass_regime_heatmap.py \
  --graph_classes er,ring,ws,ba \
  --n_list 8,10,12 \
  --p_list 0.20,0.35,0.50 \
  --seeds 1,2,3,4,5 \
  --budget_evals 6000 \
  --kind periodic --periodic_K 6 \
  --inner 60 --outer_max 200 \
  --fmt pdf

Notes
-----
- This is a pure statevector experiment, so n should be kept modest.
- "p" in --p_list corresponds to edge probability for ER graphs. For other
  graph classes we map p to a target expected degree d = p*(n-1).
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from paramham.experiment_defaults import (
    CANONICAL_SETUP,
    can_run_step,
    generate_family1d_instance,
    publication_cache_dir,
    publication_output_dir,
    vqe_fd_value_step_cost,
    vqe_id_step_cost,
)
from paramham.families import Family1D
from paramham.io import parse_float_list, parse_int_list
from paramham.maxcut import build_cut_mask, classical_Jstar, precompute_z
from paramham.plotting import COLORS, FULL_W, H_COL, _savefig, set_pub_style
from paramham.simulator import expect_J, vqe_state
from paramham.spsa import spsa_minimize

# Alias: the shared module calls it precompute_z but this experiment
# historically used the name precompute_z_big_endian.
precompute_z_big_endian = precompute_z


# ==============================================================================
# Experiment-specific helpers
# ==============================================================================


def step_auc(evals: np.ndarray, y_step: np.ndarray, x_max: float) -> float:
    """AUC of a step function y(evals) over [0, x_max] (right-continuous)."""
    evals = np.asarray(evals, float)
    y_step = np.asarray(y_step, float)
    x_max = float(x_max)

    if evals.size == 0 or x_max <= 0:
        return 0.0

    area = 0.0
    prev_x = 0.0
    prev_y = 0.0

    for x, y in zip(evals, y_step):
        x = float(min(x, x_max))
        if x <= prev_x:
            prev_y = float(y)
            continue
        area += (x - prev_x) * prev_y
        prev_x = x
        prev_y = float(y)
        if prev_x >= x_max:
            break

    if prev_x < x_max:
        area += (x_max - prev_x) * prev_y
    return float(area)


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _cache_default_dir(out: Path) -> Path:
    return publication_cache_dir("exp06")


def _cache_meta(args, n_list: List[int], p_list: List[float], seeds: List[int], graph_classes: List[str]) -> dict:
    return {
        "n_list": [int(n) for n in n_list],
        "p_list": [float(p) for p in p_list],
        "seeds": [int(s) for s in seeds],
        "graph_classes": [str(g) for g in graph_classes],
        "ws_beta": float(args.ws_beta),
        "kind": str(args.kind),
        "periodic_K": int(args.periodic_K),
        "graph_seed": int(args.graph_seed),
        "lam_min": float(args.lam_min),
        "lam_max": float(args.lam_max),
        "lam0": float(args.lam0),
        "lam_grid": int(args.lam_grid),
        "L_vqe": int(args.L_vqe),
        "budget_evals": float(args.budget_evals),
        "outer_max": int(args.outer_max),
        "inner": int(args.inner),
        "eta0": float(args.eta0),
        "eta_pow": float(args.eta_pow),
        "step_clip": float(args.step_clip),
        "fd_c_frac": float(args.fd_c_frac),
    }


def save_exp06_cache(cache_dir: Path, meta: dict, rows: List[Dict], agg_rows: List[Dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (cache_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (cache_dir / "agg_rows.json").write_text(json.dumps(agg_rows, indent=2), encoding="utf-8")


def load_exp06_cache(cache_dir: Path, meta_expected: dict):
    meta_path = cache_dir / "cache_meta.json"
    rows_path = cache_dir / "rows.json"
    agg_path = cache_dir / "agg_rows.json"
    if not meta_path.exists() or not rows_path.exists() or not agg_path.exists():
        return None
    try:
        meta_found = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        agg_rows = json.loads(agg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta_found != meta_expected:
        return None
    return rows, agg_rows


def build_heatmap_mats(
    graph_classes: List[str], n_list: List[int], p_list: List[float], agg_rows: List[Dict]
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    delta_mats: Dict[str, np.ndarray] = {}
    win_mats: Dict[str, np.ndarray] = {}
    for gc in graph_classes:
        D = np.full((len(n_list), len(p_list)), np.nan, dtype=float)
        W = np.full_like(D, np.nan)
        for row in agg_rows:
            if str(row["graph_class"]) != str(gc):
                continue
            try:
                i = n_list.index(int(row["n"]))
                j = next(k for k, p in enumerate(p_list) if abs(float(p) - float(row["p"])) < 1e-12)
            except (ValueError, StopIteration):
                continue
            D[i, j] = float(row["delta_auc_mean"])
            W[i, j] = float(row["win_rate"])
        delta_mats[gc] = D
        win_mats[gc] = W
    return delta_mats, win_mats


# ==============================================================================
# VQE energy (experiment-specific: uses cut_mask, not ZZ)
# ==============================================================================


def vqe_energy(n: int, cut_mask: np.ndarray, w: np.ndarray, params: np.ndarray, L: int) -> float:
    psi = vqe_state(n, params, L)
    J, _, _ = expect_J(psi, cut_mask, w)
    return -J


# ==============================================================================
# Outer loops (ID vs BB-FD), budgeted
# ==============================================================================


def run_outer_vqe_id_budgeted(
    n: int,
    cut_mask: np.ndarray,
    fam: Family1D,
    *,
    lam0: float,
    outer_max: int,
    inner: int,
    budget_evals: float,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    seed: int,
    L_vqe: int,
) -> Dict[str, np.ndarray]:
    """ID outer optimization (reuse hypergradient) until budget is exhausted."""

    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L_vqe
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    evals = 0.0
    bestJ = -1e18

    hist_evals = []
    hist_best = []

    step_cost = vqe_id_step_cost(inner)

    for t in range(1, int(outer_max) + 1):
        if not can_run_step(evals_used=evals, budget_evals=budget_evals, step_cost=step_cost):
            break

        w = fam.w(lam)

        def Efun(pvec):
            return vqe_energy(n, cut_mask, w, pvec, L_vqe)

        params, _bestE, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals += float(ev_in)

        psi = vqe_state(n, params, L_vqe)
        J, p_cut, _probs = expect_J(psi, cut_mask, w)
        evals += 1.0
        bestJ = max(bestJ, float(J))

        hist_evals.append(float(evals))
        hist_best.append(float(bestJ))

        # reuse hypergradient
        g = float(fam.dw_dlam(lam) @ p_cut)

        eta = eta0 / (t**eta_pow)
        step = float(np.clip(eta * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

    return {
        "evals_cum": np.asarray(hist_evals, float),
        "J_best": np.asarray(hist_best, float),
    }


def run_outer_vqe_bbfd_budgeted(
    n: int,
    cut_mask: np.ndarray,
    fam: Family1D,
    *,
    lam0: float,
    outer_max: int,
    inner: int,
    budget_evals: float,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    seed: int,
    L_vqe: int,
    fd_c_frac: float,
) -> Dict[str, np.ndarray]:
    """Black-box bilevel FD via value-probing F(lambda+-c): 2 extra inner solves per step."""

    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L_vqe
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    c_fd = float(fd_c_frac * (lam_max - lam_min))

    evals = 0.0
    bestJ = -1e18

    hist_evals = []
    hist_best = []

    step_cost = vqe_fd_value_step_cost(inner)

    for t in range(1, int(outer_max) + 1):
        if not can_run_step(evals_used=evals, budget_evals=budget_evals, step_cost=step_cost):
            break

        # ---- inner solve at current lambda ----
        w0 = fam.w(lam)

        def Efun0(pvec):
            return vqe_energy(n, cut_mask, w0, pvec, L_vqe)

        params, _bestE, ev0 = spsa_minimize(Efun0, params, bounds, iters=inner, seed=seed + 1000 * t + 0)
        evals += float(ev0)

        psi0 = vqe_state(n, params, L_vqe)
        J0, _p_cut0, _probs0 = expect_J(psi0, cut_mask, w0)
        evals += 1.0
        bestJ = max(bestJ, float(J0))
        hist_evals.append(float(evals))
        hist_best.append(float(bestJ))

        # ---- probes at lambda+-c (each with its own inner solve) ----
        lp = float(np.clip(lam + c_fd, lam_min, lam_max))
        lm = float(np.clip(lam - c_fd, lam_min, lam_max))

        if abs(lp - lm) < 1e-12:
            g = 0.0
        else:
            # warm-start from current params
            params_p = params.copy()
            params_m = params.copy()

            # plus
            wp = fam.w(lp)

            def Efunp(pvec):
                return vqe_energy(n, cut_mask, wp, pvec, L_vqe)

            params_p, _bestE_p, evp = spsa_minimize(Efunp, params_p, bounds, iters=inner, seed=seed + 1000 * t + 111)
            evals += float(evp)
            psip = vqe_state(n, params_p, L_vqe)
            Jp, _p_cut_p, _probs_p = expect_J(psip, cut_mask, wp)
            evals += 1.0
            bestJ = max(bestJ, float(Jp))
            hist_evals.append(float(evals))
            hist_best.append(float(bestJ))

            # minus
            wm = fam.w(lm)

            def Efunm(pvec):
                return vqe_energy(n, cut_mask, wm, pvec, L_vqe)

            params_m, _bestE_m, evm = spsa_minimize(Efunm, params_m, bounds, iters=inner, seed=seed + 1000 * t + 222)
            evals += float(evm)
            psim = vqe_state(n, params_m, L_vqe)
            Jm, _p_cut_m, _probs_m = expect_J(psim, cut_mask, wm)
            evals += 1.0
            bestJ = max(bestJ, float(Jm))
            hist_evals.append(float(evals))
            hist_best.append(float(bestJ))

            g = float((Jp - Jm) / (lp - lm))

        # ---- outer step ----
        eta = eta0 / (t**eta_pow)
        step = float(np.clip(eta * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

    return {
        "evals_cum": np.asarray(hist_evals, float),
        "J_best": np.asarray(hist_best, float),
    }


# ==============================================================================
# Heatmap plotting
# ==============================================================================


def _pretty_graph_name(gc: str) -> str:
    gc = str(gc).lower().strip()
    if gc in ("er", "erdos", "erdos_renyi", "gnp"):
        return "Erdos-Renyi"
    if gc in ("ring", "regular", "regular_ring"):
        return "Regular ring"
    if gc in ("ws", "watts", "watts_strogatz"):
        return "Watts-Strogatz"
    if gc in ("ba", "barabasi", "barabasi_albert"):
        return "Barabasi-Albert"
    return gc


def plot_heatmaps(
    path: Path,
    n_list: List[int],
    p_list: List[float],
    graph_classes: List[str],
    delta_mats: Dict[str, np.ndarray],
    win_mats: Dict[str, np.ndarray],
    *,
    fmt: str,
    annotate: bool = True,
    cmap: str = "RdBu_r",
):
    """Multi-panel heatmap grid with shared colorbar.
    Layout rule:
      - 1 panel  -> 1x1
      - 2-4      -> 2 columns (=> 2x2 for 4 panels)
      - >4       -> 3 columns (fallback)
    """

    set_pub_style(grid=False, base_size=8)

    n_panels = len(graph_classes)
    if n_panels <= 0:
        raise ValueError("plot_heatmaps: graph_classes must not be empty")

    # ---- Layout: force 2x2 for the common case of 4 graph classes ----
    if n_panels == 1:
        rows, cols = 1, 1
    elif n_panels <= 4:
        cols = 2
        rows = int(math.ceil(n_panels / cols))
    else:
        cols = 3
        rows = int(math.ceil(n_panels / cols))

    fig_h = rows * (H_COL - 0.15) + 0.15
    fig_w = FULL_W if cols > 1 else FULL_W / 2.0

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)

    # normalize axes to a 2D array of shape (rows, cols)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = np.asarray(axes).reshape(rows, cols)

    # symmetric color scale across all panels
    all_vals = []
    for gc in graph_classes:
        M = np.asarray(delta_mats[gc], float)
        all_vals.append(M[np.isfinite(M)])
    all_vals = np.concatenate(all_vals) if all_vals else np.asarray([0.0])
    vmax = float(np.max(np.abs(all_vals))) if all_vals.size else 1.0
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    last_im = None
    panel_axes = []

    for idx, gc in enumerate(graph_classes):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        panel_axes.append(ax)

        D = np.asarray(delta_mats[gc], float)
        W = np.asarray(win_mats[gc], float)

        last_im = ax.imshow(
            D,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
        )

        # ticks
        ax.set_xticks(np.arange(len(p_list)))
        ax.set_yticks(np.arange(len(n_list)))
        ax.set_xticklabels([f"{p:.2f}" for p in p_list])
        ax.set_yticklabels([str(n) for n in n_list])

        # only bottom row gets x-label
        if r == rows - 1:
            ax.set_xlabel(r"$p$")
        else:
            ax.set_xlabel("")

        # only left col gets y-label
        if c == 0:
            ax.set_ylabel(r"$n$")
        else:
            ax.set_ylabel("")

        ax.set_title(_pretty_graph_name(gc), pad=6)

        # subtle cell grid
        ax.set_xticks(np.arange(-0.5, len(p_list), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(n_list), 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotations
        if annotate:
            for i in range(len(n_list)):
                for j in range(len(p_list)):
                    d = D[i, j]
                    w = W[i, j]
                    if not np.isfinite(d):
                        txt = "—"
                        txt_color = COLORS["MUTED"]
                    else:
                        txt = f"{d:+.2f}\n{100 * w:.0f}%"
                        rgba = last_im.cmap(last_im.norm(d))
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        txt_color = "black" if lum > 0.62 else "white"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=txt_color)

    # hide empty axes
    for k in range(n_panels, rows * cols):
        r = k // cols
        c = k % cols
        axes[r, c].axis("off")

    # shared colorbar (attach only to used panels, not the hidden axes)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=panel_axes, shrink=0.92, pad=0.02)
        cbar.set_label(r"$\Delta\,\mathrm{AUC}_B$ (ID $-$ BB-FD)")

    _savefig(fig, path)
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--out", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--render_only", action="store_true")

    # grid over G(n,p)
    p.add_argument("--n_list", type=str, default="8,10,12")
    p.add_argument("--p_list", type=str, default="0.20,0.35,0.50")

    # graph classes
    p.add_argument("--graph_classes", type=str, default="er,ring,ws,ba", help="Comma-separated: er, ring, ws, ba")
    p.add_argument(
        "--ws_beta",
        type=float,
        default=0.30,
        help="Rewiring probability for Watts-Strogatz (density is still set by p).",
    )

    # seeds
    p.add_argument("--seeds", type=str, default="1,2,3")

    # family
    p.add_argument("--kind", type=str, default=CANONICAL_SETUP.family, choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=CANONICAL_SETUP.periodic_K)
    p.add_argument("--graph_seed", type=int, default=CANONICAL_SETUP.graph_seed)
    p.add_argument("--lam_min", type=float, default=CANONICAL_SETUP.lam_min)
    p.add_argument("--lam_max", type=float, default=CANONICAL_SETUP.lam_max)
    p.add_argument("--lam0", type=float, default=CANONICAL_SETUP.lam0)
    p.add_argument("--lam_grid", type=int, default=301)

    # VQE depth
    p.add_argument("--L_vqe", type=int, default=2)

    # optimization budget and schedules
    p.add_argument("--budget_evals", type=float, default=CANONICAL_SETUP.budget_evals)
    p.add_argument("--outer_max", type=int, default=250)
    p.add_argument("--inner", type=int, default=28)

    p.add_argument("--eta0", type=float, default=0.25)
    p.add_argument("--eta_pow", type=float, default=0.4)
    p.add_argument("--step_clip", type=float, default=0.6)

    # FD step
    p.add_argument("--fd_c_frac", type=float, default=0.10)

    # heatmap look
    p.add_argument("--no_annotate", action="store_true", help="Disable per-cell text annotations.")
    p.add_argument("--cmap", type=str, default="RdBu_r", help="Matplotlib colormap name.")

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out) if a.out is not None else publication_output_dir("exp06")
    out.mkdir(parents=True, exist_ok=True)

    n_list = parse_int_list(a.n_list)
    p_list = parse_float_list(a.p_list)
    seeds = parse_int_list(a.seeds)

    if not n_list:
        raise ValueError("--n_list must not be empty")
    if not p_list:
        raise ValueError("--p_list must not be empty")
    if not seeds:
        raise ValueError("--seeds must not be empty")

    graph_classes = [g.strip() for g in (a.graph_classes or "").split(",") if g.strip()]
    if not graph_classes:
        raise ValueError("--graph_classes must not be empty")

    B = float(a.budget_evals)
    cache_dir = Path(a.cache_dir) if a.cache_dir is not None else _cache_default_dir(out)
    meta = _cache_meta(a, n_list, p_list, seeds, graph_classes)

    cached = None if a.recompute else load_exp06_cache(cache_dir, meta)
    if cached is not None:
        rows, agg_rows = cached
        print(f"[cache] Loaded graph-class heatmap data from {cache_dir.resolve()}")
    elif a.render_only:
        raise SystemExit(f"No matching cache found in {cache_dir}")
    else:
        Z_cache: Dict[int, np.ndarray] = {}
        rows = []
        agg_rows = []

        for gc in graph_classes:
            for i, n in enumerate(n_list):
                if n not in Z_cache:
                    Z_cache[n] = precompute_z_big_endian(int(n))
                Z = Z_cache[n]

                for j, p_edge in enumerate(p_list):
                    deltas = []
                    wins = 0
                    used = 0

                    for s in seeds:
                        edges, fam = generate_family1d_instance(
                            gc,
                            int(n),
                            float(p_edge),
                            a.kind,
                            (a.lam_min, a.lam_max),
                            graph_seed=a.graph_seed,
                            periodic_K=int(a.periodic_K),
                            instance_id=f"{gc}:{n}:{p_edge}:{s}",
                            ws_beta=float(a.ws_beta),
                            safety_bounds=False,
                        )

                        if not edges or fam is None:
                            continue

                        cut_mask = build_cut_mask(edges, Z)
                        J_star, lam_star = classical_Jstar(fam, cut_mask, int(a.lam_grid))

                        hist_id = run_outer_vqe_id_budgeted(
                            int(n),
                            cut_mask,
                            fam,
                            lam0=float(a.lam0),
                            outer_max=int(a.outer_max),
                            inner=int(a.inner),
                            budget_evals=B,
                            eta0=float(a.eta0),
                            eta_pow=float(a.eta_pow),
                            step_clip=float(a.step_clip),
                            seed=int(s) + 0,
                            L_vqe=int(a.L_vqe),
                        )

                        hist_fd = run_outer_vqe_bbfd_budgeted(
                            int(n),
                            cut_mask,
                            fam,
                            lam0=float(a.lam0),
                            outer_max=int(a.outer_max),
                            inner=int(a.inner),
                            budget_evals=B,
                            eta0=float(a.eta0),
                            eta_pow=float(a.eta_pow),
                            step_clip=float(a.step_clip),
                            seed=int(s) + 100000,
                            L_vqe=int(a.L_vqe),
                            fd_c_frac=float(a.fd_c_frac),
                        )

                        y_id = np.clip(hist_id["J_best"] / J_star, 0.0, 1.5)
                        y_fd = np.clip(hist_fd["J_best"] / J_star, 0.0, 1.5)

                        auc_id = step_auc(hist_id["evals_cum"], y_id, B) / B
                        auc_fd = step_auc(hist_fd["evals_cum"], y_fd, B) / B
                        delta = float(auc_id - auc_fd)

                        deltas.append(delta)
                        wins += int(delta > 1e-12)
                        used += 1

                        def _value_at(evals_arr, y_arr, x):
                            evals_arr = np.asarray(evals_arr, float)
                            y_arr = np.asarray(y_arr, float)
                            if evals_arr.size == 0:
                                return 0.0
                            idx = np.searchsorted(evals_arr, float(x), side="right") - 1
                            if idx < 0:
                                return 0.0
                            return float(y_arr[idx])

                        final_id = _value_at(hist_id["evals_cum"], y_id, B)
                        final_fd = _value_at(hist_fd["evals_cum"], y_fd, B)

                        rows.append(
                            {
                                "graph_class": gc,
                                "n": int(n),
                                "p": float(p_edge),
                                "seed": int(s),
                                "graph_seed": int(a.graph_seed),
                                "m_edges": int(len(edges)),
                                "kind": str(a.kind),
                                "periodic_K": int(a.periodic_K),
                                "lam0": float(a.lam0),
                                "lam_star_grid": float(lam_star),
                                "J_star_grid": float(J_star),
                                "budget_evals": float(B),
                                "inner": int(a.inner),
                                "outer_max": int(a.outer_max),
                                "L_vqe": int(a.L_vqe),
                                "eta0": float(a.eta0),
                                "eta_pow": float(a.eta_pow),
                                "step_clip": float(a.step_clip),
                                "fd_c_frac": float(a.fd_c_frac),
                                "auc_id": float(auc_id),
                                "auc_fd": float(auc_fd),
                                "delta_auc": float(delta),
                                "final_id": float(final_id),
                                "final_fd": float(final_fd),
                                "delta_final": float(final_id - final_fd),
                            }
                        )

                    agg_rows.append(
                        {
                            "graph_class": gc,
                            "n": int(n),
                            "p": float(p_edge),
                            "seeds_used": int(used),
                            "delta_auc_mean": float(np.mean(deltas)) if used > 0 else float("nan"),
                            "delta_auc_std": float(np.std(deltas, ddof=1)) if used > 1 else 0.0,
                            "win_rate": float(wins / used) if used > 0 else float("nan"),
                        }
                    )

                    print(
                        f"[{_pretty_graph_name(gc):>14}] n={n:2d} p={p_edge:.2f} | seeds={used:2d} | "
                        f"ΔAUC={(float(np.mean(deltas)) if used > 0 else float('nan')):+.4f} | "
                        f"win={(100 * wins / used) if used > 0 else float('nan'):.0f}%"
                    )

        save_exp06_cache(cache_dir, meta, rows, agg_rows)
        print(f"[cache] Saved graph-class heatmap data to {cache_dir.resolve()}")

    delta_mats, win_mats = build_heatmap_mats(graph_classes, n_list, p_list, agg_rows)

    # write outputs
    write_csv(out / "runs6_graphclass_metrics.csv", rows)
    write_csv(out / "table6_graphclass_summary.csv", agg_rows)

    # heatmap
    plot_heatmaps(
        out / f"fig6_graphclass_heatmap_deltaAUC.{a.fmt}",
        n_list,
        p_list,
        graph_classes,
        delta_mats,
        win_mats,
        fmt=a.fmt,
        annotate=(not a.no_annotate),
        cmap=str(a.cmap),
    )

    # summary text
    lines = []
    lines.append("Experiment 6: Heatmap over (n,p) showing ΔAUC_B(ID - BB-FD)")
    lines.append(f"Graph classes: {', '.join(graph_classes)}")
    lines.append(f"n_list: {n_list}")
    lines.append(f"p_list: {p_list}")
    lines.append(f"seeds: {seeds} (N={len(seeds)})")
    lines.append(f"graph_seed: {a.graph_seed}")
    lines.append("")
    lines.append(f"Budget B (energy evals): {B}")
    lines.append(f"Inner iters: {a.inner} | Outer max: {a.outer_max}")
    lines.append(f"VQE depth L: {a.L_vqe}")
    lines.append(f"Family: kind={a.kind} periodic_K={a.periodic_K}")
    lines.append(f"FD step: fd_c_frac={a.fd_c_frac}")
    lines.append("")
    lines.append("Legend: cell text shows")
    lines.append("  line1: mean ΔAUC_B (ID - BB-FD)")
    lines.append("  line2: win-rate % (fraction seeds with ΔAUC_B>0)")
    lines.append("")
    lines.append("Saved outputs:")
    lines.append("  - runs6_graphclass_metrics.csv")
    lines.append("  - table6_graphclass_summary.csv")
    lines.append(f"  - fig6_graphclass_heatmap_deltaAUC.{a.fmt}")
    lines.append("  - SUMMARY.txt")

    with open(out / "SUMMARY.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("Saved to:", out.resolve())


if __name__ == "__main__":
    main()
