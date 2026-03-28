#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp07_multi_dimensional_outer_control.py
=======================================

Experiment 7 (Main / Supplement): Edge-wise outer parameters (vector λ)
-----------------------------------------------------------------------

Goal
----
Generalize the outer control from a single scalar λ to a vector of per-edge controls

    λ = (λ_e)_{e∈E},   with   H(λ) = Σ_{e∈E} w_e(λ_e) P_e,  P_e=(I - Z_i Z_j)/2.

We test whether the correlator-reuse implicit differentiation (CR-ImpDiff / "ID")
advantage persists in this higher-dimensional outer problem under matched
evaluation budgets.

Key objects (aligned with the manuscript)
----------------------------------------
Inner objective:
  J(φ, λ_vec) = ⟨H(λ_vec)⟩_φ = Σ_e w_e(λ_e) p_e(φ),    p_e=(1-⟨Z_i Z_j⟩)/2 ∈ [0,1].

Bilevel value function (target):
  F(λ_vec) = max_φ J(φ, λ_vec).

Methods (outer direction only differs; same outer update rule)
--------------------------------------------------------------
(1) ID / CR-ImpDiff (envelope / partial derivative signal)
    After ONE inner solve at current λ_vec:
        g_ID,e = ∂J/∂λ_e = w'_e(λ_e) * p_e
    i.e. the full gradient vector is obtained by reusing the same ZZ correlators
    already needed for J. (Negligible additional quantum measurement cost.)

(2) Black-box bilevel baseline in high dimension: SPSA finite difference on F
    Use a Rademacher direction Δ ∈ {±1}^m:
        g_FD = [F(λ_vec + cΔ) - F(λ_vec - cΔ)]/(2c) * Δ
    Each F(·) query requires an additional inner re-solve at the perturbed λ.
    Thus FD costs ~3 inner solves per outer step (center + plus + minus).

Fairness / cost model
---------------------
- We compare at a fixed total evaluation budget B (energy evaluations).
- For FD, best-so-far is computed over ALL value queries it actually makes
  (center, plus, minus). We do not discard evaluated candidates.
- We report performance as best-so-far value vs cumulative energy evaluations,
  and as normalized AUC_B over [0,B].

Normalization J*
----------------
In the edge-wise setting with positive weights, an informative diagnostic scale is:
  w_e^max = max_{λ∈[λmin,λmax]} w_e(λ)   (independent per edge),
  J* = max_{z∈{0,1}^n} Σ_e w_e^max * I{edge e cut by z}.
We compute w^max by scanning a λ-grid and compute J* by enumerating all 2^n bitstrings
(feasible for n<=12). J* is used only for normalization/diagnostics.

Outputs (paper-ready)
---------------------
Saved in --out directory:
  - fig7_edgewise_budget_gain_<suffix>.<fmt>
      2-panel figure: (a) best-so-far / J* vs evaluations (mean±stderr),
                      (b) per-instance ΔAUC_B vs |E| (scatter only; no extra text).
  - fig7_edgewise_steps_<suffix>.<fmt>
      Bar chart: outer steps completed within budget B (mean±stderr).
  - runs7_edgewise_metrics.csv
      Per-instance metrics (AUC, final best at budget, steps, costs, etc.).
  - table7_edgewise_summary.csv / .tex
      Mean±stderr summary across instances.
  - SUMMARY.txt
      Compact copy/paste summary.

Example
-------
python exp07_multi_dimensional_outer_control.py \
  --family periodic --periodic_K 6 \
  --n 12 --p_edge 0.45 \
  --inner_iters 28 --restarts 1 --L 2 \
  --budget_evals 5100 \
  --num_instances 20 \
  --shots 0 \
  --fmt pdf --out output/exp07

"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_style import FIG_H, FIG_W, apply_thesis_axes_style, dual_panel_size, grid_size, save_figure, use_thesis_style

from paramham.experiment_defaults import (
    CANONICAL_SETUP,
    can_run_step,
    generate_er_instance_graph,
    publication_cache_dir,
    publication_output_dir,
    restarted_fd_step_cost,
    restarted_id_step_cost,
)
from paramham.families import FamilyEdgeWise
from paramham.io import parse_int_list
from paramham.maxcut import build_cut_mask, build_ZZ_edges
from paramham.maxcut import precompute_z as precompute_z_big_endian
from paramham.plotting import COLORS, H_COL, add_panel_legend
from paramham.seeds import to_uint_seed
from paramham.simulator import vqe_state
from paramham.spsa import spsa_minimize

# ==============================================================================
# Experiment-specific constant
# ==============================================================================

H_TWO = H_COL


def _set_exp07_plot_style(grid: bool = False):
    use_thesis_style()
    plt.rcParams["axes.grid"] = bool(grid)


def _single_panel_with_footer():
    """Match the Exp2 mechanism for a single plot plus a dedicated legend row."""

    fig = plt.figure(figsize=dual_panel_size(), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.14])
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis("off")
    return fig, ax, legend_ax


def _footer_legend(legend_ax: plt.Axes, handles, labels, *, ncol: int = 2):
    return legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.2,
    )


def _compact_collage_axis(ax: plt.Axes):
    """Slightly tighten label/tick typography for square sixpack panels."""

    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.tick_params(axis="both", which="major", labelsize=9)


# ==============================================================================
# Experiment-specific helpers (kept inline)
# ==============================================================================


def _mean_stderr(x: np.ndarray, axis: int = 0):
    x = np.asarray(x, float)
    mu = np.nanmean(x, axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    n = np.sum(np.isfinite(x), axis=axis)
    se = sd / np.sqrt(np.maximum(1, n))
    return mu, se


def _step_interp(evals: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Piecewise-constant interpolation (step function):
      y(b) = last observed value with eval <= b.
    """
    evals = np.asarray(evals, float)
    vals = np.asarray(vals, float)
    grid = np.asarray(grid, float)

    if evals.size == 0:
        return np.full_like(grid, np.nan, dtype=float)

    order = np.argsort(evals)
    evals = evals[order]
    vals = vals[order]

    # Anchor at budget 0
    if evals[0] > 0.0:
        evals = np.concatenate([[0.0], evals])
        vals = np.concatenate([[vals[0]], vals])
    else:
        evals[0] = 0.0

    idx = np.searchsorted(evals, grid, side="right") - 1
    idx = np.clip(idx, 0, vals.size - 1)
    return vals[idx]


def classical_Jstar_from_wmax(cut_mask: np.ndarray, wmax: np.ndarray) -> float:
    """
    J* = max_z (cut_mask[z] @ wmax), enumerating all 2^n bitstrings.
    """
    with np.errstate(all="ignore"):
        vals = cut_mask @ wmax.astype(np.float64)
    vals = np.nan_to_num(vals, nan=-1e30, posinf=-1e30, neginf=-1e30)
    Jstar = float(np.max(vals))
    if (not np.isfinite(Jstar)) or Jstar <= 0.0:
        Jstar = 1.0
    return Jstar


# ==============================================================================
# VQE expectation (experiment-specific: different signature with shots/ZZ_edges)
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
        idx = rng.choice(np.arange(probs.size), size=shots, replace=True, p=probs)
        with np.errstate(all="ignore"):
            p_cut = np.mean(cut_mask[idx, :], axis=0).astype(np.float64)
        np.nan_to_num(p_cut, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    else:
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
# Inner solver (experiment-specific)
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
    Run 'restarts' SPSA inner solves and pick the best (lowest energy).
    Returns:
      best_params, best_J, best_p_cut, evals_used
    """
    D = init_params.size
    bounds = [(-math.pi, math.pi)] * D

    best_params = init_params.copy()
    best_E = float("inf")
    evals_total = 0

    for r in range(int(restarts)):
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

        p_star, E_star, ev = spsa_minimize(
            Efun,
            p0,
            bounds,
            iters=inner_iters,
            seed=seed_base + 17 * (r + 1),
        )
        evals_total += int(ev)

        if np.isfinite(E_star) and E_star < best_E:
            best_E = E_star
            best_params = p_star.copy()

    # evaluate final best to obtain J and p_cut (counts as ONE energy evaluation)
    eval_rng_final = np.random.default_rng(to_uint_seed(seed_base + 424242))
    J, p_cut, _ = vqe_expect(n, best_params, L, w, ZZ_edges, cut_mask, shots, eval_rng_final)
    evals_total += 1

    return best_params, float(J), p_cut.astype(np.float64), int(evals_total)


# ==============================================================================
# Outer loops (edge-wise λ) under evaluation budget B
# ==============================================================================


def auc_step(evals: np.ndarray, values: np.ndarray, budget: float) -> float:
    """
    Step-function AUC on [0, budget].
    Assumes best-so-far is held constant between recorded events.
    """
    evals = np.asarray(evals, float)
    values = np.asarray(values, float)
    budget = float(budget)
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
        return float(values[0])

    if evals[-1] < budget:
        evals = np.append(evals, budget)
        values = np.append(values, values[-1])

    dx = np.diff(evals)
    area = float(np.sum(dx * values[:-1]))
    return area / float(budget)


def _best_at_budget(evals: np.ndarray, best_norm: np.ndarray, budget: float) -> float:
    g = np.array([float(budget)], float)
    return float(_step_interp(evals, best_norm, g)[0])


def run_outer_budget_edgewise(
    mode: str,
    n: int,
    edges,
    fam: FamilyEdgeWise,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    lam_vec0: np.ndarray,
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
    fd_warmstart: str = "best",
) -> Dict[str, np.ndarray]:
    """
    Runs outer optimization until budget or outer_max.

    Returns trace dictionaries:
      events_evals: cumulative evals after each value-query completion (center, and +/− for FD)
      events_best : best-so-far normalized by J* at each event
      outer_evals : cumulative evals after each completed OUTER iteration (after update)
      evals_end   : total evals used
      lam_vec_end : final λ vector
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam_vec = np.clip(np.asarray(lam_vec0, float), lam_min, lam_max)

    D = 2 * n * L
    params = np.zeros(D, float)

    evals = 0.0
    best = -1e30

    events_evals: List[float] = []
    events_best: List[float] = []
    outer_evals: List[float] = []

    c = float(c_frac * (lam_max - lam_min))

    rng_dir = np.random.default_rng(to_uint_seed(seed + 999999))

    step_cost = (
        restarted_id_step_cost(inner_iters, restarts) if mode == "ID" else restarted_fd_step_cost(inner_iters, restarts)
    )

    for t in range(1, int(outer_max) + 1):
        if not can_run_step(evals_used=evals, budget_evals=budget_evals, step_cost=step_cost):
            break
        # --- center value query F(λ) via inner solve
        w = fam.w(lam_vec)

        params_c, Jc, p_cut_c, ev_c = inner_solve(
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
        params = params_c
        evals += float(ev_c)

        best = max(best, float(Jc))
        events_evals.append(evals)
        events_best.append(best / Jstar)

        if mode == "ID":
            # full hypergradient vector (correlator reuse): g_e = w'_e(λ_e) * p_e
            g_vec = fam.dw_dlam(lam_vec) * p_cut_c

        elif mode == "FD_SPSA":
            # random direction Δ ∈ {±1}^m
            delta = rng_dir.choice([-1.0, +1.0], size=lam_vec.size).astype(float)

            lam_p = np.clip(lam_vec + c * delta, lam_min, lam_max)
            lam_m = np.clip(lam_vec - c * delta, lam_min, lam_max)

            # + perturbation solve
            w_p = fam.w(lam_p)
            params_p, Jp, _, ev_p = inner_solve(
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
            evals += float(ev_p)
            best = max(best, float(Jp))
            events_evals.append(evals)
            events_best.append(best / Jstar)

            # - perturbation solve
            w_m = fam.w(lam_m)
            params_m, Jm, _, ev_m = inner_solve(
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
            evals += float(ev_m)
            best = max(best, float(Jm))
            events_evals.append(evals)
            events_best.append(best / Jstar)

            # SPSA-FD gradient estimate on F
            g_scalar = (float(Jp) - float(Jm)) / (2.0 * c) if c > 0 else 0.0
            g_vec = g_scalar * delta

            # optional warm-start for the NEXT center solve
            if fd_warmstart == "best":
                # pick params from the best of {center, plus, minus}
                if (Jp >= Jc) and (Jp >= Jm):
                    params = params_p
                elif (Jm >= Jc) and (Jm >= Jp):
                    params = params_m
                else:
                    params = params_c
            else:
                # keep center params (default earlier scripts)
                params = params_c

        else:
            raise ValueError("mode must be 'ID' or 'FD_SPSA'")

        # --- outer update (projected gradient ascent on box)
        eta = float(eta0 / (t**eta_pow))
        step = eta * g_vec
        step = np.clip(step, -float(step_clip), float(step_clip))
        lam_vec = np.clip(lam_vec + step, lam_min, lam_max)

        outer_evals.append(float(evals))

    return {
        "events_evals": np.asarray(events_evals, float),
        "events_best": np.asarray(events_best, float),
        "outer_evals": np.asarray(outer_evals, float),
        "evals_end": np.array([evals], float),
        "lam_vec_end": lam_vec.astype(float),
    }


def steps_within_budget(outer_evals: np.ndarray, budget: float) -> int:
    outer_evals = np.asarray(outer_evals, float)
    return int(np.sum(outer_evals <= float(budget)))


# ==============================================================================
# CSV / table writers (experiment-specific)
# ==============================================================================


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _cache_default_dir(out: Path) -> Path:
    return publication_cache_dir("exp07")


def _cache_meta(args, n_sweep: List[int]) -> dict:
    return {
        "seed0": int(args.seed0),
        "num_instances": int(args.num_instances),
        "family": str(args.family),
        "periodic_K": int(args.periodic_K),
        "n": int(args.n),
        "n_sweep": [int(n) for n in n_sweep],
        "p_edge": float(args.p_edge),
        "graph_seed": int(args.graph_seed),
        "lam_min": float(args.lam_min),
        "lam_max": float(args.lam_max),
        "lam0": float(args.lam0),
        "L": int(args.L),
        "inner_iters": int(args.inner_iters),
        "restarts": int(args.restarts),
        "shots": int(args.shots),
        "outer_max": int(args.outer_max),
        "eta0": float(args.eta0),
        "eta_pow": float(args.eta_pow),
        "step_clip": float(args.step_clip),
        "c_frac": float(args.c_frac),
        "budget_evals": float(args.budget_evals),
        "wmax_grid": int(args.wmax_grid),
        "budget_points": int(args.budget_points),
        "fd_warmstart": str(args.fd_warmstart),
    }


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_exp07_cache(cache_dir: Path, meta: dict, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (cache_dir / "payload.json").write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def load_exp07_cache(cache_dir: Path, meta_expected: dict):
    meta_path = cache_dir / "cache_meta.json"
    payload_path = cache_dir / "payload.json"
    if not meta_path.exists() or not payload_path.exists():
        return None
    try:
        meta_found = json.loads(meta_path.read_text(encoding="utf-8"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta_found != meta_expected:
        return None
    return payload


def write_table_tex(path: Path, summary_rows: List[Dict], caption: str, label: str):
    """
    Minimal booktabs LaTeX table:
      metric | ID mean±stderr | FD mean±stderr
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by exp07_multi_dimensional_outer_control.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Metric & VQE+CR-ID & VQE+FD\\\\\n")
        f.write("\\midrule\n")
        for r in summary_rows:
            f.write(
                f"{r['metric']} & {r['id_mean']:.3f}$\\pm${r['id_se']:.3f} & "
                f"{r['fd_mean']:.3f}$\\pm${r['fd_se']:.3f}\\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


# ==============================================================================
# Plotting (experiment-specific)
# ==============================================================================


def plot_2panel_budget_and_gain(
    path: Path,
    budget_grid: np.ndarray,
    best_id_grid: np.ndarray,
    best_fd_grid: np.ndarray,
    auc_gain: np.ndarray,
    m_edges: np.ndarray,
    budget_evals: float,
):
    """
    Two-panel, paper-ready figure:
      left: mean±stderr best-so-far / J* vs evaluation budget (step-interp grid)
      right: per-instance ΔAUC_B vs |E| (scatter only; no extra annotation)

    best_*_grid: (N, G)
    """
    _set_exp07_plot_style(grid=False)
    b = np.asarray(budget_grid, float)

    mu_id, se_id = _mean_stderr(best_id_grid, axis=0)
    mu_fd, se_fd = _mean_stderr(best_fd_grid, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=dual_panel_size(), constrained_layout=True)

    # Left panel: Budget curves
    ax = axs[0]
    ax.plot(b, mu_id, color=COLORS["ID"], label="VQE + CR-ID")
    ax.fill_between(b, mu_id - se_id, mu_id + se_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(b, mu_fd, color=COLORS["FD"], ls="--", label="VQE + FD")
    ax.fill_between(b, mu_fd - se_fd, mu_fd + se_fd, color=COLORS["FD"], alpha=0.18, linewidth=0)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Best-so-far $\hat F / J^*$")
    ax.set_xlim(float(b[0]), float(b[-1]))
    y_all = np.concatenate([mu_id, mu_fd])
    y0 = max(0.0, float(np.nanmin(y_all) - 0.04))
    y1 = min(1.05, float(np.nanmax(y_all) + 0.04))
    ax.set_ylim(y0, y1)
    apply_thesis_axes_style(ax, grid=False)
    add_panel_legend(ax, placement="below", ncol=2)

    # Right panel: ΔAUC vs |E|
    ax = axs[1]
    x = np.asarray(m_edges, float)
    y = np.asarray(auc_gain, float)

    ax.axhline(0.0, color=COLORS["GT"], lw=1.0, ls=":")
    ax.scatter(x, y, s=22, color="#666666", alpha=0.75, edgecolors="none")

    ax.set_xlabel(r"Number of edges $|E|$")
    ax.set_ylabel(r"$\Delta \mathrm{AUC}_B$ (CR-ID $-$ FD)")
    apply_thesis_axes_style(ax, grid=False)

    save_figure(fig, path)
    plt.close(fig)


def plot_steps_bar(
    path: Path,
    steps_id: np.ndarray,
    steps_fd: np.ndarray,
):
    """
    Bar chart: outer steps completed within budget B (mean±stderr).
    """
    _set_exp07_plot_style(grid=False)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    sid = np.asarray(steps_id, float)
    sfd = np.asarray(steps_fd, float)

    mu_id = float(np.nanmean(sid))
    mu_fd = float(np.nanmean(sfd))
    se_id = (
        float(np.nanstd(sid, ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(sid)))))
        if np.sum(np.isfinite(sid)) > 1
        else float("nan")
    )
    se_fd = (
        float(np.nanstd(sfd, ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(sfd)))))
        if np.sum(np.isfinite(sfd)) > 1
        else float("nan")
    )

    xs = np.array([0, 1], int)
    mus = np.array([mu_id, mu_fd], float)
    ses = np.array([se_id, se_fd], float)

    ax.bar(xs, mus, yerr=ses, capsize=3, width=0.6, color=["#dddddd", "#dddddd"], edgecolor=["#444444", "#444444"])
    ax.scatter([0], [mu_id], color=COLORS["ID"], s=26, zorder=5, edgecolors="white", linewidth=0.5)
    ax.scatter([1], [mu_fd], color=COLORS["FD"], s=26, zorder=5, edgecolors="white", linewidth=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(["CR-ID", "FD"])
    ax.set_ylabel(r"Outer steps within budget $B$")
    ax.set_xlim(-0.6, 1.6)

    ratio = mu_id / max(1e-12, mu_fd)
    ax.text(
        0.5,
        0.98,
        rf"steps ratio $\approx$ {ratio:.2f}$\times$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )

    apply_thesis_axes_style(ax, grid=False)
    save_figure(fig, path)
    plt.close(fig)


CURVE_STYLES = {
    "ID": {"color": COLORS["ID"], "ls": "-", "label": "VQE + CR-ID", "marker": "o", "alpha": 0.14},
    "FD": {"color": COLORS["FD"], "ls": "--", "label": "VQE + FD", "marker": "s", "alpha": 0.12},
}


def _panel_label(ax: plt.Axes, label: str):
    ax.text(0.00, 1.02, label, transform=ax.transAxes, va="bottom", ha="left", fontsize=9, fontweight="bold")


def _interesting_ylim_arrays(
    arrays: List[np.ndarray], *, y_floor: float = 0.0, y_cap: float = 1.02
) -> tuple[float, float]:
    finite = [np.asarray(a, float)[np.isfinite(a)] for a in arrays if np.asarray(a, float).size]
    finite = [a for a in finite if a.size]
    if not finite:
        return y_floor, y_cap
    lo = float(np.min(np.concatenate(finite)))
    hi = float(np.max(np.concatenate(finite)))
    span = max(hi - lo, 0.08)
    pad = 0.16 * span
    return max(y_floor, lo - pad), min(y_cap, hi + pad)


def plot_budget_curve(path: Path, budget_grid: np.ndarray, best_id_grid: np.ndarray, best_fd_grid: np.ndarray):
    _set_exp07_plot_style(grid=False)
    fig, ax, legend_ax = _single_panel_with_footer()
    b = np.asarray(budget_grid, float)
    mu_id, se_id = _mean_stderr(best_id_grid, axis=0)
    mu_fd, se_fd = _mean_stderr(best_fd_grid, axis=0)

    ax.plot(b, mu_id, color=COLORS["ID"], lw=1.9, label="VQE + CR-ID")
    ax.fill_between(b, mu_id - se_id, mu_id + se_id, color=COLORS["ID"], alpha=0.14, linewidth=0)
    ax.plot(b, mu_fd, color=COLORS["FD"], lw=1.9, ls="--", label="VQE + FD")
    ax.fill_between(b, mu_fd - se_fd, mu_fd + se_fd, color=COLORS["FD"], alpha=0.12, linewidth=0)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Best-so-far $F/J^*$")
    ax.set_xlim(float(b[0]), float(b[-1]))
    ax.set_ylim(*_interesting_ylim_arrays([mu_id - se_id, mu_id + se_id, mu_fd - se_fd, mu_fd + se_fd]))
    apply_thesis_axes_style(ax, grid=False)
    handles, labels = ax.get_legend_handles_labels()
    _footer_legend(legend_ax, handles, labels, ncol=2)
    save_figure(fig, path)
    plt.close(fig)


def plot_auc_gain_scatter(path: Path, auc_gain: np.ndarray, m_edges: np.ndarray):
    _set_exp07_plot_style(grid=False)
    fig, ax, legend_ax = _single_panel_with_footer()
    x = np.asarray(m_edges, float)
    y = np.asarray(auc_gain, float)
    pos = y > 0.0
    ax.axhline(0.0, color=COLORS["REFERENCE"], lw=1.0, ls="--", zorder=1)
    ax.scatter(x[pos], y[pos], s=28, color=COLORS["ID"], alpha=0.88, edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(x[~pos], y[~pos], s=28, color=COLORS["FD"], alpha=0.88, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xlabel(r"Number of edges $|E|$")
    ax.set_ylabel(r"$\Delta \mathrm{AUC}_B$ (CR-ID $-$ FD)")
    _footer_legend(
        legend_ax,
        [
            Line2D([], [], color=COLORS["ID"], marker="o", lw=0, markersize=5, label="CR-ID > FD"),
            Line2D([], [], color=COLORS["FD"], marker="o", lw=0, markersize=5, label="FD >= CR-ID"),
        ],
        ["CR-ID > FD", "FD >= CR-ID"],
        ncol=2,
    )
    apply_thesis_axes_style(ax, grid=False)
    save_figure(fig, path)
    plt.close(fig)


def plot_steps_bar_pretty(path: Path, steps_id: np.ndarray, steps_fd: np.ndarray):
    _set_exp07_plot_style(grid=False)
    fig, ax, legend_ax = _single_panel_with_footer()

    sid = np.asarray(steps_id, float)
    sfd = np.asarray(steps_fd, float)
    mu_id, se_id = _mean_stderr(sid)
    mu_fd, se_fd = _mean_stderr(sfd)
    xs = np.array([0, 1], int)
    ax.bar(
        xs,
        [mu_id, mu_fd],
        yerr=[se_id, se_fd],
        capsize=3,
        width=0.62,
        color=["#f7d7dc", "#d8e5f3"],
        edgecolor=[COLORS["ID"], COLORS["FD"]],
        linewidth=1.2,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(["CR-ID", "FD"])
    ax.set_ylabel(r"Outer steps within budget $B$")
    _footer_legend(
        legend_ax,
        [
            Line2D([], [], color=COLORS["ID"], lw=2.0, label="VQE + CR-ID"),
            Line2D([], [], color=COLORS["FD"], lw=2.0, ls="--", label="VQE + FD"),
        ],
        ["VQE + CR-ID", "VQE + FD"],
        ncol=2,
    )
    apply_thesis_axes_style(ax, grid=False)
    save_figure(fig, path)
    plt.close(fig)


def plot_pair_scatter(path: Path, x: np.ndarray, y: np.ndarray, *, xlabel: str, ylabel: str):
    _set_exp07_plot_style(grid=False)
    fig, ax, legend_ax = _single_panel_with_footer()
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pos = y > x + 1e-12
    neg = ~pos
    if np.any(pos):
        ax.scatter(x[pos], y[pos], s=28, color=COLORS["ID"], alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
    if np.any(neg):
        ax.scatter(x[neg], y[neg], s=28, color=COLORS["FD"], alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
    if x.size and y.size:
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
    else:
        lo, hi = 0.0, 1.0
    pad = max(0.015, 0.08 * max(hi - lo, 0.12))
    lo = max(0.0, lo - pad)
    hi = min(1.02, hi + pad)
    ax.plot([lo, hi], [lo, hi], color=COLORS["REFERENCE"], lw=1.0, ls="--", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x.size:
        win = 100.0 * float(np.mean(pos))
        ax.text(
            0.98,
            0.04,
            f"CR-ID better in {win:.1f}\\%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=COLORS["MUTED"],
        )
    _footer_legend(
        legend_ax,
        [
            Line2D([], [], color=COLORS["ID"], marker="o", lw=0, markersize=5, label="CR-ID > FD"),
            Line2D([], [], color=COLORS["FD"], marker="o", lw=0, markersize=5, label="FD >= CR-ID"),
        ],
        ["CR-ID > FD", "FD >= CR-ID"],
        ncol=2,
    )
    apply_thesis_axes_style(ax, grid=False)
    save_figure(fig, path)
    plt.close(fig)


def plot_n_sweep(path: Path, n_sweep_rows: List[Dict]):
    _set_exp07_plot_style(grid=False)
    fig, ax, legend_ax = _single_panel_with_footer()
    rows = sorted(n_sweep_rows, key=lambda r: int(r["n"]))
    n_vals = np.asarray([int(r["n"]) for r in rows], float)
    for name, mu_key, se_key in (("ID", "id_mean", "id_se"), ("FD", "fd_mean", "fd_se")):
        style = CURVE_STYLES[name]
        mu = np.asarray([float(r[mu_key]) for r in rows], float)
        se = np.asarray([float(r[se_key]) for r in rows], float)
        ax.errorbar(
            n_vals,
            mu,
            yerr=se,
            color=style["color"],
            lw=1.7,
            ls=style["ls"],
            marker=style["marker"],
            ms=3.8,
            capsize=2.5,
            label=style["label"],
            zorder=3,
        )
    lows = [float(r["id_mean"]) - float(r["id_se"]) for r in rows] + [
        float(r["fd_mean"]) - float(r["fd_se"]) for r in rows
    ]
    highs = [float(r["id_mean"]) + float(r["id_se"]) for r in rows] + [
        float(r["fd_mean"]) + float(r["fd_se"]) for r in rows
    ]
    ax.set_xticks(n_vals)
    ax.set_xlabel(r"System size $n$")
    ax.set_ylabel(r"Best-so-far at budget $B$  ($F/J^*$)")
    ax.set_ylim(*_interesting_ylim_arrays([np.asarray(lows), np.asarray(highs)]))
    apply_thesis_axes_style(ax, grid=False)
    handles, labels = ax.get_legend_handles_labels()
    _footer_legend(legend_ax, handles, labels, ncol=2)
    save_figure(fig, path)
    plt.close(fig)


def plot_sixpack_collage(
    path: Path,
    *,
    budget_grid: np.ndarray,
    best_id_grid: np.ndarray,
    best_fd_grid: np.ndarray,
    n_sweep_rows: List[Dict],
    auc_gain: np.ndarray,
    m_edges: np.ndarray,
    steps_id: np.ndarray,
    steps_fd: np.ndarray,
    best_final_id: np.ndarray,
    best_final_fd: np.ndarray,
    auc_id: np.ndarray,
    auc_fd: np.ndarray,
):
    _set_exp07_plot_style(grid=False)
    fig = plt.figure(figsize=grid_size(2), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.17])
    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])
            axes[r, c].set_box_aspect(1.0)

    b = np.asarray(budget_grid, float)
    mu_id, se_id = _mean_stderr(best_id_grid, axis=0)
    mu_fd, se_fd = _mean_stderr(best_fd_grid, axis=0)
    ax = axes[0, 0]
    ax.plot(b, mu_id, color=COLORS["ID"], lw=1.9)
    ax.fill_between(b, mu_id - se_id, mu_id + se_id, color=COLORS["ID"], alpha=0.14, linewidth=0)
    ax.plot(b, mu_fd, color=COLORS["FD"], lw=1.9, ls="--")
    ax.fill_between(b, mu_fd - se_fd, mu_fd + se_fd, color=COLORS["FD"], alpha=0.12, linewidth=0)
    ax.set_xlabel("Energy evals")
    ax.set_ylabel(r"Best-so-far $F/J^*$")
    ax.set_xlim(float(b[0]), float(b[-1]))
    ax.set_ylim(*_interesting_ylim_arrays([mu_id - se_id, mu_id + se_id, mu_fd - se_fd, mu_fd + se_fd]))
    apply_thesis_axes_style(ax, grid=False)
    _compact_collage_axis(ax)
    _panel_label(ax, "(A)")

    ax = axes[0, 1]
    rows = sorted(n_sweep_rows, key=lambda r: int(r["n"]))
    n_vals = np.asarray([int(r["n"]) for r in rows], float)
    for name, mu_key, se_key in (("ID", "id_mean", "id_se"), ("FD", "fd_mean", "fd_se")):
        style = CURVE_STYLES[name]
        mu = np.asarray([float(r[mu_key]) for r in rows], float)
        se = np.asarray([float(r[se_key]) for r in rows], float)
        ax.errorbar(
            n_vals,
            mu,
            yerr=se,
            color=style["color"],
            lw=1.7,
            ls=style["ls"],
            marker=style["marker"],
            ms=3.8,
            capsize=2.5,
            zorder=3,
        )
    lows = [float(r["id_mean"]) - float(r["id_se"]) for r in rows] + [
        float(r["fd_mean"]) - float(r["fd_se"]) for r in rows
    ]
    highs = [float(r["id_mean"]) + float(r["id_se"]) for r in rows] + [
        float(r["fd_mean"]) + float(r["fd_se"]) for r in rows
    ]
    ax.set_xticks(n_vals)
    ax.set_xlabel(r"Size $n$")
    ax.set_ylabel(r"Best at $B$ ($F/J^*$)")
    ax.set_ylim(*_interesting_ylim_arrays([np.asarray(lows), np.asarray(highs)]))
    apply_thesis_axes_style(ax, grid=False)
    _compact_collage_axis(ax)
    _panel_label(ax, "(B)")

    ax = axes[0, 2]
    x = np.asarray(m_edges, float)
    y = np.asarray(auc_gain, float)
    pos = y > 0.0
    ax.axhline(0.0, color=COLORS["REFERENCE"], lw=1.0, ls="--", zorder=1)
    ax.scatter(x[pos], y[pos], s=28, color=COLORS["ID"], alpha=0.88, edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(x[~pos], y[~pos], s=28, color=COLORS["FD"], alpha=0.88, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xlabel(r"Edges $|E|$")
    ax.set_ylabel(r"$\Delta \mathrm{AUC}_B$")
    apply_thesis_axes_style(ax, grid=False)
    _compact_collage_axis(ax)
    _panel_label(ax, "(C)")

    ax = axes[1, 0]
    mu_steps_id, se_steps_id = _mean_stderr(np.asarray(steps_id, float))
    mu_steps_fd, se_steps_fd = _mean_stderr(np.asarray(steps_fd, float))
    xs = np.array([0, 1], int)
    ax.bar(
        xs,
        [mu_steps_id, mu_steps_fd],
        yerr=[se_steps_id, se_steps_fd],
        capsize=3,
        width=0.62,
        color=["#f7d7dc", "#d8e5f3"],
        edgecolor=[COLORS["ID"], COLORS["FD"]],
        linewidth=1.2,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(["CR-ID", "FD"])
    ax.set_ylabel(r"Steps at $B$")
    apply_thesis_axes_style(ax, grid=False)
    _compact_collage_axis(ax)
    _panel_label(ax, "(D)")

    def _scatter_panel(ax, xvals, yvals, xlabel, ylabel, label):
        xvals = np.asarray(xvals, float)
        yvals = np.asarray(yvals, float)
        pos_local = yvals > xvals + 1e-12
        neg_local = ~pos_local
        if np.any(pos_local):
            ax.scatter(
                xvals[pos_local],
                yvals[pos_local],
                s=28,
                color=COLORS["ID"],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        if np.any(neg_local):
            ax.scatter(
                xvals[neg_local],
                yvals[neg_local],
                s=28,
                color=COLORS["FD"],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        lo = float(min(np.min(xvals), np.min(yvals)))
        hi = float(max(np.max(xvals), np.max(yvals)))
        pad = max(0.015, 0.08 * max(hi - lo, 0.12))
        lo = max(0.0, lo - pad)
        hi = min(1.02, hi + pad)
        ax.plot([lo, hi], [lo, hi], color=COLORS["REFERENCE"], lw=1.0, ls="--", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        apply_thesis_axes_style(ax, grid=False)
        _compact_collage_axis(ax)
        win = 100.0 * float(np.mean(pos_local))
        ax.text(
            0.98,
            0.04,
            f"CR-ID better in {win:.1f}\\%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color=COLORS["MUTED"],
        )
        _panel_label(ax, label)

    _scatter_panel(
        axes[1, 1],
        best_final_fd,
        best_final_id,
        r"FD at $B$ ($F/J^*$)",
        r"CR-ID at $B$ ($F/J^*$)",
        "(E)",
    )
    _scatter_panel(axes[1, 2], auc_fd, auc_id, r"FD $\mathrm{AUC}_B$", r"CR-ID $\mathrm{AUC}_B$", "(F)")

    handles = [
        Line2D([], [], color=COLORS["ID"], lw=1.9, ls="-", label="VQE + CR-ID"),
        Line2D([], [], color=COLORS["FD"], lw=1.9, ls="--", label="VQE + FD"),
        Line2D([], [], color=COLORS["ID"], marker="o", lw=0, markersize=5, label="CR-ID > FD"),
        Line2D([], [], color=COLORS["FD"], marker="o", lw=0, markersize=5, label="FD >= CR-ID"),
    ]
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=4,
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
# Main
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--render_only", action="store_true")

    # instances / seeds
    p.add_argument("--seed0", type=int, default=7)
    p.add_argument("--num_instances", type=int, default=8)

    # problem
    p.add_argument("--family", type=str, default=CANONICAL_SETUP.family, choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=CANONICAL_SETUP.periodic_K)
    p.add_argument("--n", type=int, default=CANONICAL_SETUP.n)
    p.add_argument("--p_edge", type=float, default=CANONICAL_SETUP.p_edge)
    p.add_argument("--graph_seed", type=int, default=CANONICAL_SETUP.graph_seed)
    p.add_argument("--lam_min", type=float, default=CANONICAL_SETUP.lam_min)
    p.add_argument("--lam_max", type=float, default=CANONICAL_SETUP.lam_max)
    p.add_argument("--lam0", type=float, default=CANONICAL_SETUP.lam0)
    p.add_argument("--n_sweep", type=str, default="8,9,10,11,12,13,14")

    # VQE / inner
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--inner_iters", type=int, default=28)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--shots", type=int, default=0, help="shots per energy evaluation (0 = exact)")

    # outer
    p.add_argument("--outer_max", type=int, default=400)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)

    # budget + normalization scan
    p.add_argument("--budget_evals", type=float, default=CANONICAL_SETUP.budget_evals)
    p.add_argument("--wmax_grid", type=int, default=801, help="grid points to scan per-edge w^max for J* diagnostic")
    p.add_argument("--budget_points", type=int, default=240, help="points in shared budget grid for plotting")

    # FD warm start
    p.add_argument(
        "--fd_warmstart",
        type=str,
        default="best",
        choices=["best", "center"],
        help="How to warm-start the next center solve in FD. 'best' uses the best of center/+/- params.",
    )

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out) if a.out is not None else publication_output_dir("exp07")
    out.mkdir(parents=True, exist_ok=True)
    n_sweep = [int(v) for v in parse_int_list(a.n_sweep)] if a.n_sweep.strip() else [int(a.n)]
    if not n_sweep:
        n_sweep = [int(a.n)]

    cache_dir = Path(a.cache_dir) if a.cache_dir is not None else _cache_default_dir(out)
    meta = _cache_meta(a, n_sweep)
    cached = None if a.recompute else load_exp07_cache(cache_dir, meta)

    def _summ(arr: np.ndarray):
        arr = np.asarray(arr, float)
        mu = float(np.nanmean(arr))
        se = (
            float(np.nanstd(arr, ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(arr)))))
            if np.sum(np.isfinite(arr)) > 1
            else float("nan")
        )
        return mu, se

    if cached is not None:
        budget_grid = np.asarray(cached["budget_grid"], float)
        run_rows = cached["run_rows"]
        best_id_grid_all = np.asarray(cached["best_id_grid_all"], float)
        best_fd_grid_all = np.asarray(cached["best_fd_grid_all"], float)
        auc_gain = np.asarray(cached["auc_gain"], float)
        m_edges = np.asarray(cached["m_edges"], float)
        steps_id = np.asarray(cached["steps_id"], float)
        steps_fd = np.asarray(cached["steps_fd"], float)
        n_sweep_rows = cached["n_sweep_rows"]
        print(f"[cache] Loaded exp07 payloads from {cache_dir.resolve()}")
    else:
        if a.render_only:
            raise SystemExit(f"No matching cache found in {cache_dir}")

        Z = precompute_z_big_endian(a.n)

        run_rows: List[Dict] = []
        budget_grid = np.linspace(0.0, float(a.budget_evals), int(a.budget_points))
        best_id_grid_list: List[np.ndarray] = []
        best_fd_grid_list: List[np.ndarray] = []
        auc_gain_list = []
        steps_id_list = []
        steps_fd_list = []
        m_edges_list = []

        for r in range(int(a.num_instances)):
            seed = int(a.seed0 + r)
            rng = np.random.default_rng(to_uint_seed(seed))
            edges = generate_er_instance_graph(a.n, a.p_edge, graph_seed=a.graph_seed, instance_id=seed)
            if not edges:
                continue

            m = len(edges)
            cut_mask = build_cut_mask(edges, Z)
            ZZ_edges = build_ZZ_edges(edges, Z)
            fam = FamilyEdgeWise(
                m=m, kind=a.family, lam_bounds=(a.lam_min, a.lam_max), rng=rng, periodic_K=a.periodic_K
            )

            wmax = fam.w_max(grid_points=a.wmax_grid)
            Jstar = classical_Jstar_from_wmax(cut_mask, wmax)
            if (not np.isfinite(Jstar)) or Jstar <= 0:
                Jstar = 1.0

            lam_vec0 = np.full(m, float(a.lam0), dtype=float)

            hist_id = run_outer_budget_edgewise(
                mode="ID",
                n=a.n,
                edges=edges,
                fam=fam,
                ZZ_edges=ZZ_edges,
                cut_mask=cut_mask,
                lam_vec0=lam_vec0,
                outer_max=a.outer_max,
                inner_iters=a.inner_iters,
                restarts=a.restarts,
                L=a.L,
                seed=seed,
                eta0=a.eta0,
                eta_pow=a.eta_pow,
                step_clip=a.step_clip,
                c_frac=a.c_frac,
                shots=a.shots,
                budget_evals=a.budget_evals,
                Jstar=Jstar,
                fd_warmstart=a.fd_warmstart,
            )

            hist_fd = run_outer_budget_edgewise(
                mode="FD_SPSA",
                n=a.n,
                edges=edges,
                fam=fam,
                ZZ_edges=ZZ_edges,
                cut_mask=cut_mask,
                lam_vec0=lam_vec0,
                outer_max=a.outer_max,
                inner_iters=a.inner_iters,
                restarts=a.restarts,
                L=a.L,
                seed=seed,
                eta0=a.eta0,
                eta_pow=a.eta_pow,
                step_clip=a.step_clip,
                c_frac=a.c_frac,
                shots=a.shots,
                budget_evals=a.budget_evals,
                Jstar=Jstar,
                fd_warmstart=a.fd_warmstart,
            )

            best_id_grid = _step_interp(hist_id["events_evals"], hist_id["events_best"], budget_grid)
            best_fd_grid = _step_interp(hist_fd["events_evals"], hist_fd["events_best"], budget_grid)
            best_id_grid_list.append(best_id_grid)
            best_fd_grid_list.append(best_fd_grid)

            auc_id = auc_step(hist_id["events_evals"], hist_id["events_best"], a.budget_evals)
            auc_fd = auc_step(hist_fd["events_evals"], hist_fd["events_best"], a.budget_evals)
            gain = float(auc_id - auc_fd)
            best_final_id = _best_at_budget(hist_id["events_evals"], hist_id["events_best"], a.budget_evals)
            best_final_fd = _best_at_budget(hist_fd["events_evals"], hist_fd["events_best"], a.budget_evals)
            steps_id_cur = steps_within_budget(hist_id["outer_evals"], a.budget_evals)
            steps_fd_cur = steps_within_budget(hist_fd["outer_evals"], a.budget_evals)

            auc_gain_list.append(gain)
            steps_id_list.append(steps_id_cur)
            steps_fd_list.append(steps_fd_cur)
            m_edges_list.append(m)

            run_rows.append(
                {
                    "instance": r,
                    "seed": seed,
                    "family": a.family,
                    "K": float(a.periodic_K) if a.family == "periodic" else float("nan"),
                    "graph_seed": float(a.graph_seed),
                    "n": float(a.n),
                    "p_edge": float(a.p_edge),
                    "m_edges": int(m),
                    "L": int(a.L),
                    "inner_iters": int(a.inner_iters),
                    "restarts": int(a.restarts),
                    "shots": int(a.shots),
                    "budget_evals": float(a.budget_evals),
                    "c_frac": float(a.c_frac),
                    "eta0": float(a.eta0),
                    "eta_pow": float(a.eta_pow),
                    "step_clip": float(a.step_clip),
                    "fd_warmstart": a.fd_warmstart,
                    "Jstar": float(Jstar),
                    "auc_id": float(auc_id),
                    "auc_fd": float(auc_fd),
                    "auc_gain": float(gain),
                    "best_final_id": float(best_final_id),
                    "best_final_fd": float(best_final_fd),
                    "steps_id": int(steps_id_cur),
                    "steps_fd": int(steps_fd_cur),
                    "evals_end_id": float(hist_id["evals_end"][0]),
                    "evals_end_fd": float(hist_fd["evals_end"][0]),
                }
            )

            print(
                f"[inst={r:02d} seed={seed:3d} |m|={m:2d}] "
                f"AUC_ID={auc_id:.4f} AUC_FD={auc_fd:.4f} gain={gain:+.4f}  "
                f"steps(ID,FD)=({steps_id_cur},{steps_fd_cur})  "
                f"final(ID,FD)=({best_final_id:.3f},{best_final_fd:.3f})"
            )

        if not run_rows:
            raise RuntimeError("No instances generated (graphs had 0 edges). Try increasing p_edge or changing seed0.")

        best_id_grid_all = np.vstack(best_id_grid_list)
        best_fd_grid_all = np.vstack(best_fd_grid_list)
        auc_gain = np.asarray(auc_gain_list, float)
        m_edges = np.asarray(m_edges_list, float)
        steps_id = np.asarray(steps_id_list, float)
        steps_fd = np.asarray(steps_fd_list, float)

        n_sweep_rows = []
        for n_cur in n_sweep:
            Z_n = precompute_z_big_endian(int(n_cur))
            vals_id = []
            vals_fd = []
            for r in range(int(a.num_instances)):
                seed = int(a.seed0 + r)
                rng = np.random.default_rng(to_uint_seed(seed))
                edges = generate_er_instance_graph(int(n_cur), a.p_edge, graph_seed=a.graph_seed, instance_id=seed)
                if not edges:
                    continue
                m = len(edges)
                cut_mask = build_cut_mask(edges, Z_n)
                ZZ_edges = build_ZZ_edges(edges, Z_n)
                fam = FamilyEdgeWise(
                    m=m, kind=a.family, lam_bounds=(a.lam_min, a.lam_max), rng=rng, periodic_K=a.periodic_K
                )
                wmax = fam.w_max(grid_points=a.wmax_grid)
                Jstar = classical_Jstar_from_wmax(cut_mask, wmax)
                if (not np.isfinite(Jstar)) or Jstar <= 0:
                    Jstar = 1.0
                lam_vec0 = np.full(m, float(a.lam0), dtype=float)

                hist_id = run_outer_budget_edgewise(
                    mode="ID",
                    n=int(n_cur),
                    edges=edges,
                    fam=fam,
                    ZZ_edges=ZZ_edges,
                    cut_mask=cut_mask,
                    lam_vec0=lam_vec0,
                    outer_max=a.outer_max,
                    inner_iters=a.inner_iters,
                    restarts=a.restarts,
                    L=a.L,
                    seed=seed,
                    eta0=a.eta0,
                    eta_pow=a.eta_pow,
                    step_clip=a.step_clip,
                    c_frac=a.c_frac,
                    shots=a.shots,
                    budget_evals=a.budget_evals,
                    Jstar=Jstar,
                    fd_warmstart=a.fd_warmstart,
                )
                hist_fd = run_outer_budget_edgewise(
                    mode="FD_SPSA",
                    n=int(n_cur),
                    edges=edges,
                    fam=fam,
                    ZZ_edges=ZZ_edges,
                    cut_mask=cut_mask,
                    lam_vec0=lam_vec0,
                    outer_max=a.outer_max,
                    inner_iters=a.inner_iters,
                    restarts=a.restarts,
                    L=a.L,
                    seed=seed,
                    eta0=a.eta0,
                    eta_pow=a.eta_pow,
                    step_clip=a.step_clip,
                    c_frac=a.c_frac,
                    shots=a.shots,
                    budget_evals=a.budget_evals,
                    Jstar=Jstar,
                    fd_warmstart=a.fd_warmstart,
                )
                vals_id.append(_best_at_budget(hist_id["events_evals"], hist_id["events_best"], a.budget_evals))
                vals_fd.append(_best_at_budget(hist_fd["events_evals"], hist_fd["events_best"], a.budget_evals))

            vals_id = np.asarray(vals_id, float)
            vals_fd = np.asarray(vals_fd, float)
            if vals_id.size == 0 or vals_fd.size == 0:
                continue
            id_mean, id_se = _summ(vals_id)
            fd_mean, fd_se = _summ(vals_fd)
            n_sweep_rows.append(
                {"n": int(n_cur), "id_mean": id_mean, "id_se": id_se, "fd_mean": fd_mean, "fd_se": fd_se}
            )

        payload = {
            "budget_grid": budget_grid,
            "run_rows": run_rows,
            "best_id_grid_all": best_id_grid_all,
            "best_fd_grid_all": best_fd_grid_all,
            "auc_gain": auc_gain,
            "m_edges": m_edges,
            "steps_id": steps_id,
            "steps_fd": steps_fd,
            "n_sweep_rows": n_sweep_rows,
        }
        save_exp07_cache(cache_dir, meta, payload)
        print(f"[cache] Saved exp07 payloads to {cache_dir.resolve()}")

    runs_csv = out / "runs7_edgewise_metrics.csv"
    write_csv(runs_csv, run_rows, fieldnames=list(run_rows[0].keys()))

    N = int(best_id_grid_all.shape[0])
    suf = f"{a.family}_n{a.n}_B{int(a.budget_evals)}_inner{a.inner_iters}_R{a.restarts}_S{a.shots}_seed0{a.seed0}_N{N}"

    best_final_id = np.array([row["best_final_id"] for row in run_rows], float)
    best_final_fd = np.array([row["best_final_fd"] for row in run_rows], float)
    auc_id = np.array([row["auc_id"] for row in run_rows], float)
    auc_fd = np.array([row["auc_fd"] for row in run_rows], float)
    steps_id_arr = np.array([row["steps_id"] for row in run_rows], float)
    steps_fd_arr = np.array([row["steps_fd"] for row in run_rows], float)

    fig_budget = out / f"fig7_edgewise_best_vs_evals_{suf}.{a.fmt}"
    plot_budget_curve(fig_budget, budget_grid=budget_grid, best_id_grid=best_id_grid_all, best_fd_grid=best_fd_grid_all)

    fig_n = out / f"fig7_edgewise_n_sweep_{suf}.{a.fmt}"
    plot_n_sweep(fig_n, n_sweep_rows=n_sweep_rows)

    fig_gain = out / f"fig7_edgewise_auc_gain_scatter_{suf}.{a.fmt}"
    plot_auc_gain_scatter(fig_gain, auc_gain=auc_gain, m_edges=m_edges)

    fig_steps = out / f"fig7_edgewise_steps_{suf}.{a.fmt}"
    plot_steps_bar_pretty(fig_steps, steps_id=steps_id_arr, steps_fd=steps_fd_arr)

    fig_pair_final = out / f"fig7_edgewise_final_pair_scatter_{suf}.{a.fmt}"
    plot_pair_scatter(
        fig_pair_final,
        best_final_fd,
        best_final_id,
        xlabel=r"VQE + FD at budget $B$  ($F/J^*$)",
        ylabel=r"VQE + CR-ID at budget $B$  ($F/J^*$)",
    )

    fig_pair_auc = out / f"fig7_edgewise_auc_pair_scatter_{suf}.{a.fmt}"
    plot_pair_scatter(
        fig_pair_auc,
        auc_fd,
        auc_id,
        xlabel=r"VQE + FD $\mathrm{AUC}_B$",
        ylabel=r"VQE + CR-ID $\mathrm{AUC}_B$",
    )

    fig_collage = out / f"fig7_edgewise_sixpack_{suf}.{a.fmt}"
    plot_sixpack_collage(
        fig_collage,
        budget_grid=budget_grid,
        best_id_grid=best_id_grid_all,
        best_fd_grid=best_fd_grid_all,
        n_sweep_rows=n_sweep_rows,
        auc_gain=auc_gain,
        m_edges=m_edges,
        steps_id=steps_id_arr,
        steps_fd=steps_fd_arr,
        best_final_id=best_final_id,
        best_final_fd=best_final_fd,
        auc_id=auc_id,
        auc_fd=auc_fd,
    )

    summary_rows = []
    for metric, arr_id, arr_fd in [
        (r"Final best-so-far / $J^*$", best_final_id, best_final_fd),
        (r"$\mathrm{AUC}_B$", auc_id, auc_fd),
        (r"Outer steps within budget $B$", steps_id_arr, steps_fd_arr),
    ]:
        idm, ids = _summ(arr_id)
        fdm, fds = _summ(arr_fd)
        summary_rows.append({"metric": metric, "id_mean": idm, "id_se": ids, "fd_mean": fdm, "fd_se": fds})

    table_csv = out / "table7_edgewise_summary.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "id_mean", "id_stderr", "fd_mean", "fd_stderr"])
        w.writeheader()
        for rrow in summary_rows:
            w.writerow(
                {
                    "metric": rrow["metric"],
                    "id_mean": f"{rrow['id_mean']:.6f}",
                    "id_stderr": f"{rrow['id_se']:.6f}",
                    "fd_mean": f"{rrow['fd_mean']:.6f}",
                    "fd_stderr": f"{rrow['fd_se']:.6f}",
                }
            )

    table_tex = out / "table7_edgewise_summary.tex"
    write_table_tex(
        table_tex,
        summary_rows,
        caption=(
            f"Experiment 7 (edge-wise outer parameters). "
            f"Mean$\\pm$stderr over $N={N}$ random instances for a shared evaluation budget "
            f"$B={int(a.budget_evals)}$ (energy evaluations)."
        ),
        label="tab:exp7_edgewise",
    )

    txt = out / "SUMMARY.txt"
    with txt.open("w", encoding="utf-8") as f:
        f.write("Experiment 7 — Edge-wise outer parameters (vector λ)\n")
        f.write(
            f"family={a.family} | K={a.periodic_K if a.family == 'periodic' else 'NA'} | n={a.n} | p_edge={a.p_edge} | graph_seed={a.graph_seed}\n"
        )
        f.write(f"inner_iters={a.inner_iters} | restarts={a.restarts} | L={a.L} | shots={a.shots}\n")
        f.write(f"budget_evals={a.budget_evals} | num_instances={N} | seed0={a.seed0}\n")
        f.write(f"n_sweep={','.join(str(int(v)) for v in n_sweep)}\n")
        f.write(f"FD warm-start: {a.fd_warmstart}\n\n")

        mu_gain = float(np.mean(auc_gain[np.isfinite(auc_gain)]))
        sem_gain = (
            float(np.std(auc_gain[np.isfinite(auc_gain)], ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(auc_gain)))))
            if np.sum(np.isfinite(auc_gain)) > 1
            else float("nan")
        )
        win = float(np.mean(auc_gain[np.isfinite(auc_gain)] > 0.0))

        f.write(f"ΔAUC_B mean (CR-ID−FD): {mu_gain:+.6f}  (sem={sem_gain:.6f})\n")
        f.write(f"win rate (ΔAUC>0): {100.0 * win:.2f}%\n\n")

        for rrow in summary_rows:
            f.write(
                f"{rrow['metric']}: CR-ID={rrow['id_mean']:.4f}±{rrow['id_se']:.4f} | "
                f"FD={rrow['fd_mean']:.4f}±{rrow['fd_se']:.4f}\n"
            )

        f.write("\nFiles:\n")
        f.write(f"  Budget curve:         {fig_budget.name}\n")
        f.write(f"  n-sweep:              {fig_n.name}\n")
        f.write(f"  AUC gain scatter:     {fig_gain.name}\n")
        f.write(f"  Steps figure:         {fig_steps.name}\n")
        f.write(f"  Final pair scatter:   {fig_pair_final.name}\n")
        f.write(f"  AUC pair scatter:     {fig_pair_auc.name}\n")
        f.write(f"  Sixpack collage:      {fig_collage.name}\n")
        f.write(f"  Runs CSV:             {runs_csv.name}\n")
        f.write(f"  Summary table:        {table_tex.name}\n")

    print("\nSaved to:", out.resolve())
    print("Budget fig:", fig_budget.name)
    print("n-sweep fig:", fig_n.name)
    print("Gain fig:", fig_gain.name)
    print("Steps fig:", fig_steps.name)
    print("Pair figs:", fig_pair_final.name, "/", fig_pair_auc.name)
    print("Collage:", fig_collage.name)
    print("Runs CSV:", runs_csv.name)
    print("Summary:", table_tex.name, "/", table_csv.name)
    print("Text:", txt.name)


if __name__ == "__main__":
    main()
