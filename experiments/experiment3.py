#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp1_readout_realism_best_mode.py
================================

Experiment 1 (paper main/supp): Readout realism --- Best-of-S and Mode
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
  - fig1_readout_best_mode_<suffix>_xIters.<fmt>   or   _xBudget.<fmt>
  - runs1_readout_metrics.csv
  - table1_readout_summary.csv / .tex
  - exp1_readout_summary.txt

Example:
  python exp1_readout_realism_best_mode.py \
    --family periodic --periodic_K 6 \
    --n 12 --p_edge 0.45 \
    --outer 30 --inner 28 --L 2 \
    --readout_shots 256 \
    --num_instances 20 \
    --xaxis iters \
    --fmt pdf --out out_exp1_readout

"""

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from paramham.families import Family1D
from paramham.graphs import generate_random_graph
from paramham.maxcut import (
    build_cut_mask,
    classical_Jstar,
)
from paramham.maxcut import (
    precompute_z as precompute_z_big_endian,
)
from paramham.metrics import mean_stderr, step_interp
from paramham.plotting import COLORS, FULL_W, _savefig, set_pub_style

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


# ==============================================================================
# Plotting
# ==============================================================================


def plot_2panel_iters(path: Path, best_id: np.ndarray, best_fd: np.ndarray, mode_id: np.ndarray, mode_fd: np.ndarray):
    """
    best_*: (N, T) cumulative best-of-S ratios (monotone)
    mode_*: (N, T) per-step mode ratios
    """
    set_pub_style(grid=False)
    T = best_id.shape[1]
    t = np.arange(1, T + 1)

    mu_b_id, se_b_id = mean_stderr(best_id, axis=0)
    mu_b_fd, se_b_fd = mean_stderr(best_fd, axis=0)
    mu_m_id, se_m_id = mean_stderr(mode_id, axis=0)
    mu_m_fd, se_m_fd = mean_stderr(mode_fd, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, H_TWO), constrained_layout=True)

    ax = axs[0]
    ax.plot(t, mu_b_id, color=COLORS["ID"], label="VQE + ID")
    ax.fill_between(t, mu_b_id - se_b_id, mu_b_id + se_b_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(t, mu_b_fd, color=COLORS["FD"], ls="--", label="VQE + BD")
    ax.fill_between(t, mu_b_fd - se_b_fd, mu_b_fd + se_b_fd, color=COLORS["FD"], alpha=0.18, linewidth=0)
    ax.set_xlabel(r"Outer iteration $t$")
    ax.set_ylabel(r"Best-of-$S$ (best cut so far) / $J^*$")
    ax.set_xlim(1, T)

    ax = axs[1]
    ax.plot(t, mu_m_id, color=COLORS["ID"], label="VQE + ID")
    ax.fill_between(t, mu_m_id - se_m_id, mu_m_id + se_m_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(t, mu_m_fd, color=COLORS["FD"], ls="--", label="VQE + BD")
    ax.fill_between(t, mu_m_fd - se_m_fd, mu_m_fd + se_m_fd, color=COLORS["FD"], alpha=0.18, linewidth=0)
    ax.set_xlabel(r"Outer iteration $t$")
    ax.set_ylabel(r"Mode cut / $J^*$")
    ax.set_xlim(1, T)

    y_all = np.concatenate([mu_b_id, mu_b_fd, mu_m_id, mu_m_fd])
    y0 = max(0.0, float(np.nanmin(y_all) - 0.04))
    y1 = min(1.05, float(np.nanmax(y_all) + 0.04))
    for ax in axs:
        ax.set_ylim(y0, y1)

    # Legend on BOTH panels (requested)
    for ax in axs:
        ax.legend(loc="lower right", frameon=False)

    _savefig(fig, path)
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
    set_pub_style(grid=False)
    b = np.asarray(budget_grid, float)

    mu_b_id, se_b_id = mean_stderr(best_id_grid, axis=0)
    mu_b_fd, se_b_fd = mean_stderr(best_fd_grid, axis=0)
    mu_m_id, se_m_id = mean_stderr(mode_id_grid, axis=0)
    mu_m_fd, se_m_fd = mean_stderr(mode_fd_grid, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, H_TWO), constrained_layout=True)

    ax = axs[0]
    ax.plot(b, mu_b_id, color=COLORS["ID"], label="VQE + ID")
    ax.fill_between(b, mu_b_id - se_b_id, mu_b_id + se_b_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(b, mu_b_fd, color=COLORS["FD"], ls="--", label="VQE + BD")
    ax.fill_between(b, mu_b_fd - se_b_fd, mu_b_fd + se_b_fd, color=COLORS["FD"], alpha=0.18, linewidth=0)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Best-of-$S$ (best cut so far) / $J^*$")
    ax.set_xlim(float(b[0]), float(b[-1]))

    ax = axs[1]
    ax.plot(b, mu_m_id, color=COLORS["ID"], label="VQE + ID")
    ax.fill_between(b, mu_m_id - se_m_id, mu_m_id + se_m_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(b, mu_m_fd, color=COLORS["FD"], ls="--", label="VQE + BD")
    ax.fill_between(b, mu_m_fd - se_m_fd, mu_m_fd + se_m_fd, color=COLORS["FD"], alpha=0.18, linewidth=0)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Mode cut / $J^*$")
    ax.set_xlim(float(b[0]), float(b[-1]))

    y_all = np.concatenate([mu_b_id, mu_b_fd, mu_m_id, mu_m_fd])
    y0 = max(0.0, float(np.nanmin(y_all) - 0.04))
    y1 = min(1.05, float(np.nanmax(y_all) + 0.04))
    for ax in axs:
        ax.set_ylim(y0, y1)

    # Legend on BOTH panels (requested)
    for ax in axs:
        ax.legend(loc="lower right", frameon=False)

    _savefig(fig, path)
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

    for t in range(1, outer + 1):
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


# ==============================================================================
# CLI + experiment driver
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="out_exp1_readout")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])

    # instances / seeds
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--num_instances", type=int, default=20)

    # problem
    p.add_argument("--family", type=str, default="quadratic", choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=6)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.45)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=4)
    p.add_argument("--grid", type=int, default=401)

    # optimization
    p.add_argument("--outer", type=int, default=30)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)

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

    return p.parse_args()


def main():
    a = parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Precompute Z once (depends only on n)
    Z = precompute_z_big_endian(a.n)

    # Collect per-instance traces
    best_id_list: List[np.ndarray] = []
    best_fd_list: List[np.ndarray] = []
    mode_id_list: List[np.ndarray] = []
    mode_fd_list: List[np.ndarray] = []
    eval_id_list: List[np.ndarray] = []
    eval_fd_list: List[np.ndarray] = []

    rows: List[Dict[str, float]] = []

    for r in range(a.num_instances):
        seed = a.seed0 + r
        rng = np.random.default_rng(to_uint_seed(seed))

        edges = generate_random_graph(a.n, a.p_edge, rng)
        if not edges:
            continue
        cut_mask = build_cut_mask(edges, Z)
        fam = Family1D(len(edges), a.family, (a.lam_min, a.lam_max), rng, periodic_K=a.periodic_K, safety_bounds=False)

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
        )

        # Normalize
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
                "family": a.family,
                "K": float(a.periodic_K) if a.family == "periodic" else float("nan"),
                "n": float(a.n),
                "p_edge": float(a.p_edge),
                "outer": float(a.outer),
                "inner": float(a.inner),
                "L": float(a.L),
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
        raise RuntimeError("No instances generated (graph had 0 edges). Try increasing p_edge or changing seed0.")

    # Stack for iteration plot
    best_id_all = np.vstack(best_id_list)
    best_fd_all = np.vstack(best_fd_list)
    mode_id_all = np.vstack(mode_id_list)
    mode_fd_all = np.vstack(mode_fd_list)

    N = best_id_all.shape[0]
    suf = f"{a.family}_n{a.n}_S{a.readout_shots}_seed0{a.seed0}_N{N}"

    if a.xaxis == "iters":
        fig_path = outdir / f"fig1_readout_best_mode_{suf}_xIters.{a.fmt}"
        plot_2panel_iters(fig_path, best_id_all, best_fd_all, mode_id_all, mode_fd_all)
    else:
        # Shared budget grid up to the smallest final budget across all runs and both methods
        B_id = np.array([row["evals_final_ID"] for row in rows], float)
        B_fd = np.array([row["evals_final_FD"] for row in rows], float)
        B_common = float(np.nanmin(np.minimum(B_id, B_fd)))
        if not np.isfinite(B_common) or B_common <= 0:
            B_common = float(np.nanmin(B_id))
        budget_grid = np.linspace(0.0, B_common, int(a.budget_points))

        best_id_grid = np.vstack([step_interp(ev, y, budget_grid) for ev, y in zip(eval_id_list, best_id_list)])
        best_fd_grid = np.vstack([step_interp(ev, y, budget_grid) for ev, y in zip(eval_fd_list, best_fd_list)])
        mode_id_grid = np.vstack([step_interp(ev, y, budget_grid) for ev, y in zip(eval_id_list, mode_id_list)])
        mode_fd_grid = np.vstack([step_interp(ev, y, budget_grid) for ev, y in zip(eval_fd_list, mode_fd_list)])

        fig_path = outdir / f"fig1_readout_best_mode_{suf}_xBudget.{a.fmt}"
        plot_2panel_budget(fig_path, budget_grid, best_id_grid, best_fd_grid, mode_id_grid, mode_fd_grid)

    # Save per-instance CSV
    import csv

    csv_path = outdir / "runs1_readout_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Summary table (mean +/- stderr over instances)
    def _summ(col: str):
        vals = np.array([row[col] for row in rows], float)
        mu = float(np.nanmean(vals))
        se = float(np.nanstd(vals, ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(vals)))))
        return mu, se

    summary = [
        ("Best-of-S final / J*",) + _summ("bestS_final_ID") + _summ("bestS_final_FD"),
        ("Mode final / J*",) + _summ("mode_final_ID") + _summ("mode_final_FD"),
        ("Best-of-S AUC (steps)",) + _summ("bestS_auc_ID") + _summ("bestS_auc_FD"),
        ("Mode AUC (steps)",) + _summ("mode_auc_ID") + _summ("mode_auc_FD"),
    ]

    table_csv = outdir / "table1_readout_summary.csv"
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ID_mean", "ID_stderr", "BD_mean", "BD_stderr"])
        for met, idm, ids, fdm, fds in summary:
            w.writerow([met, f"{idm:.6f}", f"{ids:.6f}", f"{fdm:.6f}", f"{fds:.6f}"])

    table_tex = outdir / "table1_readout_summary.tex"
    with open(table_tex, "w") as f:
        f.write("% Auto-generated by exp1_readout_realism_best_mode.py\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Metric & VQE+ID & VQE+BD\\\\\n")
        f.write("\\midrule\n")
        for met, idm, ids, fdm, fds in summary:
            f.write(f"{met} & {idm:.3f}$\\pm${ids:.3f} & {fdm:.3f}$\\pm${fds:.3f}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    txt_path = outdir / "exp1_readout_summary.txt"
    with open(txt_path, "w") as f:
        f.write(
            f"Experiment 1 (Readout realism) | family={a.family} | n={a.n} | N={N} | S_readout={a.readout_shots} | xaxis={a.xaxis}\n"
        )
        for met, idm, ids, fdm, fds in summary:
            f.write(f"{met}:  ID={idm:.4f}+/-{ids:.4f}  |  BD={fdm:.4f}+/-{fds:.4f}\n")
        f.write(f"Figure: {fig_path.name}\n")
        f.write(f"Runs: {csv_path.name}\n")

    print("Saved to:", outdir.resolve())
    print("Figure:", fig_path.name)
    print("Runs CSV:", csv_path.name)
    print("Summary tables:", table_csv.name, "/", table_tex.name)


if __name__ == "__main__":
    main()
