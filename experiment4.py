#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp1B_robustness_sweep_K.py
==========================

Experiment 1B (paper / supplement): Robustness sweep over periodic difficulty K
------------------------------------------------------------------------------

Goal (new story)
  Show that the ID advantage over the black-box FD baseline is ROBUST across a difficulty sweep.

  For the periodic response family we use the built-in "difficulty dial":
      f_e(x) = sqrt(2) cos(pi k_e x + phi_e),    k_e ~ Uniform{1,...,K}
  Larger K -> higher-frequency responses -> typically more classical envelope switches.

Metric (matched evaluation budget)
  We compare algorithms under a common evaluation budget B (energy evaluations).
  From the best-so-far value trace (vs cumulative evals), compute normalized AUC:

      AUC_alg = (1/B) ∫_0^B best_alg(e) de

  using a step-function integral over evaluation events.

  We report:
      AUC_gain = AUC_ID - AUC_FD_VALUE

  (Positive => ID is more evaluation-efficient.)

Design
  For each K in K_list:
    - sample many random instances (random graph + random family parameters)
    - run bilevel outer optimization with:
        ID        (correlator reuse hypergradient)
        FD_VALUE  (finite difference on the value function F, requiring extra inner solves)
      under the SAME evaluation budget B
    - compute per-instance AUC gain

Outputs (ready for paper/supp)
  - fig1B_auc_gain_by_K.<fmt>
      per-instance scatter (jitter) + mean ± s.e.m. per K
  - fig1B_winrate_by_K.<fmt>
      fraction of instances with AUC_gain > 0 (ID "wins")
  - exp1B_results.csv
      per-instance metrics (K, auc_id, auc_fd, auc_gain, switch_density, ...)
  - table1B_K_summary.csv
      per-K summary table
  - table1B_K_summary.tex
      LaTeX table (booktabs)
  - exp1B_summary.txt
      compact text summary

Example
  python exp1B_robustness_sweep_K.py \
    --K_list 2,3,4,5,6 --instances_per_K 8 \
    --n 12 --p_edge 0.45 \
    --inner 28 --L 2 \
    --budget_evals 5100 \
    --fmt pdf

Notes
  - Self-contained, reuses the SAME core code paths as the main demo script:
      VQE (exact statevector) + inner SPSA + outer ID vs FD_VALUE.
  - n=12 keeps the classical envelope diagnostic feasible (2^n=4096).

"""

import math
import argparse
import warnings
import logging
import csv
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ------------------------------------------------------------------------------
# Silence noisy-but-harmless fontTools timestamp logging
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")
warnings.filterwarnings("ignore", message=".*regarding as unix timestamp.*")

_ft = logging.getLogger("fontTools")
_ft.setLevel(logging.ERROR)
_ft.propagate = False
if not _ft.handlers:
    _ft.addHandler(logging.NullHandler())
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


# ==============================================================================
# 1) Plot style (match main script)
# ==============================================================================

COLORS = {
    "GT":  "#000000",
    "ID":  "#D62728",  # red
    "FD":  "#1F77B4",  # blue
    "ENV": "#000000",
}

COL_W, H_COL = 3.37, 2.8  # inches


def fig_size() -> Tuple[float, float]:
    return (COL_W, H_COL)


def set_pub_style(grid: bool = False):
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
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
    })


def _savefig(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"]:
        fig.savefig(path, dpi=600)
    else:
        fig.savefig(path)
    plt.close(fig)


# ==============================================================================
# 2) Helpers: parsing, AUC, stats, CSV/TEX
# ==============================================================================

def to_uint_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


def mean_stderr(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(x))
    if x.size < 2:
        return mu, 0.0
    se = float(np.std(x, ddof=1) / math.sqrt(x.size))
    return mu, se


def win_rate(x: np.ndarray) -> Tuple[float, int, int]:
    """
    x: array of AUC gains
    returns (rate, wins, n)
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n == 0:
        return float("nan"), 0, 0
    wins = int(np.sum(x > 0.0))
    return float(wins / n), wins, n


def auc_step(evals: np.ndarray, y: np.ndarray, B: float) -> float:
    """
    Step-function AUC for best-so-far curve y(evals), integrated on [0,B].
    y[i] is treated as constant on [evals[i], evals[i+1]).
    For x < evals[0], y(x)=0 (no area before first recorded event).
    If evals[-1] < B, the last value is held constant up to B.
    """
    evals = np.asarray(evals, dtype=float)
    y = np.asarray(y, dtype=float)
    B = float(B)
    if evals.size == 0 or B <= 0:
        return 0.0

    order = np.argsort(evals)
    evals = evals[order]
    y = y[order]

    x_prev = float(evals[0])
    if x_prev >= B:
        return 0.0

    area = 0.0
    y_prev = float(y[0])

    for i in range(1, evals.size):
        x = float(evals[i])
        if x_prev >= B:
            break
        x_clip = min(x, B)
        if x_clip > x_prev:
            area += y_prev * (x_clip - x_prev)
        x_prev = x
        y_prev = float(y[i])

    if x_prev < B:
        area += y_prev * (B - x_prev)

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


def write_tex_table(path: Path, K_rows: List[Dict], caption: str, label: str):
    """
    Writes a compact booktabs table with columns:
      K, N, switch_density (mean±se), AUC_gain (mean±se), win_rate
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r r c c c}")
    lines.append(r"\toprule")
    lines.append(r"$K$ & $N$ & switch density & AUC gain (ID--FD) & win rate \\")
    lines.append(r"\midrule")
    for rr in K_rows:
        K = int(rr["K"])
        N = int(rr["N"])
        sd = rr["switch_density_mean_se"]
        ag = rr["auc_gain_mean_se"]
        wr = rr["win_rate"]
        lines.append(f"{K:d} & {N:d} & {sd} & {ag} & {wr} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ==============================================================================
# 3) Problem instance + canonical periodic family (same as main script)
# ==============================================================================

def generate_random_graph(n: int, p_edge: float, rng: np.random.Generator):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                edges.append((i, j))
    return edges


def precompute_z_big_endian(n: int) -> np.ndarray:
    K = 1 << n
    idx = np.arange(K, dtype=np.uint32)
    Z = np.empty((n, K), dtype=np.int8)
    for q in range(n):
        bitpos = n - 1 - q
        Z[q] = 1 - 2 * ((idx >> bitpos) & 1).astype(np.int8)
    return Z


def build_cut_mask(edges, Z: np.ndarray) -> np.ndarray:
    n, K = Z.shape
    m = len(edges)
    cut = np.empty((K, m), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        cut[:, e] = 0.5 * (1.0 - (Z[i] * Z[j]).astype(np.float64))
    return cut


class Family1D:
    """
    Periodic family (difficulty via K):
      w_e(λ) = w̄_e + A_e f_e(x),   x ∈ [-1,1]
      f_e(x) = √2 cos(π k_e x + φ_e),  k_e ∈ {1,...,K}
    """
    def __init__(self, m: int, lam_bounds: Tuple[float, float], rng: np.random.Generator, K: int = 6):
        self.kind = "periodic"
        self.lam_min, self.lam_max = map(float, lam_bounds)
        self.mid = 0.5 * (self.lam_min + self.lam_max)
        self.Delta = max(1e-12, self.lam_max - self.lam_min)
        self.dx_dlam = 2.0 / self.Delta

        self.wbar = rng.uniform(2.0, 3.0, size=m).astype(float)
        self.A = rng.uniform(0.3, 0.8, size=m).astype(float)

        self.k = rng.integers(1, int(K) + 1, size=m).astype(float)
        self.phi = rng.uniform(0.0, 2 * np.pi, size=m).astype(float)

    def x(self, lam: float) -> float:
        return 2.0 * (float(lam) - self.mid) / self.Delta

    def f_df(self, x: float):
        x = float(x)
        c = math.sqrt(2.0)
        arg = math.pi * self.k * x + self.phi
        f = c * np.cos(arg)
        df = c * (-math.pi * self.k) * np.sin(arg)
        return f, df

    def w(self, lam: float) -> np.ndarray:
        f, _ = self.f_df(self.x(lam))
        w = self.wbar + self.A * f
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w.astype(float)

    def dw_dlam(self, lam: float) -> np.ndarray:
        _, df = self.f_df(self.x(lam))
        dw = self.A * df * self.dx_dlam
        dw = np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0)
        return dw.astype(float)


def classical_switch_density_and_scale(fam: Family1D, cut_mask: np.ndarray, grid_points: int) -> Tuple[int, float, float]:
    """
    Classical diagnostic from bitstrings:
      J_cl^*(λ) = max_z J(z;λ)  on a grid.
    Returns:
      switch_count, switch_density = switch_count / Δλ, J_cl_max_grid.
    """
    lams = np.linspace(fam.lam_min, fam.lam_max, int(grid_points))
    active = np.empty(lams.size, dtype=np.int32)
    J_star = np.empty(lams.size, dtype=np.float64)

    for t, lam in enumerate(lams):
        w = fam.w(float(lam)).astype(np.float64)
        with np.errstate(all="ignore"):
            vals = cut_mask @ w
        vals = np.nan_to_num(vals, nan=-1e30, posinf=-1e30, neginf=-1e30)
        idx = int(np.argmax(vals))
        active[t] = idx
        J_star[t] = float(vals[idx])

    sw = int(np.sum(active[1:] != active[:-1]))
    density = float(sw) / max(1e-12, (fam.lam_max - fam.lam_min))
    J_cl_max = float(np.max(J_star))
    return sw, density, J_cl_max


# ==============================================================================
# 4) VQE (exact statevector) + SPSA inner (same as main script)
# ==============================================================================

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=np.complex128)


def _renorm(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.vdot(psi, psi).real)
    if (not np.isfinite(nrm)) or nrm <= 0:
        psi[:] = 1.0 / np.sqrt(psi.size)
    else:
        psi /= math.sqrt(nrm)
    return psi


def _apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        psi_r = psi.reshape([2] * n)
        psi_r = np.moveaxis(psi_r, target, 0)
        block = psi_r.reshape(2, -1).astype(np.complex128, copy=False)
        np.nan_to_num(block, copy=False)
        out = gate @ block
        np.nan_to_num(out, copy=False)
        psi_r = out.reshape([2] + [2] * (n - 1))
        psi = np.moveaxis(psi_r, 0, target).reshape(-1)
    return psi


def _apply_2q(psi: np.ndarray, gate4: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    if q1 == q2:
        return psi
    with np.errstate(all="ignore"):
        a, b = sorted((q1, q2))
        psi_r = psi.reshape([2] * n)
        psi_r = np.moveaxis(psi_r, (a, b), (0, 1))
        block = psi_r.reshape(4, -1).astype(np.complex128, copy=False)
        np.nan_to_num(block, copy=False)
        out = gate4 @ block
        np.nan_to_num(out, copy=False)
        psi_r = out.reshape(2, 2, *psi_r.shape[2:])
        psi = np.moveaxis(psi_r, (0, 1), (a, b)).reshape(-1)
    return psi


def vqe_state(n: int, params: np.ndarray, L: int) -> np.ndarray:
    K = 1 << n
    psi = np.zeros(K, dtype=np.complex128)
    psi[0] = 1.0
    for l in range(L):
        ry = params[l * (2 * n): l * (2 * n) + n]
        rz = params[l * (2 * n) + n: (l + 1) * (2 * n)]
        for q in range(n):
            cy, sy = math.cos(ry[q] / 2), math.sin(ry[q] / 2)
            RY = np.array([[cy, -sy], [sy, cy]], dtype=np.complex128)
            psi = _apply_1q(psi, RY, q, n)
            cz, sz = np.exp(-0.5j * rz[q]), np.exp(+0.5j * rz[q])
            RZ = np.array([[cz, 0], [0, sz]], dtype=np.complex128)
            psi = _apply_1q(psi, RZ, q, n)
        for q in range(n):
            psi = _apply_2q(psi, CNOT, q, (q + 1) % n, n)
        psi = _renorm(psi)
    return psi


def zexp_edges(probs: np.ndarray, edges, Z: np.ndarray) -> np.ndarray:
    z = np.empty(len(edges), dtype=float)
    zz = Z.astype(float)
    with np.errstate(all="ignore"):
        for e, (i, j) in enumerate(edges):
            z[e] = float(probs @ (zz[i] * zz[j]))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(z, -1.0, 1.0)


def vqe_expect(n: int, edges, Z: np.ndarray, w: np.ndarray, params: np.ndarray, L: int):
    psi = vqe_state(n, params, L)
    probs = (psi.conj() * psi).real.astype(float)
    s = float(np.sum(probs))
    probs = probs / s if (np.isfinite(s) and s > 0) else np.full_like(probs, 1.0 / probs.size)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    z = zexp_edges(probs, edges, Z)
    p_cut = 0.5 * (1.0 - z)
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return J, psi, z


def vqe_energy(n: int, edges, Z: np.ndarray, w: np.ndarray, params: np.ndarray, L: int) -> float:
    J, _, _ = vqe_expect(n, edges, Z, w, params, L)
    return -J


def spsa_minimize(energy_fun, p0: np.ndarray, bounds, iters: int, seed: int,
                  a: float = 0.2, c: float = 0.12, A: float = 20.0,
                  alpha: float = 0.602, gamma: float = 0.101):
    rng = np.random.default_rng(to_uint_seed(seed))
    p = p0.astype(float).copy()
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    best_p, best_E = p.copy(), float("inf")
    evals = 0
    for k in range(1, iters + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k ** gamma)
        delta = rng.choice([-1.0, 1.0], size=p.size)
        Ep = float(energy_fun(np.clip(p + ck * delta, lo, hi)))
        Em = float(energy_fun(np.clip(p - ck * delta, lo, hi)))
        evals += 2
        ghat = (Ep - Em) / (2.0 * ck) * delta
        p = np.clip(p - ak * ghat, lo, hi)
        E = float(energy_fun(p))
        evals += 1
        if E < best_E:
            best_E, best_p = E, p.copy()
    return best_p, best_E, evals


# ==============================================================================
# 5) Bilevel: value eval + outer loops (ID vs FD_VALUE) under eval budget
# ==============================================================================

def value_eval(n: int, edges, Z: np.ndarray, fam: Family1D,
               lam: float, params0: np.ndarray, L: int, inner_iters: int, seed: int):
    """
    One approximate evaluation of the value function F(λ):
      - inner solve at λ (SPSA on energy)
      - return J(φ̂*(λ),λ), params*, zexp, cost (energy eval count)
    """
    w = fam.w(lam)
    bounds = [(-math.pi, math.pi)] * params0.size

    def Efun(pvec):
        return vqe_energy(n, edges, Z, w, pvec, L)

    params, _, ev_in = spsa_minimize(Efun, params0, bounds, iters=inner_iters, seed=seed)
    J, _, zexp = vqe_expect(n, edges, Z, w, params, L)
    cost = float(ev_in + 1)  # +1 counts the final expectation evaluation
    return float(J), params, zexp, cost


def run_bilevel_outer(n: int, edges, Z: np.ndarray, fam: Family1D,
                      lam0: float, L: int, inner: int,
                      eta0: float, eta_pow: float, step_clip: float,
                      mode: str, seed: int,
                      c_frac: float, budget_evals: float, outer_max: int) -> Dict[str, np.ndarray]:
    """
    Runs ID or FD_VALUE until reaching the evaluation budget (or outer_max).

    Returns event-level curves:
      evals_evt[k] = cumulative evals after the k-th completed value-evaluation
      best_evt[k]  = best-so-far value after that evaluation
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))
    params = np.zeros(2 * n * L, dtype=float)
    c = float(c_frac * (lam_max - lam_min))

    evals = 0.0
    best = -1e18

    evals_evt: List[float] = []
    best_evt: List[float] = []

    for t in range(1, int(outer_max) + 1):
        # central value evaluation at current λ
        Jc, params_c, zexp_c, cost_c = value_eval(
            n, edges, Z, fam, lam, params, L, inner, seed=seed + 1000 * t + 0
        )
        params = params_c
        evals += cost_c
        best = max(best, Jc)
        evals_evt.append(evals)
        best_evt.append(best)

        if evals >= budget_evals:
            break

        if mode == "ID":
            p_cut = 0.5 * (1.0 - zexp_c)
            g = float(fam.dw_dlam(lam) @ p_cut)

            eta = eta0 / (t ** eta_pow)
            step = float(np.clip(eta * g, -step_clip, step_clip))
            lam = float(np.clip(lam + step, lam_min, lam_max))

        elif mode == "FD_VALUE":
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))

            # +c
            Jp, _, _, cost_p = value_eval(
                n, edges, Z, fam, lp, params.copy(), L, inner, seed=seed + 1000 * t + 1
            )
            evals += cost_p
            best = max(best, Jp)
            evals_evt.append(evals)
            best_evt.append(best)

            if evals >= budget_evals:
                break

            # -c
            Jm, _, _, cost_m = value_eval(
                n, edges, Z, fam, lm, params.copy(), L, inner, seed=seed + 1000 * t + 2
            )
            evals += cost_m
            best = max(best, Jm)
            evals_evt.append(evals)
            best_evt.append(best)

            if evals >= budget_evals:
                break

            g = (Jp - Jm) / (2.0 * c) if c > 0 else 0.0
            eta = eta0 / (t ** eta_pow)
            step = float(np.clip(eta * g, -step_clip, step_clip))
            lam = float(np.clip(lam + step, lam_min, lam_max))

        else:
            raise ValueError("mode must be 'ID' or 'FD_VALUE'")

    return {
        "evals_evt": np.array(evals_evt, dtype=float),
        "best_evt":  np.array(best_evt, dtype=float),
    }


# ==============================================================================
# 6) Plots for the robustness story
# ==============================================================================

def plot_auc_gain_by_K(path: Path, rows: List[Dict], K_list: List[int]):
    """
    Main robustness figure:
      per-instance AUC gain scatter (jittered) + mean ± s.e.m. per K.
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    rng_jit = np.random.default_rng(12345)

    # Collect per-K
    for K in K_list:
        ys = np.array([r["auc_gain"] for r in rows if int(r["periodic_K"]) == int(K)], dtype=float)
        if ys.size == 0:
            continue
        x = float(K)
        jitter = rng_jit.uniform(-0.12, 0.12, size=ys.size)
        ax.scatter(x + jitter, ys, s=18, color="#555555", alpha=0.55,
                   edgecolors="none", label="_nolegend_")

        mu, se = mean_stderr(ys)
        ax.errorbar([x], [mu], yerr=[se], fmt="o", color=COLORS["ID"],
                    markersize=5.5, capsize=2.5, elinewidth=1.0, label="_nolegend_")

    ax.axhline(0.0, color=COLORS["GT"], lw=1.0, ls=":")

    ax.set_xlabel(r"Periodic difficulty $K$")
    ax.set_ylabel(r"AUC gain $\mathrm{AUC}_{\mathrm{ID}}-\mathrm{AUC}_{\mathrm{FD}}$")

    ax.set_xticks(K_list)
    ax.set_xlim(min(K_list) - 0.6, max(K_list) + 0.6)

    # Compact legend (proxy artists)
    h1 = ax.scatter([], [], s=18, color="#555555", alpha=0.55, edgecolors="none", label="Instances")
    h2 = ax.errorbar([], [], yerr=[], fmt="o", color=COLORS["ID"], markersize=5.5,
                     capsize=2.5, elinewidth=1.0, label=r"Mean $\pm$ s.e.m.")
    ax.legend(loc="upper right", frameon=False, handles=[h1, h2])

    _savefig(fig, path)


def plot_winrate_by_K(path: Path, rows: List[Dict], K_list: List[int]):
    """
    Secondary robustness plot:
      win rate = fraction of instances with AUC_gain > 0.
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    rates = []
    errs = []
    for K in K_list:
        ys = np.array([r["auc_gain"] for r in rows if int(r["periodic_K"]) == int(K)], dtype=float)
        p, wins, n = win_rate(ys)
        rates.append(p)
        # simple binomial stderr
        se = math.sqrt(max(0.0, p * (1.0 - p) / max(1, n)))
        errs.append(se)

    xs = np.array(K_list, dtype=float)
    rates = np.array(rates, dtype=float)
    errs = np.array(errs, dtype=float)

    ax.errorbar(xs, rates, yerr=errs, fmt="o", color=COLORS["ID"],
                capsize=2.5, elinewidth=1.0)
    ax.axhline(0.5, color="#888888", lw=0.8, ls=":")
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel(r"Periodic difficulty $K$")
    ax.set_ylabel(r"Win rate $\Pr(\mathrm{AUC\ gain}>0)$")
    ax.set_xticks(K_list)
    ax.set_xlim(min(K_list) - 0.6, max(K_list) + 0.6)

    _savefig(fig, path)


# ==============================================================================
# 7) Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="out_exp1B")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])

    # sweep
    p.add_argument("--K_list", type=str, default="2,3,4,5,6")
    p.add_argument("--instances_per_K", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)

    # problem size
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.45)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=0.8)
    p.add_argument("--grid", type=int, default=401)

    # VQE / bilevel
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--inner", type=int, default=28)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)

    # matched budget
    p.add_argument("--budget_evals", type=float, default=None,
                   help="common eval-budget cap (energy evals); default matches ID outer_ref*(3*inner+1)")
    p.add_argument("--id_outer_ref", type=int, default=60)
    p.add_argument("--outer_max", type=int, default=5000)

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    K_list = parse_int_list(a.K_list)
    if not K_list:
        raise ValueError("K_list is empty")

    # default budget: approximate ID budget for id_outer_ref steps
    if a.budget_evals is None:
        # inner SPSA: 3 evals/iter, plus 1 for final J -> 3*inner + 1 per value-eval
        a.budget_evals = float(a.id_outer_ref * (3 * a.inner + 1))
    B = float(a.budget_evals)

    Z = precompute_z_big_endian(int(a.n))

    rows: List[Dict] = []
    total = len(K_list) * int(a.instances_per_K)
    done = 0

    for K in K_list:
        for r in range(int(a.instances_per_K)):
            done += 1
            inst_seed = int(a.seed + 100000 * int(K) + r)
            rng = np.random.default_rng(to_uint_seed(inst_seed))

            # sample non-empty graph
            edges = generate_random_graph(int(a.n), float(a.p_edge), rng)
            retry = 0
            while (not edges) and retry < 50:
                retry += 1
                edges = generate_random_graph(int(a.n), float(a.p_edge), rng)
            if not edges:
                print(f"[warn] skip instance (empty graph) K={K} r={r}")
                continue

            cut_mask = build_cut_mask(edges, Z)
            fam = Family1D(len(edges), (a.lam_min, a.lam_max), rng, K=int(K))

            sw_count, sw_density, J_cl_max = classical_switch_density_and_scale(fam, cut_mask, int(a.grid))
            if (not np.isfinite(J_cl_max)) or (J_cl_max <= 0.0):
                J_cl_max = 1.0

            # run algorithms under matched eval budget
            hist_id = run_bilevel_outer(
                a.n, edges, Z, fam, a.lam0, a.L, a.inner,
                a.eta0, a.eta_pow, a.step_clip,
                mode="ID", seed=inst_seed, c_frac=a.c_frac, budget_evals=B, outer_max=a.outer_max
            )
            hist_fd = run_bilevel_outer(
                a.n, edges, Z, fam, a.lam0, a.L, a.inner,
                a.eta0, a.eta_pow, a.step_clip,
                mode="FD_VALUE", seed=inst_seed, c_frac=a.c_frac, budget_evals=B, outer_max=a.outer_max
            )

            # Normalize best-so-far values by classical max on grid, then compute AUC/B
            y_id = hist_id["best_evt"] / J_cl_max
            y_fd = hist_fd["best_evt"] / J_cl_max

            # guard (grid max can slightly underapprox the true max)
            y_id = np.clip(y_id, 0.0, 1.5)
            y_fd = np.clip(y_fd, 0.0, 1.5)

            auc_id = auc_step(hist_id["evals_evt"], y_id, B) / B
            auc_fd = auc_step(hist_fd["evals_evt"], y_fd, B) / B
            auc_gain = float(auc_id - auc_fd)

            rows.append({
                "periodic_K": int(K),
                "instance_seed": int(inst_seed),
                "n": int(a.n),
                "m_edges": int(len(edges)),
                "switch_count": int(sw_count),
                "switch_density": float(sw_density),
                "Jcl_max_grid": float(J_cl_max),
                "budget_evals": float(B),
                "auc_id": float(auc_id),
                "auc_fd": float(auc_fd),
                "auc_gain": float(auc_gain),
                "id_evals_end": float(hist_id["evals_evt"][-1]) if hist_id["evals_evt"].size else 0.0,
                "fd_evals_end": float(hist_fd["evals_evt"][-1]) if hist_fd["evals_evt"].size else 0.0,
                "id_best_end_norm": float(y_id[-1]) if y_id.size else 0.0,
                "fd_best_end_norm": float(y_fd[-1]) if y_fd.size else 0.0,
                "id_wins_auc": int(auc_gain > 0.0),
            })

            # lightweight progress
            if done % max(1, total // 10) == 0 or done == total:
                print(f"[{done:4d}/{total}] K={K:2d} seed={inst_seed}  switch_density={sw_density:.2f}  auc_gain={auc_gain:+.3f}")

    if not rows:
        raise RuntimeError("No results collected (all instances skipped?)")

    # Per-K summary rows
    K_summary_rows: List[Dict] = []
    txt_lines = []
    txt_lines.append(f"N_total={len(rows)} | budget_evals={B:.1f}")
    txt_lines.append("")

    all_gains = np.array([r["auc_gain"] for r in rows], dtype=float)
    all_mu, all_se = mean_stderr(all_gains)
    all_wr, all_wins, all_n = win_rate(all_gains)
    txt_lines.append(f"Overall AUC_gain mean ± s.e.m.: {all_mu:+.4f} ± {all_se:.4f}")
    txt_lines.append(f"Overall win rate (AUC_gain>0):  {all_wr*100:.1f}%  ({all_wins}/{all_n})")
    txt_lines.append("")
    txt_lines.append("Per-K summary (mean ± s.e.m.):")
    txt_lines.append("K, N, switch_density, AUC_gain, win_rate")

    for K in K_list:
        subset = [r for r in rows if int(r["periodic_K"]) == int(K)]
        if not subset:
            continue
        sd = np.array([r["switch_density"] for r in subset], dtype=float)
        ag = np.array([r["auc_gain"] for r in subset], dtype=float)

        sd_mu, sd_se = mean_stderr(sd)
        ag_mu, ag_se = mean_stderr(ag)
        wr, wins, n = win_rate(ag)

        txt_lines.append(f"{K:2d}, {n:2d}, {sd_mu:.3f} ± {sd_se:.3f}, {ag_mu:+.4f} ± {ag_se:.4f}, {wr*100:5.1f}% ({wins}/{n})")

        K_summary_rows.append({
            "K": int(K),
            "N": int(n),
            "switch_density_mean": float(sd_mu),
            "switch_density_se": float(sd_se),
            "auc_gain_mean": float(ag_mu),
            "auc_gain_se": float(ag_se),
            "win_rate": float(wr),
            "wins": int(wins),
        })

    # Write per-instance CSV
    write_csv(out / "exp1B_results.csv", rows)

    # Write per-K table CSV
    table_rows_csv = []
    for rr in K_summary_rows:
        table_rows_csv.append({
            "K": rr["K"],
            "N": rr["N"],
            "switch_density_mean": rr["switch_density_mean"],
            "switch_density_se": rr["switch_density_se"],
            "auc_gain_mean": rr["auc_gain_mean"],
            "auc_gain_se": rr["auc_gain_se"],
            "win_rate": rr["win_rate"],
            "wins": rr["wins"],
        })
    write_csv(out / "table1B_K_summary.csv", table_rows_csv)

    # Write LaTeX table (booktabs)
    tex_rows = []
    for rr in K_summary_rows:
        tex_rows.append({
            "K": rr["K"],
            "N": rr["N"],
            "switch_density_mean_se": rf"${rr['switch_density_mean']:.3f}\,\pm\,{rr['switch_density_se']:.3f}$",
            "auc_gain_mean_se": rf"${rr['auc_gain_mean']:+.4f}\,\pm\,{rr['auc_gain_se']:.4f}$",
            "win_rate": rf"${rr['win_rate']*100:.1f}\%$",
        })
    write_tex_table(
        out / "table1B_K_summary.tex",
        tex_rows,
        caption="Robustness sweep over periodic difficulty $K$ (mean $\\pm$ s.e.m. over instances). "
                "Switch density is computed from the classical envelope identity sequence on a $\\lambda$-grid.",
        label="tab:exp1B_K_sweep"
    )

    # Write text summary
    (out / "exp1B_summary.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    # Plots
    plot_auc_gain_by_K(out / f"fig1B_auc_gain_by_K.{a.fmt}", rows, K_list)
    plot_winrate_by_K(out / f"fig1B_winrate_by_K.{a.fmt}", rows, K_list)

    print("\n".join(txt_lines))
    print("Saved to:", out.resolve())


if __name__ == "__main__":
    main()
