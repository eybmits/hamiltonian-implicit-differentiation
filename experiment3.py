#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp1_readout_realism_best_mode.py
================================

Experiment 1 (paper main/supp): Readout realism — Best-of-S and Mode
-------------------------------------------------------------------

Reviewer question:
  "Expectation value is better... but do I get better *solutions* on hardware?"

What we simulate:
  At each outer iteration t:
    1) run an inner VQE solve at the current λ_t   (approximate value query F(λ_t))
    2) perform a FIXED readout budget of S bitstring samples from the resulting state
       and compute two practical readout metrics:
         - Best-of-S:  max cut value among the S samples
         - Mode cut:   cut value of the most frequent sampled bitstring

We compare:
  - ID        : correlator-reuse implicit differentiation (CR-ImpDiff)
  - BD        : black-box finite difference on the VALUE function F(λ) (requires re-solves at λ±c)
                (Legend label requested: "VQE + BD")

Minimal paper output:
  - One 2-panel figure (Best-of-S | Mode), for the periodic family (default), aggregated over instances
    as mean ± stderr.

Key fairness choice (matches "fixed readout shots per outer step"):
  - We apply the readout budget S ONCE PER OUTER STEP for BOTH methods, at the *center* candidate state.
    BD performs extra perturbed inner re-solves internally; we do not allocate extra readout shots to those
    perturbed solves, because the goal here is to compare "what solution do I get if I read out my current candidate
    each iteration with a fixed readout budget".

Normalization:
  - We normalize cut values by a classical diagnostic upper bound
        J* = max_{λ in grid} max_{z in {0,1}^n} J(z;λ),
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

import math
import argparse
import warnings
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------------------------------------------------------------
# Silence noisy-but-harmless messages
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")
warnings.filterwarnings("ignore", message=".*regarding as unix timestamp.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

_ft = logging.getLogger("fontTools")
_ft.setLevel(logging.ERROR)
_ft.propagate = False
if not _ft.handlers:
    _ft.addHandler(logging.NullHandler())
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


# ==============================================================================
# 1) NPJ/Nature-ish plotting (same style family as your main script)
# ==============================================================================

COLORS = {
    "ID": "#D62728",  # red
    "FD": "#1F77B4",  # blue  (kept key name; legend label will say "BD")
    "GT": "#000000",
}

FULL_W = 6.95
H_TWO = 2.6

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
        "lines.linewidth": 1.6,
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

def plot_2panel_iters(path: Path,
                      best_id: np.ndarray, best_fd: np.ndarray,
                      mode_id: np.ndarray, mode_fd: np.ndarray):
    """
    best_*: (N, T) cumulative best-of-S ratios (monotone)
    mode_*: (N, T) per-step mode ratios
    """
    set_pub_style(grid=False)
    T = best_id.shape[1]
    t = np.arange(1, T + 1)

    mu_b_id, se_b_id = _mean_stderr(best_id, axis=0)
    mu_b_fd, se_b_fd = _mean_stderr(best_fd, axis=0)
    mu_m_id, se_m_id = _mean_stderr(mode_id, axis=0)
    mu_m_fd, se_m_fd = _mean_stderr(mode_fd, axis=0)

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

def plot_2panel_budget(path: Path,
                       budget_grid: np.ndarray,
                       best_id_grid: np.ndarray, best_fd_grid: np.ndarray,
                       mode_id_grid: np.ndarray, mode_fd_grid: np.ndarray):
    """
    *_grid: (N, G) traces already interpolated onto a shared budget grid.
    """
    set_pub_style(grid=False)
    b = np.asarray(budget_grid, float)

    mu_b_id, se_b_id = _mean_stderr(best_id_grid, axis=0)
    mu_b_fd, se_b_fd = _mean_stderr(best_fd_grid, axis=0)
    mu_m_id, se_m_id = _mean_stderr(mode_id_grid, axis=0)
    mu_m_fd, se_m_fd = _mean_stderr(mode_fd_grid, axis=0)

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
# 2) Instance generation and canonical family
# ==============================================================================

def to_uint_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)

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
    """
    cut_mask[z, e] = (1 - Z_i Z_j)/2 for bitstring z and edge e.
    """
    n, K = Z.shape
    m = len(edges)
    cut = np.empty((K, m), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        cut[:, e] = 0.5 * (1.0 - (Z[i] * Z[j]).astype(np.float64))
    return cut

class Family1D:
    """
    Normalized shape families with mean-zero and unit RMS over x ~ Unif[-1,1].
    w_e(λ)=wbar_e + A_e f_e(x(λ))
    """
    def __init__(self, m: int, kind: str, lam_bounds: Tuple[float, float], rng: np.random.Generator, K: int = 6):
        self.kind = kind
        self.lam_min, self.lam_max = map(float, lam_bounds)
        self.mid = 0.5 * (self.lam_min + self.lam_max)
        self.Delta = max(1e-12, self.lam_max - self.lam_min)
        self.dx_dlam = 2.0 / self.Delta

        self.wbar = rng.uniform(2.0, 3.0, size=m).astype(float)
        self.A = rng.uniform(0.3, 0.8, size=m).astype(float)

        if kind in ("linear", "quadratic"):
            self.s = rng.choice([-1.0, +1.0], size=m).astype(float)
        elif kind == "periodic":
            self.k = rng.integers(1, K + 1, size=m).astype(float)
            self.phi = rng.uniform(0.0, 2*np.pi, size=m).astype(float)
        else:
            raise ValueError("kind must be linear, quadratic, or periodic")

    def x(self, lam: float) -> float:
        return 2.0 * (float(lam) - self.mid) / self.Delta

    def f_df(self, x: float):
        x = float(x)
        if self.kind == "linear":
            c = math.sqrt(3.0)
            f = c * self.s * x
            df = c * self.s
        elif self.kind == "quadratic":
            c = math.sqrt(45.0 / 4.0)
            f = c * self.s * (x*x - 1.0/3.0)
            df = c * self.s * (2.0 * x)
        else:
            c = math.sqrt(2.0)
            arg = math.pi * self.k * x + self.phi
            f = c * np.cos(arg)
            df = c * (-math.pi * self.k) * np.sin(arg)
        return f, df

    def w(self, lam: float) -> np.ndarray:
        f, _ = self.f_df(self.x(lam))
        w = self.wbar + self.A * f
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

    def dw_dlam(self, lam: float) -> np.ndarray:
        _, df = self.f_df(self.x(lam))
        dw = self.A * df * self.dx_dlam
        return np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

def classical_Jstar_max(fam: Family1D, cut_mask: np.ndarray, grid: int) -> float:
    """
    Diagnostic max over λ-grid and all bitstrings: J*.
    """
    lams = np.linspace(fam.lam_min, fam.lam_max, grid)
    best = -1e30
    for lam in lams:
        w = fam.w(float(lam))
        with np.errstate(all="ignore"):
            vals = cut_mask @ w
        vals = np.nan_to_num(vals, nan=-1e30, posinf=-1e30, neginf=-1e30)
        b = float(np.max(vals))
        if b > best:
            best = b
    return float(best if np.isfinite(best) and best > 0 else 1.0)


# ==============================================================================
# 3) VQE (exact statevector) + inner SPSA
# ==============================================================================

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=np.complex128)

def _renorm(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.vdot(psi, psi).real)
    if (not np.isfinite(nrm)) or nrm <= 0:
        psi[:] = 1.0 / math.sqrt(psi.size)
    else:
        psi /= math.sqrt(nrm)
    return psi

def _apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        psi_r = psi.reshape([2]*n)
        psi_r = np.moveaxis(psi_r, target, 0)
        block = psi_r.reshape(2, -1).astype(np.complex128, copy=False)
        np.nan_to_num(block, copy=False)
        out = gate @ block
        np.nan_to_num(out, copy=False)
        psi_r = out.reshape([2] + [2]*(n-1))
        psi = np.moveaxis(psi_r, 0, target).reshape(-1)
    return psi

def _apply_2q(psi: np.ndarray, gate4: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    if q1 == q2:
        return psi
    with np.errstate(all="ignore"):
        a, b = sorted((q1, q2))
        psi_r = psi.reshape([2]*n)
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
        ry = params[l*(2*n): l*(2*n) + n]
        rz = params[l*(2*n) + n: (l+1)*(2*n)]
        for q in range(n):
            cy, sy = math.cos(ry[q]/2), math.sin(ry[q]/2)
            RY = np.array([[cy, -sy], [sy, cy]], dtype=np.complex128)
            psi = _apply_1q(psi, RY, q, n)
            cz, sz = np.exp(-0.5j*rz[q]), np.exp(+0.5j*rz[q])
            RZ = np.array([[cz, 0], [0, sz]], dtype=np.complex128)
            psi = _apply_1q(psi, RZ, q, n)
        for q in range(n):
            psi = _apply_2q(psi, CNOT, q, (q+1) % n, n)
        psi = _renorm(psi)
    return psi

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

def spsa_minimize(energy_fun, p0, bounds, iters, seed,
                  a=0.2, c=0.12, A=20.0, alpha=0.602, gamma=0.101):
    rng = np.random.default_rng(to_uint_seed(seed))
    p = p0.astype(float).copy()
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    best_p, best_E = p.copy(), float("inf")
    evals = 0
    for k in range(1, iters + 1):
        ak = a / ((k + A)**alpha)
        ck = c / (k**gamma)
        delta = rng.choice([-1.0, 1.0], size=p.size)
        Ep = float(energy_fun(np.clip(p + ck*delta, lo, hi)))
        Em = float(energy_fun(np.clip(p - ck*delta, lo, hi)))
        evals += 2
        ghat = (Ep - Em) / (2.0 * ck) * delta
        p = np.clip(p - ak*ghat, lo, hi)
        E = float(energy_fun(p))
        evals += 1
        if E < best_E:
            best_E, best_p = E, p.copy()
    return best_p, best_E, evals


# ==============================================================================
# 4) Readout metrics + outer loops
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

def run_outer_with_readout(n: int, edges, Z: np.ndarray, fam: Family1D, cut_mask: np.ndarray,
                           lam0: float, outer: int, inner: int, L: int,
                           seed: int, eta0: float, eta_pow: float, step_clip: float,
                           mode: str, c_frac: float, readout_shots: int):
    """
    Returns arrays per outer step:
      - evals_cum: cumulative energy evaluations
      - best_of_S: best sampled cut (S shots) at each step
      - mode_cut:  mode cut (S shots) at each step
    Note: readout is computed only at the CENTER candidate (λ_t) once per outer step.
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
        def Efun(pvec): return vqe_energy(n, edges, Z, w, pvec, L)
        params, _, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000*t)
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
            def Efun_p(pvec): return vqe_energy(n, edges, Z, w_p, pvec, L)
            p_p, _, evp = spsa_minimize(Efun_p, params, bounds, iters=inner, seed=seed + 1000*t + 17)
            evals += evp
            _Jp, _, _ = vqe_expect(n, edges, Z, w_p, p_p, L)
            evals += 1.0

            # -c solve
            w_m = fam.w(lm)
            def Efun_m(pvec): return vqe_energy(n, edges, Z, w_m, pvec, L)
            p_m, _, evm = spsa_minimize(Efun_m, params, bounds, iters=inner, seed=seed + 1000*t + 29)
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
# 5) CLI + experiment driver
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
    p.add_argument("--xaxis", type=str, default="iters", choices=["iters", "budget"],
                   help="Plot x-axis: 'iters' (outer iteration) or 'budget' (energy evaluations).")
    p.add_argument("--budget_points", type=int, default=220,
                   help="Number of points for shared budget grid when --xaxis=budget.")

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
        fam = Family1D(len(edges), a.family, (a.lam_min, a.lam_max), rng, K=a.periodic_K)

        J_star = classical_Jstar_max(fam, cut_mask, a.grid)
        if not np.isfinite(J_star) or J_star <= 0:
            J_star = 1.0

        hist_id = run_outer_with_readout(
            a.n, edges, Z, fam, cut_mask,
            a.lam0, a.outer, a.inner, a.L,
            seed, a.eta0, a.eta_pow, a.step_clip,
            "ID", a.c_frac, a.readout_shots
        )
        hist_fd = run_outer_with_readout(
            a.n, edges, Z, fam, cut_mask,
            a.lam0, a.outer, a.inner, a.L,
            seed, a.eta0, a.eta_pow, a.step_clip,
            "FD_VALUE", a.c_frac, a.readout_shots
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

        rows.append({
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
        })

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

        best_id_grid = np.vstack([_step_interp(ev, y, budget_grid) for ev, y in zip(eval_id_list, best_id_list)])
        best_fd_grid = np.vstack([_step_interp(ev, y, budget_grid) for ev, y in zip(eval_fd_list, best_fd_list)])
        mode_id_grid = np.vstack([_step_interp(ev, y, budget_grid) for ev, y in zip(eval_id_list, mode_id_list)])
        mode_fd_grid = np.vstack([_step_interp(ev, y, budget_grid) for ev, y in zip(eval_fd_list, mode_fd_list)])

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

    # Summary table (mean ± stderr over instances)
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
        f.write(f"Experiment 1 (Readout realism) | family={a.family} | n={a.n} | N={N} | S_readout={a.readout_shots} | xaxis={a.xaxis}\n")
        for met, idm, ids, fdm, fds in summary:
            f.write(f"{met}:  ID={idm:.4f}±{ids:.4f}  |  BD={fdm:.4f}±{fds:.4f}\n")
        f.write(f"Figure: {fig_path.name}\n")
        f.write(f"Runs: {csv_path.name}\n")

    print("Saved to:", outdir.resolve())
    print("Figure:", fig_path.name)
    print("Runs CSV:", csv_path.name)
    print("Summary tables:", table_csv.name, "/", table_tex.name)

if __name__ == "__main__":
    main()
