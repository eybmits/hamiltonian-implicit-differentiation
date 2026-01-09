#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_allinone_id_vs_fd_bilevel_pubplots.py
=========================================

Single-file, paper-grade demo: VQE + parametrized Max-Cut family + ID vs FD on the *value function* F(λ)
---------------------------------------------------------------------------------------------------------

This script is the "Tier-0 / 0B" core demo for the paper / thesis:
  - 0B mechanistic 1D explanation: classical envelope + X-ray spectrum with switch points
  - 0A budget efficiency: best-so-far value versus cumulative energy evaluations
  - (NEW) budget advantage plot: cumulative energy evaluations versus outer iteration

NOTATION (aligned with the write-up)
- Inner objective:      J(φ,λ) = ⟨H(λ)⟩_φ,   H(λ)=Σ_e w_e(λ) C_e,  C_e=(I-Z_i Z_j)/2.
- Value function:       F(λ)   = max_φ J(φ,λ)     (bilevel target; approximated by the inner solve).
- Classical diagnostic: J_cl^*(λ) = max_z J(z;λ)  (bitstring envelope; used only as a diagnostic scale).

FAIRNESS / CORE LOGIC (Claim 1)
- Evaluating F(λ) is expensive: it requires an (approx.) inner solve at λ.
- ID (implicit diff / envelope principle; correlator reuse):
    after ONE inner solve at λ_t, compute  g_ID(λ_t)=∂_λ J(φ*(λ_t),λ_t)=Σ_e w'_e(λ_t) p_e,
    reusing the same ZZ correlators that were already measured for J.
- FD baseline is black-box on the value function:
    g_FD(λ_t) ≈ [F(λ_t+c) - F(λ_t-c)]/(2c),
    where each F(·) query requires an additional inner solve at λ±c.
  => FD has ~2 additional inner solves per outer step vs ID (≈3× evaluation cost in this setup).

PLOTTING (NPJ/Nature-ish, no titles)
- Fig1: envelope + trajectories with a shaded zoom band + inset (BEST points, semi-transparent).
- Fig2: X-ray spectrum (background + active branches) + switch points.
- Fig3: best-so-far value vs cumulative energy evaluations, truncated to a common budget B.
- Fig4: cumulative energy evaluations vs outer iteration (shows per-step cost ratio).

Run (example):
  python demo_allinone_id_vs_fd_bilevel_pubplots.py --kind periodic --n 12 --outer 100 --inner 30 --fmt pdf
"""

import math
import argparse
import warnings
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ------------------------------------------------------------------------------
# Silence known noisy-but-harmless messages (fontTools timestamp logging)
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
# 1) Minimal NPJ/Nature-ish plotting
# ==============================================================================

COLORS = {
    "GT":  "#000000",
    "ID":  "#D62728",  # red
    "FD":  "#1F77B4",  # blue
    "ENV": "#000000",
}

COL_W, H_COL = 3.37, 2.8  # inches (single-column-ish)


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
        "savefig.transparent": False,  # avoids alpha/transparent PDF artefacts
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _savefig(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    # high dpi helps rasterized artists even for PDFs
    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"]:
        fig.savefig(path, dpi=600)
    else:
        fig.savefig(path)


def plot_envelope_and_expectations(path: Path, lam_grid, J_cl_star, hist_id, hist_fd):
    """
    Fig1:
      - main axis: full classical envelope J_cl^*(λ) + all trajectory points
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    lam_grid = np.asarray(lam_grid, dtype=float)
    J_cl_star = np.asarray(J_cl_star, dtype=float)

    # Full classical envelope (diagnostic)
    ax.plot(lam_grid, J_cl_star, color=COLORS["ENV"], lw=2.0,
            label=r"Envelope $J_{\mathrm{cl}}^*(\lambda)$", zorder=1)

    # Scatter full trajectories (no connecting lines in (λ, value))
    ax.scatter(hist_id["lam_pre"], hist_id["J"], s=10, marker="o", color=COLORS["ID"],
               alpha=0.25, edgecolors="none", label="VQE + ID", zorder=2)
    ax.scatter(hist_fd["lam_pre"], hist_fd["J"], s=10, marker="s", color=COLORS["FD"],
               alpha=0.25, edgecolors="none", label=r"VQE + FD (black-box $F$)", zorder=2)

    ax.set_xlabel(r"Control parameter $\lambda$")
    ax.set_ylabel(r"Value estimate $\hat F(\lambda)$")
    ax.set_xlim(float(lam_grid[0]), float(lam_grid[-1]))
    ax.legend(loc="lower left", frameon=False)

    _savefig(fig, path)
    plt.close(fig)

def _truncate_step_to_budget(evals: np.ndarray, y: np.ndarray, budget: float):
    """
    Keep the prefix of (evals,y) up to 'budget', then append a final point at exactly 'budget'
    holding the last value. Used for fair comparison at a common evaluation budget.
    """
    evals = np.asarray(evals, float)
    y = np.asarray(y, float)
    budget = float(budget)

    m = evals <= budget
    ev = evals[m]
    yy = y[m]

    if ev.size == 0:
        # nothing before budget -> hold the first value across the budget range
        return np.array([0.0, budget]), np.array([y[0], y[0]])

    # append a final point exactly at budget (hold last value)
    if ev[-1] < budget:
        ev = np.append(ev, budget)
        yy = np.append(yy, yy[-1])

    return ev, yy


def _steps_completed_within_budget(evals_cum: np.ndarray, budget: float) -> int:
    evals_cum = np.asarray(evals_cum, float)
    budget = float(budget)
    # number of outer iterations whose completed cost is <= budget
    return int(np.searchsorted(evals_cum, budget, side="right"))


def plot_bestJ_vs_evals(path: Path, hist_id, hist_fd, J_cl_max=None, budget: float = None):
    """
    Fig3: Best-so-far value vs cumulative energy evaluations.

    If 'budget' is provided, both curves are truncated to that common budget.
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    if budget is None:
        budget = float(min(hist_id["evals_cum"][-1], hist_fd["evals_cum"][-1]))
    budget = float(budget)

    ev_id, y_id = _truncate_step_to_budget(hist_id["evals_cum"], hist_id["J_best"], budget)
    ev_fd, y_fd = _truncate_step_to_budget(hist_fd["evals_cum"], hist_fd["J_best"], budget)

    ax.plot(ev_id, y_id, color=COLORS["ID"], lw=1.7, label="VQE + ID")
    ax.plot(ev_fd, y_fd, color=COLORS["FD"], lw=1.7, ls="--",
            label=r"VQE + FD (black-box $F$)")

    if J_cl_max is not None and np.isfinite(J_cl_max):
        ax.axhline(float(J_cl_max), color=COLORS["GT"], lw=1.0, ls=":",
                   label=r"$\max_\lambda J_{\mathrm{cl}}^*(\lambda)$ (grid)")

    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Best-so-far value $\hat F$")
    ax.set_xlim(0.0, budget)

    # Legend only (no annotation to avoid overlap)
    ax.legend(loc="lower right", frameon=False)

    _savefig(fig, path)
    plt.close(fig)


def plot_evals_vs_outer(path: Path, hist_id, hist_fd, budget: float):
    """
    Fig4: Cumulative energy evaluations vs outer iteration.

    Shows the per-iteration cost gap directly.
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    budget = float(budget)

    t_id = np.arange(1, hist_id["evals_cum"].size + 1, dtype=int)
    t_fd = np.arange(1, hist_fd["evals_cum"].size + 1, dtype=int)

    ax.plot(t_id, hist_id["evals_cum"], color=COLORS["ID"], lw=1.6, label="VQE + ID")
    ax.plot(t_fd, hist_fd["evals_cum"], color=COLORS["FD"], lw=1.6, ls="--",
            label=r"VQE + FD (black-box $F$)")

    ax.axhline(budget, color=COLORS["GT"], lw=1.0, ls=":")

    # Budget intersection (completed steps within budget)
    tid_B = _steps_completed_within_budget(hist_id["evals_cum"], budget)
    tfd_B = _steps_completed_within_budget(hist_fd["evals_cum"], budget)

    # Markers (subtle)
    if tid_B > 0:
        ax.scatter([tid_B], [hist_id["evals_cum"][tid_B - 1]], s=18, color=COLORS["ID"],
                   edgecolors="white", linewidth=0.5, zorder=5)
    if tfd_B > 0:
        ax.scatter([tfd_B], [hist_fd["evals_cum"][tfd_B - 1]], s=18, color=COLORS["FD"],
                   edgecolors="white", linewidth=0.5, zorder=5)

    ax.set_xlabel(r"Outer iteration $t$")
    ax.set_ylabel("Energy evaluations")
    ax.set_xlim(0.0, max(t_id[-1], t_fd[-1]) + 1.0)
    ax.set_ylim(0.0, 1.05 * float(max(hist_id["evals_cum"][-1], hist_fd["evals_cum"][-1], budget)))

    ax.legend(loc="upper left", frameon=False)

    _savefig(fig, path)
    plt.close(fig)


def plot_lambda_trajectories(path: Path, hist_id, hist_fd, lam_true=None, lam_bounds=None):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
    ax.plot(np.arange(len(hist_id["lam_pre"])), hist_id["lam_pre"], color=COLORS["ID"], lw=1.4, label="VQE + ID")
    ax.plot(np.arange(len(hist_fd["lam_pre"])), hist_fd["lam_pre"], color=COLORS["FD"], lw=1.4, ls="--",
            label=r"VQE + FD (black-box $F$)")
    if lam_true is not None and np.isfinite(lam_true):
        ax.axhline(float(lam_true), color=COLORS["GT"], lw=1.0, ls=":", label=r"$\lambda^*$ (grid)")
    if lam_bounds is not None:
        ax.set_ylim(lam_bounds)
    ax.set_xlabel("Outer iteration $t$")
    ax.set_ylabel(r"$\lambda_t$")
    ax.legend(loc="best", frameon=False)
    _savefig(fig, path)
    plt.close(fig)


def plot_xray(path: Path, lams, all_J, J_cl_star, active_ids, switch_lams, switch_vals):
    """
    Old-style X-ray:
      - background: all non-active lines (rasterized)
      - active: all unique envelope-contributing lines (rasterized)
      - legend: only envelope and switch points
      - y-range: zoomed out (full spread)
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    K = all_J.shape[0]
    uniq = np.unique(active_ids)
    is_active = np.zeros(K, dtype=bool)
    is_active[uniq] = True

    ax.plot(lams, all_J[~is_active].T, color="#aaaaaa", alpha=0.05, lw=0.30,
            rasterized=True, label="_nolegend_")
    ax.plot(lams, all_J[is_active].T, color=COLORS["FD"], alpha=0.55, lw=1.0,
            rasterized=True, label="_nolegend_")

    ax.plot(lams, J_cl_star, color=COLORS["ENV"], lw=2.0, ls="--",
            label=r"Envelope $J_{\mathrm{cl}}^*(\lambda)$", zorder=5)
    if switch_lams.size:
        ax.scatter(switch_lams, switch_vals, color=COLORS["ID"], s=26,
                   edgecolors="white", linewidth=0.6, label="Switch points", zorder=10)

    ymin = float(np.min(all_J))
    ymax = float(np.max(all_J))
    yr = max(1e-9, ymax - ymin)
    ax.set_ylim(ymin - 0.02 * yr, ymax + 0.02 * yr)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$J(z;\lambda)$")
    ax.legend(loc="lower right", frameon=False)
    _savefig(fig, path)
    plt.close(fig)


# ==============================================================================
# 2) Problem instance + canonical family
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
    n, K = Z.shape
    m = len(edges)
    cut = np.empty((K, m), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        cut[:, e] = 0.5 * (1.0 - (Z[i] * Z[j]).astype(np.float64))
    return cut


class Family1D:
    """
    Canonical bounded families:
      w_e(λ) = w̄_e + A_e f_e(x),   x = 2(λ-mid)/Δ ∈ [-1,1]

    with mean-zero, RMS-normalized response functions:
      linear:     f = √3 s x
      quadratic:  f = √(45/4) s (x^2 - 1/3)
      periodic:   f = √2 cos(π k x + φ) ,   k ∈ {1,...,K}
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
            self.k = None
            self.phi = None
        else:
            self.s = None
            self.k = rng.integers(1, K + 1, size=m).astype(float)
            self.phi = rng.uniform(0.0, 2 * np.pi, size=m).astype(float)

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
            f = c * self.s * (x * x - 1.0 / 3.0)
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
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w.astype(float)

    def dw_dlam(self, lam: float) -> np.ndarray:
        _, df = self.f_df(self.x(lam))
        dw = self.A * df * self.dx_dlam
        dw = np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0)
        return dw.astype(float)


# ==============================================================================
# 3) VQE (exact) + SPSA inner  (with robust matmul)
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
    zz = (Z.astype(float))
    with np.errstate(all="ignore"):
        for e, (i, j) in enumerate(edges):
            z[e] = float(probs @ (zz[i] * zz[j]))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(z, -1.0, 1.0)


def vqe_expect(n, edges, Z: np.ndarray, w: np.ndarray, params: np.ndarray, L: int):
    psi = vqe_state(n, params, L)
    probs = (psi.conj() * psi).real.astype(float)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or (s <= 0):
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    z = zexp_edges(probs, edges, Z)
    p_cut = 0.5 * (1.0 - z)
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return J, psi, z


def vqe_energy(n, edges, Z: np.ndarray, w: np.ndarray, params: np.ndarray, L: int) -> float:
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
# 4) Classical envelope/spectrum + outer loops (ID vs FD on F)
# ==============================================================================

def envelope_spectrum(fam: Family1D, cut_mask: np.ndarray, grid_points: int):
    lams = np.linspace(fam.lam_min, fam.lam_max, grid_points)
    K = cut_mask.shape[0]
    all_J = np.empty((K, grid_points), dtype=np.float32)
    J_cl_star = np.empty(grid_points, dtype=np.float32)
    active = np.empty(grid_points, dtype=np.int32)
    for t, lam in enumerate(lams):
        w = fam.w(float(lam)).astype(np.float64)
        with np.errstate(all="ignore"):
            vals = cut_mask @ w
        vals = np.nan_to_num(vals, nan=-1e30, posinf=-1e30, neginf=-1e30).astype(np.float32)
        all_J[:, t] = vals
        idx = int(np.argmax(vals))
        active[t] = idx
        J_cl_star[t] = vals[idx]
    sw = np.where(active[1:] != active[:-1])[0] + 1
    return lams, all_J, J_cl_star, active, lams[sw], J_cl_star[sw]


def readout_metrics(rng, psi, cut_vals, shots):
    if shots <= 0:
        return float("nan"), float("nan")
    probs = (psi.conj() * psi).real.astype(float)
    s = float(np.sum(probs))
    probs = probs / s if (np.isfinite(s) and s > 0) else np.full_like(probs, 1.0 / probs.size)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    idx = rng.choice(np.arange(probs.size), size=shots, replace=True, p=probs)
    best = float(np.max(cut_vals[idx]))
    counts = np.bincount(idx, minlength=probs.size)
    mode = int(np.argmax(counts))
    return best, float(cut_vals[mode])


def run_outer(n, edges, Z, fam: Family1D, cut_mask,
              lam0, outer, inner, L, seed, eta0, eta_pow, step_clip,
              mode: str, readout_shots: int, c_frac: float):
    rng_read = np.random.default_rng(to_uint_seed(seed + 1234567))
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    hist = {k: [] for k in ["lam_pre", "lam", "J", "J_best", "evals_cum", "J_best_cut", "J_mode_cut"]}
    evals = 0.0
    best = -1e18

    c = c_frac * (lam_max - lam_min)

    for t in range(1, outer + 1):
        hist["lam_pre"].append(lam)

        # INNER SOLVE at current λ  -> one approximate value-evaluation of F(λ)
        w = fam.w(lam)

        def Efun(pvec): return vqe_energy(n, edges, Z, w, pvec, L)

        params, _, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals += ev_in

        # objective at λ_t  (≈ F(λ_t))
        J, psi, zexp = vqe_expect(n, edges, Z, w, params, L)
        evals += 1.0
        best = max(best, float(J))

        # optional readout realism
        if readout_shots > 0:
            with np.errstate(all="ignore"):
                cut_vals = (cut_mask @ w.astype(np.float64)).astype(float)
            jb, jm = readout_metrics(rng_read, psi, cut_vals, readout_shots)
        else:
            jb, jm = float("nan"), float("nan")

        # OUTER SIGNAL
        if mode == "ID":
            p_cut = 0.5 * (1.0 - zexp)
            g = float(fam.dw_dlam(lam) @ p_cut)

        elif mode == "FD_VALUE":
            # FD on VALUE FUNCTION F(λ) -> extra inner solves at λ±c
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))

            # +c
            w_p = fam.w(lp)

            def Efun_p(pvec): return vqe_energy(n, edges, Z, w_p, pvec, L)

            p_p, _, evp = spsa_minimize(Efun_p, params, bounds, iters=inner, seed=seed + 1000 * t + 17)
            evals += evp
            Jp, _, _ = vqe_expect(n, edges, Z, w_p, p_p, L)
            evals += 1.0

            # -c
            w_m = fam.w(lm)

            def Efun_m(pvec): return vqe_energy(n, edges, Z, w_m, pvec, L)

            p_m, _, evm = spsa_minimize(Efun_m, params, bounds, iters=inner, seed=seed + 1000 * t + 29)
            evals += evm
            Jm, _, _ = vqe_expect(n, edges, Z, w_m, p_m, L)
            evals += 1.0

            # FAIR best-so-far: FD has evaluated these candidates too
            best = max(best, float(Jp), float(Jm))

            g = (Jp - Jm) / (2.0 * c) if c > 0 else 0.0
        else:
            raise ValueError("mode must be 'ID' or 'FD_VALUE'")

        # MATCHED OUTER UPDATE RULE
        eta = eta0 / (t ** eta_pow)
        step = float(eta * g)
        if step_clip is not None:
            step = float(np.clip(step, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

        hist["lam"].append(lam)
        hist["J"].append(float(J))
        hist["J_best"].append(float(best))
        hist["evals_cum"].append(float(evals))
        hist["J_best_cut"].append(float(jb))
        hist["J_mode_cut"].append(float(jm))

    for k in hist:
        hist[k] = np.array(hist[k], float)
    return hist


# ==============================================================================
# 5) Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="experiment1")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--seed", type=int, default=7)

    # instance / family
    p.add_argument("--kind", type=str, default="quadratic", choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=6)

    # problem size
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.45)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=4.5)
    p.add_argument("--grid", type=int, default=401)

    # bilevel
    p.add_argument("--outer", type=int, default=100)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)
    p.add_argument("--readout_shots", type=int, default=0)

    # plotting / budget
    p.add_argument("--budget_evals", type=float, default=None,
                   help="Common evaluation budget B for the budget plots. Default: B = evals used by ID after --outer steps.")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(to_uint_seed(a.seed))
    edges = generate_random_graph(a.n, a.p_edge, rng)
    if not edges:
        raise RuntimeError("Graph has 0 edges; increase p_edge or change seed.")

    Z = precompute_z_big_endian(a.n)
    cut_mask = build_cut_mask(edges, Z)
    fam = Family1D(len(edges), a.kind, (a.lam_min, a.lam_max), rng, K=a.periodic_K)

    # classical diagnostic envelope (bitstrings)
    lams, all_J, J_cl_star, active_ids, sw_lams, sw_vals = envelope_spectrum(fam, cut_mask, a.grid)
    lam_true = float(lams[int(np.argmax(J_cl_star))])
    J_cl_max = float(np.max(J_cl_star))

    hist_id = run_outer(a.n, edges, Z, fam, cut_mask, a.lam0, a.outer, a.inner, a.L,
                        a.seed, a.eta0, a.eta_pow, a.step_clip, "ID", a.readout_shots, a.c_frac)
    hist_fd = run_outer(a.n, edges, Z, fam, cut_mask, a.aam0 if hasattr(a, "aam0") else a.lam0, a.outer, a.inner, a.L,
                        a.seed, a.eta0, a.eta_pow, a.step_clip, "FD_VALUE", a.readout_shots, a.c_frac)

    # Default budget B: the evaluation cost spent by ID after --outer steps (so t_ID(B)=outer).
    B = float(a.budget_evals) if a.budget_evals is not None else float(hist_id["evals_cum"][-1])

    suf = f"{a.kind}_n{a.n}_seed{a.seed}"
    plot_envelope_and_expectations(out / f"envelope_expectations_{suf}.{a.fmt}",
                                   lams, J_cl_star, hist_id, hist_fd)
    plot_bestJ_vs_evals(out / f"bestJ_vs_evals_{suf}.{a.fmt}", hist_id, hist_fd, J_cl_max=J_cl_max, budget=B)
    plot_evals_vs_outer(out / f"evals_vs_outer_{suf}.{a.fmt}", hist_id, hist_fd, budget=B)
    plot_lambda_trajectories(out / f"lambda_trajectories_{suf}.{a.fmt}",
                             hist_id, hist_fd, lam_true=lam_true, lam_bounds=(a.lam_min, a.lam_max))
    plot_xray(out / f"xray_envelope_{suf}.{a.fmt}", lams, all_J, J_cl_star, active_ids, sw_lams, sw_vals)

    if a.readout_shots > 0:
        # best-of-S and mode overlays (simple; no titles)
        set_pub_style(False)
        fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
        ax.plot(lams, J_cl_star, color=COLORS["ENV"], lw=2.0, label=r"$J_{\mathrm{cl}}^*(\lambda)$")
        ax.scatter(hist_id["lam_pre"], hist_id["J_best_cut"], color=COLORS["ID"], s=14, marker="o",
                   alpha=0.65, edgecolors="none", label="ID best-of-S")
        ax.scatter(hist_fd["lam_pre"], hist_fd["J_best_cut"], color=COLORS["FD"], s=14, marker="s",
                   alpha=0.65, edgecolors="none", label="FD best-of-S")
        ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("Best sampled cut")
        ax.legend(loc="lower right", frameon=False)
        _savefig(fig, out / f"best_cut_{suf}.{a.fmt}"); plt.close(fig)

        set_pub_style(False)
        fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
        ax.plot(lams, J_cl_star, color=COLORS["ENV"], lw=2.0, label=r"$J_{\mathrm{cl}}^*(\lambda)$")
        ax.scatter(hist_id["lam_pre"], hist_id["J_mode_cut"], color=COLORS["ID"], s=14, marker="o",
                   alpha=0.65, edgecolors="none", label="ID mode")
        ax.scatter(hist_fd["lam_pre"], hist_fd["J_mode_cut"], color=COLORS["FD"], s=14, marker="s",
                   alpha=0.65, edgecolors="none", label="FD mode")
        ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("Mode cut")
        ax.legend(loc="lower right", frameon=False)
        _savefig(fig, out / f"mode_cut_{suf}.{a.fmt}"); plt.close(fig)

    print("Saved to:", out.resolve())
    print(f"classical grid argmax λ*≈{lam_true:.4f}, max J_cl^*≈{J_cl_max:.6f}")
    print(f"ID best(value)={hist_id['J_best'][-1]:.6f}, evals={int(hist_id['evals_cum'][-1])}")
    print(f"FD best(value)={hist_fd['J_best'][-1]:.6f}, evals={int(hist_fd['evals_cum'][-1])}")
    print(f"Budget used for budget-plots: B={B:.0f} evals")


if __name__ == "__main__":
    main()
