#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_allinone_id_vs_fd_bilevel_pubplots_final.py
================================================

COMPLETE PAPER-GRADE DEMO: VQE + parametrized Max-Cut family + ID vs FD
-----------------------------------------------------------------------
FINAL REFINED PLOTTING SUITE (VERSION 6):

  1. Envelope: Legend with WHITE BOX. Transparent points.
  2. Efficiency: Advantage Zone + EXACTLY ONE Marker at t=20.
  3. X-Ray: Red/Black Segments + Full Ghost Curves + Legend Box.
  4. Cost Gap: Waste Zone is PURE GRAY (#808080) filled (no hatch).
  5. Trajectory: Black Optimum, Start 'x' in Legend, Square marker for FD.

Run (example):
  python demo_allinone_id_vs_fd_bilevel_pubplots_final.py --kind periodic --n 12 --outer 100 --inner 30 --fmt pdf
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
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ------------------------------------------------------------------------------
# Silence known noisy-but-harmless messages
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")
warnings.filterwarnings("ignore", message=".*regarding as unix timestamp.*")

logging.getLogger("fontTools").setLevel(logging.ERROR)


# ==============================================================================
# 1) Pub-Style Settings & Colors
# ==============================================================================

COLORS = {
    "GT":  "#000000",
    "ID":  "#D62728",  # red
    "FD":  "#1F77B4",  # blue
    "ENV": "#333333",
    "ZONE": "#2ca02c", # green (hardly used now)
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
# 2) THE 5 ADJUSTED PLOTTING FUNCTIONS
# ==============================================================================

def plot_envelope_improved(path: Path, lam_grid, J_cl_star, hist_id, hist_fd):
    """
    Fig1: Envelope with Time-Encoded Gradient
    Adjustments: 
    - Legend now has a WHITE BOX (like Plot 2).
    - Transparent points (alpha=0.6).
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    lam_grid = np.asarray(lam_grid, dtype=float)
    J_cl_star = np.asarray(J_cl_star, dtype=float)

    # Background
    ax.plot(lam_grid, J_cl_star, color="black", lw=1.5, alpha=0.8, zorder=1)

    # Time-Encoded Scatter with Transparency
    t_id = np.arange(len(hist_id["J"]))
    t_fd = np.arange(len(hist_fd["J"]))
    
    # ID (Reds)
    ax.scatter(hist_id["lam_pre"], hist_id["J"], s=15, c=t_id, cmap="Reds", 
               vmin=0, vmax=len(t_id)*1.2, marker="o", edgecolors="black", linewidth=0.3, 
               alpha=0.6, zorder=3)
    
    # FD (Blues)
    ax.scatter(hist_fd["lam_pre"], hist_fd["J"], s=15, c=t_fd, cmap="Blues", 
               vmin=0, vmax=len(t_fd)*1.2, marker="s", edgecolors="black", linewidth=0.3, 
               alpha=0.6, zorder=2)

    ax.set_xlabel(r"Control parameter $\lambda$")
    ax.set_ylabel(r"Value estimate $\hat F(\lambda)$")
    ax.set_xlim(float(lam_grid[0]), float(lam_grid[-1]))
    
    # Legend - Lower Right WITH WHITE BOX
    line_env = mlines.Line2D([], [], color='black', lw=1.5, label='Envelope')
    dot_id = mlines.Line2D([], [], color='#D62728', marker='o', ls='None', ms=5, label='VQE + ID')
    dot_fd = mlines.Line2D([], [], color='#1F77B4', marker='s', ls='None', ms=5, label='VQE + FD')
    ax.legend(handles=[line_env, dot_id, dot_fd], loc="lower right", 
              frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=7)

    _savefig(fig, path); plt.close(fig)


def plot_efficiency_improved(path: Path, hist_id, hist_fd, J_cl_max=None, budget: float = None):
    """
    Fig3: Efficiency with Advantage Zone
    Adjustments: EXACTLY ONE marker at t=20.
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    if budget is None:
        budget = float(min(hist_id["evals_cum"][-1], hist_fd["evals_cum"][-1]))
    budget = float(budget)
    
    ev_id, y_id = _truncate_step_to_budget(hist_id["evals_cum"], hist_id["J_best"], budget)
    ev_fd, y_fd = _truncate_step_to_budget(hist_fd["evals_cum"], hist_fd["J_best"], budget)

    # Advantage Zone
    y_fd_interp = np.interp(ev_id, ev_fd, y_fd)
    ax.fill_between(ev_id, y_id, y_fd_interp, where=(y_id > y_fd_interp),
                    color=COLORS["ID"], alpha=0.1, interpolate=True, label="Advantage Zone")

    # Curves
    ax.plot(ev_id, y_id, color=COLORS["ID"], lw=1.8, label="VQE + ID")
    ax.plot(ev_fd, y_fd, color=COLORS["FD"], lw=1.8, ls="--", label=r"VQE + FD")

    if J_cl_max is not None:
        ax.axhline(J_cl_max, color="gray", lw=1.0, ls=":", label=r"Grid Max $J^*$")
        # 99% Threshold
        thresh = 0.99 * J_cl_max
        idx = np.argmax(y_id >= thresh)
        if y_id[idx] >= thresh:
            cx = ev_id[idx]
            ax.vlines(cx, 0, thresh, color=COLORS["ID"], lw=1.0, alpha=0.5, linestyles="-.")
            ax.text(cx, thresh*0.85, f"99% @ {int(cx)}", color=COLORS["ID"], 
                    fontsize=7, ha="right", rotation=90)

    # SINGLE Marker at t=20
    target_step = 20
    for name, h, col, mk in [("ID", hist_id, COLORS["ID"], "o"), ("FD", hist_fd, COLORS["FD"], "s")]:
        if len(h["evals_cum"]) > target_step:
            e, j = h["evals_cum"][target_step], h["J_best"][target_step]
            if e <= budget:
                ax.scatter(e, j, s=20, color="white", edgecolors=col, marker=mk, zorder=10, lw=0.8)
                ax.annotate(f"t={target_step}", (e, j), xytext=(0, -15 if name=="FD" else 10), 
                            textcoords="offset points", ha='center', fontsize=6, color=col)

    ax.set_xlabel("Cumulative Energy Evaluations")
    ax.set_ylabel(r"Best-so-far Value $\hat F$")
    ax.set_xlim(0.0, budget)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    _savefig(fig, path); plt.close(fig)


def plot_xray_improved(path: Path, lams, all_J, J_cl_star, active_ids, switch_lams, switch_vals):
    """
    Fig2: RED/BLACK X-RAY
    Adjustments: Legend with White Box, Red/Black segments, Full ghost curves.
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    # 1. Very faint background
    K = all_J.shape[0]
    sample = np.linspace(0, K-1, min(K, 500), dtype=int)
    ax.plot(lams, all_J[sample].T, color="#cccccc", alpha=0.05, lw=0.3, rasterized=True, zorder=0)

    # 2. Identify Contributors
    unique_ids = np.unique(active_ids)
    
    first_appearance = [(np.where(active_ids == uid)[0][0], uid) for uid in unique_ids]
    first_appearance.sort()
    sorted_uids = [u for _, u in first_appearance]
    
    curve_colors = [COLORS["ID"], "black"]

    for i, uid in enumerate(sorted_uids):
        col = curve_colors[i % 2]
        # Full Ghost Curve
        ax.plot(lams, all_J[uid], color=col, lw=1.0, ls="--", alpha=0.4, zorder=2, label="_nolegend_")

    # 3. Active Segments (Thick)
    changes = np.where(active_ids[1:] != active_ids[:-1])[0] + 1
    bounds = np.concatenate(([0], changes, [len(lams)]))
    
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k+1]
        uid = active_ids[s]
        
        try:
            c_idx = sorted_uids.index(uid)
            col = curve_colors[c_idx % 2]
        except ValueError: continue
        
        s_p = max(0, s)
        e_p = min(len(lams), e + 1)
        ax.plot(lams[s_p:e_p], all_J[uid, s_p:e_p], color=col, lw=2.0, alpha=1.0, zorder=4)

    # 4. Switch Points
    if switch_lams.size > 0:
        ax.scatter(switch_lams, switch_vals, color="white", s=30, edgecolors="black", zorder=10)

    ax.set_xlabel(r"Control parameter $\lambda$")
    ax.set_ylabel(r"Energy landscape $J(z;\lambda)$")
    
    y_max = np.max(all_J)
    y_env_min = np.min(J_cl_star)
    range_y = y_max - y_env_min
    ax.set_ylim(y_env_min - 0.4 * range_y, y_max + 0.05 * range_y)
    ax.set_xlim(lams[0], lams[-1])

    # Legend WITH WHITE BOX
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Active Branch'),
        Line2D([0], [0], color='black', lw=1, ls='--', alpha=0.5, label='Full Curve'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markersize=6, label='Switch Point')
    ]
    ax.legend(handles=legend_elements, loc="lower right", 
              frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=7)

    _savefig(fig, path); plt.close(fig)


def plot_cost_gap_improved(path: Path, hist_id, hist_fd, budget: float):
    """
    Fig4: Cost Gap
    Adjustments: Waste Zone is PURE GRAY (#808080) filled, no hatching.
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    t_id, t_fd = np.arange(1, hist_id["evals_cum"].size + 1), np.arange(1, hist_fd["evals_cum"].size + 1)
    e_id, e_fd = hist_id["evals_cum"], hist_fd["evals_cum"]

    slope_id = e_id[-1] / t_id[-1]
    slope_fd = e_fd[-1] / t_fd[-1]
    
    # Waste Zone: Pure Gray, transparent, no edges
    common_t = min(t_id[-1], t_fd[-1])
    ax.fill_between(t_id[:common_t], e_id[:common_t], e_fd[:common_t], 
                    color="#808080",    # PURE GRAY hex code
                    alpha=0.2,          # Transparent
                    linewidth=0.0,      # No outline
                    label="Cost Overhead")

    ax.plot(t_id, e_id, color=COLORS["ID"], lw=2.0, label=f"VQE + ID (Slope $\\approx$ {slope_id:.1f})")
    ax.plot(t_fd, e_fd, color=COLORS["FD"], lw=2.0, ls="--", label=f"VQE + FD (Slope $\\approx$ {slope_fd:.1f})")

    idx_id = np.searchsorted(e_id, budget)
    idx_fd = np.searchsorted(e_fd, budget)
    
    ax.axhline(budget, color="black", ls=":", lw=1.0)
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
    ax.legend(loc="upper left", frameon=False)
    _savefig(fig, path); plt.close(fig)


def plot_trajectory_improved(path: Path, hist_id, hist_fd, lam_true, lam_bounds):
    """
    Fig5: Trajectory
    Adjustments:
    - Optimum is BLACK.
    - Start 'x' explicitly in Legend.
    - Blue line ends with Square marker.
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
    t = np.arange(len(hist_id["lam_pre"]))

    # Optimum Line: BLACK
    ax.axhline(lam_true, color="black", lw=1.0, alpha=0.8, ls="-", label=r"Optimum $\lambda^*$")
    
    # Curves
    ax.plot(t, hist_id["lam_pre"], color=COLORS["ID"], lw=1.5, label="VQE + ID")
    ax.plot(t, hist_fd["lam_pre"], color=COLORS["FD"], lw=1.5, ls="--", label="VQE + FD")

    # Start Marker with Label (x)
    ax.scatter([0], [hist_id["lam_pre"][0]], color="black", s=20, marker="x", zorder=5, label="Start")
    
    # End Markers
    # ID: Circle
    ax.scatter([t[-1]], [hist_id["lam_pre"][-1]], color=COLORS["ID"], s=20, edgecolors="white", lw=0.5, zorder=5)
    # FD: Square
    ax.scatter([t[-1]], [hist_fd["lam_pre"][-1]], color=COLORS["FD"], s=20, marker="s", edgecolors="white", lw=0.5, zorder=5)
    
    ax.set_ylim(lam_bounds)
    ax.set_xlabel("Outer iteration $t$")
    ax.set_ylabel(r"Parameter $\lambda_t$")
    # Legend with 2 columns to fit 'Start' nicely
    ax.legend(loc="upper right", frameon=False, ncol=2)
    _savefig(fig, path); plt.close(fig)


# ==============================================================================
# 3) Simulation Logic (Original 1:1)
# ==============================================================================

def to_uint_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)

def generate_random_graph(n: int, p_edge: float, rng: np.random.Generator):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge: edges.append((i, j))
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
            self.k, self.phi = None, None
        else:
            self.s = None
            self.k = rng.integers(1, K + 1, size=m).astype(float)
            self.phi = rng.uniform(0.0, 2 * np.pi, size=m).astype(float)
    def x(self, lam: float) -> float:
        return 2.0 * (float(lam) - self.mid) / self.Delta
    def f_df(self, x: float):
        x = float(x)
        if self.kind == "linear":
            c, f, df = math.sqrt(3.0), math.sqrt(3.0)*self.s*x, math.sqrt(3.0)*self.s
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
        return np.nan_to_num(self.wbar + self.A * f).astype(float)
    def dw_dlam(self, lam: float) -> np.ndarray:
        _, df = self.f_df(self.x(lam))
        return np.nan_to_num(self.A * df * self.dx_dlam).astype(float)

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)

def _renorm(psi):
    nrm = float(np.vdot(psi, psi).real)
    psi[:] = 1.0/np.sqrt(psi.size) if (not np.isfinite(nrm) or nrm<=0) else psi/math.sqrt(nrm)
    return psi

def _apply_1q(psi, gate, target, n):
    with np.errstate(all="ignore"):
        psi_r = np.moveaxis(psi.reshape([2]*n), target, 0)
        out = gate @ np.nan_to_num(psi_r.reshape(2, -1).astype(np.complex128, copy=False))
        psi = np.moveaxis(np.nan_to_num(out).reshape([2]+[2]*(n-1)), 0, target).reshape(-1)
    return psi

def _apply_2q(psi, gate4, q1, q2, n):
    if q1 == q2: return psi
    with np.errstate(all="ignore"):
        a, b = sorted((q1, q2))
        psi_r = np.moveaxis(psi.reshape([2]*n), (a, b), (0, 1))
        out = gate4 @ np.nan_to_num(psi_r.reshape(4, -1).astype(np.complex128, copy=False))
        psi = np.moveaxis(np.nan_to_num(out).reshape(2, 2, *psi_r.shape[2:]), (0, 1), (a, b)).reshape(-1)
    return psi

def vqe_state(n, params, L):
    psi = np.zeros(1<<n, dtype=np.complex128); psi[0] = 1.0
    for l in range(L):
        ry = params[l*2*n : l*2*n+n]; rz = params[l*2*n+n : (l+1)*2*n]
        for q in range(n):
            c, s = math.cos(ry[q]/2), math.sin(ry[q]/2)
            psi = _apply_1q(psi, np.array([[c, -s], [s, c]], dtype=np.complex128), q, n)
            psi = _apply_1q(psi, np.array([[np.exp(-0.5j*rz[q]), 0], [0, np.exp(0.5j*rz[q])]], dtype=np.complex128), q, n)
        for q in range(n): psi = _apply_2q(psi, CNOT, q, (q+1)%n, n)
        psi = _renorm(psi)
    return psi

def vqe_expect(n, edges, Z, w, params, L):
    psi = vqe_state(n, params, L)
    probs = (psi.conj()*psi).real.astype(float)
    s = np.sum(probs)
    probs = probs/s if (np.isfinite(s) and s>0) else np.full_like(probs, 1.0/probs.size)
    z = np.zeros(len(edges), float)
    zz = Z.astype(float)
    for e, (i,j) in enumerate(edges): z[e] = float(probs @ (zz[i]*zz[j]))
    z = np.clip(np.nan_to_num(z), -1.0, 1.0)
    J = float(0.5*(1.0-z) @ w)
    return (0.0 if not np.isfinite(J) else J), psi, z

def vqe_energy(n, edges, Z, w, params, L):
    return -vqe_expect(n, edges, Z, w, params, L)[0]

def spsa_minimize(energy_fun, p0, bounds, iters, seed, a=0.2, c=0.12, A=20.0, alpha=0.602, gamma=0.101):
    rng = np.random.default_rng(to_uint_seed(seed))
    p = p0.astype(float).copy()
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    best_p, best_E = p.copy(), float("inf")
    evals = 0
    for k in range(1, iters + 1):
        ak, ck = a/((k+A)**alpha), c/(k**gamma)
        delta = rng.choice([-1.0, 1.0], size=p.size)
        Ep = float(energy_fun(np.clip(p + ck*delta, lo, hi)))
        Em = float(energy_fun(np.clip(p - ck*delta, lo, hi)))
        evals += 2
        ghat = (Ep - Em)/(2.0*ck) * delta
        p = np.clip(p - ak*ghat, lo, hi)
        E = float(energy_fun(p)); evals += 1
        if E < best_E: best_E, best_p = E, p.copy()
    return best_p, best_E, evals

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

def run_outer(n, edges, Z, fam, cut_mask, lam0, outer, inner, L, seed, eta0, eta_pow, step_clip, mode, readout_shots, c_frac):
    rng_read = np.random.default_rng(to_uint_seed(seed + 1234567))
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))
    params = np.zeros(2*n*L, float)
    bounds = [(-math.pi, math.pi)] * params.size
    hist = {k: [] for k in ["lam_pre", "lam", "J", "J_best", "evals_cum", "J_best_cut", "J_mode_cut"]}
    evals, best = 0.0, -1e18
    c = c_frac * (lam_max - lam_min)

    for t in range(1, outer + 1):
        hist["lam_pre"].append(lam)
        w = fam.w(lam)
        def Efun(p): return vqe_energy(n, edges, Z, w, p, L)
        params, _, ev = spsa_minimize(Efun, params, bounds, inner, seed + 1000*t)
        evals += ev
        J, psi, zexp = vqe_expect(n, edges, Z, w, params, L)
        evals += 1.0; best = max(best, J)
        
        jb, jm = (readout_metrics(rng_read, psi, (cut_mask@w).astype(float), readout_shots) if readout_shots>0 else (float("nan"), float("nan")))

        if mode == "ID":
            g = float(fam.dw_dlam(lam) @ (0.5*(1.0-zexp)))
        elif mode == "FD_VALUE":
            lp, lm = np.clip(lam+c, lam_min, lam_max), np.clip(lam-c, lam_min, lam_max)
            wp = fam.w(lp)
            pp, _, evp = spsa_minimize(lambda p: vqe_energy(n, edges, Z, wp, p, L), params, bounds, inner, seed+1000*t+17)
            evals += evp + 1.0; Jp = vqe_expect(n, edges, Z, wp, pp, L)[0]
            wm = fam.w(lm)
            pm, _, evm = spsa_minimize(lambda p: vqe_energy(n, edges, Z, wm, p, L), params, bounds, inner, seed+1000*t+29)
            evals += evm + 1.0; Jm = vqe_expect(n, edges, Z, wm, pm, L)[0]
            best = max(best, Jp, Jm)
            g = (Jp - Jm)/(2.0*c) if c>0 else 0.0
        else: raise ValueError(f"Unknown mode: {mode}")

        step = float(np.clip((eta0/(t**eta_pow)) * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))
        hist["lam"].append(lam); hist["J"].append(J); hist["J_best"].append(best); hist["evals_cum"].append(evals)
        hist["J_best_cut"].append(jb); hist["J_mode_cut"].append(jm)

    for k in hist: hist[k] = np.array(hist[k], float)
    return hist

def readout_metrics(rng, psi, cut_vals, shots):
    probs = (psi.conj() * psi).real.astype(float); s = np.sum(probs)
    probs = probs/s if (np.isfinite(s) and s>0) else np.full_like(probs, 1.0/probs.size)
    idx = rng.choice(np.arange(probs.size), size=shots, p=probs)
    return float(np.max(cut_vals[idx])), float(cut_vals[np.argmax(np.bincount(idx, minlength=probs.size))])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="experiment_final")
    p.add_argument("--fmt", type=str, default="pdf")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--kind", type=str, default="quadratic")
    p.add_argument("--periodic_K", type=int, default=6)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--outer", type=int, default=100)
    p.add_argument("--inner", type=int, default=30)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)
    p.add_argument("--readout_shots", type=int, default=0)
    p.add_argument("--budget_evals", type=float, default=None)
    args = p.parse_args()
    
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(to_uint_seed(args.seed))
    edges = generate_random_graph(args.n, 0.45, rng)
    Z = precompute_z_big_endian(args.n)
    mask = build_cut_mask(edges, Z)
    fam = Family1D(len(edges), args.kind, (-5.0, 5.0), rng, K=args.periodic_K)

    print("Precomputing envelope...")
    lams, all_J, J_star, active, sw_l, sw_v = envelope_spectrum(fam, mask, 401)
    lam_true = float(lams[np.argmax(J_star)])
    J_max = float(np.max(J_star))

    print("Running ID vs FD...")
    h_id = run_outer(args.n, edges, Z, fam, mask, 4, args.outer, args.inner, args.L, args.seed, args.eta0, args.eta_pow, args.step_clip, "ID", args.readout_shots, args.c_frac)
    h_fd = run_outer(args.n, edges, Z, fam, mask, 4, args.outer, args.inner, args.L, args.seed, args.eta0, args.eta_pow, args.step_clip, "FD_VALUE", args.readout_shots, args.c_frac)

    B = float(args.budget_evals) if args.budget_evals is not None else float(h_id["evals_cum"][-1])
    suf = f"{args.kind}_n{args.n}_seed{args.seed}"

    print("Generating High-End Plots...")
    plot_envelope_improved(out/f"1_envelope_zoom_{suf}.{args.fmt}", lams, J_star, h_id, h_fd)
    plot_xray_improved(out/f"2_xray_segments_bw_{suf}.{args.fmt}", lams, all_J, J_star, active, sw_l, sw_v)
    plot_efficiency_improved(out/f"3_efficiency_zone_{suf}.{args.fmt}", h_id, h_fd, J_cl_max=J_max, budget=B)
    plot_cost_gap_improved(out/f"4_cost_gap_waste_{suf}.{args.fmt}", h_id, h_fd, budget=B)
    plot_trajectory_improved(out/f"5_trajectory_target_{suf}.{args.fmt}", h_id, h_fd, lam_true, (-5.0, 5.0))

    print(f"Done! Results in {out.resolve()}")

if __name__ == "__main__":
    main()