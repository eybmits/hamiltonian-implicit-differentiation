#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp08_vqe_vs_qaoa_readout_bridge.py
===================================

Experiment 8: VQE vs QAOA under a matched outer protocol,
and how expectation improvements translate to sampled solutions
----------------------------------------------------------------------------------

Paper story (why this exists)
  In Exp. 2A we compare bilevel outer-optimization performance across ansatze.
  But in combinatorial optimization we ultimately care about sampling *good bitstrings*
  with a finite readout budget.

  This script is the "readout realism" bridge (cf. your Experiment 3):
    - Track best-so-far expectation value along the outer trajectory
    - Track best-so-far sampled solution quality along the same trajectory
      under a fixed readout shot budget S per outer step.

  Optional (recommended for rebuttal / supplement):
    - Track a tail metric that saturates much more slowly than best-of-256, e.g.
        Hit-rate:   P(sample >= (1-eps) J*)
      or
        Top-k tail: P(sample is in top-k% of bitstrings by cut value).

Correct bilevel / hypergradient choices (minimal but correct)
  We optimize the value function F(lam)=max_theta J(theta,lam) via an outer update in lam.
  At an inner optimum theta*(lam), the envelope theorem gives:
      dF/dlam = partial_lam J(theta*(lam), lam).

  VQE (lam-independent state at fixed parameters)
      |psi_VQE(phi)> does not depend on lam  =>  partial_lam J is exactly the "reuse term"
          partial_lam J = Sum_e w'_e(lam) p_e(phi)
      and can be computed without extra objective probes.

  QAOA (problem-dependent state)
      |psi_QAOA(theta,lam)> depends on lam via the cost unitary U_C(lam)=exp(-i gamma H_C(lam)).
      Therefore partial_lam J includes an additional state-dependence term.
      Here we use the *safe* option for QAOA:
          partial_lam J(theta*,lam) approx [J(theta*,lam+c) - J(theta*,lam-c)] / (2c),
      i.e. fixed-theta central finite differences (2 extra objective calls per outer step).

What we report (paper-friendly)
  Main figure (single column, two panels; vs ENERGY EVALUATION BUDGET):
    (a) best-so-far expectation (normalized by J*)
    (b) best-so-far sampled cut (best-of-S readout, normalized by J*)

  Optional figures:
    - Tail probability vs budget (hit-rate or top-k)
    - Readout shot sweep at fixed budget B (best-of-S for S in {16,32,64,256,...})
    - Cost plot: cumulative energy evaluations vs outer step
    - Tradeoff scatter at budget B with y=x diagonal:
        "above diagonal" = stronger tail advantage (sampling closes expectation gap)

Outputs (in --out folder)
  - fig8_expect_and_readout_vs_evals.<fmt>
  - fig8_tailprob_vs_evals.<fmt>                (if --tail_metric != none)
  - fig8_readout_shots_sweep.<fmt>              (if --supp_shots not empty)
  - fig8_evals_vs_outer.<fmt>                   (can disable with --no_cost_plot)
  - fig8_tradeoff_scatter.<fmt>                 (can disable with --no_tradeoff_plot)
  - runs8_bridge_metrics.csv
  - SUMMARY.txt

Example
  python exp08_vqe_vs_qaoa_readout_bridge.py \\
    --kind periodic --periodic_K 6 --seeds 7,8,9,10,11,12,13,14 \\
    --readout_shots 256 --supp_shots 16,32,64 \\
    --tail_metric hit --hit_eps 0.10 \\
    --fmt pdf

Notes
  - Inner solver: SPSA uses 3 objective calls per iteration (Ep, Em, E).
  - Readout shots are NOT counted as "energy evaluations" (budget-first framing).
  - CLEAN PLOTS: no in-plot annotation boxes; only the legend is used to identify methods.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paramham.experiment_defaults import (
    CANONICAL_SETUP,
    can_run_step,
    generate_er_family1d_instance,
    publication_cache_dir,
    publication_output_dir,
    qaoa_full_step_cost,
    vqe_id_step_cost,
)
from paramham.families import Family1D
from paramham.io import parse_int_list
from paramham.maxcut import build_cut_mask, classical_Jstar
from paramham.maxcut import precompute_z as precompute_z_big_endian
from paramham.plotting import COL_W, COLORS, FULL_W, H_COL, _savefig, add_figure_legend, add_panel_legend, set_pub_style
from paramham.qaoa import qaoa_energy, qaoa_state

# ---------------------------------------------------------------------------
# paramham imports (shared library)
# ---------------------------------------------------------------------------
from paramham.seeds import to_uint_seed
from paramham.simulator import (
    expect_J,
    vqe_state,
)
from paramham.spsa import spsa_minimize

# ==============================================================================
# Experiment-specific helpers
# ==============================================================================


def fig_size() -> Tuple[float, float]:
    return (COL_W, H_COL)


def _panel_label(ax: plt.Axes, label: str):
    ax.text(0.00, 1.02, label, transform=ax.transAxes, va="bottom", ha="left", fontsize=9, fontweight="bold")


def step_sample(evals: np.ndarray, y_step: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Sample a step function y(evals) on a grid.
    For grid < evals[0], output 0.
    """
    evals = np.asarray(evals, float)
    y_step = np.asarray(y_step, float)
    grid = np.asarray(grid, float)
    if evals.size == 0:
        return np.zeros_like(grid)
    idx = np.searchsorted(evals, grid, side="right") - 1
    out = np.zeros_like(grid, dtype=float)
    m = idx >= 0
    out[m] = y_step[idx[m]]
    return out


def step_value_at(evals: np.ndarray, y_step: np.ndarray, x: float) -> float:
    evals = np.asarray(evals, float)
    y_step = np.asarray(y_step, float)
    if evals.size == 0:
        return 0.0
    idx = np.searchsorted(evals, float(x), side="right") - 1
    if idx < 0:
        return 0.0
    return float(y_step[idx])


def step_auc(evals: np.ndarray, y_step: np.ndarray, x_max: float) -> float:
    """
    AUC of a step function y(evals) over [0, x_max] (piecewise-constant, right-continuous).
    Returns raw area (not normalized).
    """
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


def sample_indices(rng: np.random.Generator, probs: np.ndarray, shots: int) -> np.ndarray:
    shots = int(shots)
    if shots <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(probs.size, size=shots, replace=True, p=probs).astype(np.int64)


def best_of_prefix(idx: np.ndarray, cut_vals: np.ndarray, shots: int) -> float:
    shots = int(shots)
    if shots <= 0:
        return float("nan")
    if idx.size < shots:
        raise ValueError("idx must have at least 'shots' entries.")
    return float(np.max(cut_vals[idx[:shots]]))


def tail_probability(
    probs: np.ndarray, cut_vals: np.ndarray, *, metric: str, J_star: float, hit_eps: float, topk_frac: float
) -> Tuple[float, float]:
    """
    Returns (p_tail, threshold) where
      - metric == 'hit' : threshold = (1-hit_eps)*J_star
      - metric == 'topk': threshold = quantile_{1-topk_frac}(cut_vals)
    and p_tail = P(cut >= threshold) under 'probs'.

    This is computed EXACTLY from the state distribution (no shot noise).
    """
    metric = str(metric).lower().strip()
    if metric == "hit":
        thr = (1.0 - float(hit_eps)) * float(J_star)
    elif metric == "topk":
        q = 1.0 - float(topk_frac)
        q = float(np.clip(q, 0.0, 1.0))
        thr = float(np.quantile(cut_vals, q))
    else:
        return float("nan"), float("nan")

    m = cut_vals >= thr
    p = float(np.sum(probs[m]))
    p = float(np.clip(p, 0.0, 1.0))
    return p, float(thr)


def _mean_stderr(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(Y, axis=0)
    if Y.shape[0] > 1:
        se = np.std(Y, axis=0, ddof=1) / math.sqrt(Y.shape[0])
    else:
        se = np.zeros_like(mu)
    return mu, se


def vqe_energy(n: int, cut_mask: np.ndarray, w: np.ndarray, params: np.ndarray, L: int) -> float:
    psi = vqe_state(n, params, L)
    J, _, _ = expect_J(psi, cut_mask, w)
    return -J


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
    return publication_cache_dir("exp08")


def _cache_meta(args, seeds: List[int], shot_list: List[int]) -> dict:
    return {
        "seeds": [int(s) for s in seeds],
        "n_sweep": [int(s) for s in parse_int_list(getattr(args, "n_sweep", ""))],
        "n": int(args.n),
        "p_edge": float(args.p_edge),
        "graph_seed": int(args.graph_seed),
        "kind": str(args.kind),
        "periodic_K": int(args.periodic_K),
        "lam_min": float(args.lam_min),
        "lam_max": float(args.lam_max),
        "lam0": float(args.lam0),
        "grid": int(args.grid),
        "L_vqe": int(args.L_vqe),
        "p_qaoa": int(args.p_qaoa),
        "outer": int(args.outer),
        "inner": int(args.inner),
        "budget_evals": float(args.budget_evals),
        "eta0": float(args.eta0),
        "eta_pow": float(args.eta_pow),
        "step_clip": float(args.step_clip),
        "fd_c_frac": float(args.fd_c_frac),
        "readout_shots": int(args.readout_shots),
        "shot_list": [int(s) for s in shot_list],
        "tail_metric": str(args.tail_metric),
        "hit_eps": float(args.hit_eps),
        "topk_frac": float(args.topk_frac),
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


def save_exp08_cache(cache_dir: Path, meta: dict, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (cache_dir / "payload.json").write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def load_exp08_cache(cache_dir: Path, meta_expected: dict):
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


# ==============================================================================
# Outer loops
# ==============================================================================


def run_outer_vqe_id(
    n: int,
    cut_mask: np.ndarray,
    fam: Family1D,
    *,
    J_star: float,
    lam0: float,
    outer: int,
    inner: int,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    seed: int,
    L_vqe: int,
    readout_shots: int,
    shot_list: List[int],
    readout_seed: int,
    tail_metric: str,
    hit_eps: float,
    topk_frac: float,
    budget_evals: float,
):
    """
    VQE outer optimization using exact reuse hypergradient:
      g = dw/dlam . p_cut

    Additionally tracks:
      - best-of-S readout for S in shot_list
      - optional tail probability metric from the exact state distribution
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L_vqe
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    shot_list = sorted({int(s) for s in shot_list if (int(s) > 0 and int(s) <= int(readout_shots))})
    if int(readout_shots) not in shot_list:
        shot_list.append(int(readout_shots))
    shot_list = sorted(set(shot_list))
    S_max = int(readout_shots)  # fixed per-step budget; smaller S use prefixes

    rng_read = np.random.default_rng(to_uint_seed(readout_seed))

    hist = {
        "lam": [],
        "J": [],
        "J_best": [],
        "evals_cum": [],
        "ro_best": [],
        "ro_best_sofar": [],
        "tail_prob": [],
        "tail_best_sofar": [],
    }
    # per-S best-so-far
    bestR = {S: -1e18 for S in shot_list}

    evals = 0.0
    bestJ = -1e18
    bestTail = -1.0

    step_cost = vqe_id_step_cost(inner)

    for t in range(1, int(outer) + 1):
        if not can_run_step(evals_used=evals, budget_evals=budget_evals, step_cost=step_cost):
            break
        w = fam.w(lam)

        def Efun(pvec):
            return vqe_energy(n, cut_mask, w, pvec, L_vqe)

        params, _, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals += float(ev_in)

        # expectation at lam
        psi = vqe_state(n, params, L_vqe)
        J, p_cut, probs = expect_J(psi, cut_mask, w)
        evals += 1.0
        bestJ = max(bestJ, float(J))

        # precompute per-bitstring cut values for readout metrics
        cut_vals = (cut_mask @ w).astype(np.float64)  # shape (K,)

        # readout sample once at max shots, then reuse prefixes for smaller S
        idx = sample_indices(rng_read, probs, S_max)

        for S in shot_list:
            rb = best_of_prefix(idx, cut_vals, S)
            bestR[S] = max(bestR[S], float(rb))
            # store per-S running best
            key = f"ro_best_sofar_S{S}"
            if key not in hist:
                hist[key] = []
            hist[key].append(float(bestR[S]))

        # main readout metric (S = readout_shots)
        rb_main = best_of_prefix(idx, cut_vals, int(readout_shots))
        hist["ro_best"].append(float(rb_main))
        hist["ro_best_sofar"].append(float(bestR[int(readout_shots)]))

        # tail probability (exact, no shot noise)
        if str(tail_metric).lower() != "none":
            p_tail, _thr = tail_probability(
                probs, cut_vals, metric=tail_metric, J_star=J_star, hit_eps=hit_eps, topk_frac=topk_frac
            )
            bestTail = max(bestTail, float(p_tail))
            hist["tail_prob"].append(float(p_tail))
            hist["tail_best_sofar"].append(float(bestTail))
        else:
            hist["tail_prob"].append(float("nan"))
            hist["tail_best_sofar"].append(float("nan"))

        # hypergradient (reuse term; exact for VQE)
        g = float(fam.dw_dlam(lam) @ p_cut)

        # outer step
        eta = eta0 / (t**eta_pow)
        step = float(np.clip(eta * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

        hist["lam"].append(lam)
        hist["J"].append(float(J))
        hist["J_best"].append(float(bestJ))
        hist["evals_cum"].append(float(evals))

    for k in hist:
        hist[k] = np.asarray(hist[k], float)
    return hist


def run_outer_qaoa_full(
    n: int,
    edges,
    cut_mask: np.ndarray,
    fam: Family1D,
    *,
    J_star: float,
    lam0: float,
    outer: int,
    inner: int,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    seed: int,
    p_qaoa: int,
    fd_c_frac: float,
    readout_shots: int,
    shot_list: List[int],
    readout_seed: int,
    tail_metric: str,
    hit_eps: float,
    topk_frac: float,
    budget_evals: float,
):
    """
    QAOA outer optimization using fixed-theta FD hypergradient:
      g_full approx [J(theta,lam+c) - J(theta,lam-c)] / (2c)
    which includes explicit + state-dependence terms.

    Additionally tracks:
      - best-of-S readout for S in shot_list
      - optional tail probability metric from the exact state distribution
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * p_qaoa
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    shot_list = sorted({int(s) for s in shot_list if (int(s) > 0 and int(s) <= int(readout_shots))})
    if int(readout_shots) not in shot_list:
        shot_list.append(int(readout_shots))
    shot_list = sorted(set(shot_list))
    S_max = int(readout_shots)  # fixed per-step budget; smaller S use prefixes

    rng_read = np.random.default_rng(to_uint_seed(readout_seed))

    c_fd = float(fd_c_frac * (lam_max - lam_min))

    hist = {
        "lam": [],
        "J": [],
        "J_best": [],
        "evals_cum": [],
        "ro_best": [],
        "ro_best_sofar": [],
        "tail_prob": [],
        "tail_best_sofar": [],
    }
    bestR = {S: -1e18 for S in shot_list}

    evals = 0.0
    bestJ = -1e18
    bestTail = -1.0

    step_cost = qaoa_full_step_cost(inner)

    for t in range(1, int(outer) + 1):
        if not can_run_step(evals_used=evals, budget_evals=budget_evals, step_cost=step_cost):
            break
        w = fam.w(lam)

        def Efun(pvec):
            return qaoa_energy(n, edges, cut_mask, w, pvec, p_qaoa)

        params, _, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals += float(ev_in)

        # expectation at lam
        psi = qaoa_state(n, edges, w, params, p_qaoa)
        J, _p_cut, probs = expect_J(psi, cut_mask, w)
        evals += 1.0
        bestJ = max(bestJ, float(J))

        cut_vals = (cut_mask @ w).astype(np.float64)

        # readout (sample once, reuse prefixes)
        idx = sample_indices(rng_read, probs, S_max)

        for S in shot_list:
            rb = best_of_prefix(idx, cut_vals, S)
            bestR[S] = max(bestR[S], float(rb))
            key = f"ro_best_sofar_S{S}"
            if key not in hist:
                hist[key] = []
            hist[key].append(float(bestR[S]))

        rb_main = best_of_prefix(idx, cut_vals, int(readout_shots))
        hist["ro_best"].append(float(rb_main))
        hist["ro_best_sofar"].append(float(bestR[int(readout_shots)]))

        # tail probability (exact)
        if str(tail_metric).lower() != "none":
            p_tail, _thr = tail_probability(
                probs, cut_vals, metric=tail_metric, J_star=J_star, hit_eps=hit_eps, topk_frac=topk_frac
            )
            bestTail = max(bestTail, float(p_tail))
            hist["tail_prob"].append(float(p_tail))
            hist["tail_best_sofar"].append(float(bestTail))
        else:
            hist["tail_prob"].append(float("nan"))
            hist["tail_best_sofar"].append(float("nan"))

        # fixed-theta FD hypergradient (2 extra objective evals)
        lp = float(np.clip(lam + c_fd, lam_min, lam_max))
        lm = float(np.clip(lam - c_fd, lam_min, lam_max))
        if abs(lp - lm) < 1e-12:
            g = 0.0
        else:
            wp = fam.w(lp)
            wm = fam.w(lm)

            psi_p = qaoa_state(n, edges, wp, params, p_qaoa)
            Jp, _, _ = expect_J(psi_p, cut_mask, wp)

            psi_m = qaoa_state(n, edges, wm, params, p_qaoa)
            Jm, _, _ = expect_J(psi_m, cut_mask, wm)

            evals += 2.0
            g = float((Jp - Jm) / (lp - lm))

        # outer step
        eta = eta0 / (t**eta_pow)
        step = float(np.clip(eta * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

        hist["lam"].append(lam)
        hist["J"].append(float(J))
        hist["J_best"].append(float(bestJ))
        hist["evals_cum"].append(float(evals))

    for k in hist:
        hist[k] = np.asarray(hist[k], float)
    return hist


# ==============================================================================
# Plotting helpers
# ==============================================================================


CURVE_STYLES = {
    "VQE": {"color": COLORS["VQE"], "ls": "-", "label": "VQE + ID", "alpha": 0.14, "marker": "o"},
    "QAOA": {"color": COLORS["QAOA"], "ls": "--", "label": "QAOA + FD", "alpha": 0.10, "marker": "s"},
}


def _curve_grid(budget: float) -> np.ndarray:
    return np.linspace(0.0, float(budget), 320)


def _curve_stats(
    curves: Dict[str, List[Dict]], budget: float
) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    E = _curve_grid(budget)
    stats = {}
    for name, runs in curves.items():
        if not runs:
            continue
        Y = np.vstack([step_sample(r["evals"], r["best"], E) for r in runs])
        stats[name] = _mean_stderr(Y)
    return E, stats


def _interesting_curve_ylim(
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    y_floor: float = 0.0,
    y_cap: float = 1.02,
    min_span: float = 0.12,
) -> Tuple[float, float]:
    lower_parts = []
    upper_parts = []
    for mu, se in stats.values():
        mu = np.asarray(mu, float)
        se = np.asarray(se, float)
        mask = (mu + se) > 1e-6
        if np.any(mask):
            lower_parts.append(mu[mask] - se[mask])
            upper_parts.append(mu[mask] + se[mask])
    if not lower_parts:
        return y_floor, y_cap
    lo = float(np.min(np.concatenate(lower_parts)))
    hi = float(np.max(np.concatenate(upper_parts)))
    span = max(hi - lo, min_span)
    pad = 0.14 * span
    lo = max(y_floor, lo - pad)
    hi = min(y_cap, hi + pad)
    if hi - lo < min_span:
        center = 0.5 * (lo + hi)
        lo = max(y_floor, center - 0.5 * min_span)
        hi = min(y_cap, center + 0.5 * min_span)
    return lo, hi


def _draw_curve_panel(
    ax: plt.Axes,
    curves: Dict[str, List[Dict]],
    *,
    budget: float,
    ylabel: str,
    ylim: Tuple[float, float] | None = None,
    show_xlabel: bool = True,
):
    E, stats = _curve_stats(curves, budget)
    handles = []
    labels = []
    for name in ("VQE", "QAOA"):
        if name not in stats:
            continue
        style = CURVE_STYLES[name]
        mu, se = stats[name]
        line = ax.plot(E, mu, color=style["color"], lw=1.9, ls=style["ls"], label=style["label"], zorder=3)[0]
        ax.fill_between(E, mu - se, mu + se, color=style["color"], alpha=style["alpha"], lw=0, zorder=2)
        handles.append(line)
        labels.append(style["label"])

    ax.set_xlim(0.0, float(budget))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Energy evaluations" if show_xlabel else "")
    ax.set_ylim(*(ylim if ylim is not None else _interesting_curve_ylim(stats)))
    return handles, labels


def _uplift_stats(
    curves_expect: Dict[str, List[Dict]], curves_readout: Dict[str, List[Dict]], budget: float
) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    E = _curve_grid(budget)
    stats = {}
    for name in ("VQE", "QAOA"):
        runs_exp = curves_expect.get(name, [])
        runs_ro = curves_readout.get(name, [])
        if not runs_exp or not runs_ro:
            continue
        Y = []
        for r_exp, r_ro in zip(runs_exp, runs_ro):
            y_exp = step_sample(r_exp["evals"], r_exp["best"], E)
            y_ro = step_sample(r_ro["evals"], r_ro["best"], E)
            Y.append(y_ro - y_exp)
        stats[name] = _mean_stderr(np.vstack(Y))
    return E, stats


def _draw_uplift_panel(
    ax: plt.Axes,
    curves_expect: Dict[str, List[Dict]],
    curves_readout: Dict[str, List[Dict]],
    *,
    budget: float,
):
    E, stats = _uplift_stats(curves_expect, curves_readout, budget)
    handles = []
    labels = []
    for name in ("VQE", "QAOA"):
        if name not in stats:
            continue
        style = CURVE_STYLES[name]
        mu, se = stats[name]
        line = ax.plot(E, mu, color=style["color"], lw=1.9, ls=style["ls"], label=style["label"], zorder=3)[0]
        ax.fill_between(E, mu - se, mu + se, color=style["color"], alpha=style["alpha"], lw=0, zorder=2)
        handles.append(line)
        labels.append(style["label"])

    all_vals = []
    for mu, se in stats.values():
        all_vals.append(mu - se)
        all_vals.append(mu + se)
    if all_vals:
        lo = float(np.min(np.concatenate(all_vals)))
        hi = float(np.max(np.concatenate(all_vals)))
    else:
        lo, hi = -0.05, 0.05
    span = max(hi - lo, 0.08)
    pad = 0.18 * span
    ax.axhline(0.0, color=COLORS["REFERENCE"], lw=1.0, ls="--", zorder=1)
    ax.set_xlim(0.0, float(budget))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Readout uplift $(\mathrm{best\!-\!of\!-\!}S - J)/J^*$")
    return handles, labels


def _draw_n_sweep_panel(
    ax: plt.Axes,
    n_sweep_stats: List[Dict[str, float]],
    *,
    readout_shots: int,
):
    if not n_sweep_stats:
        return [], []

    n_vals = np.asarray([int(r["n"]) for r in n_sweep_stats], float)
    use_expectation = "vqe_mu" in n_sweep_stats[0]
    if use_expectation:
        vqe_mu_key, vqe_se_key = "vqe_mu", "vqe_se"
        qaoa_mu_key, qaoa_se_key = "qaoa_mu", "qaoa_se"
        ylabel = r"Best-so-far at budget $B$  ($J/J^*$)"
    else:
        vqe_mu_key, vqe_se_key = "vqe_ro_mu", "vqe_ro_se"
        qaoa_mu_key, qaoa_se_key = "qaoa_ro_mu", "qaoa_ro_se"
        ylabel = rf"Best-of-{int(readout_shots)} at budget $B$  ($J/J^*$)"

    handles = []
    labels = []
    for name, mu_key, se_key in (
        ("VQE", vqe_mu_key, vqe_se_key),
        ("QAOA", qaoa_mu_key, qaoa_se_key),
    ):
        style = CURVE_STYLES[name]
        mu = np.asarray([float(r[mu_key]) for r in n_sweep_stats], float)
        se = np.asarray([float(r[se_key]) for r in n_sweep_stats], float)
        handle = ax.errorbar(
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
        handles.append(handle.lines[0])
        labels.append(style["label"])

    lows = [float(r[vqe_mu_key]) - float(r[vqe_se_key]) for r in n_sweep_stats] + [
        float(r[qaoa_mu_key]) - float(r[qaoa_se_key]) for r in n_sweep_stats
    ]
    highs = [float(r[vqe_mu_key]) + float(r[vqe_se_key]) for r in n_sweep_stats] + [
        float(r[qaoa_mu_key]) + float(r[qaoa_se_key]) for r in n_sweep_stats
    ]
    lo = min(lows)
    hi = max(highs)
    span = max(hi - lo, 0.08)
    pad = 0.18 * span
    ax.set_xticks(n_vals)
    ax.set_xlabel(r"System size $n$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(max(0.0, lo - pad), min(1.02, hi + pad))
    return handles, labels


def plot_metric_vs_evals(
    path: Path,
    curves: Dict[str, List[Dict]],
    *,
    budget: float,
    ylabel: str,
    ylim: Tuple[float, float] | None = None,
):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
    _draw_curve_panel(ax, curves, budget=budget, ylabel=ylabel, ylim=ylim, show_xlabel=True)
    leg = add_panel_legend(ax, placement="below", ncol=2)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    _savefig(fig, path)


def plot_uplift_vs_evals(
    path: Path,
    curves_expect: Dict[str, List[Dict]],
    curves_readout: Dict[str, List[Dict]],
    *,
    budget: float,
):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
    _draw_uplift_panel(ax, curves_expect, curves_readout, budget=budget)
    leg = add_panel_legend(ax, placement="below", ncol=2)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    _savefig(fig, path)


def plot_expect_and_readout_vs_evals(
    path: Path,
    curves_expect: Dict[str, List[Dict]],
    curves_readout: Dict[str, List[Dict]],
    budget: float,
    readout_shots: int,
    annotate: bool = False,
):
    """Legacy combined budget figure kept for compatibility."""
    del readout_shots, annotate
    set_pub_style(grid=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, H_COL), constrained_layout=True)
    _draw_curve_panel(ax1, curves_expect, budget=budget, ylabel=r"Best-so-far expectation $J/J^*$", show_xlabel=True)
    handles, labels = _draw_curve_panel(
        ax2, curves_readout, budget=budget, ylabel=r"Best sampled cut $J/J^*$", show_xlabel=True
    )
    leg = add_figure_legend(fig, handles, labels, ncol=2)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    _savefig(fig, path)


def plot_tailprob_vs_evals(
    path: Path, curves_tail: Dict[str, List[Dict]], budget: float, tail_label: str, annotate: bool = False
):
    del annotate
    plot_metric_vs_evals(
        path,
        curves_tail,
        budget=budget,
        ylabel=tail_label,
        ylim=_interesting_curve_ylim(_curve_stats(curves_tail, budget)[1]),
    )


def plot_evals_vs_outer(path: Path, evals_mean: Dict[str, np.ndarray]):
    """Optional: mean cumulative energy evals vs outer iteration."""
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    for name in ("VQE", "QAOA"):
        if name not in evals_mean:
            continue
        ev_mean = np.asarray(evals_mean[name], float)
        t = np.arange(1, ev_mean.size + 1)
        style = CURVE_STYLES[name]
        ax.plot(t, ev_mean, color=style["color"], lw=1.9, ls=style["ls"], label=style["label"], zorder=3)

    ax.set_xlabel(r"Outer iteration $t$")
    ax.set_ylabel("Cumulative energy evaluations")
    leg = add_panel_legend(ax, placement="below", ncol=2)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    _savefig(fig, path)


def _draw_pair_scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    positive_label: str = "VQE > QAOA",
    negative_label: str = "QAOA >= VQE",
):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pos = y > x + 1e-12
    neg = ~pos

    handles = []
    labels = []

    if np.any(pos):
        ax.scatter(
            x[pos],
            y[pos],
            s=28,
            alpha=0.92,
            color=COLORS["ID"],
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    handles.append(Line2D([], [], color=COLORS["ID"], marker="o", lw=0, markersize=5, label=positive_label))
    labels.append(positive_label)

    if np.any(neg):
        ax.scatter(
            x[neg],
            y[neg],
            s=28,
            alpha=0.92,
            color=COLORS["FD"],
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    handles.append(Line2D([], [], color=COLORS["FD"], marker="o", lw=0, markersize=5, label=negative_label))
    labels.append(negative_label)

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
            f"VQE better in {win:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=COLORS["MUTED"],
        )

    return handles, labels


def plot_pair_scatter(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    positive_label: str = "VQE > QAOA",
    negative_label: str = "QAOA >= VQE",
):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
    handles, labels = _draw_pair_scatter_panel(
        ax,
        x,
        y,
        xlabel=xlabel,
        ylabel=ylabel,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    leg = add_panel_legend(ax, placement="below", ncol=2, handles=handles, labels=labels)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    _savefig(fig, path)


def plot_tradeoff_scatter(
    path: Path,
    x_vqe: np.ndarray,
    y_vqe: np.ndarray,
    x_qaoa: np.ndarray,
    y_qaoa: np.ndarray,
    xlabel: str,
    ylabel: str,
    annotate: bool = False,
):
    del annotate
    x = np.concatenate([np.asarray(x_vqe, float), np.asarray(x_qaoa, float)])
    y = np.concatenate([np.asarray(y_vqe, float), np.asarray(y_qaoa, float)])
    plot_pair_scatter(
        path,
        x,
        y,
        xlabel=xlabel,
        ylabel=ylabel,
        positive_label="Readout > expectation",
        negative_label="Expectation >= readout",
    )


def _draw_shots_panel(
    ax: plt.Axes,
    shots: List[int],
    stats_vqe: List[Tuple[float, float]],
    stats_qaoa: List[Tuple[float, float]],
):
    shots = [int(s) for s in shots]
    x = np.asarray(shots, float)
    handles = []
    labels = []

    for name, stats_list in (("VQE", stats_vqe), ("QAOA", stats_qaoa)):
        style = CURVE_STYLES[name]
        mu = np.asarray([m for (m, _se) in stats_list], float)
        se = np.asarray([_se for (_m, _se) in stats_list], float)
        handle = ax.errorbar(
            x,
            mu,
            yerr=se,
            color=style["color"],
            lw=1.6,
            ls=style["ls"],
            marker=style["marker"],
            ms=3.6,
            capsize=2.5,
            label=style["label"],
            zorder=3,
        )
        handles.append(handle.lines[0])
        labels.append(style["label"])

    y_low = min(float(m - s) for m, s in stats_vqe + stats_qaoa)
    y_high = max(float(m + s) for m, s in stats_vqe + stats_qaoa)
    span = max(y_high - y_low, 0.08)
    pad = 0.18 * span
    ax.set_ylim(max(0.0, y_low - pad), min(1.02, y_high + pad))
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(r"Readout shots per outer step $S$")
    ax.set_ylabel(r"Best-of-$S$ at budget $B$  ($J/J^*$)")
    return handles, labels


def plot_readout_shots_sweep(
    path: Path,
    shots: List[int],
    stats_vqe: List[Tuple[float, float]],
    stats_qaoa: List[Tuple[float, float]],
    budget: float,
    annotate: bool = False,
):
    del budget, annotate
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
    _draw_shots_panel(ax, shots, stats_vqe, stats_qaoa)
    leg = add_panel_legend(ax, placement="below", ncol=2)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    _savefig(fig, path)


def plot_n_sweep_budget(path: Path, n_sweep_stats: List[Dict[str, float]], *, readout_shots: int):
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)
    _draw_n_sweep_panel(ax, n_sweep_stats, readout_shots=readout_shots)
    leg = add_panel_legend(ax, placement="below", ncol=2)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    _savefig(fig, path)


def plot_sixpack_collage(
    path: Path,
    *,
    curves_expect: Dict[str, List[Dict]],
    curves_readout: Dict[str, List[Dict]],
    budget: float,
    n_sweep_stats: List[Dict[str, float]],
    readout_shots: int,
    shots_sorted: List[int],
    stats_vqe: List[Tuple[float, float]],
    stats_qaoa: List[Tuple[float, float]],
    objective_x: np.ndarray,
    objective_y: np.ndarray,
    readout_x: np.ndarray,
    readout_y: np.ndarray,
):
    set_pub_style(grid=False, base_size=8)
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 2 * H_COL + 1.0), constrained_layout=False)
    axes = np.asarray(axes)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.985, wspace=0.36, hspace=0.42)

    _draw_curve_panel(
        axes[0, 0], curves_expect, budget=budget, ylabel=r"Best-so-far expectation $J/J^*$", show_xlabel=True
    )
    _panel_label(axes[0, 0], "(A)")

    _draw_curve_panel(axes[0, 1], curves_readout, budget=budget, ylabel=r"Best sampled cut $J/J^*$", show_xlabel=True)
    _panel_label(axes[0, 1], "(B)")

    _draw_n_sweep_panel(axes[0, 2], n_sweep_stats, readout_shots=readout_shots)
    _panel_label(axes[0, 2], "(C)")

    _draw_shots_panel(axes[1, 0], shots_sorted, stats_vqe, stats_qaoa)
    _panel_label(axes[1, 0], "(D)")

    _draw_pair_scatter_panel(
        axes[1, 1],
        objective_x,
        objective_y,
        xlabel=r"QAOA + FD at budget $B$  ($J/J^*$)",
        ylabel=r"VQE + ID at budget $B$  ($J/J^*$)",
    )
    _panel_label(axes[1, 1], "(E)")

    _draw_pair_scatter_panel(
        axes[1, 2],
        readout_x,
        readout_y,
        xlabel=r"QAOA + FD best-of-$S$ at $B$  ($J/J^*$)",
        ylabel=r"VQE + ID best-of-$S$ at $B$  ($J/J^*$)",
    )
    _panel_label(axes[1, 2], "(F)")

    handles = [
        Line2D([], [], color=COLORS["VQE"], lw=1.9, ls="-", label="VQE + ID"),
        Line2D([], [], color=COLORS["QAOA"], lw=1.9, ls="--", label="QAOA + FD"),
        Line2D([], [], color=COLORS["ID"], marker="o", lw=0, markersize=5, label="VQE > QAOA"),
        Line2D([], [], color=COLORS["FD"], marker="o", lw=0, markersize=5, label="QAOA >= VQE"),
    ]
    labels = [h.get_label() for h in handles]
    leg = add_figure_legend(fig, handles, labels, ncol=4)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    _savefig(fig, path)


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

    # instance set
    p.add_argument("--seeds", type=str, default="7,8,9,10,11,12,13,14")
    p.add_argument("--n", type=int, default=CANONICAL_SETUP.n)
    p.add_argument("--p_edge", type=float, default=CANONICAL_SETUP.p_edge)
    p.add_argument("--graph_seed", type=int, default=CANONICAL_SETUP.graph_seed)

    # family
    p.add_argument("--kind", type=str, default=CANONICAL_SETUP.family, choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=CANONICAL_SETUP.periodic_K)
    p.add_argument("--lam_min", type=float, default=CANONICAL_SETUP.lam_min)
    p.add_argument("--lam_max", type=float, default=CANONICAL_SETUP.lam_max)
    p.add_argument("--lam0", type=float, default=CANONICAL_SETUP.lam0)
    p.add_argument("--grid", type=int, default=401)

    # ansatz depths
    p.add_argument("--L_vqe", type=int, default=2)
    p.add_argument("--p_qaoa", type=int, default=3)

    # budgets
    p.add_argument("--outer", type=int, default=10)
    p.add_argument("--inner", type=int, default=100)
    p.add_argument("--budget_evals", type=float, default=CANONICAL_SETUP.budget_evals)

    # outer schedule
    p.add_argument("--eta0", type=float, default=0.25)
    p.add_argument("--eta_pow", type=float, default=0.4)
    p.add_argument("--step_clip", type=float, default=0.6)

    # QAOA FD step
    p.add_argument("--fd_c_frac", type=float, default=0.10)

    # readout realism
    p.add_argument(
        "--readout_shots",
        type=int,
        default=128,
        help="Main readout budget S per outer step for the main 2-panel figure.",
    )
    p.add_argument(
        "--supp_shots",
        type=str,
        default="2,4,8,16,32,64,128",
        help="Comma-separated additional shot budgets for a sweep plot (supplement). Set to '' to disable.",
    )

    # tail metric (optional, recommended)
    p.add_argument(
        "--tail_metric",
        type=str,
        default="hit",
        choices=["none", "hit", "topk"],
        help="Tail metric to plot: 'hit' = P(cut >= (1-eps)J*), 'topk' = P(cut in top-k%% bitstrings).",
    )
    p.add_argument(
        "--hit_eps", type=float, default=0.10, help="eps for hit-rate threshold (1-eps)J*. Default 0.10 => 90%% of J*."
    )
    p.add_argument(
        "--topk_frac",
        type=float,
        default=0.01,
        help="k for top-k tail probability (fraction, not percent). Default 0.01 => top 1%%.",
    )

    # plots toggles (default ON for paper convenience)
    p.set_defaults(cost_plot=True)
    p.add_argument(
        "--no_cost_plot", action="store_false", dest="cost_plot", help="Disable the evals-vs-outer cost plot."
    )
    p.set_defaults(tradeoff_plot=True)
    p.add_argument(
        "--no_tradeoff_plot",
        action="store_false",
        dest="tradeoff_plot",
        help="Disable the matched-budget objective/readout pair-scatter plots.",
    )
    p.set_defaults(shots_sweep_plot=True)
    p.add_argument(
        "--no_shots_sweep_plot",
        action="store_false",
        dest="shots_sweep_plot",
        help="Disable the readout-shot sweep plot (if supp_shots provided).",
    )
    p.set_defaults(collage_plot=True)
    p.add_argument(
        "--no_collage_plot", action="store_false", dest="collage_plot", help="Disable the 2x3 collage render."
    )

    # kept for compatibility; plots ignore annotate anyway (legend-only policy)
    p.add_argument("--no_annotate", action="store_true")
    p.add_argument(
        "--n_sweep", type=str, default="8,9,10,11,12,13,14", help="Comma-separated system sizes for panel (C)."
    )

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out) if a.out is not None else publication_output_dir("exp08")
    out.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(a.seeds)
    if not seeds:
        raise ValueError("No seeds provided.")

    # readout shot list for sweep (includes main S)
    supp_raw = parse_int_list(a.supp_shots)
    # Supplement is meant for *smaller* shot budgets (e.g. 16/32/64) under the SAME per-step budget S.
    supp = [int(x) for x in supp_raw if (int(x) > 0 and int(x) <= int(a.readout_shots))]
    ignored = [int(x) for x in supp_raw if int(x) > int(a.readout_shots)]
    if ignored:
        print(f"[warn] ignoring supp_shots > readout_shots (per-step budget): {ignored}")
    shot_list = sorted(set([int(a.readout_shots)] + supp))
    if not shot_list:
        shot_list = [int(a.readout_shots)]
    n_sweep = parse_int_list(a.n_sweep)
    if not n_sweep:
        n_sweep = [int(a.n)]
    n_sweep = [int(nv) for nv in n_sweep]

    cache_dir = Path(a.cache_dir) if a.cache_dir is not None else _cache_default_dir(out)
    meta = _cache_meta(a, seeds, shot_list)
    cached = None if a.recompute else load_exp08_cache(cache_dir, meta)

    if cached is not None:
        curves_expect = cached["curves_expect"]
        curves_readout_main = cached["curves_readout_main"]
        curves_tail = cached["curves_tail"]
        curves_readout_byS = {int(k): v for k, v in cached["curves_readout_byS"].items()}
        evals_curves = cached["evals_curves"]
        n_sweep_stats = cached.get("n_sweep_stats", [])
        rows = cached["rows"]
        print(f"[cache] Loaded exp08 payloads from {cache_dir.resolve()}")
    else:
        if a.render_only:
            raise SystemExit(f"No matching cache found in {cache_dir}")

        Z = precompute_z_big_endian(int(a.n))

        # step curves (budget-first)
        curves_expect = {"VQE": [], "QAOA": []}
        curves_readout_main = {"VQE": [], "QAOA": []}
        curves_tail = {"VQE": [], "QAOA": []}

        # per-shot sweep curves (for budget-B extraction)
        curves_readout_byS = {S: {"VQE": [], "QAOA": []} for S in shot_list}

        # store evals curves for cost plot
        evals_curves = {"VQE": [], "QAOA": []}

        rows = []

        for s in seeds:
            edges, fam = generate_er_family1d_instance(
                int(a.n),
                float(a.p_edge),
                a.kind,
                (a.lam_min, a.lam_max),
                graph_seed=a.graph_seed,
                periodic_K=int(a.periodic_K),
                instance_id=s,
                safety_bounds=False,
            )
            if not edges or fam is None:
                print(f"[warn] skip seed={s} (empty graph)")
                continue

            cut_mask = build_cut_mask(edges, Z)
            J_star, lam_star = classical_Jstar(fam, cut_mask, int(a.grid))

            hist_vqe = run_outer_vqe_id(
                a.n,
                cut_mask,
                fam,
                J_star=J_star,
                lam0=a.lam0,
                outer=a.outer,
                inner=a.inner,
                eta0=a.eta0,
                eta_pow=a.eta_pow,
                step_clip=a.step_clip,
                seed=s + 0,
                L_vqe=a.L_vqe,
                readout_shots=a.readout_shots,
                shot_list=shot_list,
                readout_seed=s + 4242,
                tail_metric=a.tail_metric,
                hit_eps=a.hit_eps,
                topk_frac=a.topk_frac,
                budget_evals=a.budget_evals,
            )

            hist_qaoa = run_outer_qaoa_full(
                a.n,
                edges,
                cut_mask,
                fam,
                J_star=J_star,
                lam0=a.lam0,
                outer=a.outer,
                inner=a.inner,
                eta0=a.eta0,
                eta_pow=a.eta_pow,
                step_clip=a.step_clip,
                seed=s + 100000,
                p_qaoa=a.p_qaoa,
                fd_c_frac=a.fd_c_frac,
                readout_shots=a.readout_shots,
                shot_list=shot_list,
                readout_seed=s + 8888,
                tail_metric=a.tail_metric,
                hit_eps=a.hit_eps,
                topk_frac=a.topk_frac,
                budget_evals=a.budget_evals,
            )

            # normalize curves
            vqe_exp = np.clip(hist_vqe["J_best"] / J_star, 0.0, 1.5)
            qaoa_exp = np.clip(hist_qaoa["J_best"] / J_star, 0.0, 1.5)

            vqe_ro_main = np.clip(hist_vqe[f"ro_best_sofar_S{int(a.readout_shots)}"] / J_star, 0.0, 1.5)
            qaoa_ro_main = np.clip(hist_qaoa[f"ro_best_sofar_S{int(a.readout_shots)}"] / J_star, 0.0, 1.5)

            curves_expect["VQE"].append({"evals": hist_vqe["evals_cum"], "best": vqe_exp})
            curves_expect["QAOA"].append({"evals": hist_qaoa["evals_cum"], "best": qaoa_exp})

            curves_readout_main["VQE"].append({"evals": hist_vqe["evals_cum"], "best": vqe_ro_main})
            curves_readout_main["QAOA"].append({"evals": hist_qaoa["evals_cum"], "best": qaoa_ro_main})

            # per-shot curves (for sweep)
            for S in shot_list:
                vqe_ro_S = np.clip(hist_vqe[f"ro_best_sofar_S{S}"] / J_star, 0.0, 1.5)
                qaoa_ro_S = np.clip(hist_qaoa[f"ro_best_sofar_S{S}"] / J_star, 0.0, 1.5)
                curves_readout_byS[S]["VQE"].append({"evals": hist_vqe["evals_cum"], "best": vqe_ro_S})
                curves_readout_byS[S]["QAOA"].append({"evals": hist_qaoa["evals_cum"], "best": qaoa_ro_S})

            # tail curves
            if str(a.tail_metric).lower() != "none":
                vqe_tail = np.clip(hist_vqe["tail_best_sofar"], 0.0, 1.0)
                qaoa_tail = np.clip(hist_qaoa["tail_best_sofar"], 0.0, 1.0)
                curves_tail["VQE"].append({"evals": hist_vqe["evals_cum"], "best": vqe_tail})
                curves_tail["QAOA"].append({"evals": hist_qaoa["evals_cum"], "best": qaoa_tail})

            # cost curves
            evals_curves["VQE"].append(hist_vqe["evals_cum"])
            evals_curves["QAOA"].append(hist_qaoa["evals_cum"])

            row = {
                "seed": int(s),
                "kind": a.kind,
                "graph_seed": int(a.graph_seed),
                "n": int(a.n),
                "m_edges": int(len(edges)),
                "J_star_grid": float(J_star),
                "lam_star_grid": float(lam_star),
                "outer": int(a.outer),
                "inner": int(a.inner),
                "budget_evals": float(a.budget_evals),
                "L_vqe": int(a.L_vqe),
                "p_qaoa": int(a.p_qaoa),
                "eta0": float(a.eta0),
                "eta_pow": float(a.eta_pow),
                "step_clip": float(a.step_clip),
                "fd_c_frac": float(a.fd_c_frac),
                "readout_shots": int(a.readout_shots),
                "supp_shots": a.supp_shots,
                "tail_metric": str(a.tail_metric),
                "hit_eps": float(a.hit_eps),
                "topk_frac": float(a.topk_frac),
                "vqe_exp_final": float(vqe_exp[-1]),
                "qaoa_exp_final": float(qaoa_exp[-1]),
                "vqe_ro_final": float(vqe_ro_main[-1]),
                "qaoa_ro_final": float(qaoa_ro_main[-1]),
                "vqe_evals_final": float(hist_vqe["evals_cum"][-1]),
                "qaoa_evals_final": float(hist_qaoa["evals_cum"][-1]),
            }
            if str(a.tail_metric).lower() != "none":
                row.update(
                    {
                        "vqe_tail_final": float(vqe_tail[-1]),
                        "qaoa_tail_final": float(qaoa_tail[-1]),
                    }
                )
            rows.append(row)

            msg = (
                f"[seed={s}] exp_final: VQE={vqe_exp[-1]:.3f} | QAOA={qaoa_exp[-1]:.3f} || "
                f"readout_final(S={a.readout_shots}): VQE={vqe_ro_main[-1]:.3f} | QAOA={qaoa_ro_main[-1]:.3f}"
            )
            if str(a.tail_metric).lower() != "none":
                msg += f" || tail_final: VQE={vqe_tail[-1]:.3f} | QAOA={qaoa_tail[-1]:.3f}"
            print(msg)

        n_sweep_stats = []

        def _stderr(vals: np.ndarray) -> float:
            return float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0

        for n_value in n_sweep:
            if int(n_value) == int(a.n):
                vals_v = np.asarray(
                    [step_value_at(r["evals"], r["best"], float(a.budget_evals)) for r in curves_expect["VQE"]],
                    float,
                )
                vals_q = np.asarray(
                    [step_value_at(r["evals"], r["best"], float(a.budget_evals)) for r in curves_expect["QAOA"]],
                    float,
                )
                n_sweep_stats.append(
                    {
                        "n": int(n_value),
                        "vqe_mu": float(np.mean(vals_v)),
                        "vqe_se": _stderr(vals_v),
                        "qaoa_mu": float(np.mean(vals_q)),
                        "qaoa_se": _stderr(vals_q),
                    }
                )
                continue

            vals_v = []
            vals_q = []
            Z_n = precompute_z_big_endian(int(n_value))
            for s in seeds:
                edges, fam = generate_er_family1d_instance(
                    int(n_value),
                    float(a.p_edge),
                    a.kind,
                    (a.lam_min, a.lam_max),
                    graph_seed=a.graph_seed,
                    periodic_K=int(a.periodic_K),
                    instance_id=s,
                    safety_bounds=False,
                )
                if not edges or fam is None:
                    continue

                cut_mask = build_cut_mask(edges, Z_n)
                J_star, _lam_star = classical_Jstar(fam, cut_mask, int(a.grid))

                hist_vqe = run_outer_vqe_id(
                    int(n_value),
                    cut_mask,
                    fam,
                    J_star=J_star,
                    lam0=a.lam0,
                    outer=a.outer,
                    inner=a.inner,
                    eta0=a.eta0,
                    eta_pow=a.eta_pow,
                    step_clip=a.step_clip,
                    seed=s + 0,
                    L_vqe=a.L_vqe,
                    readout_shots=a.readout_shots,
                    shot_list=shot_list,
                    readout_seed=s + 4242,
                    tail_metric=a.tail_metric,
                    hit_eps=a.hit_eps,
                    topk_frac=a.topk_frac,
                    budget_evals=a.budget_evals,
                )
                hist_qaoa = run_outer_qaoa_full(
                    int(n_value),
                    edges,
                    cut_mask,
                    fam,
                    J_star=J_star,
                    lam0=a.lam0,
                    outer=a.outer,
                    inner=a.inner,
                    eta0=a.eta0,
                    eta_pow=a.eta_pow,
                    step_clip=a.step_clip,
                    seed=s + 100000,
                    p_qaoa=a.p_qaoa,
                    fd_c_frac=a.fd_c_frac,
                    readout_shots=a.readout_shots,
                    shot_list=shot_list,
                    readout_seed=s + 8888,
                    tail_metric=a.tail_metric,
                    hit_eps=a.hit_eps,
                    topk_frac=a.topk_frac,
                    budget_evals=a.budget_evals,
                )
                vals_v.append(
                    step_value_at(
                        hist_vqe["evals_cum"],
                        np.clip(hist_vqe["J_best"] / J_star, 0.0, 1.5),
                        float(a.budget_evals),
                    )
                )
                vals_q.append(
                    step_value_at(
                        hist_qaoa["evals_cum"],
                        np.clip(hist_qaoa["J_best"] / J_star, 0.0, 1.5),
                        float(a.budget_evals),
                    )
                )

            vals_v = np.asarray(vals_v, float)
            vals_q = np.asarray(vals_q, float)
            if vals_v.size == 0 or vals_q.size == 0:
                continue
            n_sweep_stats.append(
                {
                    "n": int(n_value),
                    "vqe_mu": float(np.mean(vals_v)),
                    "vqe_se": _stderr(vals_v),
                    "qaoa_mu": float(np.mean(vals_q)),
                    "qaoa_se": _stderr(vals_q),
                }
            )

        payload = {
            "curves_expect": curves_expect,
            "curves_readout_main": curves_readout_main,
            "curves_tail": curves_tail,
            "curves_readout_byS": curves_readout_byS,
            "evals_curves": evals_curves,
            "n_sweep_stats": n_sweep_stats,
            "rows": rows,
        }
        save_exp08_cache(cache_dir, meta, payload)
        print(f"[cache] Saved exp08 payloads to {cache_dir.resolve()}")

    if not rows:
        raise RuntimeError("No runs completed.")

    B = float(a.budget_evals)

    # mean evals-vs-outer (cost plot)
    evals_mean = {}
    for name, arrs in evals_curves.items():
        if not arrs:
            continue
        M = np.vstack(arrs)  # (N, outer)
        evals_mean[name] = np.mean(M, axis=0)

    # helpers
    def budget_stats(curves: Dict[str, List[Dict]], name: str) -> Tuple[float, float]:
        vals = [step_value_at(r["evals"], r["best"], B) for r in curves[name]]
        vals = np.asarray(vals, float)
        mu = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        return mu, se

    def auc_stats(curves: Dict[str, List[Dict]], name: str) -> Tuple[float, float]:
        vals = [step_auc(r["evals"], r["best"], B) / (B * 1.0) for r in curves[name]]
        vals = np.asarray(vals, float)
        mu = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        return mu, se

    # expectation + readout stats @B
    vqe_exp_mu, vqe_exp_se = budget_stats(curves_expect, "VQE")
    qaoa_exp_mu, qaoa_exp_se = budget_stats(curves_expect, "QAOA")
    vqe_ro_mu, vqe_ro_se = budget_stats(curves_readout_main, "VQE")
    qaoa_ro_mu, qaoa_ro_se = budget_stats(curves_readout_main, "QAOA")

    vqe_exp_auc, vqe_exp_auc_se = auc_stats(curves_expect, "VQE")
    qaoa_exp_auc, qaoa_exp_auc_se = auc_stats(curves_expect, "QAOA")
    vqe_ro_auc, vqe_ro_auc_se = auc_stats(curves_readout_main, "VQE")
    qaoa_ro_auc, qaoa_ro_auc_se = auc_stats(curves_readout_main, "QAOA")

    # gap / closure summaries
    gap_exp = vqe_exp_mu - qaoa_exp_mu
    gap_ro = vqe_ro_mu - qaoa_ro_mu
    gap_closure = float("nan")
    if abs(gap_exp) > 1e-12 and np.isfinite(gap_exp) and np.isfinite(gap_ro):
        gap_closure = 1.0 - (gap_ro / gap_exp)

    # tail factor: how much sampling boosts over expectation
    tail_factor_vqe = float(vqe_ro_mu / max(vqe_exp_mu, 1e-12))
    tail_factor_qaoa = float(qaoa_ro_mu / max(qaoa_exp_mu, 1e-12))

    # tail metric stats (optional)
    tail_mu = tail_se = tail_auc_mu = tail_auc_se = None
    if str(a.tail_metric).lower() != "none" and curves_tail["VQE"]:
        vqe_tail_mu, vqe_tail_se = budget_stats(curves_tail, "VQE")
        qaoa_tail_mu, qaoa_tail_se = budget_stats(curves_tail, "QAOA")
        vqe_tail_auc, vqe_tail_auc_se = auc_stats(curves_tail, "VQE")
        qaoa_tail_auc, qaoa_tail_auc_se = auc_stats(curves_tail, "QAOA")
        tail_mu = (vqe_tail_mu, qaoa_tail_mu)
        tail_se = (vqe_tail_se, qaoa_tail_se)
        tail_auc_mu = (vqe_tail_auc, qaoa_tail_auc)
        tail_auc_se = (vqe_tail_auc_se, qaoa_tail_auc_se)

    # shot sweep stats (optional)
    sweep_stats = []
    for S in shot_list:
        mu_v, se_v = budget_stats(curves_readout_byS[S], "VQE")
        mu_q, se_q = budget_stats(curves_readout_byS[S], "QAOA")
        sweep_stats.append((S, mu_v, se_v, mu_q, se_q))

    shots_sorted = sorted(set(shot_list))
    stats_vqe = [budget_stats(curves_readout_byS[S], "VQE") for S in shots_sorted]
    stats_qaoa = [budget_stats(curves_readout_byS[S], "QAOA") for S in shots_sorted]

    if str(a.tail_metric).lower() == "hit":
        tail_label = r"Best-so-far hit-rate $P(\mathrm{cut}\geq (1-\epsilon)J^*)$"
    elif str(a.tail_metric).lower() == "topk":
        tail_label = rf"Best-so-far top-$k$ rate $P(\mathrm{{cut}}\in \mathrm{{top}}\ {100 * a.topk_frac:.1f}\%)$"
    else:
        tail_label = ""

    objective_x = np.array([step_value_at(r["evals"], r["best"], B) for r in curves_expect["QAOA"]], float)
    objective_y = np.array([step_value_at(r["evals"], r["best"], B) for r in curves_expect["VQE"]], float)
    readout_x = np.array([step_value_at(r["evals"], r["best"], B) for r in curves_readout_main["QAOA"]], float)
    readout_y = np.array([step_value_at(r["evals"], r["best"], B) for r in curves_readout_main["VQE"]], float)

    # write CSV + summary
    write_csv(out / "runs8_bridge_metrics.csv", rows)

    lines = []
    lines.append(
        f"Experiment 8 | kind={a.kind} | seeds={len(rows)} | budget B={B:.1f} evals | graph_seed={a.graph_seed}"
    )
    lines.append(
        f"Settings: eta0={a.eta0} eta_pow={a.eta_pow} step_clip={a.step_clip} "
        f"fd_c_frac={a.fd_c_frac} | readout_shots={a.readout_shots} | supp_shots={a.supp_shots}"
    )
    lines.append("")

    lines.append("Best-so-far ratios at budget B (mean +/- stderr):")
    lines.append(f"  Expectation  ID(VQE):   {vqe_exp_mu:.4f} +/- {vqe_exp_se:.4f}")
    lines.append(f"  Expectation  ID(QAOA):  {qaoa_exp_mu:.4f} +/- {qaoa_exp_se:.4f}")
    lines.append(f"  Readout best ID(VQE):   {vqe_ro_mu:.4f} +/- {vqe_ro_se:.4f}")
    lines.append(f"  Readout best ID(QAOA):  {qaoa_ro_mu:.4f} +/- {qaoa_ro_se:.4f}")

    lines.append("")
    lines.append("Normalized AUC over [0,B] (mean +/- stderr):")
    lines.append(f"  AUC Expectation ID(VQE):  {vqe_exp_auc:.4f} +/- {vqe_exp_auc_se:.4f}")
    lines.append(f"  AUC Expectation ID(QAOA): {qaoa_exp_auc:.4f} +/- {qaoa_exp_auc_se:.4f}")
    lines.append(f"  AUC Readout     ID(VQE):  {vqe_ro_auc:.4f} +/- {vqe_ro_auc_se:.4f}")
    lines.append(f"  AUC Readout     ID(QAOA): {qaoa_ro_auc:.4f} +/- {qaoa_ro_auc_se:.4f}")

    lines.append("")
    lines.append("Gaps at budget B:")
    lines.append(f"  Delta_expect = VQE - QAOA = {gap_exp:.4f}")
    lines.append(f"  Delta_readout= VQE - QAOA = {gap_ro:.4f}")
    if np.isfinite(gap_closure):
        lines.append(f"  Gap-closure from sampling = 1 - Delta_readout/Delta_expect = {gap_closure:.3f}")

    lines.append("")
    lines.append("Tail factor (readout/expectation) at budget B:")
    lines.append(f"  tail_factor(VQE)  = {tail_factor_vqe:.3f}")
    lines.append(f"  tail_factor(QAOA) = {tail_factor_qaoa:.3f}")

    if tail_mu is not None:
        vqe_tail_mu, qaoa_tail_mu = tail_mu
        vqe_tail_se, qaoa_tail_se = tail_se
        vqe_tail_auc, qaoa_tail_auc = tail_auc_mu
        vqe_tail_auc_se, qaoa_tail_auc_se = tail_auc_se

        lines.append("")
        if str(a.tail_metric).lower() == "hit":
            lines.append(f"Tail metric: hit-rate P(cut >= (1-eps)J*) with eps={a.hit_eps:.3f}")
        else:
            lines.append(f"Tail metric: top-k tail P(cut in top {100 * a.topk_frac:.2f}% bitstrings)")
        lines.append(f"  Tail@B ID(VQE):   {vqe_tail_mu:.4f} +/- {vqe_tail_se:.4f}")
        lines.append(f"  Tail@B ID(QAOA):  {qaoa_tail_mu:.4f} +/- {qaoa_tail_se:.4f}")
        lines.append(f"  AUC Tail ID(VQE): {vqe_tail_auc:.4f} +/- {vqe_tail_auc_se:.4f}")
        lines.append(f"  AUC Tail ID(QAOA):{qaoa_tail_auc:.4f} +/- {qaoa_tail_auc_se:.4f}")

    if sweep_stats and a.supp_shots.strip() and a.shots_sweep_plot:
        lines.append("")
        lines.append("Readout shot sweep at budget B (best-of-S):")
        for S, mu_v, se_v, mu_q, se_q in sweep_stats:
            lines.append(f"  S={S:4d} | VQE {mu_v:.4f} +/- {se_v:.4f} | QAOA {mu_q:.4f} +/- {se_q:.4f}")

    lines.append("")
    lines.append("Saved outputs:")
    lines.append(f"  - fig8_expect_vs_evals.{a.fmt}")
    lines.append(f"  - fig8_readout_vs_evals.{a.fmt}")
    lines.append(f"  - fig8_n_sweep.{a.fmt}")
    if str(a.tail_metric).lower() != "none":
        lines.append(f"  - fig8_tailprob_vs_evals.{a.fmt}")
    if a.supp_shots.strip() and a.shots_sweep_plot:
        lines.append(f"  - fig8_readout_shots_sweep.{a.fmt}")
    if a.cost_plot:
        lines.append(f"  - fig8_evals_vs_outer.{a.fmt}")
    if a.tradeoff_plot:
        lines.append(f"  - fig8_objective_pair_scatter.{a.fmt}")
        lines.append(f"  - fig8_readout_pair_scatter.{a.fmt}")
    if a.collage_plot and str(a.tail_metric).lower() != "none":
        lines.append(f"  - fig8_sixpack_collage.{a.fmt}")
    lines.append("  - runs8_bridge_metrics.csv")
    lines.append("  - SUMMARY.txt")

    with open(out / "SUMMARY.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- plots (legend-only; no annotation boxes)
    plot_metric_vs_evals(
        out / f"fig8_expect_vs_evals.{a.fmt}",
        curves_expect,
        budget=B,
        ylabel=r"Best-so-far expectation $J/J^*$",
    )
    plot_metric_vs_evals(
        out / f"fig8_readout_vs_evals.{a.fmt}",
        curves_readout_main,
        budget=B,
        ylabel=rf"Best-of-{int(a.readout_shots)} readout $J/J^*$",
    )
    plot_n_sweep_budget(out / f"fig8_n_sweep.{a.fmt}", n_sweep_stats, readout_shots=int(a.readout_shots))

    if str(a.tail_metric).lower() != "none" and curves_tail["VQE"]:
        plot_tailprob_vs_evals(
            out / f"fig8_tailprob_vs_evals.{a.fmt}", curves_tail, budget=B, tail_label=tail_label, annotate=False
        )

    if a.supp_shots.strip() and a.shots_sweep_plot:
        plot_readout_shots_sweep(
            out / f"fig8_readout_shots_sweep.{a.fmt}", shots_sorted, stats_vqe, stats_qaoa, budget=B, annotate=False
        )

    if a.cost_plot:
        plot_evals_vs_outer(out / f"fig8_evals_vs_outer.{a.fmt}", evals_mean)

    if a.tradeoff_plot:
        plot_pair_scatter(
            out / f"fig8_objective_pair_scatter.{a.fmt}",
            objective_x,
            objective_y,
            xlabel=r"QAOA + FD at budget $B$  ($J/J^*$)",
            ylabel=r"VQE + ID at budget $B$  ($J/J^*$)",
        )
        plot_pair_scatter(
            out / f"fig8_readout_pair_scatter.{a.fmt}",
            readout_x,
            readout_y,
            xlabel=rf"QAOA + FD best-of-{int(a.readout_shots)} at $B$  ($J/J^*$)",
            ylabel=rf"VQE + ID best-of-{int(a.readout_shots)} at $B$  ($J/J^*$)",
        )

    if a.collage_plot:
        plot_sixpack_collage(
            out / f"fig8_sixpack_collage.{a.fmt}",
            curves_expect=curves_expect,
            curves_readout=curves_readout_main,
            budget=B,
            n_sweep_stats=n_sweep_stats,
            readout_shots=int(a.readout_shots),
            shots_sorted=shots_sorted,
            stats_vqe=stats_vqe,
            stats_qaoa=stats_qaoa,
            objective_x=objective_x,
            objective_y=objective_y,
            readout_x=readout_x,
            readout_y=readout_y,
        )

    print("\n".join(lines))
    print("Saved to:", out.resolve())


if __name__ == "__main__":
    main()
