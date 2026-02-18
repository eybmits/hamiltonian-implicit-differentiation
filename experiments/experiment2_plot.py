#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp0A_budget_efficiency.py (FINAL VERSION: t=20 MARKER VISUALIZATION)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from paramham.families import Family1D
from paramham.graphs import generate_random_graph
from paramham.io import parse_int_list, parse_str_list, write_csv
from paramham.maxcut import build_cut_mask, build_ZZ_edges, classical_Jstar
from paramham.maxcut import precompute_z as precompute_z_big_endian
from paramham.metrics import mean_stderr, step_interp
from paramham.plotting import COLORS, _savefig, set_pub_style
from paramham.seeds import to_uint_seed
from paramham.simulator import vqe_state
from paramham.spsa import spsa_minimize


def stderr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def auc_trapz(budgets: np.ndarray, values: np.ndarray) -> float:
    budgets = np.asarray(budgets, dtype=float)
    values = np.asarray(values, dtype=float)
    if budgets.size < 2:
        return float("nan")
    return float(np.trapz(values, budgets))


def estimate_J_and_zexp(psi, w, ZZ, shots, rng):
    w = np.asarray(w, dtype=np.float64)
    ZZ = np.asarray(ZZ, dtype=np.int8)
    m, K = ZZ.shape
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    if shots is None or shots <= 0:
        zexp = ZZ.astype(np.float64) @ probs
    else:
        idx = rng.choice(np.arange(K), size=int(shots), replace=True, p=probs)
        zexp = np.mean(ZZ[:, idx].astype(np.float64), axis=1)
    zexp = np.clip(np.nan_to_num(zexp, nan=0.0), -1.0, 1.0)
    p_cut = 0.5 * (1.0 - zexp)
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return J, zexp


def vqe_eval(n, params, L, w, ZZ, shots, rng_meas):
    psi = vqe_state(n, params, L)
    return estimate_J_and_zexp(psi, w, ZZ, shots, rng_meas)


def classical_Jstar_max(fam, cut_mask, grid_points):
    J_star, _ = classical_Jstar(fam, cut_mask, grid_points)
    return J_star


@dataclass
class RunHist:
    evals: np.ndarray
    best: np.ndarray
    best_norm: np.ndarray
    final_best_at_budget: float
    auc_at_budget: float


def run_outer(
    *, mode, n, edges, ZZ, fam, lam0, outer, inner, L, shots, seed, eta0, eta_pow, step_clip, c_frac, J_star_max, budget
):
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))
    D = 2 * n * L
    params = np.zeros(D, dtype=np.float64)
    bounds = [(-math.pi, math.pi)] * D
    rng_meas = np.random.default_rng(to_uint_seed(seed + 99991))
    evals_cum = 0.0
    best = -1e18
    evals_trace, best_trace = [], []
    c = float(c_frac * (lam_max - lam_min))

    for t in range(1, outer + 1):
        w = fam.w(lam)

        def energy_fun(pvec):
            J_hat, _ = vqe_eval(n, pvec, L, w, ZZ, shots, rng_meas)
            return -J_hat

        params, _, ev_in = spsa_minimize(energy_fun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals_cum += ev_in
        J, zexp = vqe_eval(n, params, L, w, ZZ, shots, rng_meas)
        evals_cum += 1.0
        best = max(best, float(J))

        if mode == "ID":
            p_cut = 0.5 * (1.0 - zexp)
            g = float(fam.dw_dlam(lam) @ p_cut)
        elif mode == "FD_VALUE":
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))
            w_p = fam.w(lp)

            def efp(pvec):
                J_hat, _ = vqe_eval(n, pvec, L, w_p, ZZ, shots, rng_meas)
                return -J_hat

            p_p, _, evp = spsa_minimize(efp, params, bounds, iters=inner, seed=seed + 1000 * t + 17)
            evals_cum += evp
            Jp, _ = vqe_eval(n, p_p, L, w_p, ZZ, shots, rng_meas)
            evals_cum += 1.0
            w_m = fam.w(lm)

            def efm(pvec):
                J_hat, _ = vqe_eval(n, pvec, L, w_m, ZZ, shots, rng_meas)
                return -J_hat

            p_m, _, evm = spsa_minimize(efm, params, bounds, iters=inner, seed=seed + 1000 * t + 29)
            evals_cum += evm
            Jm, _ = vqe_eval(n, p_m, L, w_m, ZZ, shots, rng_meas)
            evals_cum += 1.0
            best = max(best, float(Jp), float(Jm))
            g = float((Jp - Jm) / (2.0 * c)) if c > 0 else 0.0
            try:
                lam_new_cand = float(np.clip(lam + (eta0 / (t**eta_pow)) * g, lam_min, lam_max))
                cand = [(lam, params), (lp, p_p), (lm, p_m)]
                _, p_closest = min(cand, key=lambda pr: abs(pr[0] - lam_new_cand))
                params = np.asarray(p_closest, dtype=np.float64).copy()
            except Exception:
                pass
        else:
            raise ValueError("mode?")

        eta_t = eta0 / (t**eta_pow)
        step = float(eta_t * g)
        if step_clip is not None:
            step = float(np.clip(step, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))
        evals_trace.append(float(evals_cum))
        best_trace.append(float(best))

    evals_arr = np.asarray(evals_trace, dtype=float)
    best_arr = np.asarray(best_trace, dtype=float)
    best_norm = best_arr / max(1e-12, float(J_star_max))
    B = float(budget)
    budgets = np.linspace(0.0, B, 200)
    best_on_grid = step_interp(evals_arr, best_norm, budgets)
    final_best = float(best_on_grid[-1])
    auc = auc_trapz(budgets, best_on_grid) / max(1e-12, B)
    return RunHist(
        evals=evals_arr, best=best_arr, best_norm=best_norm, final_best_at_budget=final_best, auc_at_budget=float(auc)
    )


@dataclass
class AggCurves:
    budgets: np.ndarray
    mean_id: np.ndarray
    se_id: np.ndarray
    mean_fd: np.ndarray
    se_fd: np.ndarray
    final_id_mean: float
    final_id_se: float
    final_fd_mean: float
    final_fd_se: float
    auc_id_mean: float
    auc_id_se: float
    auc_fd_mean: float
    auc_fd_se: float
    budget_used: float
    id_p20: Optional[Tuple[float, float]]
    fd_p20: Optional[Tuple[float, float]]


def aggregate_runs(runs_id, runs_fd, budget, budget_points):
    B = float(budget)
    budgets = np.linspace(0.0, B, int(budget_points))
    Y_id = np.stack([step_interp(r.evals, r.best_norm, budgets) for r in runs_id], axis=0)
    Y_fd = np.stack([step_interp(r.evals, r.best_norm, budgets) for r in runs_fd], axis=0)
    m_id, se_id_arr = mean_stderr(Y_id, axis=0)
    m_fd, se_fd_arr = mean_stderr(Y_fd, axis=0)
    final_id = np.array([r.final_best_at_budget for r in runs_id], dtype=float)
    final_fd = np.array([r.final_best_at_budget for r in runs_fd], dtype=float)
    auc_id = np.array([r.auc_at_budget for r in runs_id], dtype=float)
    auc_fd = np.array([r.auc_at_budget for r in runs_fd], dtype=float)

    target_idx = 19

    def get_avg_coord(runs, idx):
        xs, ys = [], []
        for r in runs:
            if idx < r.evals.size:
                xs.append(r.evals[idx])
                ys.append(r.best_norm[idx])
        if not xs:
            return None
        return (float(np.mean(xs)), float(np.mean(ys)))

    return AggCurves(
        budgets=budgets,
        mean_id=m_id,
        se_id=se_id_arr,
        mean_fd=m_fd,
        se_fd=se_fd_arr,
        final_id_mean=float(np.mean(final_id)),
        final_id_se=stderr(final_id),
        final_fd_mean=float(np.mean(final_fd)),
        final_fd_se=stderr(final_fd),
        auc_id_mean=float(np.mean(auc_id)),
        auc_id_se=stderr(auc_id),
        auc_fd_mean=float(np.mean(auc_fd)),
        auc_fd_se=stderr(auc_fd),
        budget_used=B,
        id_p20=get_avg_coord(runs_id, target_idx),
        fd_p20=get_avg_coord(runs_fd, target_idx),
    )


def plot_budget_curves(path, curves_by_shots, shots_list):
    set_pub_style(grid=True, base_size=9)
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.25
    mpl.rcParams["grid.linestyle"] = "--"

    if len(shots_list) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True, sharey=True)
        axes = list(axs)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0), constrained_layout=True)
        axes = [ax]

    for ax, shots in zip(axes, shots_list):
        C = curves_by_shots[shots]
        x = C.budgets
        ax.axhline(1.0, color=COLORS["GT"], lw=1.0, ls=":", alpha=0.6, label="_nolegend_")
        ax.plot(x, C.mean_fd, color=COLORS["FD"], lw=1.5, ls="--", label="VQE + FD")
        ax.fill_between(x, C.mean_fd - C.se_fd, C.mean_fd + C.se_fd, color=COLORS["FD"], alpha=0.15, linewidth=0)
        ax.plot(x, C.mean_id, color=COLORS["ID"], lw=1.8, ls="-", label="VQE + ID")
        ax.fill_between(x, C.mean_id - C.se_id, C.mean_id + C.se_id, color=COLORS["ID"], alpha=0.2, linewidth=0)
        ax.set_xlim(0, C.budget_used)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Energy evaluations")
        if ax is axes[0]:
            ax.set_ylabel(r"Norm. Approx. Ratio $\hat{F} / J_{\mathrm{cl}}^*$")

        if C.id_p20:
            x_id, y_id = C.id_p20
            ax.plot(
                x_id,
                y_id,
                marker="o",
                color=COLORS["ID"],
                markersize=6,
                zorder=10,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )
            ax.text(x_id, y_id + 0.05, "t=20", color=COLORS["ID"], fontsize=8, ha="center", fontweight="bold")
        if C.fd_p20:
            x_fd, y_fd = C.fd_p20
            if x_fd <= C.budget_used:
                ax.plot(
                    x_fd,
                    y_fd,
                    marker="s",
                    color=COLORS["FD"],
                    markersize=6,
                    zorder=10,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                )
                ax.text(x_fd, y_fd - 0.08, "t=20", color=COLORS["FD"], fontsize=8, ha="center", fontweight="bold")
                if C.id_p20:
                    ax.annotate(
                        "",
                        xy=(C.id_p20[0], C.id_p20[1]),
                        xytext=(x_fd, y_fd),
                        arrowprops=dict(arrowstyle="-", color="gray", linestyle=":", linewidth=1.0, alpha=0.6),
                    )
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    _savefig(fig, path)
    plt.close(fig)


def write_table_latex(path, table):
    lines = [r"\begin{tabular}{l c c c}", r"\hline", r"Family & Shots & VQE--ID & VQE--FD \\", r"\hline"]
    for (kind, shots_lbl), d in table.items():
        mid, seid = d["ID"]
        mfd, sefd = d["FD"]
        lines.append(f"{kind} & {shots_lbl} & {mid:.3f} $\\pm$ {seid:.3f} & {mfd:.3f} $\\pm$ {sefd:.3f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="out_exp0A")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.45)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=0.8)
    p.add_argument("--families", type=str, default="linear,quadratic,periodic")
    p.add_argument("--periodic_K", type=int, default=6)
    p.add_argument("--shots_list", type=str, default="0,256")
    p.add_argument("--outer", type=int, default=100)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)
    p.add_argument("--grid", type=int, default=401)
    p.add_argument("--budget", type=float, default=2000.0)
    p.add_argument("--budget_points", type=int, default=500)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    families = parse_str_list(args.families)
    shots_list = parse_int_list(args.shots_list)
    Z = precompute_z_big_endian(args.n)
    table_rows = []
    latex_table = {}

    for kind in families:
        curves_by_shots = {}
        for shots in shots_list:
            runs_id, runs_fd = [], []
            for i in range(args.num_seeds):
                seed = args.seed0 + i
                rng_inst = np.random.default_rng(to_uint_seed(seed))
                edges = generate_random_graph(args.n, args.p_edge, rng_inst)
                if not edges:
                    continue
                ZZ = build_ZZ_edges(edges, Z)
                cut_mask = build_cut_mask(edges, Z)
                fam = Family1D(len(edges), kind, (args.lam_min, args.lam_max), rng_inst, periodic_K=args.periodic_K)
                J_star_max = classical_Jstar_max(fam, cut_mask, args.grid)
                budget = float(args.budget)
                runs_id.append(
                    run_outer(
                        mode="ID",
                        n=args.n,
                        edges=edges,
                        ZZ=ZZ,
                        fam=fam,
                        lam0=args.lam0,
                        outer=args.outer,
                        inner=args.inner,
                        L=args.L,
                        shots=shots,
                        seed=to_uint_seed(seed + 111),
                        eta0=args.eta0,
                        eta_pow=args.eta_pow,
                        step_clip=args.step_clip,
                        c_frac=args.c_frac,
                        J_star_max=J_star_max,
                        budget=budget,
                    )
                )
                runs_fd.append(
                    run_outer(
                        mode="FD_VALUE",
                        n=args.n,
                        edges=edges,
                        ZZ=ZZ,
                        fam=fam,
                        lam0=args.lam0,
                        outer=args.outer,
                        inner=args.inner,
                        L=args.L,
                        shots=shots,
                        seed=to_uint_seed(seed + 222),
                        eta0=args.eta0,
                        eta_pow=args.eta_pow,
                        step_clip=args.step_clip,
                        c_frac=args.c_frac,
                        J_star_max=J_star_max,
                        budget=budget,
                    )
                )

            min_cover = min(float(np.max(r.evals)) for r in (runs_id + runs_fd) if r.evals.size)
            budget_used = min(float(args.budget), min_cover)
            curves = aggregate_runs(runs_id, runs_fd, budget=budget_used, budget_points=args.budget_points)
            curves_by_shots[shots] = curves
            shots_lbl = "exact" if shots <= 0 else str(shots)
            table_rows.append(
                {
                    "family": kind,
                    "shots": shots_lbl,
                    "budget": f"{budget_used:.0f}",
                    "ID_mean": f"{curves.final_id_mean:.4f}",
                    "ID_stderr": f"{curves.final_id_se:.4f}",
                    "FD_mean": f"{curves.final_fd_mean:.4f}",
                    "FD_stderr": f"{curves.final_fd_se:.4f}",
                }
            )
            latex_table[(kind, shots_lbl)] = {
                "ID": (curves.final_id_mean, curves.final_id_se),
                "FD": (curves.final_fd_mean, curves.final_fd_se),
            }
            print(f"[done] {kind:9s} {shots_lbl:>5s} | ID {curves.final_id_mean:.3f} | FD {curves.final_fd_mean:.3f}")

        if len(shots_list) == 2:
            plot_budget_curves(out / f"fig0A_budget_{kind}.{args.fmt}", curves_by_shots, shots_list)
        else:
            for shots in shots_list:
                plot_budget_curves(
                    out / f"fig0A_budget_{kind}_shots{shots}.{args.fmt}", {shots: curves_by_shots[shots]}, [shots]
                )

    write_csv(out / "table0A_summary.csv", table_rows)
    write_table_latex(out / "table0A_summary.tex", latex_table)
    print("\nSaved to:", out.resolve())


if __name__ == "__main__":
    main()
