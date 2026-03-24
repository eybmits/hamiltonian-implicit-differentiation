"""Entry-point checks for the cleaned public experiment surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

PUBLIC_SCRIPTS = [
    "exp01_id_vs_fd_core_demo.py",
    "exp02_budget_efficiency_multiseed.py",
    "exp03_readout_realism_best_mode.py",
    "exp04_robustness_sweep_periodic_k.py",
    "exp05_inner_budget_ablation.py",
    "exp06_graphclass_regime_heatmap.py",
    "exp07_multi_dimensional_outer_control.py",
    "exp08_vqe_vs_qaoa_readout_bridge.py",
]

REMOVED_SCRIPTS = [
    "exp03c_vqe_vs_qaoa_readout_bridge.py",
    "exp06_edgewise_lambda_vector.py",
    "exp07_vqe_vs_qaoa_readout_bridge.py",
    "exp08_id_vs_fd_np_graphclass_heatmap.py",
    "experiment1.py",
    "experiment1_plot.py",
    "experiment2.py",
    "experiment2_plot.py",
    "experiment3.py",
    "experiment4.py",
    "experiment5.py",
    "experiment6.py",
    "experiment7.py",
    "experiment8.py",
]


def _run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, str(EXPERIMENTS_DIR / script_name), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


@pytest.mark.parametrize("script_name", PUBLIC_SCRIPTS)
def test_public_experiment_scripts_expose_help(script_name: str):
    proc = _run_help(script_name)
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


@pytest.mark.parametrize("script_name", REMOVED_SCRIPTS)
def test_removed_legacy_scripts_are_absent(script_name: str):
    assert not (EXPERIMENTS_DIR / script_name).exists()


def test_public_surface_contains_exactly_eight_experiments():
    files = sorted(p.name for p in EXPERIMENTS_DIR.glob("exp*.py") if not p.name.startswith("_"))
    assert files == PUBLIC_SCRIPTS


def test_exp01_help_lists_suite_flag():
    proc = _run_help("exp01_id_vs_fd_core_demo.py")
    assert proc.returncode == 0, proc.stderr
    assert "--suite" in proc.stdout


def test_exp02_help_lists_cache_and_render_flags():
    proc = _run_help("exp02_budget_efficiency_multiseed.py")
    assert proc.returncode == 0, proc.stderr
    assert "--cache_dir" in proc.stdout
    assert "--recompute" in proc.stdout
    assert "--render_only" in proc.stdout
    assert "--ymin" in proc.stdout


def test_exp03_help_lists_family_metric_and_cache_flags():
    proc = _run_help("exp03_readout_realism_best_mode.py")
    assert proc.returncode == 0, proc.stderr
    assert "--families" in proc.stdout
    assert "--metric" in proc.stdout
    assert "--cache_dir" in proc.stdout
    assert "--render_only" in proc.stdout


def test_exp04_help_lists_cache_flags():
    proc = _run_help("exp04_robustness_sweep_periodic_k.py")
    assert proc.returncode == 0, proc.stderr
    assert "--cache_dir" in proc.stdout
    assert "--recompute" in proc.stdout
    assert "--render_only" in proc.stdout


def test_exp05_help_lists_cache_flags():
    proc = _run_help("exp05_inner_budget_ablation.py")
    assert proc.returncode == 0, proc.stderr
    assert "--cache_dir" in proc.stdout
    assert "--recompute" in proc.stdout
    assert "--render_only" in proc.stdout


def test_exp06_graphclass_help_lists_cache_flags():
    proc = _run_help("exp06_graphclass_regime_heatmap.py")
    assert proc.returncode == 0, proc.stderr
    assert "--cache_dir" in proc.stdout
    assert "--recompute" in proc.stdout
    assert "--render_only" in proc.stdout


def test_exp07_multidim_help_lists_cache_and_nsweep_flags():
    proc = _run_help("exp07_multi_dimensional_outer_control.py")
    assert proc.returncode == 0, proc.stderr
    assert "--cache_dir" in proc.stdout
    assert "--recompute" in proc.stdout
    assert "--render_only" in proc.stdout
    assert "--n_sweep" in proc.stdout


def test_exp08_bridge_help_lists_cache_and_collage_flags():
    proc = _run_help("exp08_vqe_vs_qaoa_readout_bridge.py")
    assert proc.returncode == 0, proc.stderr
    assert "--cache_dir" in proc.stdout
    assert "--recompute" in proc.stdout
    assert "--render_only" in proc.stdout
    assert "--no_collage_plot" in proc.stdout


def test_makefile_supports_final_dry_run():
    proc = subprocess.run(
        ["make", "-n", "final"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "exp01_id_vs_fd_core_demo.py" in proc.stdout
    assert "exp08_vqe_vs_qaoa_readout_bridge.py" in proc.stdout
