"""Entry-point compatibility checks for experiment scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

CANONICAL_SCRIPTS = [
    "exp01_id_vs_fd_core_demo.py",
    "exp02_budget_efficiency_multiseed.py",
    "exp03_readout_realism_best_mode.py",
]

LEGACY_TO_CANONICAL = [
    ("experiment1.py", "exp01_id_vs_fd_core_demo.py"),
    ("experiment1_plot.py", "exp01_id_vs_fd_core_demo_refined_plots.py"),
    ("experiment2.py", "exp02_budget_efficiency_multiseed.py"),
    ("experiment2_plot.py", "exp02_budget_efficiency_t20_variant.py"),
    ("experiment3.py", "exp03_readout_realism_best_mode.py"),
    ("experiment4.py", "exp04_robustness_sweep_periodic_k.py"),
    ("experiment5.py", "exp05_inner_budget_ablation.py"),
    ("experiment6.py", "exp06_edgewise_lambda_vector.py"),
    ("experiment7.py", "exp07_vqe_vs_qaoa_readout_bridge.py"),
    ("experiment8.py", "exp08_id_vs_fd_np_graphclass_heatmap.py"),
]


def _run_help(script_name: str, *, warnings_default: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if warnings_default:
        env["PYTHONWARNINGS"] = "default"
    return subprocess.run(
        [sys.executable, str(EXPERIMENTS_DIR / script_name), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


@pytest.mark.parametrize("script_name", CANONICAL_SCRIPTS)
def test_canonical_experiment_scripts_expose_help(script_name: str):
    proc = _run_help(script_name)
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


@pytest.mark.parametrize("legacy_name,canonical_name", LEGACY_TO_CANONICAL)
def test_legacy_experiment_wrappers_expose_help_and_warn(legacy_name: str, canonical_name: str):
    proc = _run_help(legacy_name, warnings_default=True)
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
    assert canonical_name in proc.stderr
    assert "deprecated" in proc.stderr.lower()
