"""Central reproduction runner for the publication experiment suite."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_MPLCONFIGDIR = "/tmp/mpl"

EXPERIMENT_TARGETS = (
    "exp01",
    "exp02",
    "exp03",
    "exp04",
    "exp05",
    "exp06",
    "exp07",
    "exp08",
)


def _command(*parts: str) -> tuple[str, ...]:
    return tuple(parts)


TARGET_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "test": (
        _command("-m", "pytest"),
    ),
    "exp01": (
        _command("experiments/exp01_id_vs_fd_core_demo.py", "--suite", "--fmt", "pdf", "--out", "output/exp01"),
    ),
    "exp01-rerender": (
        _command("experiments/exp01_id_vs_fd_core_demo.py", "--suite", "--fmt", "pdf", "--out", "output/exp01"),
    ),
    "exp02": (
        _command(
            "experiments/exp02_budget_efficiency_multiseed.py",
            "--fmt",
            "pdf",
            "--out",
            "output/exp02",
            "--cache_dir",
            "output/cache/exp02",
        ),
    ),
    "exp02-rerender": (
        _command(
            "experiments/exp02_budget_efficiency_multiseed.py",
            "--render_only",
            "--fmt",
            "pdf",
            "--out",
            "output/exp02",
            "--cache_dir",
            "output/cache/exp02",
        ),
    ),
    "exp03": (
        _command(
            "experiments/exp03_readout_realism_best_mode.py",
            "--xaxis",
            "iters",
            "--fmt",
            "pdf",
            "--out",
            "output/exp03/iters",
            "--cache_dir",
            "output/cache/exp03/iters",
        ),
        _command(
            "experiments/exp03_readout_realism_best_mode.py",
            "--xaxis",
            "budget",
            "--fmt",
            "pdf",
            "--out",
            "output/exp03/budget",
            "--cache_dir",
            "output/cache/exp03/budget",
        ),
    ),
    "exp03-rerender": (
        _command(
            "experiments/exp03_readout_realism_best_mode.py",
            "--xaxis",
            "iters",
            "--render_only",
            "--fmt",
            "pdf",
            "--out",
            "output/exp03/iters",
            "--cache_dir",
            "output/cache/exp03/iters",
        ),
        _command(
            "experiments/exp03_readout_realism_best_mode.py",
            "--xaxis",
            "budget",
            "--render_only",
            "--fmt",
            "pdf",
            "--out",
            "output/exp03/budget",
            "--cache_dir",
            "output/cache/exp03/budget",
        ),
    ),
    "exp04": (
        _command(
            "experiments/exp04_robustness_sweep_periodic_k.py",
            "--fmt",
            "pdf",
            "--out",
            "output/exp04",
            "--cache_dir",
            "output/cache/exp04",
        ),
    ),
    "exp04-rerender": (
        _command(
            "experiments/exp04_robustness_sweep_periodic_k.py",
            "--render_only",
            "--fmt",
            "pdf",
            "--out",
            "output/exp04",
            "--cache_dir",
            "output/cache/exp04",
        ),
    ),
    "exp05": (
        _command(
            "experiments/exp05_inner_budget_ablation.py",
            "--fmt",
            "pdf",
            "--out",
            "output/exp05",
            "--cache_dir",
            "output/cache/exp05",
        ),
    ),
    "exp05-rerender": (
        _command(
            "experiments/exp05_inner_budget_ablation.py",
            "--render_only",
            "--fmt",
            "pdf",
            "--out",
            "output/exp05",
            "--cache_dir",
            "output/cache/exp05",
        ),
    ),
    "exp06": (
        _command(
            "experiments/exp06_graphclass_regime_heatmap.py",
            "--fmt",
            "pdf",
            "--out",
            "output/exp06",
            "--cache_dir",
            "output/cache/exp06",
            "--n_list",
            "8,10,12,14",
            "--p_list",
            "0.20,0.35,0.50,0.65",
        ),
    ),
    "exp06-rerender": (
        _command(
            "experiments/exp06_graphclass_regime_heatmap.py",
            "--render_only",
            "--fmt",
            "pdf",
            "--out",
            "output/exp06",
            "--cache_dir",
            "output/cache/exp06",
            "--n_list",
            "8,10,12,14",
            "--p_list",
            "0.20,0.35,0.50,0.65",
        ),
    ),
    "exp07": (
        _command(
            "experiments/exp07_multi_dimensional_outer_control.py",
            "--fmt",
            "pdf",
            "--out",
            "output/exp07",
            "--cache_dir",
            "output/cache/exp07",
        ),
    ),
    "exp07-rerender": (
        _command(
            "experiments/exp07_multi_dimensional_outer_control.py",
            "--render_only",
            "--fmt",
            "pdf",
            "--out",
            "output/exp07",
            "--cache_dir",
            "output/cache/exp07",
        ),
    ),
    "exp08": (
        _command(
            "experiments/exp08_vqe_vs_qaoa_readout_bridge.py",
            "--fmt",
            "pdf",
            "--out",
            "output/exp08",
            "--cache_dir",
            "output/cache/exp08",
        ),
    ),
    "exp08-rerender": (
        _command(
            "experiments/exp08_vqe_vs_qaoa_readout_bridge.py",
            "--render_only",
            "--fmt",
            "pdf",
            "--out",
            "output/exp08",
            "--cache_dir",
            "output/cache/exp08",
        ),
    ),
}

RERENDER_TARGETS = tuple(f"{name}-rerender" for name in EXPERIMENT_TARGETS)
PUBLIC_TARGETS = ("test",) + EXPERIMENT_TARGETS + RERENDER_TARGETS + ("all", "final", "rerender")


def _target_commands(target: str) -> tuple[tuple[str, ...], ...]:
    if target in {"all", "final"}:
        expanded: list[tuple[str, ...]] = []
        for name in EXPERIMENT_TARGETS:
            expanded.extend(_target_commands(name))
        return tuple(expanded)
    if target == "rerender":
        expanded = []
        for name in RERENDER_TARGETS:
            expanded.extend(_target_commands(name))
        return tuple(expanded)
    try:
        return TARGET_COMMANDS[target]
    except KeyError as exc:
        raise KeyError(f"Unknown target: {target}") from exc


def _runner_env() -> dict[str, str]:
    env = os.environ.copy()
    src_entry = str(SRC_DIR.resolve())
    existing = env.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    normalized_parts = [src_entry]
    seen = {src_entry}
    for part in parts:
        candidate = Path(part)
        resolved = str((REPO_ROOT / candidate).resolve()) if not candidate.is_absolute() else str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized_parts.append(part)
    env["PYTHONPATH"] = os.pathsep.join(normalized_parts)
    env.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _format_command(command: tuple[str, ...], env: dict[str, str]) -> str:
    env_prefix = f"PYTHONPATH={shlex.quote(env['PYTHONPATH'])} MPLCONFIGDIR={shlex.quote(env['MPLCONFIGDIR'])}"
    python_cmd = (sys.executable, *command)
    return f"{env_prefix} {shlex.join(python_cmd)}"


def _run_target(target: str, *, dry_run: bool) -> int:
    commands = _target_commands(target)
    env = _runner_env()
    for command in commands:
        formatted = _format_command(command, env)
        if dry_run:
            print(formatted)
            continue
        print(f"[run] {formatted}", flush=True)
        subprocess.run([sys.executable, *command], cwd=REPO_ROOT, env=env, check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical reproduction runner for the publication experiment suite."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=PUBLIC_TARGETS,
        help="suite target to run; default: all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved commands without executing them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run_target(args.target, dry_run=bool(args.dry_run))
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode) if exc.returncode is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
