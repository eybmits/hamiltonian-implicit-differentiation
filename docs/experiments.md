# Experiment Reference

This repository exposes exactly 8 public experiment entrypoints.

## Canonical Defaults

Shared final defaults from `paramham.experiment_defaults`:

- `budget_evals = 2000`
- `kind/family = periodic`
- `periodic_K = 6`
- `n = 12`
- `p_edge = 0.45`
- `lam_min = -5.0`
- `lam_max = 5.0`
- `lam0 = 0.8`
- `graph_seed = 7`

Sweep experiments apply these defaults only to non-swept parameters.

## Public Experiments

### Exp 1: Core ID vs FD Demo

- Script: `experiments/exp01_id_vs_fd_core_demo.py`
- Standard command: `make exp01`
- Output: `output/exp01`
- Notes: renders `linear`, `quadratic`, `periodic`, plus the spectrum-compare collage

### Exp 2: Systematic Cost Advantage

- Script: `experiments/exp02_budget_efficiency_multiseed.py`
- Standard command: `make exp02`
- Output: `output/exp02`
- Notes: final matched-budget grid with the `t=20` markers

### Exp 3: Readout Realism

- Script: `experiments/exp03_readout_realism_best_mode.py`
- Standard command: `make exp03`
- Output: `output/exp03/iters` and `output/exp03/budget`

### Exp 4: Robustness Sweep

- Script: `experiments/exp04_robustness_sweep_periodic_k.py`
- Standard command: `make exp04`
- Output: `output/exp04`

### Exp 5: Inner-Budget Ablation

- Script: `experiments/exp05_inner_budget_ablation.py`
- Standard command: `make exp05`
- Output: `output/exp05`

### Exp 6: Graph-Class Regime Heatmap

- Script: `experiments/exp06_graphclass_regime_heatmap.py`
- Standard command: `make exp06`
- Output: `output/exp06`

### Exp 7: Multi-Dimensional Outer Control

- Script: `experiments/exp07_multi_dimensional_outer_control.py`
- Standard command: `make exp07`
- Output: `output/exp07`

### Exp 8: VQE vs QAOA Readout Bridge

- Script: `experiments/exp08_vqe_vs_qaoa_readout_bridge.py`
- Standard command: `make exp08`
- Output: `output/exp08`

## Internal Helpers

Two internal helper modules remain in `experiments/` for implementation reuse:

- `_exp01_core_demo_impl.py`
- `_exp02_budget_efficiency_impl.py`

They are not part of the public experiment surface and are invoked only through the public entrypoints above.
