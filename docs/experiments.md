# Experiment Reference

This repository ships two entrypoint layers:

- canonical scripts: `experiments/exp*.py`
- legacy wrappers: `experiments/experiment*.py` (deprecated, removed in `v0.3.0`)

## Canonical Experiments

### Exp 01: Core ID vs FD Demo
- Script: `experiments/exp01_id_vs_fd_core_demo.py`
- Purpose: single-instance bilevel demo and publication plots
- Default output dir: `outputs/exp01_id_vs_fd_core_demo`

### Exp 01 (Refined Plots)
- Script: `experiments/exp01_id_vs_fd_core_demo_refined_plots.py`
- Purpose: styling variant for Exp 01 figures
- Default output dir: `outputs/exp01_id_vs_fd_core_demo_refined_plots`

### Exp 02: Budget Efficiency (Multiseed)
- Script: `experiments/exp02_budget_efficiency_multiseed.py`
- Purpose: ID vs FD efficiency across seeds/families
- Default output dir: `outputs/exp02_budget_efficiency_multiseed`

### Exp 02 (t=20 Variant)
- Script: `experiments/exp02_budget_efficiency_t20_variant.py`
- Purpose: variant plots with `t=20` marker emphasis
- Default output dir: `outputs/exp02_budget_efficiency_t20_variant`

### Exp 03: Readout Realism
- Script: `experiments/exp03_readout_realism_best_mode.py`
- Purpose: expectation improvements versus sampled-solution quality
- Default output dir: `outputs/exp03_readout_realism_best_mode`

### Exp 04: Robustness Sweep
- Script: `experiments/exp04_robustness_sweep_periodic_k.py`
- Purpose: robustness over periodic difficulty parameter `K`
- Default output dir: `outputs/exp04_robustness_sweep_periodic_k`

### Exp 05: Inner-Budget Ablation
- Script: `experiments/exp05_inner_budget_ablation.py`
- Purpose: sensitivity to inner optimizer budget and restarts
- Default output dir: `outputs/exp05_inner_budget_ablation`

### Exp 06: Edge-wise Lambda Vector
- Script: `experiments/exp06_edgewise_lambda_vector.py`
- Purpose: high-dimensional outer optimization (`lambda` per edge)
- Default output dir: `outputs/exp06_edgewise_lambda_vector`

### Exp 07: VQE vs QAOA Readout Bridge
- Script: `experiments/exp07_vqe_vs_qaoa_readout_bridge.py`
- Purpose: compare expectation and sampled-readout progress
- Default output dir: `outputs/exp07_vqe_vs_qaoa_readout_bridge`

### Exp 08: ID vs FD Heatmap over Graph Classes
- Script: `experiments/exp08_id_vs_fd_np_graphclass_heatmap.py`
- Purpose: `(n,p)` heatmap studies across graph families
- Default output dir: `outputs/exp08_id_vs_fd_np_graphclass_heatmap`

## Legacy-to-Canonical Mapping

| Legacy script | Canonical replacement |
| --- | --- |
| `experiment1.py` | `exp01_id_vs_fd_core_demo.py` |
| `experiment1_plot.py` | `exp01_id_vs_fd_core_demo_refined_plots.py` |
| `experiment2.py` | `exp02_budget_efficiency_multiseed.py` |
| `experiment2_plot.py` | `exp02_budget_efficiency_t20_variant.py` |
| `experiment3.py` | `exp03_readout_realism_best_mode.py` |
| `experiment4.py` | `exp04_robustness_sweep_periodic_k.py` |
| `experiment5.py` | `exp05_inner_budget_ablation.py` |
| `experiment6.py` | `exp06_edgewise_lambda_vector.py` |
| `experiment7.py` | `exp07_vqe_vs_qaoa_readout_bridge.py` |
| `experiment8.py` | `exp08_id_vs_fd_np_graphclass_heatmap.py` |
