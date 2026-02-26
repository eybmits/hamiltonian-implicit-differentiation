# paramham

Research code and Python package for studying **implicit differentiation (ID)** versus **black-box finite differences (FD)** in bilevel optimization of parametrized Max-Cut Hamiltonians.

## Scope

`paramham` provides reusable simulation utilities plus experiment scripts used for paper-style studies:

- graph and Hamiltonian family generation
- VQE and QAOA statevector simulation
- SPSA-based inner optimization
- bilevel outer-loop metrics and plotting utilities

## Installation

Install from source for development:

```bash
pip install -e ".[dev]"
```

After publishing, install from PyPI:

```bash
pip install paramham
```

If `paramham` is unavailable on PyPI in your environment, use the fallback name documented in the release notes.

## Quickstart

Run unit tests and lint checks:

```bash
ruff check src tests experiments
pytest -v
```

Run one core experiment:

```bash
python experiments/exp01_id_vs_fd_core_demo.py --kind periodic --n 12 --outer 30 --inner 20
```

## Experiments

Canonical script names:

| Script | Purpose |
| --- | --- |
| `exp01_id_vs_fd_core_demo.py` | Core single-instance ID vs FD demo |
| `exp01_id_vs_fd_core_demo_refined_plots.py` | Refined plotting variant of Exp 01 |
| `exp02_budget_efficiency_multiseed.py` | Multi-seed budget efficiency analysis |
| `exp02_budget_efficiency_t20_variant.py` | Budget efficiency variant with t=20 marker |
| `exp03_readout_realism_best_mode.py` | Readout realism (best-of-S and mode) |
| `exp04_robustness_sweep_periodic_k.py` | Robustness sweep over periodic difficulty `K` |
| `exp05_inner_budget_ablation.py` | Inner-budget ablation (`iters x restarts`) |
| `exp06_edgewise_lambda_vector.py` | Edge-wise outer-parameter experiment |
| `exp07_vqe_vs_qaoa_readout_bridge.py` | VQE vs QAOA readout bridge |
| `exp08_id_vs_fd_np_graphclass_heatmap.py` | `(n,p)` heatmap across graph classes |

Legacy `experiment*.py` entrypoints are kept as deprecation wrappers through `v0.2.x` and are removed in `v0.3.0`.

## Reproducibility and Outputs

- default outputs are stored under `outputs/` (per-script subdirectory)
- runs should always set explicit seeds for deterministic comparisons
- detailed conventions are documented in [docs/reproducibility.md](docs/reproducibility.md)

## Documentation Index

- [Experiment Reference](docs/experiments.md)
- [Reproducibility Guide](docs/reproducibility.md)
- [Contributing](CONTRIBUTING.md)
- [Release Process](RELEASING.md)
- [Changelog](CHANGELOG.md)

## License

MIT
