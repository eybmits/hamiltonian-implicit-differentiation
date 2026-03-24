# paramham

`paramham` is a research codebase and publication artifact repository for studying **implicit differentiation (ID)** versus **black-box finite differences (FD)** in bilevel optimization of parametrized Max-Cut Hamiltonians.

This repository is intentionally trimmed to the final paper-facing experiment suite:

- exactly **8 public experiments**
- one clean output tree under `output/exp01` to `output/exp08`
- cache-backed rerender support under `output/cache`
- a single standard runner interface through `make`

## Installation

```bash
pip install -e ".[dev]"
```

## Standard Commands

Run from the repository root:

```bash
make test
make exp01
make exp02
make final
make rerender
```

The standard environment is handled by the `Makefile` (`PYTHONPATH=src`, `MPLCONFIGDIR=/tmp/mpl`).

## Canonical Experiment Suite

| Experiment | Public script | Output folder |
| --- | --- | --- |
| `Exp 1` | `experiments/exp01_id_vs_fd_core_demo.py` | `output/exp01` |
| `Exp 2` | `experiments/exp02_budget_efficiency_multiseed.py` | `output/exp02` |
| `Exp 3` | `experiments/exp03_readout_realism_best_mode.py` | `output/exp03` |
| `Exp 4` | `experiments/exp04_robustness_sweep_periodic_k.py` | `output/exp04` |
| `Exp 5` | `experiments/exp05_inner_budget_ablation.py` | `output/exp05` |
| `Exp 6` | `experiments/exp06_graphclass_regime_heatmap.py` | `output/exp06` |
| `Exp 7` | `experiments/exp07_multi_dimensional_outer_control.py` | `output/exp07` |
| `Exp 8` | `experiments/exp08_vqe_vs_qaoa_readout_bridge.py` | `output/exp08` |

There are no legacy wrapper scripts in the public repo surface.

## Canonical Final Defaults

Unless an experiment sweeps over that variable, the final paper runs use:

- `budget_evals = 2000`
- `kind/family = periodic`
- `periodic_K = 6`
- `n = 12`
- `p_edge = 0.45`
- `lam_min = -5.0`
- `lam_max = 5.0`
- `lam0 = 0.8`
- `graph_seed = 7`

Experiment-specific sweep settings are documented in [docs/experiments.md](docs/experiments.md).

## Output Policy

Versioned paper artifacts live in:

- `output/exp01` to `output/exp08`
- `output/cache/exp02` to `output/cache/exp08`

Each experiment output folder contains:

- final figure PDFs
- CSV and TeX exports where applicable
- `SUMMARY.txt`

## Documentation

- [Experiment Reference](docs/experiments.md)
- [Reproducibility Guide](docs/reproducibility.md)
- [Output Manifest](output/README.md)
- [Contributing](CONTRIBUTING.md)
- [Release Process](RELEASING.md)
- [Changelog](CHANGELOG.md)

## License

MIT
