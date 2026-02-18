# Parametrized Hamiltonians -- Implicit Differentiation

Research code for studying **implicit differentiation (ID)** versus **black-box finite differences (FD)** in bilevel optimisation of parametrised quantum Hamiltonians, with a focus on Max-Cut VQE.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

Each experiment script lives in `experiments/` and can be run standalone:

```bash
python experiments/experiment1.py --help
python experiments/experiment2.py --num_seeds 5 --outer 30
```

## Experiments

| Script | Description |
|--------|-------------|
| `experiment1.py` | Single-instance ID vs FD demo with envelope, X-ray, and budget plots |
| `experiment1_plot.py` | Refined plotting variant of experiment 1 |
| `experiment2.py` | Multi-seed budget efficiency (ID vs BB-FD) across families |
| `experiment2_plot.py` | Variant of experiment 2 with t=20 marker visualisation |
| `experiment3.py` | Readout realism: best-of-S and mode metrics |
| `experiment4.py` | Robustness sweep over periodic difficulty K |
| `experiment5.py` | Inner-budget ablation (iterations x restarts) |
| `experiment6.py` | Edge-wise outer parameters (vector lambda) |
| `experiment7.py` | VQE vs QAOA readout bridge |
| `experiment8.py` | Heatmap: ID vs BB-FD across graph classes |

## Package structure

Shared code extracted into `src/paramham/`:

- **seeds** -- reproducible uint32 seed conversion
- **io** -- CSV/LaTeX table writers, CLI list parsers
- **plotting** -- publication-style matplotlib setup
- **graphs** -- random graph generators (Erdos-Renyi, ring, Watts-Strogatz, Barabasi-Albert)
- **maxcut** -- Z-basis precomputation, cut masks, classical diagnostics
- **families** -- parametrised weight families (1D scalar and edge-wise)
- **simulator** -- statevector VQE ansatz and expectation values
- **qaoa** -- QAOA ansatz (from experiment 7)
- **spsa** -- SPSA optimiser with NaN guards
- **metrics** -- step interpolation, AUC, mean/stderr, win rate, tail probability

## License

MIT
