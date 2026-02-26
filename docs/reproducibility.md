# Reproducibility Guide

## Baseline Environment

- Python: `3.10` to `3.12`
- Install: `pip install -e ".[dev]"`

## Standard Validation

Run before and after experiment changes:

```bash
ruff check src tests experiments
ruff format --check src tests experiments
pytest -v
```

## Reproducible Runs

1. Always pass explicit seeds (`--seed`, `--seed0`, or `--seeds`).
2. Use explicit output directory (`--out outputs/<experiment-id>/<run-id>`).
3. Record invocation command with commit hash.

Example:

```bash
git rev-parse HEAD
python experiments/exp02_budget_efficiency_multiseed.py \
  --kinds periodic,linear \
  --seeds 1,2,3 \
  --outer 30 \
  --inner 28 \
  --out outputs/exp02_budget_efficiency_multiseed/run_2026-02-26_a
```

## Output Convention

Recommended output layout:

- `outputs/<experiment-id>/<run-id>/fig*.pdf`
- `outputs/<experiment-id>/<run-id>/*.csv`
- `outputs/<experiment-id>/<run-id>/*summary*.txt`

Do not commit generated artifacts unless intentionally versioned for publication.

## Determinism Notes

- Stochasticity is controlled through NumPy RNG seeds and SPSA seeds.
- If shot-noise is enabled (`--shots > 0`), repeated runs with identical seeds should remain deterministic.
- Different Python or NumPy versions can slightly change floating-point behavior; pin versions for strict reproduction.
