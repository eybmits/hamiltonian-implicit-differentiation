# Reproducibility Guide

## Environment

- Python `>=3.10`
- install with `pip install -e ".[dev]"`

## Validation

```bash
make test
```

## Standard Reproduction Interface

The repository-standard interface is the `Makefile`.

Per-experiment runs:

```bash
make exp01
make exp02
make exp03
make exp04
make exp05
make exp06
make exp07
make exp08
```

Full suite:

```bash
make final
```

Cache-backed rerender:

```bash
make rerender
```

## Output Layout

- final figures and tables: `output/exp01` to `output/exp08`
- cached rerender payloads: `output/cache/exp02` to `output/cache/exp08`

`Exp 1` is cheap enough that `make exp01-rerender` simply reruns it.

## Canonical Final Defaults

Unless an experiment sweeps over the variable:

- `budget_evals = 2000`
- `kind/family = periodic`
- `periodic_K = 6`
- `n = 12`
- `p_edge = 0.45`
- `lam_min = -5.0`
- `lam_max = 5.0`
- `lam0 = 0.8`
- `graph_seed = 7`

## Artifact Policy

The publication repository intentionally keeps:

- final plot PDFs
- CSV and TeX exports
- `SUMMARY.txt`
- current cache directories for rerenderable experiments

The repository does not keep exploratory or pre-renaming artifacts.

## Push Checklist

Before publishing or pushing:

1. Run `make test`.
2. Run `make rerender` if you changed only styling or docs around existing artifacts.
3. Confirm `output/exp01` to `output/exp08` contain the expected final files.
4. Confirm `git status` contains no legacy experiment wrappers or exploratory artifact directories.
