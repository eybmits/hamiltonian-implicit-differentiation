# Appendix Reproduction Note

The publication-facing repository exposes exactly eight public experiments and a
single standard runner interface through `paramham-reproduce`.

## Environment

```bash
git clone https://github.com/eybmits/parameterized-hamiltonians-id.git
cd parameterized-hamiltonians-id

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Validation

```bash
make test
```

## Canonical Reproduction Commands

Full final suite:

```bash
paramham-reproduce all
```

Cache-backed rerender after style-only changes:

```bash
paramham-reproduce rerender
```

Per-experiment targets remain available when needed:

```bash
paramham-reproduce exp01
paramham-reproduce exp02
paramham-reproduce exp03
paramham-reproduce exp04
paramham-reproduce exp05
paramham-reproduce exp06
paramham-reproduce exp07
paramham-reproduce exp08
```

The `Makefile` remains available as a thin wrapper around the same target registry:

```bash
make final
make rerender
```

## Output Convention

Final paper artifacts are written to:

- `output/exp01` to `output/exp08`

Cache-backed rerender payloads are written to:

- `output/cache/exp02` to `output/cache/exp08`

Each experiment directory contains the checked-in paper PDFs together with
tables and `SUMMARY.txt`.
