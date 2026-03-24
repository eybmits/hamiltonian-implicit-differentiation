# Appendix Reproduction Note

The publication-facing repository exposes exactly eight public experiments and a
single standard runner interface through `make`.

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

Per experiment:

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

Full final suite:

```bash
make final
```

Cache-backed rerender after style-only changes:

```bash
make rerender
```

## Output Convention

Final paper artifacts are written to:

- `output/exp01` to `output/exp08`

Cache-backed rerender payloads are written to:

- `output/cache/exp02` to `output/cache/exp08`

Each experiment directory contains the checked-in paper PDFs together with
tables and `SUMMARY.txt`.
