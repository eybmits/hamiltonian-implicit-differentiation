# Contributing

## Development Setup

```bash
pip install -e ".[dev]"
```

## Required Checks

```bash
ruff check src tests experiments
ruff format --check src tests experiments
pytest -v
```

## Contribution Guidelines

- keep experiment entrypoints in `experiments/`
- keep reusable logic in `src/paramham/`
- add tests for behavior changes
- document interface/output changes in `README.md` and `docs/`
- update `CHANGELOG.md` for user-visible changes

## Pull Requests

Include:

- scope summary
- test evidence (commands + pass/fail)
- migration note for renamed/removed interfaces
