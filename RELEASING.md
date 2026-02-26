# Releasing

## Versioning Policy

- Semantic Versioning (`MAJOR.MINOR.PATCH`)
- `v0.2.x` keeps legacy experiment wrappers
- remove legacy wrappers in `v0.3.0`

## Pre-Release Checklist

```bash
ruff check src tests experiments
ruff format --check src tests experiments
pytest -v
python -m build
twine check dist/*
```

## GitHub Release Steps

1. update version in `pyproject.toml` and `src/paramham/__init__.py`
2. update `CHANGELOG.md`
3. commit and push to `main`
4. create and push tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

5. GitHub Actions `release.yml` builds and publishes artifacts to PyPI

## PyPI Trusted Publishing

Configure a PyPI trusted publisher for this repository and workflow:

- owner/repo: `eybmits/parameterized-hamiltonians-id`
- workflow: `.github/workflows/release.yml`
- environment: `pypi`

No API token is required when trusted publishing is configured correctly.

## Package Name Fallback

Primary distribution name is `paramham`.

If `paramham` is unavailable on PyPI, switch to `paramham-id` consistently in:

- `pyproject.toml` (`[project].name`)
- README install instructions
- release notes
