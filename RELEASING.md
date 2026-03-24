# Releasing

## Versioning Policy

- Semantic Versioning (`MAJOR.MINOR.PATCH`)
- `v0.3.x` exposes exactly 8 public experiment entrypoints and no legacy wrappers
- checked-in publication artifacts live under `output/exp01` to `output/exp08`
- rerender caches live under `output/cache/exp02` to `output/cache/exp08`

## Pre-Release Checklist

```bash
make test
ruff check src tests experiments
ruff format --check src tests experiments
python -m pytest -v
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

## Publication Repo Checklist

- ensure `README.md`, `docs/experiments.md`, and `docs/reproducibility.md` all describe the same canonical `Exp 1` to `Exp 8` suite
- ensure `output/exp01` to `output/exp08` contain the intended final PDFs and `SUMMARY.txt`
- ensure `output/cache/exp02` to `output/cache/exp08` are current if cache-backed rerendering is part of the release
- ensure no legacy output folders or deprecated experiment wrappers remain in `git status`

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
