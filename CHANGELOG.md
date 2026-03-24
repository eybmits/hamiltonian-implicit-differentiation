# Changelog

All notable changes to this project are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning.

## [0.3.0] - 2026-03-23

### Added
- single `Makefile` runner interface for the publication-facing experiment suite
- canonical checked-in output layout under `output/exp01` to `output/exp08`
- canonical rerender cache layout under `output/cache/exp02` to `output/cache/exp08`
- publication-oriented README and reproducibility documentation aligned with the final paper artifact set

### Changed
- public experiment surface reduced to exactly 8 canonical entrypoints
- final experiment numbering fixed as `Exp 1` to `Exp 8`, with:
  - `Exp 6` = graph-class regime heatmap
  - `Exp 7` = multi-dimensional outer control
  - `Exp 8` = VQE vs QAOA readout bridge
- final output directories, summaries, and tables normalized to the canonical paper artifact tree
- package version advanced to `0.3.0`

### Removed
- legacy wrapper scripts and alias entrypoints from the public repository surface
- exploratory, pre-renaming, and superseded artifact directories from the tracked publication layout

## [0.2.0] - 2026-02-26

### Added
- release documentation (`RELEASING.md`), contribution guide, reproducibility docs
- canonical experiment naming reference in `docs/experiments.md`
- CI package build validation (`python -m build` + `twine check`)
- GitHub release workflow for PyPI publishing via trusted publishing
- wrapper compatibility tests for legacy `experiment*.py` entrypoints

### Changed
- canonical experiment scripts renamed from `experiment*.py` to descriptive `exp*.py` names
- legacy script names retained as deprecation wrappers (supported through `v0.2.x`)
- package metadata expanded for publishability (classifiers, URLs, extras)
- package version advanced to `0.2.0`

### Deprecated
- `experiments/experiment*.py` wrappers are deprecated and scheduled for removal in `v0.3.0`
