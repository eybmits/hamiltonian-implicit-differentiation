# Changelog

All notable changes to this project are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning.

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
