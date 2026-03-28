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

## Final Standardized Configuration Summary

The table below matches the actual `paramham-reproduce` targets and the final
paper-facing defaults from the repository entry points. Script IDs refer to the
public experiment entry points; exact commands are listed above and in
`src/paramham/reproduce.py`.

| Thesis item | Script ID | Final setting summary |
| --- | --- | --- |
| `1` (core ID vs. FD demo) | `exp01` | Single-instance mechanism suite for the `linear`, `quadratic`, and `periodic` families, plus the spectrum-compare collage; `seed=7`, `graph_seed=7`, `n=12`, `p_edge=0.45`, `outer=100`, `inner=30`, `L=2`, exact evaluation (`readout_shots=0`), `B=2000`. |
| `2` (systematic cost advantage) | `exp02` | Matched-budget comparison over the `linear`, `quadratic`, and `periodic` families; `seed0=1`, `num_seeds=5`, `graph_seed=7`, `n=12`, `p_edge=0.45`, `periodic_K=6`, `shots in {0,256}`, `outer=100`, `inner=10`, `L=2`, `B=2000`. |
| `3` (readout realism) | `exp03` | Family-grid readout study with `20` instances per family; standardized reruns for both `xaxis=iters` and `xaxis=budget`; `seed0=1`, `graph_seed=7`, `n=12`, `p_edge=0.45`, `periodic_K=6`, `readout_shots=128`, `iters` render uses the standard `bestS` family grid, the canonical budget-axis paper figure is the combined `bestS`/`mode` double figure, `outer=100`, `inner=10`, `L=2`, `B=2000`. |
| `4` (robustness sweep) | `exp04` | Periodic family with `K in {2,3,4,5,6}`; `instances_per_K=10`, `seed=7`, `graph_seed=7`, `n=12`, `p_edge=0.45`, `inner=10`, `L=2`, exact evaluation, `B=2000`. |
| `5` (inner-budget ablation) | `exp05` | Periodic family; `seed0=7`, `num_seeds=7`, `graph_seed=7`, `n=12`, `p_edge=0.45`, `periodic_K=6`, `inner_iters in {14,28,42}`, `restarts in {1,2,4}`, exact evaluation (`shots=0`), `L=2`, `B=2000`. |
| `6` (graph-class regime heatmap) | `exp06` | Regime maps over graph classes `{ER, ring, WS, BA}`; `n in {8,10,12,14}`, `p in {0.20,0.35,0.50,0.65}`, `seeds={1,2,3}`, periodic family with `K=6`, `graph_seed=7`, `L_vqe=2`, `inner=28`, `outer_max=250`, `fd_c_frac=0.10`, exact evaluation, `B=2000`. |
| `7` (multi-dimensional outer control) | `exp07` | Edge-wise `lambda`-vector with one outer parameter per edge; `seed0=7`, `num_instances=8`, `n in {8,9,10,11,12,13,14}`, periodic family with `K=6`, `graph_seed=7`, `inner_iters=28`, `restarts=1`, `outer_max=400`, exact evaluation (`shots=0`), `L=2`, `B=2000`. |
| `8` (architecture study: VQE versus QAOA) | `exp08` | Architecture comparison across expectation-value, readout, tail-metric, shot-sweep, scatter-summary, and six-panel outputs; `seeds={7,...,14}`, `n=12`, `p_edge=0.45`, `graph_seed=7`, periodic family with `K=6`, `L_vqe=2`, `p_qaoa=3`, `outer=10`, `inner=100`, `readout_shots=128`, supplementary shots `{2,4,8,16,32,64,128}`, `tail_metric=hit`, `hit_eps=0.10`, `B=2000`. |

For `exp03`, the repository also contains additional cache-backed rerenders
such as mode-only panels. The combined best-of-`S`/mode budget figure is part
of the standardized `paramham-reproduce exp03` target.

## Output Convention

Final paper artifacts are written to:

- `output/exp01` to `output/exp08`

Cache-backed rerender payloads are written to:

- `output/cache/exp02` to `output/cache/exp08`

Each experiment directory contains the checked-in paper PDFs together with
tables and `SUMMARY.txt`.
