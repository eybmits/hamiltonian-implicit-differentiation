PYTHON ?= python
ENV = PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl

.PHONY: help test final rerender clean exp01 exp02 exp03 exp04 exp05 exp06 exp07 exp08 exp01-rerender exp02-rerender exp03-rerender exp04-rerender exp05-rerender exp06-rerender exp07-rerender exp08-rerender

help:
	@printf "Targets:\n"
	@printf "  make test\n"
	@printf "  make exp01 ... make exp08\n"
	@printf "  make final\n"
	@printf "  make exp01-rerender ... make exp08-rerender\n"
	@printf "  make rerender\n"
	@printf "  make clean\n"

test:
	$(ENV) $(PYTHON) -m pytest

final: exp01 exp02 exp03 exp04 exp05 exp06 exp07 exp08

rerender: exp01-rerender exp02-rerender exp03-rerender exp04-rerender exp05-rerender exp06-rerender exp07-rerender exp08-rerender

exp01:
	$(ENV) $(PYTHON) experiments/exp01_id_vs_fd_core_demo.py --suite --fmt pdf --out output/exp01

exp01-rerender:
	$(ENV) $(PYTHON) experiments/exp01_id_vs_fd_core_demo.py --suite --fmt pdf --out output/exp01

exp02:
	$(ENV) $(PYTHON) experiments/exp02_budget_efficiency_multiseed.py --fmt pdf --out output/exp02 --cache_dir output/cache/exp02

exp02-rerender:
	$(ENV) $(PYTHON) experiments/exp02_budget_efficiency_multiseed.py --render_only --fmt pdf --out output/exp02 --cache_dir output/cache/exp02

exp03:
	$(ENV) $(PYTHON) experiments/exp03_readout_realism_best_mode.py --xaxis iters --fmt pdf --out output/exp03/iters --cache_dir output/cache/exp03/iters
	$(ENV) $(PYTHON) experiments/exp03_readout_realism_best_mode.py --xaxis budget --fmt pdf --out output/exp03/budget --cache_dir output/cache/exp03/budget

exp03-rerender:
	$(ENV) $(PYTHON) experiments/exp03_readout_realism_best_mode.py --xaxis iters --render_only --fmt pdf --out output/exp03/iters --cache_dir output/cache/exp03/iters
	$(ENV) $(PYTHON) experiments/exp03_readout_realism_best_mode.py --xaxis budget --render_only --fmt pdf --out output/exp03/budget --cache_dir output/cache/exp03/budget

exp04:
	$(ENV) $(PYTHON) experiments/exp04_robustness_sweep_periodic_k.py --fmt pdf --out output/exp04 --cache_dir output/cache/exp04

exp04-rerender:
	$(ENV) $(PYTHON) experiments/exp04_robustness_sweep_periodic_k.py --render_only --fmt pdf --out output/exp04 --cache_dir output/cache/exp04

exp05:
	$(ENV) $(PYTHON) experiments/exp05_inner_budget_ablation.py --fmt pdf --out output/exp05 --cache_dir output/cache/exp05

exp05-rerender:
	$(ENV) $(PYTHON) experiments/exp05_inner_budget_ablation.py --render_only --fmt pdf --out output/exp05 --cache_dir output/cache/exp05

exp06:
	$(ENV) $(PYTHON) experiments/exp06_graphclass_regime_heatmap.py --fmt pdf --out output/exp06 --cache_dir output/cache/exp06 --n_list 8,10,12,14 --p_list 0.20,0.35,0.50,0.65

exp06-rerender:
	$(ENV) $(PYTHON) experiments/exp06_graphclass_regime_heatmap.py --render_only --fmt pdf --out output/exp06 --cache_dir output/cache/exp06 --n_list 8,10,12,14 --p_list 0.20,0.35,0.50,0.65

exp07:
	$(ENV) $(PYTHON) experiments/exp07_multi_dimensional_outer_control.py --fmt pdf --out output/exp07 --cache_dir output/cache/exp07

exp07-rerender:
	$(ENV) $(PYTHON) experiments/exp07_multi_dimensional_outer_control.py --render_only --fmt pdf --out output/exp07 --cache_dir output/cache/exp07

exp08:
	$(ENV) $(PYTHON) experiments/exp08_vqe_vs_qaoa_readout_bridge.py --fmt pdf --out output/exp08 --cache_dir output/cache/exp08

exp08-rerender:
	$(ENV) $(PYTHON) experiments/exp08_vqe_vs_qaoa_readout_bridge.py --render_only --fmt pdf --out output/exp08 --cache_dir output/cache/exp08

clean:
	rm -rf output/exp01 output/exp02 output/exp03 output/exp04 output/exp05 output/exp06 output/exp07 output/exp08 output/cache
