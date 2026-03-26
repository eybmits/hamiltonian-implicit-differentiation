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
	$(ENV) $(PYTHON) -m paramham.reproduce test

final:
	$(ENV) $(PYTHON) -m paramham.reproduce final

rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce rerender

exp01:
	$(ENV) $(PYTHON) -m paramham.reproduce exp01

exp01-rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce exp01-rerender

exp02:
	$(ENV) $(PYTHON) -m paramham.reproduce exp02

exp02-rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce exp02-rerender

exp03:
	$(ENV) $(PYTHON) -m paramham.reproduce exp03

exp03-rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce exp03-rerender

exp04:
	$(ENV) $(PYTHON) -m paramham.reproduce exp04

exp04-rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce exp04-rerender

exp05:
	$(ENV) $(PYTHON) -m paramham.reproduce exp05

exp05-rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce exp05-rerender

exp06:
	$(ENV) $(PYTHON) -m paramham.reproduce exp06

exp06-rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce exp06-rerender

exp07:
	$(ENV) $(PYTHON) -m paramham.reproduce exp07

exp07-rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce exp07-rerender

exp08:
	$(ENV) $(PYTHON) -m paramham.reproduce exp08

exp08-rerender:
	$(ENV) $(PYTHON) -m paramham.reproduce exp08-rerender

clean:
	rm -rf output/exp01 output/exp02 output/exp03 output/exp04 output/exp05 output/exp06 output/exp07 output/exp08 output/cache
