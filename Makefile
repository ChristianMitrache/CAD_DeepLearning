ENV_PY := $(HOME)/mamba/envs/cad-dl/bin/python
ENV_BIN := $(HOME)/mamba/envs/cad-dl/bin

.PHONY: setup doctor pyright lint format check

setup:
	./scripts/setup.sh

# Just the node/simdjson repair step — useful when pyright suddenly dies with
# "Library not loaded: libsimdjson.*.dylib" after a brew upgrade.
doctor:
	@bash -c 'source ./scripts/setup.sh && doctor_node'

pyright: doctor
	$(ENV_BIN)/pyright

lint:
	$(ENV_BIN)/ruff check src

format:
	$(ENV_BIN)/ruff format src
	$(ENV_BIN)/ruff check --fix src

check: lint pyright
