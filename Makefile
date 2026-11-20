PYTHON ?= python
TAG ?= nov20
MODEL ?= $(OPENAI_MODEL)
.PHONY: test compile prep baseline init evolve dashboard env

test:
	$(PYTHON) -m unittest discover -s tests -v
compile:
	$(PYTHON) -m compileall -q autoresearch_evo prepare.py train.py
prep:
	uv run prepare.py
baseline:
	uv run train.py
init:
	uv run python -m autoresearch_evo init --tag $(TAG)
evolve:
	uv run python -m autoresearch_evo run --provider openai --model $(MODEL) --hours 8
dashboard:
	uv run python -m autoresearch_evo dashboard
env:
	bash scripts/check_environment.sh
