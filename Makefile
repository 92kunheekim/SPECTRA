.PHONY: help setup lint format test train clean
help:
	@echo "setup   - install package + dev deps"
	@echo "lint    - ruff check"
	@echo "format  - ruff format"
	@echo "test    - run pytest"
	@echo "train   - train full fusion model"
setup:
	python -m pip install -U pip && pip install -e . && pip install ruff pytest pre-commit
lint:
	ruff check spectra tests
format:
	ruff format spectra tests
test:
	pytest -q
train:
	python -m spectra.training.train --config configs/model/full_fusion.yaml
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
