## Convenience Makefile for eo-processor-mcp development

VENV_DIR ?= $(shell if [ -d ".venv" ]; then echo ".venv"; elif [ -d "venv" ]; then echo "venv"; else echo ".venv"; fi)

PYTHON := $(if $(wildcard $(VENV_DIR)/bin/python),$(VENV_DIR)/bin/python,python3)
PIP := $(PYTHON) -m pip
PYTEST := $(if $(wildcard $(VENV_DIR)/bin/pytest),$(VENV_DIR)/bin/pytest,pytest)
RUFF := $(if $(wildcard $(VENV_DIR)/bin/ruff),$(VENV_DIR)/bin/ruff,$(PYTHON) -m ruff)
COVERAGE := $(if $(wildcard $(VENV_DIR)/bin/coverage),$(VENV_DIR)/bin/coverage,coverage)

.PHONY: help install install-editable install-dev run-package format lint test coverage clean

help:
	@echo "Available targets:"
	@echo "  install           Install package (pip install .)"
	@echo "  install-editable  Install package in editable mode (pip install -e .)"
	@echo "  install-dev       Install dev dependencies (editable with dev extras)."
	@echo "  run-package       Run the installed package (python -m eo_processor_mcp)."
	@echo "  format            Run ruff format on repository files."
	@echo "  lint              Run ruff check and try to fix auto-fixable issues."
	@echo "  test              Run the full test suite (pytest)."
	@echo "  coverage          Run coverage report for tests."
	@echo "  clean             Remove common build/test artifacts."

install:
	@echo "Installing package..."
	$(PIP) install .

install-editable:
	@echo "Installing package in editable mode..."
	$(PIP) install -e .

install-dev:
	@echo "Installing dev dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

run-package:
	@echo "Running eo-processor-mcp package (python -m eo_processor_mcp)"
	$(PYTHON) -m eo_processor_mcp

format:
	@echo "Formatting source with ruff"
	$(RUFF) format eo_processor_mcp/ tests/ || true

lint:
	@echo "Running ruff checks and attempting to auto-fix"
	$(RUFF) check eo_processor_mcp/ tests/ --fix || true

test:
	@echo "Running full test suite"
	$(PYTEST) -v

coverage:
	@echo "Running tests with coverage"
	$(COVERAGE) run -m pytest && $(COVERAGE) report -m

clean:
	@echo "Cleaning build/test artifacts"
	rm -rf build/ dist/ .pytest_cache/ $(VENV_DIR)/ __pycache__/ .cache/ .ruff_cache/ .coverage
