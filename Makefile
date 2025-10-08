#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = cache_simulator
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pip install poetry
	poetry install --with special,dev

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*pytest_cache" -exec rm -rf {} +
	find . -type d -name "*mypy_cache" -exec rm -rf {} +
	find . -type d -name "*ruff_cache" -exec rm -rf {} +

## Validate rules and format using pre-commit-hooks, ruff and mypy
.PHONY: validate
validate:
	poetry run pre-commit run --all-files

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	poetry run ruff format --check
	poetry run ruff check

## Format source code with ruff
.PHONY: format
format:
	poetry run ruff check --fix
	poetry run ruff format

## Install pre-commit hooks
.PHONY: hooks
hooks:
	poetry run pre-commit autoupdate
	poetry run pre-commit clean
	poetry run pre-commit install
	poetry run pre-commit install -f --hook-type commit-msg --hook-type pre-commit

## Run mypy checking
.PHONY: mypy
mypy:
	poetry run pre-commit run mypy --all-files

## Generate baseline file for detect-secrets
.PHONY: scan
scan:
	poetry run detect-secrets scan > .secrets.baseline

## Run tests
.PHONY: test
test:
	python -m pytest --cov

.PHONY: precommit
precommit:
	poetry run pre-commit run -a -v

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:  conda activate $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
