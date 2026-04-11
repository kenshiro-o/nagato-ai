SHELL=bash
SRC_DIR := nagatoai_core
TEST_DIR := tests
PYLINT_THRESHOLD := 6

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  lint       		- Run linter on Python files"
	@echo "  lint-ci    		- Run linter and fail if score is below $(PYLINT_THRESHOLD)"
	@echo "  fmt        		- Format Python files using Black"
	@echo "  install    		- Install project dependencies using uv"
	@echo "  install-ci 		- Install project dependencies using uv in a CI environment"
	@echo "  test       		- Run tests using pytest"

.PHONY: lint
lint:
	uv run pylint $(SRC_DIR) $(TEST_DIR)

.PHONY: lint-ci
lint-ci:
	uv run pylint --fail-under=$(PYLINT_THRESHOLD) $(shell find $(SRC_DIR) -name "*.py")
	uv run pylint --fail-under=$(PYLINT_THRESHOLD) $(shell find $(TEST_DIR) -name "*.py")

.PHONY: fmt
fmt:
	uv run isort $(SRC_DIR) $(TEST_DIR)
	uv run black $(SRC_DIR) $(TEST_DIR)

.PHONY: install
install:
	uv sync

.PHONY: install-ci
install-ci:
	uv sync

.PHONY: test
test:
	uv run pytest

.PHONY: test-debug
test-debug:
	uv run pytest --log-cli-level=DEBUG
