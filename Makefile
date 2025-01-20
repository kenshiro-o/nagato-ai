SHELL=bash
POETRY := poetry
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
	@echo "  install    		- Install project dependencies using Poetry"
	@echo "  install-ci 		- Install project dependencies using Poetry in a CI environment"
	@echo "  test       		- Run tests using pytest"

.PHONY: lint
lint:
	$(POETRY) run pylint $(SRC_DIR) $(TEST_DIR)

.PHONY: lint-ci
lint-ci:
	$(POETRY) run pylint --fail-under=$(PYLINT_THRESHOLD) $(shell find $(SRC_DIR) -name "*.py")
	$(POETRY) run pylint --fail-under=$(PYLINT_THRESHOLD) $(shell find $(TEST_DIR) -name "*.py")

.PHONY: fmt
fmt:
	$(POETRY) run isort $(SRC_DIR) $(TEST_DIR)
	$(POETRY) run black $(SRC_DIR) $(TEST_DIR)

.PHONY: install
install:
	$(POETRY) install

.PHONY: install-ci
install-ci:
	$(POETRY) install --no-interaction --no-root

.PHONY: test
test:
	$(POETRY) run pytest
