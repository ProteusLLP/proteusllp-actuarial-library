# Makefile for Proteus Actuarial Library

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  lint           - Run ruff linting"
	@echo "  format         - Run ruff formatting"
	@echo "  format-check   - Check ruff formatting without making changes"
	@echo "  typecheck      - Run mypy type checking"
	@echo "  security       - Run bandit security scanning"
	@echo "  deadcode       - Run vulture dead code detection"
	@echo "  static-analysis - Run all static analysis tools"
	@echo "  test           - Run pytest with coverage"
	@echo "  check-examples - Check that examples compile"
	@echo "  build          - Build the package"
	@echo "  clean          - Clean build artifacts"

# Static analysis targets
.PHONY: lint
lint:
	pdm run ruff check pal tests examples

.PHONY: format
format:
	pdm run ruff format pal tests examples

.PHONY: format-check
format-check:
	pdm run ruff format --check pal tests examples

.PHONY: typecheck
typecheck:
	pdm run mypy pal

.PHONY: security
security:
	pdm run bandit -r pal

.PHONY: deadcode
deadcode:
	pdm run vulture pal

.PHONY: static-analysis
static-analysis: lint format-check typecheck security deadcode
	@echo "All static analysis checks completed"

# Test targets
.PHONY: test
test:
	mkdir -p coverage
	pdm run pytest -v --cov=pal --cov-report=xml:coverage/coverage.xml

.PHONY: check-examples
check-examples:
	@for example in $(wildcard examples/*.py); do \
		echo "Running $$example..."; \
		PAL_SUPPRESS_PLOTS=true pdm run python $$example || exit 1; \
	done

# Build targets
.PHONY: build
build:
	mkdir -p dist
	pdm build

.PHONY: clean
clean:
	rm -rf dist/
	rm -rf coverage/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +