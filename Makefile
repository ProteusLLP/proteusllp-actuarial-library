# Makefile for Proteus Actuarial Library

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  lint           - Run ruff linting"
	@echo "  lint-fix       - Auto-fix lint issues with ruff"
	@echo "  format         - Run ruff formatting"
	@echo "  format-check   - Check ruff formatting without making changes"
	@echo "  typecheck      - Run pyright type checking"
	@echo "  security       - Run bandit security scanning"
	@echo "  deadcode       - Run vulture dead code detection"
	@echo "  static-analysis - Run all static analysis tools (lint, format, typecheck, security, deadcode)"
	@echo "  test           - Run pytest with coverage"
	@echo "  check-notebooks - Execute all notebooks to verify they work"
	@echo "  check          - Run all checks (static-analysis + tests + notebooks)"
	@echo "  build          - Build the package"
	@echo "  clean          - Clean build artifacts"

# Static analysis targets
.PHONY: lint
lint:
	pdm run ruff check pal tests examples

.PHONY: lint-fix
lint-fix:
	pdm run ruff check --fix pal tests examples

.PHONY: format
format:
	pdm run ruff format pal tests examples

.PHONY: format-check
format-check:
	pdm run ruff format --check pal tests examples

.PHONY: typecheck
typecheck:
	pyright

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

.PHONY: check-notebooks
check-notebooks:
	@echo "Executing notebooks to verify they work..."
	@for notebook in $(wildcard examples/*.ipynb); do \
		echo "Executing $$notebook..."; \
		PAL_SUPPRESS_PLOTS=true pdm run jupyter nbconvert --to notebook --execute \
			--ExecutePreprocessor.timeout=300 \
			--output-dir=/tmp \
			"$$notebook" || exit 1; \
	done
	@echo "All notebooks executed successfully"

# All checks combined
.PHONY: check
check: static-analysis test check-notebooks
	@echo "✓ All checks passed successfully!"

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