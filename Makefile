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
	@echo "  test-fast      - Run pytest without coverage"
	@echo "  coverage       - Run tests and show HTML coverage report"
	@echo "  coverage-report - Show coverage report (requires running tests first)"
	@echo "  check-notebooks - Execute all notebooks to verify they work"
	@echo "  check          - Run all checks (static-analysis + tests + notebooks)"
	@echo "  build          - Build the package"
	@echo "  clean          - Clean build artifacts"

# Static analysis targets
.PHONY: lint
lint:
	ruff check src/pal tests examples

.PHONY: lint-fix
lint-fix:
	ruff check --fix src/pal tests examples

.PHONY: format
format:
	ruff format src/pal tests examples

.PHONY: format-check
format-check:
	ruff format --check src/pal tests examples

.PHONY: typecheck
typecheck:
	pyright

.PHONY: security
security:
	bandit -r src/pal

.PHONY: deadcode
deadcode:
	vulture src/pal

.PHONY: static-analysis
static-analysis: lint format-check typecheck security deadcode
	@echo "All static analysis checks completed"

# Test targets
.PHONY: test
test:
	pytest -v --cov=pal --cov-report=xml --cov-report=term

.PHONY: test-fast
test-fast:
	pytest -v

.PHONY: coverage
coverage:
	pytest --cov=pal --cov-report=html --cov-report=term
	@echo "\nOpening coverage report in browser..."
	@python -c "import webbrowser; webbrowser.open('htmlcov/index.html')" 2>/dev/null || \
		(command -v xdg-open >/dev/null && xdg-open htmlcov/index.html) || \
		(command -v open >/dev/null && open htmlcov/index.html) || \
		echo "Please open htmlcov/index.html in your browser"

.PHONY: coverage-report
coverage-report:
	@if [ -f htmlcov/index.html ]; then \
		python -c "import webbrowser; webbrowser.open('htmlcov/index.html')" 2>/dev/null || \
		(command -v xdg-open >/dev/null && xdg-open htmlcov/index.html) || \
		(command -v open >/dev/null && open htmlcov/index.html) || \
		echo "Please open htmlcov/index.html in your browser"; \
	else \
		echo "Coverage report not found. Run 'make coverage' first."; \
		exit 1; \
	fi

.PHONY: check-notebooks
check-notebooks:
	@echo "Executing notebooks to verify they work..."
	@for notebook in $(wildcard examples/*.ipynb); do \
		echo "Executing $$notebook..."; \
		PAL_SUPPRESS_PLOTS=true jupyter nbconvert --to notebook --execute \
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
	python -m build

.PHONY: clean
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +