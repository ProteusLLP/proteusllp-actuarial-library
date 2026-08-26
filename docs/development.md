<!--pytest-codeblocks:skipfile-->

# Development Guide

PAL can be developed in a standard Python environment or in the supplied Docker devcontainers. The devcontainer is convenient for local development but is not a prerequisite for cloud coding agents or other automated environments.

Coding agents should read [`AGENTS.md`](../AGENTS.md) before making changes. The project configuration, Makefile and CI workflows are authoritative when prose documentation differs from executable configuration.

## Standard Python setup

Create or activate a Python environment supported by `pyproject.toml`, then install PAL in editable mode with the development extras:

```bash
python -m pip install -e ".[test,dev,docs]"
```

Verify the environment with:

```bash
python --version
python -m pytest --version
make help
```

For GPU support in a compatible CUDA environment:

```bash
python -m pip install -e ".[test,dev,docs,gpu]"
```

## Devcontainer setup

For a reproducible local environment, open the repository in VS Code and choose **Dev Containers: Reopen in Container**. The devcontainer installs the project in editable mode.

The established local container may be named `pal-devcontainer`. If it is already running, commands can also be invoked from the host, for example:

```bash
docker exec pal-devcontainer make test-fast
```

Do not assume that named container exists in a fresh cloud or CI environment.

## Development commands

The `Makefile` is the canonical interface for routine validation:

```bash
make help
make lint
make format-check
make typecheck
make static-analysis
make test-fast
make test
make check
make build
```

Use focused tests while iterating:

```bash
pytest tests/test_variables.py
pytest tests/test_variables.py::test_name
```

Documentation code blocks can be exercised with `pytest-codeblocks` as configured by the project and CI.

## Dependencies

Dependencies are declared in `pyproject.toml` using standard Python project metadata.

- `gpu`: CUDA/CuPy support.
- `docs`: documentation build dependencies.
- `test`: pytest and test tooling.
- `dev`: linting, type checking, security and development tools.

After changing dependencies, reinstall the relevant editable extras.

## Static analysis

PAL uses Ruff for linting/formatting, Pyright for type checking, Bandit for security checks and Vulture for dead-code detection. Run these through `make static-analysis` so local validation matches project configuration.

## CPU and GPU development

PAL has separate CPU and GPU CI paths. A backend-sensitive change is not fully validated by CPU tests alone. Follow the root and `src/pal/AGENTS.md` guidance and run the relevant GPU workflow or equivalent CUDA tests.

## Versioning and releases

Versions are generated from Git tags by `setuptools-scm`; do not edit a version constant manually.

Use PEP 440-compliant release tags such as:

- `v0.0.1a1`
- `v0.0.1b1`
- `v0.0.1rc1`
- `v0.0.1`

A normal release process is:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Then create the corresponding GitHub Release. The release workflow publishes the package to PyPI.

## Troubleshooting

If dependencies are missing, reinstall the editable package:

```bash
python -m pip install -e ".[test,dev,docs]"
```

If a devcontainer configuration changed, rebuild the container. In a non-container environment, diagnose the Python environment directly rather than attempting to find `pal-devcontainer`.

## See also

- [Usage Guide](usage.md)
- [Examples](../examples/)
- [Repository agent guide](../AGENTS.md)
- [Type-system architecture](structure.md)
