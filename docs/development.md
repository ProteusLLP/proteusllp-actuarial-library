<!--pytest-codeblocks:skipfile-->

# Development Guide

This project uses pip for dependency installation and Docker devcontainers for development.

## Getting Started

### Prerequisites
- Docker
- VS Code with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Setup Development Environment

1. **Open in devcontainer**:
   - Open the project in VS Code
   - Command Palette → "Dev Containers: Reopen in Container"
   - Wait for the container to build.

2. **Verify setup**:
   ```bash
   python --version
   pip --version
   pytest --version
   ```

The devcontainer installs the project in editable mode with the test, development, and documentation extras:

```bash
pip install -e ".[test,dev,docs]"
```

## Managing Dependencies

Dependencies are declared in `pyproject.toml` using standard Python project metadata.

- **Core dependencies**: required runtime dependencies such as NumPy, SciPy, Plotly and pandas.
- **Optional dependencies**:
  - `gpu`: CUDA support with `cupy-cuda12x`.
  - `docs`: documentation build dependencies.
  - `test`: testing tools.
  - `dev`: development and static-analysis tools.

After changing `pyproject.toml`, reinstall the project and the extras you need. For a full development environment:

```bash
pip install -e ".[test,dev,docs]"
```

For GPU support:

```bash
pip install -e ".[gpu]"
```

Or combine extras when required:

```bash
pip install -e ".[test,dev,docs,gpu]"
```

## Versioning

Versions are automatically managed from Git tags by `setuptools-scm`; no manual version update is required.

- `dynamic = ["version"]` in `pyproject.toml` enables dynamic versioning.
- A tag such as `v1.0.0` produces version `1.0.0`.
- Commits between releases receive an automatically generated development version.

### Creating a release

1. Tag the release: `git tag v1.0.0`
2. Push the tag: `git push origin v1.0.0`
3. Create a GitHub Release from that tag. The release workflow publishes the package to PyPI.

### Check the installed version

```bash
pip show proteusllp-actuarial-library
```

## Release Process

### Use PEP 440-Compliant Versions

Use a PEP 440-compliant version format, for example:

- `v0.0.1a1` (alpha)
- `v0.0.1b1` (beta)
- `v0.0.1rc1` (release candidate)
- `v0.0.1.dev1` (development)
- `v0.0.1.post1` (post-release)

Avoid arbitrary suffixes such as `-test`.

For pre-release versions, use GitHub's "Set as pre-release" option when creating the release.

## Running Tests

### From VS Code Terminal

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_variables.py

# Run with verbose output
pytest -v

# Run tests with coverage
pytest --cov=pal

# Run tests in parallel
pytest -n auto

# Execute documentation code blocks
pytest --codeblocks
```

### From VS Code Test Explorer

1. Open the Test Explorer panel.
2. Click "Configure Python Tests" if prompted.
3. Select `pytest` as the test framework.

## Static Analysis and Type Checking

PAL uses Pyright for static type checking and Ruff for linting and formatting.

```bash
make lint
make format
make typecheck
```

The same tools can also be run directly from the activated project environment.

## Development Commands

Run the Makefile commands from inside the devcontainer:

```bash
make help
make lint
make format
make typecheck
make test
make build
```

## Container Architecture

The project uses separate CPU and GPU development containers. Both install PAL into the container using pip in editable mode, so changes to the source tree are immediately available.

## Troubleshooting

### Dependencies Not Found

Reinstall the editable package with the required extras:

```bash
pip install -e ".[test,dev,docs]"
```

If the devcontainer itself has changed, rebuild it with "Dev Containers: Rebuild Container".

### Container Won't Start

Rebuild the relevant container without cache if necessary.

## See Also

- [Usage Guide](usage.md) - Comprehensive examples and API documentation
- [Examples](../examples/) - Example scripts showing library usage
- [Main README](../README.md) - Project overview and installation

## References

- [pip Documentation](https://pip.pypa.io/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [Dev Containers Documentation](https://containers.dev/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
