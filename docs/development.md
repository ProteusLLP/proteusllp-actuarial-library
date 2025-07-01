# Development Guide

This project uses PDM for dependency management, uv for fast installs, and Docker devcontainers for development.

## Getting Started

### Prerequisites
- Docker
- VS Code with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Setup Development Environment

1. **Open in devcontainer**:
   - Open the project in VS Code
   - Command Palette → "Dev Containers: Reopen in Container"
   - Wait for container to build (first time takes several minutes).

2. **Verify setup**:
   ```bash
   python --version  # Should show Python 3.13.x
   pdm --version
   uv --version
   pytest --version
   ```

## Managing Dependencies

### Dependency Groups

This project uses PDM dependency groups defined in `pyproject.toml`:

- **Core dependencies**: Required runtime dependencies (numpy, scipy, plotly, pandas)
- **Optional dependencies**: 
  - `gpu`: CUDA support with cupy-cuda12x
- **Development dependencies**:
  - `test`: Testing tools (pytest, pytest-cov, pytest-xdist)
  - `dev`: Development tools (jupyter, jupyterlab)

**Important**: The devcontainer automatically installs ALL dependency groups (core, optional, and dev groups) to provide a complete development environment. This means you have access to all testing, development, and GPU dependencies without manual installation.

### Adding New Dependencies

Use PDM CLI to manage dependencies:

```bash
# Add regular dependencies
pdm add "new-package>=1.0"

# Add optional dependencies (like GPU)
pdm add -G gpu "cupy-cuda12x"

# Add development/test dependencies  
pdm add -dG test "pytest-mock"

# Remove dependencies
pdm remove new-package
```

After adding dependencies, rebuild the container:
- Command Palette → "Dev Containers: Rebuild Container"

### Installing GPU Dependencies

GPU dependencies are in a separate group:
```bash
pdm install -G gpu
```

## Versioning

**Versions are automatically managed from git tags** - no manual updates needed!

### How it works:
- Version is determined from git tags using SCM (Source Code Management)
- `dynamic = ["version"]` in pyproject.toml enables this
- Create releases by tagging: `git tag v1.0.0` → version becomes `1.0.0`
- Between tags: automatic dev versions like `1.0.0.dev5+g1a2b3c4.d20250630` - this tag would mean that you're on a development version of the package which is 5 commits ahead of `v1.0.0`, with git commit hash beginning `1a2bc4` and built on date `20250630`. `pdm` likely uses `git describe --tags` to generate this string.

### Creating a release:
1. **Tag the release**: `git tag v1.0.0`
2. **Push the tag**: `git push origin v1.0.0`  
3. **Create GitHub Release** from the tag → triggers automatic PyPI publishing

### Check current version:
```bash
pdm show --version  # Shows current computed version
```

## Release Process

### Creating GitHub Releases

When creating a GitHub release, follow these steps to avoid common issues. If successful, you should CI should be triggered and a release built and pushed to pypi with the associated tag.

#### Create and Push a Valid Tag

GitHub releases require an existing tag. Create one from the command line:

```bash
# Create a PEP 440-compliant tag
git tag v0.0.1a1
git push origin v0.0.1a1
```

#### Use PEP 440-Compliant Versions

Use a PEP-440-compliant version format. This project uses PDM for Python package versioning, which follows PEP 440. Use these valid version formats:

- `v0.0.1a1` (alpha)
- `v0.0.1b1` (beta) 
- `v0.0.1rc1` (release candidate)
- `v0.0.1.dev1` (development)
- `v0.0.1.post1` (post-release)

**Avoid** arbitrary suffixes like `-test` that aren't PEP 440 compliant.

#### Mark Pre-releases Appropriately

For pre-release versions, use GitHub's "Set as pre-release" checkbox instead of adding non-standard suffixes to your tag name.

### Troubleshooting Release Issues

#### "tag name can't be blank, tag name is not well-formed"

This error occurs when:
- The tag doesn't exist in the repository
- The tag hasn't been pushed to GitHub
- The tag name isn't PEP 440 compliant

**Solution**: Create and push a valid tag first, then create the GitHub release.

#### "published releases must have a valid tag"

This indicates the tag exists locally but hasn't been pushed to GitHub.

**Solution**: Push the tag with `git push origin <tag-name>`

## Running Tests

### From VS Code Terminal

Basic test commands:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_variables.py

# Run with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=pal

# Run tests in parallel (faster)
pytest -n auto
```

### From VS Code Test Explorer

1. Open the Test Explorer panel (Testing icon in sidebar)
2. Click "Configure Python Tests" if prompted
3. Select "pytest" as the test framework
4. Tests will appear in the explorer for easy running/debugging

### Test Structure

Tests are organized in the `tests/` directory:
- `test_*.py` files contain test cases
- Each test function starts with `test_`
- Use `pytest.ini` for configuration

## Container Architecture

The project uses a multi-stage Dockerfile:

- **base**: Python 3.13 + system dependencies
- **deps**: All PDM dependencies installed (cached layer)
- **dev**: Development tools + Jupyter + non-root user
- **ci**: Test runners + CI tools

## Troubleshooting

### "pdm.lock out of date" Error

This happens when dependencies change:

1. Sync dependencies: `pdm sync`
2. Or rebuild container: "Dev Containers: Rebuild Container"

### Container Won't Start

Try rebuilding without cache:
```bash
docker build --no-cache --target dev -t proteus-dev .
```

### Dependencies Not Found

Make sure you rebuilt the container after adding dependencies with `pdm add`. VS Code may use cached layers that don't include new dependencies.

## See Also

- [Usage Guide](usage.md) - Comprehensive examples and API documentation  
- [Examples](../examples/) - Example scripts showing library usage
- [Main README](../README.md) - Project overview and installation

## References

- [PDM Documentation](https://pdm-project.org/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Dev Containers Documentation](https://containers.dev/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)