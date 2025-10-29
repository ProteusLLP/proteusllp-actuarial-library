# Claude Development Guide

This file contains project-specific information to help Claude work effectively with this codebase.

## Development Environment

This project uses Docker for consistent development environments.

### Command Execution Guidelines

**Run INSIDE the Docker container (`pal-devcontainer`):**
- All `make` commands (lint, test, typecheck, etc.)
- Python/PDM commands
- Package builds

**Run on the HOST machine:**
- Git commands (`git add`, `git commit`, `git push`, etc.)
- GitHub CLI (`gh` commands for issues, PRs, etc.)
- File editing (though this works in both environments)

### Docker Commands

The container name is configured in `.devcontainer/devcontainer.json` as `pal-devcontainer`:

```bash
# All development commands (run inside container):
# Lint code
docker exec pal-devcontainer make lint

# Auto-fix lint issues  
docker exec pal-devcontainer make lint-fix

# Run tests
docker exec pal-devcontainer make test

# Type checking
docker exec pal-devcontainer make typecheck
```

### Python Execution

**IMPORTANT**: This project uses PDM for dependency management. Always use `pdm run` to execute Python commands to ensure you're using the correct virtual environment:

```bash
# Run Python scripts (CORRECT)
docker exec pal-devcontainer pdm run python script.py

# Run examples (CORRECT) 
docker exec pal-devcontainer pdm run python examples/example_catastrophes.py

# Wrong - uses system Python (INCORRECT)
docker exec pal-devcontainer python script.py
```

### Finding Your Container
```bash
# List running containers 
docker ps

# Filter by the devcontainer name
docker ps --filter name=pal-devcontainer
```

## Code Style

**IMPORTANT**: 
- Do not add cruft comments like "# mean() removed - use numpy.mean() instead". When removing code, just remove it cleanly without leaving placeholder comments.
- **No nested imports** - Always import modules at the top of the file, not inside functions. Nested imports hide import errors and delay them until runtime instead of catching them at load time.

**See [STYLE_GUIDE.md](./STYLE_GUIDE.md) for all coding standards including:**
- Line length (88 characters max) - IMPORTANT: Never exceed 88 characters
- Type annotations
- Comments (explain WHY, not WHAT)
- Docstring format
- Import ordering
- No trailing whitespace at end of lines
- No unnecessary blank lines

### Linting Workflow
1. Run: `docker exec pal-devcontainer make lint`
2. For auto-fixable errors: `docker exec pal-devcontainer make lint-fix`  
3. Manually fix remaining errors following [STYLE_GUIDE.md](./STYLE_GUIDE.md)
4. **All lint errors must be fixed** (no warnings allowed in CI)

## Project Structure

- `pal/` - Main library code
- `tests/` - Test suite  
- `examples/` - Usage examples
- `pyproject.toml` - Project configuration including ruff lint rules
- `STYLE_GUIDE.md` - **All coding standards and examples**
- `docs/structure.md` - **Type system architecture and design principles**