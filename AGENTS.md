# PAL Agent Guide

This is the canonical repository-level guidance for coding agents working on the Proteus Actuarial Library (PAL).
Tool-specific instruction files should point here rather than duplicate these rules.

## What PAL Is

PAL is a Python library for simulation-based actuarial and financial modelling. It provides stochastic variables, probability distributions, copulas, frequency-severity models, reinsurance contracts, risk measures and optional CuPy GPU acceleration.

The installed package is `pal`; the PyPI distribution is `proteusllp-actuarial-library`.

## Sources of Truth

When instructions disagree, use these sources in this order:

1. `pyproject.toml` for supported Python versions, dependencies, Ruff configuration and package metadata.
2. `Makefile` and CI workflows for the commands that must pass.
3. This file and any more-specific nested `AGENTS.md` for architectural and behavioural rules.
4. `STYLE_GUIDE.md`, `docs/structure.md` and `docs/development.md` for additional explanation.

Do not copy configuration values into new guidance when they can be referenced from the authoritative configuration instead.

## Repository Map

- `src/pal/`: library implementation. Read `src/pal/AGENTS.md` before changing library code.
- `tests/`: automated tests. Read `tests/AGENTS.md` before adding or changing tests.
- `docs/`: Sphinx documentation and tutorials. Read `docs/AGENTS.md` before changing documentation.
- `examples/`: executable end-to-end examples.
- `.github/workflows/`: CPU, GPU and other CI workflows.
- `pyproject.toml`: package, dependency and static-analysis configuration.
- `Makefile`: canonical local development commands.
- `docs/structure.md`: type-system architecture and dependency direction.

## Development Environment

PAL uses standard Python packaging and also provides CPU/GPU devcontainers.

A fresh environment can be prepared with:

<!--pytest-codeblocks:skip-->

```bash
pip install -e ".[test,dev,docs]"
```

For GPU work, install the `gpu` extra in a CUDA-capable environment.

When the `pal-devcontainer` container is already running, commands may be run through `docker exec pal-devcontainer ...`. Do not assume that container exists in cloud or CI environments.

## Validation

Prefer the narrowest useful check while iterating, then run the full relevant checks before finishing.

<!--pytest-codeblocks:skip-->

```bash
make lint
make format-check
make typecheck
make test-fast
make static-analysis
make check
```

Run focused tests with `pytest path/to/test.py` or a specific test node. Documentation examples are executable and should remain valid.

For documentation, remember that `pytest-codeblocks` treats fenced code blocks as independent by default. Use `<!--pytest-codeblocks:cont-->` before a block that intentionally depends on state created by an earlier block. Use `<!--pytest-codeblocks:skip-->` for illustrative shell commands or examples that should not execute in CI; do not use skips merely to hide a broken executable example. See `docs/AGENTS.md` for the documentation-specific rule.

For GPU changes, also validate the GPU workflow or equivalent CUDA tests. CPU success alone is not sufficient for code that changes backend-sensitive behaviour.

## Definition of Done

A change is complete when:

- behaviour is covered by tests appropriate to the risk of the change;
- public API changes have accurate user-facing docstrings and documentation;
- static analysis and relevant tests pass;
- CPU and GPU semantics remain consistent where the feature supports both;
- numerical changes are checked against an independent theoretical or trusted reference where practical;
- no unrelated refactoring, compatibility break or documentation churn is introduced.

## General Rules

- Preserve backwards compatibility unless the task explicitly changes the public API.
- Prefer small, focused changes over broad rewrites.
- Do not leave comments describing deleted or superseded code.
- Keep imports at module scope unless there is a documented architectural reason not to.
- Use type annotations for public interfaces; do not duplicate type information in docstrings.
- Comments should explain non-obvious reasons, numerical choices or domain rules rather than narrate the code.
- Never commit secrets or generated local environment state.

## Before Architectural Changes

Read `docs/structure.md`. In particular, PAL's protocol/type layer must remain independent of concrete implementations, and dependencies should continue to flow from lower to higher architectural layers.
