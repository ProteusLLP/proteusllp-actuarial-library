# GitHub Copilot Instructions for PAL

Before changing code, read the repository root `AGENTS.md` and any more-specific `AGENTS.md` in the directory being modified.

Key expectations:

- Treat `pyproject.toml` as authoritative for Python support, Ruff configuration and dependencies.
- Treat `Makefile` and CI workflows as authoritative for validation commands.
- Preserve PAL's public API and stochastic/coupling semantics unless the task explicitly changes them.
- Read `docs/structure.md` before architectural or type-system changes.
- For numerical code, add independent mathematical/reference validation where practical; do not rely only on implementation round trips.
- For backend-sensitive code, preserve CPU/GPU semantics and avoid unnecessary host/device transfers.
- For public APIs and documentation, use actuarial/statistical user language rather than unnecessary backend implementation terminology.
- Run focused checks while iterating and the full relevant static-analysis/test suite before finishing.

Do not duplicate configuration values in comments or instructions when the authoritative project configuration can be referenced directly.
