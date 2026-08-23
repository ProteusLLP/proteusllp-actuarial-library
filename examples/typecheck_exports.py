"""Static type-checking guardrail for pal exports.

This file exists purely so pyright verifies that `pal` exports the symbols that
users (and our examples/tests) commonly import from the top-level package.

If `src/pal/__init__.pyi` drifts from the runtime API, pyright should fail here.
"""

# pyright: reportUnusedImport=false

from pal import (  # noqa: F401
    FreqSevSims,
    ProteusVariable,
    StochasticScalar,
    XoL,
    XoLTower,
    config,
    copulas,
    distributions,
    frequency_severity,
    maths,
    risk_measures,
    set_default_n_sims,
    set_random_seed,
    stats,
)
