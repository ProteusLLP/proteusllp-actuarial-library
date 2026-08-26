"""Static type-checking guardrail for PAL's public module namespaces.

This file exists purely so pyright verifies the module-oriented imports that
users and examples should rely on from the top-level package.
"""

# pyright: reportUnusedImport=false

from pal import (  # noqa: F401
    config,
    contracts,
    copulas,
    distributions,
    frequency_severity,
    maths,
    risk_measures,
    set_default_n_sims,
    set_random_seed,
    stats,
    stochastic_scalar,
    variables,
)

# Domain objects are deliberately accessed through their module namespaces.
_ = (
    contracts.XoL,
    contracts.XoLTower,
    frequency_severity.FreqSevSims,
    stochastic_scalar.StochasticScalar,
    variables.ProteusVariable,
    distributions.Gamma,
    copulas.GaussianCopula,
)
