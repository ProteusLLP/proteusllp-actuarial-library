"""Static type-checking guardrail for PAL's public module namespaces.

This file exists purely so pyright verifies the documented imports that users
and examples should rely on from the installed package.
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
    variables,
)
from pal.variables import StochasticScalar

# Domain objects are accessed through module namespaces; core variable types are
# imported directly from pal.variables.
_ = (
    contracts.XoL,
    contracts.XoLTower,
    frequency_severity.FreqSevSims,
    StochasticScalar,
    variables.ProteusVariable,
    distributions.Gamma,
    copulas.GaussianCopula,
)
