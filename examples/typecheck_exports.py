"""Static type-checking guardrail for PAL's documented public imports.

This file exists purely so pyright verifies the imports that users and examples
should rely on from the installed package.
"""

# pyright: reportUnusedImport=false

from pal import (
    FreqSevSims,
    FrequencySeverityModel,
    ProteusVariable,
    StochasticScalar,
    config,
    set_default_n_sims,
    set_random_seed,
)
from pal.contracts import XoL, XoLTower
from pal.copulas import GaussianCopula
from pal.distributions import Gamma

_ = (
    config,
    set_default_n_sims,
    set_random_seed,
    XoL,
    XoLTower,
    GaussianCopula,
    Gamma,
    FreqSevSims,
    FrequencySeverityModel,
    ProteusVariable,
    StochasticScalar,
)
