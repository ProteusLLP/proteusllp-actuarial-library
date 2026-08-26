"""Proteus Actuarial Library (PAL).

A simple, fast and lightweight framework for building simulation-based
actuarial and financial models.

PAL is designed to look after the complicated stuff, such as copulas and
simulation re-ordering, providing easy to use objects and clear syntax.

PAL is based on the scientific python stack of numpy and scipy for fast
performance. It can optionally run on a GPU using the cupy package for
extremely fast performance. It is designed for interoperability with
numpy and ndarrays.

See: http://github.com/ProteusLLP/proteus-actuarial-library
"""

from . import distributions as distributions
from .config import config, set_default_n_sims, set_random_seed
from .empirical import Empirical as _Empirical
from .hyperexponential import HyperExponential as _HyperExponential
from .multivariate_distributions import (
    Dirichlet as _Dirichlet,
    GeneralizedDirichlet as _GeneralizedDirichlet,
    InverseWishart as _InverseWishart,
    InvertedDirichlet as _InvertedDirichlet,
    InvertedGeneralizedDirichlet as _InvertedGeneralizedDirichlet,
    MatrixDistributionBase as _MatrixDistributionBase,
    Multinomial as _Multinomial,
    MultivariateDistributionBase as _MultivariateDistributionBase,
    MultivariateNormal as _MultivariateNormal,
    MultivariateStudentsT as _MultivariateStudentsT,
    Wishart as _Wishart,
)

__all__ = [
    "api",
    "config",
    "distributions",
    "set_default_n_sims",
    "set_random_seed",
]

# Empirical and HyperExponential have vector-valued parameters and are
# implemented in separate modules. Expose them through the standard
# distributions namespace and named-distribution generator APIs.
setattr(distributions, "Empirical", _Empirical)
distributions.AVAILABLE_DISCRETE_DISTRIBUTIONS["empirical"] = _Empirical
setattr(distributions, "HyperExponential", _HyperExponential)
distributions.AVAILABLE_CONTINUOUS_DISTRIBUTIONS["hyperexponential"] = _HyperExponential

# ``variables`` depends on ``frequency_severity``, which in turn depends on the
# univariate distributions module. Attach the multivariate API after those modules
# have initialized to avoid making that established import cycle recursive.
for _distribution_name, _distribution in {
    "Dirichlet": _Dirichlet,
    "GeneralizedDirichlet": _GeneralizedDirichlet,
    "InverseWishart": _InverseWishart,
    "InvertedDirichlet": _InvertedDirichlet,
    "InvertedGeneralizedDirichlet": _InvertedGeneralizedDirichlet,
    "MatrixDistributionBase": _MatrixDistributionBase,
    "Multinomial": _Multinomial,
    "MultivariateDistributionBase": _MultivariateDistributionBase,
    "MultivariateNormal": _MultivariateNormal,
    "MultivariateStudentsT": _MultivariateStudentsT,
    "Wishart": _Wishart,
}.items():
    setattr(distributions, _distribution_name, _distribution)

del _distribution_name, _distribution

# Runtime API discovery is imported last so it sees the complete public
# distributions namespace, including the aliases attached above.
from . import api as api  # noqa: E402
