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
from .multivariate_distributions import Dirichlet as _Dirichlet
from .multivariate_distributions import GeneralizedDirichlet as _GeneralizedDirichlet
from .multivariate_distributions import InverseWishart as _InverseWishart
from .multivariate_distributions import InvertedDirichlet as _InvertedDirichlet
from .multivariate_distributions import InvertedGeneralizedDirichlet as _InvertedGeneralizedDirichlet
from .multivariate_distributions import MatrixDistributionBase as _MatrixDistributionBase
from .multivariate_distributions import Multinomial as _Multinomial
from .multivariate_distributions import MultivariateDistributionBase as _MultivariateDistributionBase
from .multivariate_distributions import MultivariateNormal as _MultivariateNormal
from .multivariate_distributions import MultivariateStudentsT as _MultivariateStudentsT
from .multivariate_distributions import Wishart as _Wishart
from .stochastic_scalar import StochasticScalar as _StochasticScalar
from .variables import ProteusVariable as _ProteusVariable

# Empirical and HyperExponential have vector-valued parameters and are
# implemented in separate modules. Expose them through the standard
# distributions namespace and named-distribution generator APIs. ``setattr``
# is deliberate here: these are runtime aliases on the distributions module.
setattr(distributions, "Empirical", _Empirical)
distributions.AVAILABLE_DISCRETE_DISTRIBUTIONS["empirical"] = _Empirical
setattr(distributions, "HyperExponential", _HyperExponential)
distributions.AVAILABLE_CONTINUOUS_DISTRIBUTIONS["hyperexponential"] = _HyperExponential

# Attach multivariate classes to the standard distributions namespace without
# also exporting them from the package root.
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

# ``copulas`` historically imports these two classes from the package root.
# Make them available only while the module initialises, then remove them again
# so users cannot rely on the old top-level shortcuts.
ProteusVariable = _ProteusVariable
StochasticScalar = _StochasticScalar
from . import copulas as copulas  # noqa: E402

del ProteusVariable, StochasticScalar

# Import the public module namespaces explicitly so ``import pal; pal.contracts``
# and ``from pal import contracts`` are both reliable and discoverable.
# Runtime API discovery is imported last conceptually; it is independent of the
# ordering within this late-import block once the aliases above are installed.
from . import api as api  # noqa: E402
from . import contracts as contracts  # noqa: E402
from . import couplings as couplings  # noqa: E402
from . import frequency_severity as frequency_severity  # noqa: E402
from . import maths as maths  # noqa: E402
from . import multivariate_distributions as multivariate_distributions  # noqa: E402
from . import risk_measures as risk_measures  # noqa: E402
from . import stats as stats  # noqa: E402
from . import stochastic_scalar as stochastic_scalar  # noqa: E402
from . import variables as variables  # noqa: E402

__all__ = [
    "api",
    "config",
    "contracts",
    "copulas",
    "couplings",
    "distributions",
    "frequency_severity",
    "maths",
    "multivariate_distributions",
    "risk_measures",
    "set_default_n_sims",
    "set_random_seed",
    "stats",
    "stochastic_scalar",
    "variables",
]
