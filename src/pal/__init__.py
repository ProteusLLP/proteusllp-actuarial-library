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
from .config import *
from .contracts import *
from .distributions import *
from .frequency_severity import *
from .hyperexponential import HyperExponential
from .multivariate_distributions import *
from .stats import *
from .variables import *

# HyperExponential has vector-valued mixture parameters and is implemented in a
# separate module. Expose it through the standard distributions namespace and
# named-distribution generator API.
setattr(distributions, "HyperExponential", HyperExponential)
distributions.AVAILABLE_CONTINUOUS_DISTRIBUTIONS["hyperexponential"] = HyperExponential

# ``variables`` depends on ``frequency_severity``, which in turn depends on the
# univariate distributions module. Attach the multivariate API after those modules
# have initialized to avoid making that established import cycle recursive.
for _distribution_name in (
    "Dirichlet",
    "GeneralizedDirichlet",
    "InverseWishart",
    "InvertedDirichlet",
    "InvertedGeneralizedDirichlet",
    "MatrixDistributionBase",
    "Multinomial",
    "MultivariateDistributionBase",
    "MultivariateNormal",
    "MultivariateStudentsT",
    "Wishart",
):
    setattr(distributions, _distribution_name, globals()[_distribution_name])

del _distribution_name