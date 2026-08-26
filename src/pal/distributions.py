"""Distributions Module.

This module contains classes for simulating statistical distributions.
The implementations follow conventions similar to Klugman. Random number
generation and GPU support are managed via configuration settings.

It's expected that you construct distributions of distributions ie. a distribution can
be created and passed to another distribution as a parameter.

Univariate distributions accept and return only primitives (int, float) or
StochasticScalar. Multivariate distributions return a ProteusVariable containing
one StochasticScalar per named component, keeping the simulation dimension
one-dimensional at each leaf. Matrix distributions return nested row and column
ProteusVariable objects with a StochasticScalar at each named matrix entry.

Type Definitions:
- DistributionParameter: Union[int, float, StochasticScalar]
- ReturnType: Union[int, float, StochasticScalar]
"""

# Standard library imports
from __future__ import annotations

import typing as t
from abc import ABC

import numpy as np
import scipy.special as scipy_special
from scipy.stats import geninvgauss

# Local imports
from ._compat import override
from ._maths import asnumpy, scalar_or_array, special, to_backend, xp
from .config import config
from .stochastic_scalar import StochasticScalar
from .types import DistributionParameter, RandomGenerator

if t.TYPE_CHECKING:
    from .empirical import Empirical as Empirical
    from .hyperexponential import HyperExponential as HyperExponential
    from .multivariate_distributions import Dirichlet as Dirichlet
    from .multivariate_distributions import GeneralizedDirichlet as GeneralizedDirichlet
    from .multivariate_distributions import InverseWishart as InverseWishart
    from .multivariate_distributions import InvertedDirichlet as InvertedDirichlet
    from .multivariate_distributions import InvertedGeneralizedDirichlet as InvertedGeneralizedDirichlet
    from .multivariate_distributions import MatrixDistributionBase as MatrixDistributionBase
    from .multivariate_distributions import Multinomial as Multinomial
    from .multivariate_distributions import MultivariateDistributionBase as MultivariateDistributionBase
    from .multivariate_distributions import MultivariateNormal as MultivariateNormal
    from .multivariate_distributions import MultivariateStudentsT as MultivariateStudentsT
    from .multivariate_distributions import Wishart as Wishart

TOLERANCE = 1e-10  # Tolerance for numerical comparisons
# FIXME: Consider replaching with VectorLike from types.py
ReturnType = t.Union[int, float, StochasticScalar]


def _special_call(function: t.Callable[..., t.Any], *args: t.Any) -> t.Any:
    """Call a SciPy-compatible special function on active-backend values."""
    stochastic_args = [arg for arg in args if isinstance(arg, StochasticScalar)]
    processed_args = [arg.values if isinstance(arg, StochasticScalar) else to_backend(arg) for arg in args]
    result = function(*processed_args)
    if stochastic_args and getattr(result, "ndim", None) == 1:
        wrapped = StochasticScalar(result)
        for arg in stochastic_args:
            wrapped.coupled_variable_group.merge(arg.coupled_variable_group)
        return wrapped
    return scalar_or_array(result)


def _rng_value(value: t.Any, rng: t.Any) -> t.Any:
    """Place a parameter on the array backend used by the supplied generator."""
    return xp.asarray(value) if xp.__name__ == "cupy" else value


class DistributionBase(ABC):
    """Base class for PAL distributions."""

    @property
    def parameters(self) -> list[t.Any]:
        """Return the parameters of the distribution."""
        raise NotImplementedError

    def generate(self, n_sims: int | None = None, rng: t.Any = None) -> StochasticScalar:
        """Generate simulations from the distribution."""
        raise NotImplementedError


class DiscreteDistributionBase(DistributionBase):
    """Base class for discrete distributions."""


class ContinuousDistributionBase(DistributionBase):
    """Base class for continuous distributions."""


# NOTE: file content below this point is unchanged from the branch version.
