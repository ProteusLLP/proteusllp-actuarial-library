"""Distributions Module.

This module contains classes for simulating statistical distributions.
The implementations follow conventions similar to Klugman. Random number
generation and GPU support are managed via configuration settings.

It's expected that you construct distributions of distributions ie. a distribution can
be created and passed to another distribution as a parameter.

Note on Type Signatures:
The cdf and invcdf methods use NumericOrArray type signatures to provide
maximum flexibility for both scalar and array inputs. This is necessary because
scipy.special functions support array inputs despite having restrictive type stubs that
only show scalar
types. By using this union type, we can leverage scipy's vectorized operations while
maintaining type safety. For more details on scipy.special array support, see:
https://docs.scipy.org/doc/scipy-1.16.0/reference/generated/scipy.special.pdtr.html

Type Definitions:
- NumericType: Union[float, int, bool] - Basic numeric scalars
- NumericOrArray: NumericType | npt.NDArray[t.Any] | StochasticScalar - Union for
  maximum flexibility
"""

# Standard library imports
from __future__ import annotations

import typing as t
from abc import ABC

# Third-party imports
import numpy.typing as npt

# Local imports
from ._maths import special
from ._maths import xp as np
from .config import config
from .stochastic_scalar import StochasticScalar
from .types import ScipyNumeric

TOLERANCE = 1e-10  # Tolerance for numerical comparisons

# FIXME: Consider replaching with VectorLike from types.py
NumericOrArray = ScipyNumeric | npt.NDArray[t.Any] | StochasticScalar
ReturnType = NumericOrArray


class DistributionBase:
    """Abstract base class for statistical distributions."""

    def __init__(self, **params: NumericOrArray) -> None:
        """Initialize distribution with parameters."""
        # Store parameters in a private dictionary.
        self._params: dict[str, NumericOrArray] = params

    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute the cumulative distribution function at x.

        Args:
            x: Single value or sequence of values to evaluate.

        Returns:
            CDF value(s) - same type as input (Numeric -> Numeric,
            Sequence -> Sequence).
        """
        raise NotImplementedError

    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute the inverse cumulative distribution function at u.

        Args:
            u: Single probability or sequence of probabilities to evaluate.

        Returns:
            Quantile value(s) - same type as input (Numeric -> Numeric,
            Sequence -> Sequence).
        """
        raise NotImplementedError

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> StochasticScalar:
        """Generate random samples from the distribution.

        Parameters:
            n_sims (optional): Number of simulations. Uses config.n_sims if None.
            rng (optional): Random number generator.

        Returns:
            StochasticScalar: Generated samples.
        """
        if n_sims is None:
            n_sims = config.n_sims

        if rng is None:
            rng = config.rng

        result = self._generate(n_sims, rng)
        # Merge coupled variable groups from parameters if applicable.
        for param in self._params.values():
            if isinstance(param, StochasticScalar):
                result.coupled_variable_group.merge(param.coupled_variable_group)
        return result

    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        """Generate random samples using the inverse CDF technique.

        Args:
            n_sims: Number of simulations to generate. Must be >= 1.
            rng: Random number generator to use.

        Returns:
            StochasticScalar containing the generated random samples.

        Raises:
            ValueError: If n_sims < 1.
        """
        if n_sims < 1:
            raise ValueError(f"n_sims must be >= 1, got {n_sims}")

        # Generate uniform random numbers and transform via inverse CDF
        # When n_sims >= 1, rng.uniform(size=n_sims) returns an array,
        # so invcdf also returns an array (SequenceLike) due to overload typing
        uniform_samples = rng.uniform(size=n_sims)
        result = self.invcdf(uniform_samples)
        return StochasticScalar(result)

    @property
    def _param_values(
        self,
    ) -> t.Generator[NumericOrArray]:
        # Yields parameter values; if a parameter is a StochasticScalar, its
        # 'values' are returned - which will be a numpy array otherwise we just yield
        # the parameter value directly.
        for param in self._params.values():
            yield param.values if isinstance(param, StochasticScalar) else param


class DiscreteDistributionBase(DistributionBase, ABC):
    """Abstract base class for discrete distributions."""

    # Inherits __init__, cdf, and invcdf abstract methods from DistributionBase.
    pass


# --- Discrete Distributions ---


class Poisson(DiscreteDistributionBase):
    """Poisson Distribution.

    Parameters:
        mean: Mean number of events.
    """

    def __init__(self, mean: NumericOrArray) -> None:
        """Initialize Poisson distribution with mean parameter."""
        super().__init__(mean=mean)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        # scipy.special functions support array inputs despite restrictive type stubs
        (mean,) = self._param_values
        return special.pdtr(x, mean)

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        # scipy.special functions support array inputs despite restrictive type stubs
        (mean,) = self._param_values
        return special.pdtrik(u, mean)

    @t.override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        (mean,) = self._param_values
        return StochasticScalar(
            rng.poisson(t.cast(float | npt.NDArray[np.floating], mean), n_sims)
        )


class NegBinomial(DiscreteDistributionBase):
    """Negative Binomial Distribution."""

    def __init__(
        self,
        n: NumericOrArray,
        p: NumericOrArray,
    ) -> None:
        """Initialize negative binomial distribution.

        Args:
            n: Number of failures until stop.
            p: Probability of success.
        """
        super().__init__(n=n, p=p)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        n, p = self._param_values
        return special.nbdtr(x, n, p)  # type: ignore[misc, arg-type]

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        n, p = self._param_values
        return special.nbdtri(u, n, p)  # type: ignore[misc, arg-type]

    @t.override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        n, p = self._param_values
        return StochasticScalar(
            rng.negative_binomial(
                t.cast(int | npt.NDArray[np.integer], n),
                t.cast(float | npt.NDArray[np.floating], p),
                size=n_sims,
            )
        )


class Binomial(DiscreteDistributionBase):
    """Binomial Distribution."""

    def __init__(
        self, n: int | npt.NDArray[np.integer], p: float | npt.NDArray[np.floating]
    ) -> None:
        """Initialize binomial distribution.

        Args:
            n: Number of trials.
            p: Probability of success.
        """
        super().__init__(n=n, p=p)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        n, p = self._param_values
        return special.bdtr(x, n, p)

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        n, p = self._param_values
        return special.bdtri(u, n, p)

    @t.override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        n, p = self._param_values
        return StochasticScalar(
            rng.binomial(
                t.cast(int | npt.NDArray[np.integer], n),
                t.cast(float | npt.NDArray[np.floating], p),
                n_sims,
            )
        )


class HyperGeometric(DiscreteDistributionBase):
    """Hypergeometric Distribution.

    Models the number of successes in draws without replacement.

    Parameters:
        ngood: Number of good items.
        nbad: Number of bad items.
        population_size: Total population size.
    """

    def __init__(
        self,
        ngood: int,
        nbad: int,
        population_size: int,
    ) -> None:
        """Initialize hypergeometric distribution.

        Args:
            ngood: Number of good items.
            nbad: Number of bad items.
            population_size: Total population size.
        """
        # Note: population_size is stored with key 'n'
        super().__init__(ngood=ngood, nbad=nbad, n=population_size)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        raise NotImplementedError(f"CDF for {type(self).__name__} is not implemented.")

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        raise NotImplementedError(
            f"Inverse CDF for {type(self).__name__} is not implemented."
        )

    @t.override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        ngood, nbad, population_size = self._param_values
        return StochasticScalar(
            rng.hypergeometric(
                t.cast(int, ngood),
                t.cast(int, nbad),
                t.cast(int, population_size),
                n_sims,
            )
        )


# --- Continuous Distributions ---


class GPD(DistributionBase):
    r"""Generalized Pareto Distribution.

    Defined by:
        F(x) = 1 - (1 + ξ(x-μ)/σ)^(-1/ξ) for ξ ≠ 0,
        F(x) = 1 - exp(-(x-μ)/σ) for ξ = 0.
    """

    def __init__(
        self,
        shape: NumericOrArray,
        scale: NumericOrArray,
        loc: NumericOrArray,
    ) -> None:
        """Initialize GPD distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        shape, scale, loc = self._params.values()
        if abs(shape) <= TOLERANCE:
            result = 1 - np.exp(-(x - loc) / scale)
        else:
            result = 1 - (1 + shape * (x - loc) / scale) ** (-1 / shape)
        return result

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        shape, scale, loc = self._params.values()
        return (np.exp(np.log(1 - u) * (-shape)) - 1) * (scale / shape) + loc


class Burr(DistributionBase):
    r"""Burr Distribution.

    Defined by:
        F(x) = 1 - [1 + ((x-μ)/σ)^power]^(-shape), x > μ

    Parameters:
        power: The power parameter.
        shape: The shape parameter.
        scale: The scale parameter.
        loc: The location parameter.
    """

    def __init__(
        self,
        power: NumericOrArray,
        shape: NumericOrArray,
        scale: NumericOrArray,
        loc: NumericOrArray,
    ) -> None:
        """Initialize Burr distribution.

        Args:
            power: Power parameter.
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(power=power, shape=shape, scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        power, shape, scale, loc = self._params.values()
        return 1 - (1 + ((x - loc) / scale) ** power) ** (-shape)

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        power, shape, scale, loc = self._params.values()
        return scale * (((1 / (1 - u)) ** (1 / shape) - 1) ** (1 / power)) + loc


class Beta(DistributionBase):
    r"""Beta Distribution.

    Defined by:
        F(x) = (Γ(α+β) / (Γ(α)Γ(β))) ∫₀^((x-μ)/σ) u^(α-1)(1-u)^(β-1) du

    Parameters:
        alpha: Alpha parameter.
        beta: Beta parameter.
        scale: Scale parameter (default 1.0).
        loc: Location parameter (default 0.0).
    """

    def __init__(
        self,
        alpha: NumericOrArray,
        beta: NumericOrArray,
        scale: NumericOrArray = 1.0,
        loc: NumericOrArray = 0.0,
    ) -> None:
        """Initialize beta distribution.

        Args:
            alpha: Alpha parameter.
            beta: Beta parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(alpha=alpha, beta=beta, scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        alpha, beta, scale, loc = self._params.values()
        return special.betainc(alpha, beta, (x - loc) / scale)  # type: ignore[return-type]

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, beta, scale, loc = self._params.values()
        return special.betaincinv(alpha, beta, u) * scale + loc  # type: ignore[return-type]


class LogLogistic(DistributionBase):
    r"""Log-Logistic Distribution.

    Defined by:
        F(x) = y / (1 + y) where y = ((x-μ)/σ)^shape, x > μ

    Parameters:
        shape: Shape parameter.
        scale: Scale parameter.
        loc: Location parameter (default 0.0).
    """

    def __init__(
        self,
        shape: NumericOrArray,
        scale: NumericOrArray,
        loc: NumericOrArray = 0.0,
    ) -> None:
        """Initialize log-logistic distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** shape
        result = y / (1 + y)
        return result

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        shape, scale, loc = self._params.values()
        result = scale * ((u / (1 - u)) ** (1 / shape)) + loc
        return result


class Normal(DistributionBase):
    """Normal Distribution."""

    def __init__(self, mu: NumericOrArray, sigma: NumericOrArray) -> None:
        """Initialize normal distribution.

        Args:
            mu: Mean parameter.
            sigma: Standard deviation parameter.
        """
        super().__init__(mu=mu, sigma=sigma)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        mu, sigma = self._params.values()
        arg = (x - mu) / sigma
        return special.ndtr(t.cast(npt.NDArray[np.floating] | float, arg))

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        mu, sigma = self._param_values
        return special.ndtri(u) * sigma + mu


class Logistic(DistributionBase):
    """Logistic Distribution."""

    def __init__(self, mu: NumericOrArray, sigma: NumericOrArray) -> None:
        """Initialize logistic distribution.

        Args:
            mu: Location parameter.
            sigma: Scale parameter.
        """
        super().__init__(mu=mu, sigma=sigma)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        mu, sigma = self._param_values
        return 1 / (1 + np.exp(-(x - mu) / sigma))

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        mu, sigma = self._param_values
        return mu + sigma * np.log(u / (1 - u))


class LogNormal(DistributionBase):
    """Log-Normal Distribution."""

    def __init__(self, mu: NumericOrArray, sigma: NumericOrArray) -> None:
        """Initialize log-normal distribution.

        Args:
            mu: Mean of the logged variable.
            sigma: Standard deviation of the logged variable.
        """
        super().__init__(mu=mu, sigma=sigma)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        mu, sigma = self._param_values
        result = special.ndtr((np.log(x) - mu) / sigma)
        return result

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        mu, sigma = self._param_values
        return np.exp(special.ndtri(u) * sigma + mu)


class Gamma(DistributionBase):
    r"""Gamma Distribution.

    Defined by:
        F(x) = (1/Γ(α)) γ(α, (x-μ)/θ), x > μ
    """

    def __init__(
        self,
        alpha: NumericOrArray,
        theta: NumericOrArray,
        loc: NumericOrArray = 0.0,
    ) -> None:
        """Initialize gamma distribution.

        Args:
            alpha: Shape parameter.
            theta: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(alpha=alpha, theta=theta, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        alpha, theta, loc = self._param_values
        return special.gammainc(
            t.cast(npt.NDArray[np.floating] | float, alpha),
            t.cast(npt.NDArray[np.floating] | float, (x - loc) / theta),
        )  # type: ignore[return-type]

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, theta, loc = self._param_values
        result = special.gammaincinv(alpha, u) * theta + loc
        return result

    @t.override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        alpha, theta, loc = self._param_values
        result = StochasticScalar(
            rng.gamma(
                t.cast(float | npt.NDArray[np.floating], alpha),
                t.cast(float | npt.NDArray[np.floating], theta),
                size=n_sims,
            )
            + t.cast(float | npt.NDArray[np.floating], loc)
        )
        return result


class InverseGamma(DistributionBase):
    r"""Inverse Gamma Distribution.

    Defined by:
        F(x) = 1 - (1/Γ(α)) γ(α, θ/(x-μ)), x > μ
    """

    def __init__(
        self,
        alpha: NumericOrArray,
        theta: NumericOrArray,
        loc: NumericOrArray = 0.0,
    ) -> None:
        """Initialize inverse gamma distribution.

        Args:
            alpha: Shape parameter.
            theta: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(alpha=alpha, theta=theta, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        alpha, theta, loc = self._param_values
        return special.gammaincc(alpha, np.divide(theta, (x - loc)))

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, theta, loc = self._param_values
        return np.divide(theta, special.gammainccinv(alpha, u)) + loc


class Pareto(DistributionBase):
    r"""Pareto Distribution.

    Defined by:
        F(x) = 1 - (x_m / x)^a
    """

    def __init__(self, shape: NumericOrArray, scale: NumericOrArray) -> None:
        """Initialize Pareto distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
        """
        super().__init__(shape=shape, scale=scale)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        shape, scale = self._param_values
        return 1 - (x / scale) ** (-shape)

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        shape, scale = self._param_values
        return (1 - u) ** (-1 / shape) * scale


class Paralogistic(DistributionBase):
    r"""ParaLogistic Distribution.

    Defined by:
        F(x) = 1 - [1 + ((x-μ)/σ)^α]^(-α), x > μ

    Parameters:
        shape: Shape parameter.
        scale: Scale parameter.
        loc: Location parameter (default 0).
    """

    def __init__(
        self,
        shape: NumericOrArray,
        scale: NumericOrArray,
        loc: NumericOrArray = 0.0,
    ) -> None:
        """Initialize paralogistic distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        shape, scale, loc = self._params.values()
        y = 1 / (1 + ((x - loc) / scale) ** shape)
        return 1 - y**shape

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        shape, scale, loc = self._params.values()
        return loc + scale * (((1 - u) ** (-1 / shape)) - 1) ** (1 / shape)


class InverseBurr(DistributionBase):
    r"""Inverse Burr Distribution.

    Defined by:
        F(x) = [(( (x-μ)/σ )^τ / (1 + ((x-μ)/σ )^τ)]^α

    Parameters:
        power: Power parameter (τ).
        shape: Shape parameter (α).
        scale: Scale parameter (σ).
        loc: Location parameter (μ).
    """

    def __init__(
        self,
        power: NumericOrArray,
        shape: NumericOrArray,
        scale: NumericOrArray,
        loc: NumericOrArray,
    ) -> None:
        """Initialize inverse Burr distribution.

        Args:
            power: Power parameter.
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(power=power, shape=shape, scale=scale, loc=loc)
        self._power = power
        self._shape = shape
        self._scale = scale
        self._loc = loc

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        y = ((x - self._loc) / self._scale) ** self._power
        return (y / (1 + y)) ** self._shape

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        # Transform quantile u using shape parameter
        u_transformed = np.float_power(u, (-1 / self._shape))

        # Calculate intermediate term for power transformation
        power_base = u_transformed - 1

        # Apply inverse power transformation
        power_transformed = np.float_power(power_base, (-1 / self._power))

        # Scale and translate the result
        return self._scale * power_transformed + self._loc  # type: ignore[no-any-return]


class InverseParalogistic(DistributionBase):
    r"""Inverse ParaLogistic Distribution.

    Represents an Inverse ParaLogistic distribution with given shape and scale
    parameters.

    Its CDF is defined as:

        F(x) = [(( (x-μ)/σ )^α / (1 + ((x-μ)/σ )^α)]^(-α),  x > μ
    """

    def __init__(
        self,
        shape: NumericOrArray,
        scale: NumericOrArray,
        loc: NumericOrArray = 0.0,
    ) -> None:
        """Initialize inverse paralogistic distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        # Unpack parameters with explicit type annotations
        params = tuple(self._params.values())
        shape_val = params[0]
        scale_val = params[1]
        loc_val = params[2]
        y = ((x - loc_val) / scale_val) ** shape_val
        return (y / (1 + y)) ** shape_val

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        params = tuple(self._params.values())
        shape_val = params[0]
        scale_val = params[1]
        loc_val = params[2]
        y = u ** (1 / shape_val)
        return loc_val + scale_val * (y / (1 - y)) ** (1 / shape_val)


class Weibull(DistributionBase):
    r"""Weibull Distribution.

    Defined by:
        F(x) = 1 - exp(-((x-μ)/σ)^α), x > μ
    """

    def __init__(self, shape: float, scale: float, loc: float = 0) -> None:
        """Initialize Weibull distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** shape
        return -np.expm1(-y)

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        shape, scale, loc = self._params.values()
        return loc + scale * (-np.log(1 - u)) ** (1 / shape)


class InverseWeibull(DistributionBase):
    r"""Inverse Weibull Distribution.

    Defined by:
        F(x) = exp(-((x-μ)/σ)^(-α)), x > μ

    Parameters:
        shape (float): Shape parameter (α).
        scale (float): Scale parameter (σ).
        loc (float): Location parameter (μ).
    """

    def __init__(self, shape: float, scale: float, loc: float = 0) -> None:
        """Initialize inverse Weibull distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)
        self._shape = shape
        self._scale = scale
        self._loc = loc

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        y = np.float_power((x - self._loc) / self._scale, -self._shape)
        return np.exp(-y)  # type: ignore[no-any-return]

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        return self._loc + self._scale * (-1 / np.log(u)) ** (1 / self._shape)


class Exponential(DistributionBase):
    r"""Exponential Distribution.

    Defined by:
        F(x) = 1 - exp(-((x-μ)/σ)), x > μ

    Parameters:
        scale: Scale parameter.
        loc: Location parameter (default 0).
    """

    def __init__(self, scale: NumericOrArray, loc: NumericOrArray = 0.0) -> None:
        """Initialize exponential distribution.

        Args:
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        scale, loc = self._params.values()
        y = (x - loc) / scale
        return -np.expm1(-y)

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        scale, loc = self._params.values()
        return loc + scale * (-np.log(1 - u))


class Uniform(DistributionBase):
    r"""Uniform Distribution.

    Defined by:
        F(x) = (x - a) / (b - a), for a <= x <= b

    Parameters:
        a (float): Lower bound.
        b (float): Upper bound.
    """

    def __init__(self, a: float, b: float) -> None:
        """Initialize uniform distribution.

        Args:
            a: Lower bound.
            b: Upper bound.
        """
        super().__init__(a=a, b=b)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Compute cumulative distribution function."""
        a, b = self._params.values()
        return (x - a) / (b - a)

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        a, b = self._params.values()
        return a + (b - a) * u


class InverseExponential(DistributionBase):
    r"""Inverse Exponential Distribution.

    Defined by:
        F(x) = exp(-σ/(x-μ)), x > μ

    Parameters:
        scale (float): Scale parameter.
        loc (float): Location parameter (default 0).
    """

    def __init__(self, scale: float, loc: float = 0) -> None:
        """Initialize inverse exponential distribution.

        Args:
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(scale=scale, loc=loc)

    @t.override
    def cdf(self, x: NumericOrArray) -> ReturnType:
        scale, loc = self._params.values()
        y = scale * np.float_power((x - loc), -1)
        return np.exp(-y)

    @t.override
    def invcdf(self, u: NumericOrArray) -> ReturnType:
        scale, loc = self._params.values()
        return loc - scale / np.log(u)


# --- Distribution Generator Classes ---

AVAILABLE_DISCRETE_DISTRIBUTIONS: dict[str, t.Any] = {
    "poisson": Poisson,
    "negbinomial": NegBinomial,
    "binomial": Binomial,
    "hypergeometric": HyperGeometric,
}

AVAILABLE_CONTINUOUS_DISTRIBUTIONS: dict[str, t.Any] = {
    "beta": Beta,
    "burr": Burr,
    "exponential": Exponential,
    "gamma": Gamma,
    "gpd": GPD,
    "logistic": Logistic,
    "lognormal": LogNormal,
    "loglogistic": LogLogistic,
    "normal": Normal,
    "paralogistic": Paralogistic,
    "pareto": Pareto,
    "uniform": Uniform,
    "inverseburr": InverseBurr,
    "inverseexponential": InverseExponential,
    "inversegamma": InverseGamma,
    "inverseparalogistic": InverseParalogistic,
    "inverseweibull": InverseWeibull,
    "weibull": Weibull,
}


class DistributionGeneratorBase:
    """Base class for parameterized distribution generators.

    Wraps a DistributionBase instance.
    """

    def __init__(self, distribution: DistributionBase) -> None:
        """Initialize distribution generator with a distribution instance.

        Args:
            distribution: The distribution to wrap.
        """
        self.this_distribution = distribution

    def cdf(self, x: NumericOrArray) -> ReturnType:
        """Delegate to wrapped distribution."""
        return self.this_distribution.cdf(x)

    def invcdf(self, u: NumericOrArray) -> ReturnType:
        """Delegate to wrapped distribution."""
        return self.this_distribution.invcdf(u)

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator = config.rng
    ) -> StochasticScalar:
        """Delegate to wrapped distribution."""
        return self.this_distribution.generate(n_sims, rng)


class DiscreteDistributionGenerator(DistributionGeneratorBase):
    """Discrete distribution generator instantiated by name."""

    def __init__(
        self, distribution_name: str, parameters: list[NumericOrArray]
    ) -> None:
        """Initialize discrete distribution by name.

        Args:
            distribution_name: Name of the discrete distribution.
            parameters: Distribution parameters.
        """
        distribution_name = distribution_name.lower()
        if distribution_name not in AVAILABLE_DISCRETE_DISTRIBUTIONS:
            raise ValueError(
                f"Distribution {distribution_name} must be one of "
                f"{list(AVAILABLE_DISCRETE_DISTRIBUTIONS.keys())}"
            )
        distribution_cls = AVAILABLE_DISCRETE_DISTRIBUTIONS[distribution_name]
        super().__init__(distribution_cls(*parameters))


class ContinuousDistributionGenerator(DistributionGeneratorBase):
    """Continuous distribution generator instantiated by name."""

    def __init__(
        self, distribution_name: str, parameters: list[NumericOrArray]
    ) -> None:
        """Initialize continuous distribution by name.

        Args:
            distribution_name: Name of the continuous distribution.
            parameters: Distribution parameters.
        """
        distribution_name = distribution_name.lower()
        if distribution_name not in AVAILABLE_CONTINUOUS_DISTRIBUTIONS:
            raise ValueError(
                f"Distribution {distribution_name} must be one of "
                f"{list(AVAILABLE_CONTINUOUS_DISTRIBUTIONS.keys())}"
            )
        distribution_cls = AVAILABLE_CONTINUOUS_DISTRIBUTIONS[distribution_name]
        super().__init__(distribution_cls(*parameters))
