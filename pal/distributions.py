"""
Distributions Module

This module contains classes for simulating statistical distributions.
The implementations follow conventions similar to Klugman. Random number
generation and GPU support are managed via configuration settings.
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, TypeVar

# Local imports
from .config import config, xp as np, _use_gpu
from .stochastic_scalar import StochasticScalar

# Use cupyx if GPU is enabled
if _use_gpu:
    import cupyx.scipy.special as special  # type: ignore
else:
    import scipy.special as special  # type: ignore

# Type aliases
NumberType = Union[float, int]
NumberOrStochasticScalar = TypeVar(
    "NumberOrStochasticScalar", NumberType, StochasticScalar
)


class DistributionBase(ABC):
    """Abstract base class for statistical distributions."""

    def __init__(self, **params: Any) -> None:
        # Store parameters in a private dictionary.
        self._params: Dict[str, Any] = params

    @property
    def _param_values(self):
        # Yields parameter values; if a parameter is a StochasticScalar, its 'values' are returned.
        for param in self._params.values():
            yield param.values if isinstance(param, StochasticScalar) else param

    @abstractmethod
    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Compute the cumulative distribution function at x."""
        pass

    @abstractmethod
    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Compute the inverse cumulative distribution function at u."""
        pass

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator = config.rng
    ) -> StochasticScalar:
        """
        Generate random samples from the distribution.

        Parameters:
            n_sims (int, optional): Number of simulations. Uses config.n_sims if None.
            rng (np.random.Generator, optional): Random number generator.

        Returns:
            StochasticScalar: Generated samples.
        """
        if n_sims is None:
            n_sims = config.n_sims

        result = self._generate(n_sims, rng)
        # Merge coupled variable groups from parameters if applicable.
        for param in self._params.values():
            if isinstance(param, StochasticScalar):
                result.coupled_variable_group.merge(param.coupled_variable_group)
        return result

    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        # Default generation method using the inverse CDF technique.
        return StochasticScalar(self.invcdf(rng.uniform(size=n_sims)))


class DiscreteDistributionBase(DistributionBase, ABC):
    """Abstract base class for discrete distributions."""

    # Inherits __init__, cdf, and invcdf abstract methods from DistributionBase.
    pass


# --- Discrete Distributions ---


class Poisson(DiscreteDistributionBase):
    """Poisson Distribution.

    Parameters:
        mean (float): Mean number of events.
    """

    def __init__(self, mean: float) -> None:
        super().__init__(mean=mean)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        return special.pdtr(x, self._params["mean"])

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        return special.pdtrik(u, self._params["mean"])

    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        (mean,) = self._param_values
        return StochasticScalar(rng.poisson(mean, n_sims))


class NegBinomial(DiscreteDistributionBase):
    """Negative Binomial Distribution.

    Parameters:
        n (float or StochasticScalar): Number of failures until stop.
        p (float or StochasticScalar): Probability of success.
    """

    def __init__(
        self, n: Union[float, StochasticScalar], p: Union[float, StochasticScalar]
    ) -> None:
        super().__init__(n=n, p=p)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        n, p = self._param_values
        return special.nbdtr(x, n, p)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        n, p = self._param_values
        return special.nbdtri(u, n, p)

    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        n, p = self._param_values
        return StochasticScalar(rng.negative_binomial(n, p, n_sims))


class Binomial(DiscreteDistributionBase):
    """Binomial Distribution.

    Parameters:
        n (int): Number of trials.
        p (float): Probability of success.
    """

    def __init__(self, n: int, p: float) -> None:
        super().__init__(n=n, p=p)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        n, p = self._param_values
        return special.bdtr(x, n, p)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        n, p = self._param_values
        return special.bdtri(u, n, p)

    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        n, p = self._param_values
        return StochasticScalar(rng.binomial(n, p, n_sims))


class HyperGeometric(DiscreteDistributionBase):
    """Hypergeometric Distribution.

    Models the number of successes in draws without replacement.

    Parameters:
        ngood (int): Number of good items.
        nbad (int): Number of bad items.
        population_size (int): Total population size.
    """

    def __init__(self, ngood: int, nbad: int, population_size: int) -> None:
        # Note: population_size is stored with key 'n'
        super().__init__(ngood=ngood, nbad=nbad, n=population_size)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        raise NotImplementedError("CDF for HyperGeometric is not implemented.")

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        raise NotImplementedError("Inverse CDF for HyperGeometric is not implemented.")

    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        ngood, nbad, n = self._param_values
        return StochasticScalar(rng.hypergeometric(ngood, nbad, n, n_sims))


# --- Continuous Distributions ---


class GPD(DistributionBase):
    r"""Generalized Pareto Distribution.

    Defined by:
        F(x) = 1 - (1 + ξ(x-μ)/σ)^(-1/ξ) for ξ ≠ 0,
        F(x) = 1 - exp(-(x-μ)/σ) for ξ = 0.

    Parameters:
        shape: ξ, the shape parameter.
        scale: σ, the scale parameter.
        loc: μ, the location parameter.
    """

    def __init__(
        self,
        shape: NumberOrStochasticScalar,
        scale: NumberOrStochasticScalar,
        loc: NumberOrStochasticScalar,
    ) -> None:
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale, loc = self._params.values()
        if shape != 0:
            return 1 - (1 + shape * (x - loc) / scale) ** (-1 / shape)
        else:
            return 1 - np.exp(-(x - loc) / scale)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
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
        power: NumberOrStochasticScalar,
        shape: NumberOrStochasticScalar,
        scale: NumberOrStochasticScalar,
        loc: NumberOrStochasticScalar,
    ) -> None:
        super().__init__(power=power, shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        power, shape, scale, loc = self._params.values()
        return 1 - (1 + ((x - loc) / scale) ** power) ** (-shape)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        power, shape, scale, loc = self._params.values()
        return scale * (((1 / (1 - u)) ** (1 / shape) - 1) ** (1 / power)) + loc


class Beta(DistributionBase):
    r"""Beta Distribution.

    Defined by:
        F(x) = (Γ(α+β) / (Γ(α)Γ(β))) ∫₀^((x-μ)/σ) u^(α-1)(1-u)^(β-1) du

    Parameters:
        alpha (float): Alpha parameter.
        beta (float): Beta parameter.
        scale (float): Scale parameter (default 1).
        loc (float): Location parameter (default 0).
    """

    def __init__(
        self, alpha: float, beta: float, scale: float = 1, loc: float = 0
    ) -> None:
        super().__init__(alpha=alpha, beta=beta, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        alpha, beta, scale, loc = self._params.values()
        return special.betainc(alpha, beta, (x - loc) / scale)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        alpha, beta, scale, loc = self._params.values()
        return special.betaincinv(alpha, beta, u) * scale + loc


class LogLogistic(DistributionBase):
    r"""Log-Logistic Distribution.

    Defined by:
        F(x) = y / (1 + y) where y = ((x-μ)/σ)^shape, x > μ

    Parameters:
        shape (float): Shape parameter.
        scale (float): Scale parameter.
        loc (float): Location parameter (default 0).
    """

    def __init__(self, shape: float, scale: float, loc: float = 0) -> None:
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** shape
        return y / (1 + y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale, loc = self._params.values()
        return scale * ((u / (1 - u)) ** (1 / shape)) + loc


class Normal(DistributionBase):
    """Normal Distribution.

    Parameters:
        mu (float): Mean.
        sigma (float): Standard deviation.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        super().__init__(mu=mu, sigma=sigma)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        mu, sigma = self._param_values
        return special.ndtr((x - mu) / sigma)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        mu, sigma = self._param_values
        return special.ndtri(u) * sigma + mu


class Logistic(DistributionBase):
    """Logistic Distribution.

    Parameters:
        mu (float): Location parameter.
        sigma (float): Scale parameter.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        super().__init__(mu=mu, sigma=sigma)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        mu, sigma = self._param_values
        return 1 / (1 + np.exp(-(x - mu) / sigma))

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        mu, sigma = self._param_values
        return mu + sigma * np.log(u / (1 - u))


class LogNormal(DistributionBase):
    """Log-Normal Distribution.

    Parameters:
        mu (float): Mean of the logged variable.
        sigma (float): Standard deviation of the logged variable.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        super().__init__(mu=mu, sigma=sigma)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        mu, sigma = self._param_values
        return special.ndtr((np.log(x) - mu) / sigma)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        mu, sigma = self._param_values
        return np.exp(special.ndtri(u) * sigma + mu)


class Gamma(DistributionBase):
    r"""Gamma Distribution.

    Defined by:
        F(x) = (1/Γ(α)) γ(α, (x-μ)/θ), x > μ

    Parameters:
        alpha: Shape parameter.
        theta: Scale parameter.
        loc: Location parameter (default 0).
    """

    def __init__(self, alpha: float, theta: float, loc: float = 0) -> None:
        super().__init__(alpha=alpha, theta=theta, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        alpha, theta, loc = self._param_values
        return special.gammainc(alpha, (x - loc) / theta)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        alpha, theta, loc = self._param_values
        return special.gammaincinv(alpha, u) * theta + loc

    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        alpha, theta, loc = self._param_values
        return StochasticScalar(rng.gamma(alpha, theta, size=n_sims) + loc)


class InverseGamma(DistributionBase):
    r"""Inverse Gamma Distribution.

    Defined by:
        F(x) = 1 - (1/Γ(α)) γ(α, θ/(x-μ)), x > μ

    Parameters:
        alpha: Shape parameter.
        theta: Scale parameter.
        loc: Location parameter (default 0).
    """

    def __init__(self, alpha: float, theta: float, loc: float = 0) -> None:
        super().__init__(alpha=alpha, theta=theta, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        alpha, theta, loc = self._param_values
        return special.gammaincc(alpha, np.divide(theta, (x - loc)))

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        alpha, theta, loc = self._param_values
        return np.divide(theta, special.gammainccinv(alpha, u)) + loc


class Pareto(DistributionBase):
    r"""Pareto Distribution.

    Defined by:
        F(x) = 1 - (x_m / x)^a

    Parameters:
        shape (float): Shape parameter.
        scale (float): Scale parameter (x_m).
    """

    def __init__(self, shape: float, scale: float) -> None:
        super().__init__(shape=shape, scale=scale)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale = self._param_values
        return 1 - (x / scale) ** (-shape)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale = self._param_values
        return (1 - u) ** (-1 / shape) * scale


class Paralogistic(DistributionBase):
    r"""ParaLogistic Distribution.

    Defined by:
        F(x) = 1 - [1 + ((x-μ)/σ)^α]^(-α), x > μ

    Parameters:
        shape (float): Shape parameter.
        scale (float): Scale parameter.
        loc (float): Location parameter (default 0).
    """

    def __init__(self, shape: float, scale: float, loc: float = 0) -> None:
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale, loc = self._params.values()
        y = 1 / (1 + ((x - loc) / scale) ** shape)
        return 1 - y**shape

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale, loc = self._params.values()
        return loc + scale * (((1 - u) ** (-1 / shape)) - 1) ** (1 / shape)


class InverseBurr(DistributionBase):
    r"""Inverse Burr Distribution.

    Defined by:
        F(x) = [(( (x-μ)/σ )^τ / (1 + ((x-μ)/σ )^τ)]^α

    Parameters:
        power (float): Power parameter (τ).
        shape (float): Shape parameter (α).
        scale (float): Scale parameter (σ).
        loc (float): Location parameter (μ).
    """

    def __init__(self, power: float, shape: float, scale: float, loc: float) -> None:
        super().__init__(power=power, shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        power, shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** power
        return (y / (1 + y)) ** shape

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        power, shape, scale, loc = self._params.values()
        return (
            scale
            * (np.float_power((np.float_power(u, (-1 / shape)) - 1), (-1 / power)))
            + loc
        )


class InverseParalogistic(DistributionBase):
    r"""Inverse ParaLogistic Distribution.

    Represents an Inverse ParaLogistic distribution with given shape and scale parameters.
    Its CDF is defined as:

        F(x) = [(( (x-μ)/σ )^α / (1 + ((x-μ)/σ )^α)]^(-α),  x > μ

    Parameters:
        shape (float): The shape parameter.
        scale (float): The scale parameter.
        loc (float): The location parameter (default 0).
    """

    def __init__(self, shape: float, scale: float, loc: float = 0) -> None:
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        # Unpack parameters with explicit type annotations
        params = tuple(self._params.values())
        shape_val: float = params[0]
        scale_val: float = params[1]
        loc_val: float = params[2]
        y = ((x - loc_val) / scale_val) ** shape_val
        return (y / (1 + y)) ** shape_val

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        params = tuple(self._params.values())
        shape_val: float = params[0]
        scale_val: float = params[1]
        loc_val: float = params[2]
        y = u ** (1 / shape_val)
        return loc_val + scale_val * (y / (1 - y)) ** (1 / shape_val)


class Weibull(DistributionBase):
    r"""Weibull Distribution.

    Defined by:
        F(x) = 1 - exp(-((x-μ)/σ)^α), x > μ

    Parameters:
        shape (float): Shape parameter (α).
        scale (float): Scale parameter (σ).
        loc (float): Location parameter (μ).
    """

    def __init__(self, shape: float, scale: float, loc: float = 0) -> None:
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** shape
        return -np.expm1(-y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
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
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale, loc = self._params.values()
        y = np.float_power((x - loc) / scale, -shape)
        return np.exp(-y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        shape, scale, loc = self._params.values()
        return loc + scale * (-1 / np.log(u)) ** (1 / shape)


class Exponential(DistributionBase):
    r"""Exponential Distribution.

    Defined by:
        F(x) = 1 - exp(-((x-μ)/σ)), x > μ

    Parameters:
        scale (float): Scale parameter.
        loc (float): Location parameter (default 0).
    """

    def __init__(self, scale: float, loc: float = 0) -> None:
        super().__init__(scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        scale, loc = self._params.values()
        y = (x - loc) / scale
        return -np.expm1(-y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
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
        super().__init__(a=a, b=b)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        a, b = self._params.values()
        return (x - a) / (b - a)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
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
        super().__init__(scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        scale, loc = self._params.values()
        y = scale * np.float_power((x - loc), -1)
        return np.exp(-y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        scale, loc = self._params.values()
        return loc - scale / np.log(u)


# --- Distribution Generator Classes ---

AVAILABLE_DISCRETE_DISTRIBUTIONS: Dict[str, Any] = {
    "poisson": Poisson,
    "negbinomial": NegBinomial,
    "binomial": Binomial,
    "hypergeometric": HyperGeometric,
}

AVAILABLE_CONTINUOUS_DISTRIBUTIONS: Dict[str, Any] = {
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


class DistributionGeneratorBase(ABC):
    """
    Base class for parameterized distribution generators.
    Wraps a DistributionBase instance.
    """

    def __init__(self, distribution: DistributionBase) -> None:
        self.this_distribution = distribution

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        return self.this_distribution.cdf(x)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        return self.this_distribution.invcdf(u)

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator = config.rng
    ) -> StochasticScalar:
        return self.this_distribution.generate(n_sims, rng)


class DiscreteDistributionGenerator(DistributionGeneratorBase):
    """Discrete distribution generator instantiated by name."""

    def __init__(self, distribution_name: str, parameters: list) -> None:
        distribution_name = distribution_name.lower()
        if distribution_name not in AVAILABLE_DISCRETE_DISTRIBUTIONS:
            raise ValueError(
                f"Distribution {distribution_name} must be one of {list(AVAILABLE_DISCRETE_DISTRIBUTIONS.keys())}"
            )
        distribution_cls = AVAILABLE_DISCRETE_DISTRIBUTIONS[distribution_name]
        super().__init__(distribution_cls(*parameters))


class ContinuousDistributionGenerator(DistributionGeneratorBase):
    """Continuous distribution generator instantiated by name."""

    def __init__(self, distribution_name: str, parameters: list) -> None:
        distribution_name = distribution_name.lower()
        if distribution_name not in AVAILABLE_CONTINUOUS_DISTRIBUTIONS:
            raise ValueError(
                f"Distribution {distribution_name} must be one of {list(AVAILABLE_CONTINUOUS_DISTRIBUTIONS.keys())}"
            )
        distribution_cls = AVAILABLE_CONTINUOUS_DISTRIBUTIONS[distribution_name]
        super().__init__(distribution_cls(*parameters))
