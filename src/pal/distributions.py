"""Distributions Module.

This module contains classes for simulating statistical distributions.
The implementations follow conventions similar to Klugman. Random number
generation and GPU support are managed via configuration settings.

It's expected that you construct distributions of distributions ie. a distribution can
be created and passed to another distribution as a parameter.

Note on Type Signatures:
Distributions accept and return only primitives (int, float) or StochasticScalar.
The DistributionParameter type alias is Union[int, float, StochasticScalar].
Internally, scipy.special functions may operate on arrays extracted from
StochasticScalar.values, but the public API never exposes raw numpy arrays.

Type Definitions:
- DistributionParameter: Union[int, float, StochasticScalar]
- ReturnType: Union[int, float, StochasticScalar]
"""

# Standard library imports
from __future__ import annotations

import typing as t
from abc import ABC

import numpy as np

# Local imports
from ._compat import override
from ._maths import special, xp
from .config import config
from .stochastic_scalar import StochasticScalar
from .types import DistributionParameter

TOLERANCE = 1e-10  # Tolerance for numerical comparisons
# FIXME: Consider replaching with VectorLike from types.py
ReturnType = t.Union[int, float, StochasticScalar]


class DistributionBase:
    """Abstract base class for statistical distributions."""

    def __init__(self, **params: DistributionParameter) -> None:
        """Initialize distribution with parameters."""
        # Store parameters in a private dictionary.
        self._params: dict[str, DistributionParameter] = params

    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute the cumulative distribution function at x.

        Args:
            x: Single value or sequence of values to evaluate.

        Returns:
            CDF value(s) - same type as input (Numeric -> Numeric,
            Sequence -> Sequence).
        """
        raise NotImplementedError

    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute the inverse cumulative distribution function at u.

        Args:
            u: Single probability or sequence of probabilities to evaluate.

        Returns:
            Quantile value(s) - same type as input (Numeric -> Numeric,
            Sequence -> Sequence).
        """
        raise NotImplementedError

    def generate(self, n_sims: int | None = None, rng: np.random.Generator | None = None) -> StochasticScalar:
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
        uniform_samples = StochasticScalar(rng.uniform(size=n_sims))
        result = self.invcdf(uniform_samples)
        return StochasticScalar(result)

    @property
    def _param_values(
        self,
    ) -> t.Generator[t.Any, None, None]:
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
    r"""Poisson Distribution.

    The probability mass function (PMF) is:

    .. math::

        P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots

    where :math:`\lambda > 0` is the mean (and variance) of the distribution.

    The cumulative distribution function is:

    .. math::

        F(k) = e^{-\lambda} \sum_{i=0}^{\lfloor k \rfloor} \frac{\lambda^i}{i!}

    Parameters:
        mean: Mean number of events :math:`\lambda`.
    """

    def __init__(self, mean: DistributionParameter) -> None:
        """Initialize Poisson distribution with mean parameter."""
        super().__init__(mean=mean)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        # scipy.special functions support array inputs despite restrictive type stubs
        (mean,) = self._param_values
        return special.pdtr(x, mean)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        # scipy.special functions support array inputs despite restrictive type stubs
        (mean,) = self._param_values
        return special.pdtrik(u, mean)

    @override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        (mean,) = self._param_values
        return StochasticScalar(rng.poisson(mean, n_sims))


class NegBinomial(DiscreteDistributionBase):
    r"""Negative Binomial Distribution.

    The probability mass function (PMF) is:

    .. math::

        P(X = k) = \binom{k + r - 1}{k} p^r (1-p)^k, \quad k = 0, 1, 2, \ldots

    where :math:`r > 0` is the number of failures until stop and :math:`0 < p < 1`
    is the probability of success.

    Often used to model overdispersed count data.
    """

    def __init__(
        self,
        n: DistributionParameter,
        p: DistributionParameter,
    ) -> None:
        """Initialize negative binomial distribution.

        Args:
            n: Number of failures until stop.
            p: Probability of success.
        """
        super().__init__(n=n, p=p)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        n, p = self._param_values
        return special.nbdtr(x, n, p)  # type: ignore[misc, arg-type]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        n, p = self._param_values
        return special.nbdtri(u, n, p)  # type: ignore[misc, arg-type]

    @override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        n, p = self._param_values
        return StochasticScalar(rng.negative_binomial(n, p, size=n_sims))


class Binomial(DiscreteDistributionBase):
    r"""Binomial Distribution.

    The probability mass function (PMF) is:

    .. math::

        P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n

    where :math:`n` is the number of trials and :math:`0 \leq p \leq 1` is the
    probability of success on each trial.

    Models the number of successes in a fixed number of independent Bernoulli trials.
    """

    def __init__(self, n: DistributionParameter, p: DistributionParameter) -> None:
        """Initialize binomial distribution.

        Args:
            n: Number of trials.
            p: Probability of success.
        """
        super().__init__(n=n, p=p)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        n, p = self._param_values
        return special.bdtr(x, n, p)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        n, p = self._param_values
        return special.bdtri(u, n, p)  # type: ignore[return-value]

    @override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        n, p = self._param_values
        return StochasticScalar(rng.binomial(n, p, n_sims))


class HyperGeometric(DiscreteDistributionBase):
    r"""Hypergeometric Distribution.

    The probability mass function (PMF) is:

    .. math::

        P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}

    where :math:`N` is the population size, :math:`K` is the number of success
    states in the population, :math:`n` is the number of draws, and :math:`k`
    is the number of observed successes.

    Models the number of successes in draws without replacement from a finite population.

    Parameters:
        ngood: Number of good items :math:`K`.
        nbad: Number of bad items :math:`N-K`.
        n_draws: Number of items drawn :math:`n`.
    """

    def __init__(
        self,
        ngood: int,
        nbad: int,
        n_draws: int,
    ) -> None:
        """Initialize hypergeometric distribution.

        Args:
            ngood: Number of good items.
            nbad: Number of bad items.
            n_draws: Number of items drawn.
        """
        # Note: n_draws is stored with key 'n'
        super().__init__(ngood=ngood, nbad=nbad, n=n_draws)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        if xp.__name__ == "cupy":
            raise NotImplementedError("HyperGeometric CDF is not supported on GPU.")

        # Use scipy.stats because scipy.special does not expose hypergeom CDF directly
        from scipy.stats import hypergeom

        ngood, nbad, n_draws = self._param_values
        m = ngood + nbad
        n = ngood
        n_total = n_draws
        return hypergeom.cdf(x, m, n, n_total)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        if xp.__name__ == "cupy":
            raise NotImplementedError("HyperGeometric inverse CDF is not supported on GPU.")

        from scipy.stats import hypergeom

        ngood, nbad, n_draws = self._param_values
        m = ngood + nbad
        n = ngood
        return hypergeom.ppf(u, m, n, n_draws)  # type: ignore[return-value]

    @override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        ngood, nbad, n_draws = self._param_values
        return StochasticScalar(
            rng.hypergeometric(
                t.cast(int, ngood),
                t.cast(int, nbad),
                t.cast(int, n_draws),
                n_sims,
            )
        )


class Bernoulli(Binomial):
    r"""Bernoulli Distribution.

    The probability mass function (PMF) is:

    .. math::

        P(X = k) = p^k (1-p)^{1-k}, \quad k = 0, 1

    where :math:`0 \leq p \leq 1` is the probability of success.

    Models a single trial with two possible outcomes: success (1) or failure (0).
    """

    def __init__(self, p: DistributionParameter) -> None:
        """Initialize Bernoulli distribution.

        Args:
            p: Probability of success.
        """
        super().__init__(n=1, p=p)


# --- Continuous Distributions ---


class GPD(DistributionBase):
    r"""Generalized Pareto Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \begin{cases}
        1 - \left(1 + \frac{\xi(x-\mu)}{\sigma}\right)^{-1/\xi} & \text{for } \xi \neq 0 \\
        1 - \exp\left(-\frac{x-\mu}{\sigma}\right) & \text{for } \xi = 0
        \end{cases}

    where :math:`\xi` is the shape parameter, :math:`\sigma` is the scale parameter,
    and :math:`\mu` is the location parameter.
    """

    def __init__(
        self,
        shape: DistributionParameter,
        scale: DistributionParameter,
        loc: DistributionParameter,
    ) -> None:
        """Initialize GPD distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        shape, scale, loc = self._params.values()
        if abs(shape) <= TOLERANCE:
            result = 1 - np.exp(-(x - loc) / scale)
        else:
            result = 1 - (1 + shape * (x - loc) / scale) ** (-1 / shape)
        return result  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        shape, scale, loc = self._params.values()
        return (np.exp(np.log(1 - u) * (-shape)) - 1) * (scale / shape) + loc  # type: ignore[return-value]


class Burr(DistributionBase):
    r"""Burr Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = 1 - \left[1 + \left(\frac{x-\mu}{\sigma}\right)^c\right]^{-k}, \quad x > \mu

    where :math:`c` is the power parameter, :math:`k` is the shape parameter,
    :math:`\sigma` is the scale parameter, and :math:`\mu` is the location parameter.

    Parameters:
        power: The power parameter :math:`c`.
        shape: The shape parameter :math:`k`.
        scale: The scale parameter :math:`\sigma`.
        loc: The location parameter :math:`\mu`.
    """

    def __init__(
        self,
        power: DistributionParameter,
        shape: DistributionParameter,
        scale: DistributionParameter,
        loc: DistributionParameter,
    ) -> None:
        """Initialize Burr distribution.

        Args:
            power: Power parameter.
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(power=power, shape=shape, scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        power, shape, scale, loc = self._params.values()
        return 1 - (1 + ((x - loc) / scale) ** power) ** (-shape)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        power, shape, scale, loc = self._params.values()
        return scale * (((1 / (1 - u)) ** (1 / shape) - 1) ** (1 / power)) + loc


class Beta(DistributionBase):
    r"""Beta Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = I_{(x-\mu)/\sigma}(\alpha, \beta) =
            \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
            \int_0^{(x-\mu)/\sigma} t^{\alpha-1}(1-t)^{\beta-1} dt

    where :math:`I_x(\alpha, \beta)` is the regularized incomplete beta function,
    :math:`\Gamma` is the gamma function, :math:`\alpha` and :math:`\beta` are shape parameters,
    :math:`\sigma` is the scale parameter, and :math:`\mu` is the location parameter.

    Parameters:
        alpha: Alpha shape parameter :math:`\alpha > 0`.
        beta: Beta shape parameter :math:`\beta > 0`.
        scale: Scale parameter :math:`\sigma` (default 1.0).
        loc: Location parameter :math:`\mu` (default 0.0).
    """

    def __init__(
        self,
        alpha: DistributionParameter,
        beta: DistributionParameter,
        scale: DistributionParameter = 1.0,
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize beta distribution.

        Args:
            alpha: Alpha parameter.
            beta: Beta parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(alpha=alpha, beta=beta, scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        alpha, beta, scale, loc = self._params.values()
        return special.betainc(alpha, beta, (x - loc) / scale)  # type: ignore[return-type]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, beta, scale, loc = self._params.values()
        return special.betaincinv(alpha, beta, u) * scale + loc  # type: ignore[return-type]

    @override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        alpha, beta, scale, loc = self._param_values
        return StochasticScalar(rng.beta(alpha, beta, n_sims) * scale + loc)


class LogLogistic(DistributionBase):
    r"""Log-Logistic Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \frac{y}{1 + y}, \quad \text{where } y = \left(\frac{x-\mu}{\sigma}\right)^\alpha, \quad x > \mu

    where :math:`\alpha` is the shape parameter, :math:`\sigma` is the scale parameter,
    and :math:`\mu` is the location parameter.

    Parameters:
        shape: Shape parameter :math:`\alpha`.
        scale: Scale parameter :math:`\sigma`.
        loc: Location parameter :math:`\mu` (default 0.0).
    """

    def __init__(
        self,
        shape: DistributionParameter,
        scale: DistributionParameter,
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize log-logistic distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** shape
        result = y / (1 + y)
        return result

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        shape, scale, loc = self._params.values()
        result = scale * ((u / (1 - u)) ** (1 / shape)) + loc
        return result


class Normal(DistributionBase):
    r"""Normal (Gaussian) Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \Phi\left(\frac{x - \mu}{\sigma}\right) =
            \frac{1}{2}\left[1 + \text{erf}\left(
            \frac{x - \mu}{\sigma\sqrt{2}}\right)\right]

    where :math:`\Phi` is the standard normal CDF, :math:`\mu` is the mean,
    and :math:`\sigma > 0` is the standard deviation.

    The probability density function is:

    .. math::

        f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
    """

    def __init__(self, mu: DistributionParameter, sigma: DistributionParameter) -> None:
        """Initialize normal distribution.

        Args:
            mu: Mean parameter.
            sigma: Standard deviation parameter.
        """
        super().__init__(mu=mu, sigma=sigma)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        mu, sigma = self._params.values()
        arg = (x - mu) / sigma
        return special.ndtr(arg)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        mu, sigma = self._param_values
        return special.ndtri(u) * sigma + mu


class Logistic(DistributionBase):
    r"""Logistic Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \frac{1}{1 + e^{-(x-\mu)/\sigma}}

    where :math:`\mu` is the location parameter and :math:`\sigma > 0` is the
    scale parameter.

    The logistic distribution has heavier tails than the normal distribution.
    """

    def __init__(self, mu: DistributionParameter, sigma: DistributionParameter) -> None:
        """Initialize logistic distribution.

        Args:
            mu: Location parameter.
            sigma: Scale parameter.
        """
        super().__init__(mu=mu, sigma=sigma)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        mu, sigma = self._param_values
        return 1 / (1 + np.exp(-(x - mu) / sigma))

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        mu, sigma = self._param_values
        return mu + sigma * np.log(u / (1 - u))


class LogNormal(DistributionBase):
    r"""Log-Normal Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \Phi\left(\frac{\ln(x) - \mu}{\sigma}\right)

    where :math:`\Phi` is the standard normal CDF, :math:`\mu` is the mean of
    the logarithm of the variable, and :math:`\sigma > 0` is the standard deviation
    of the logarithm.

    If :math:`Y = \ln(X)` is normally distributed with mean :math:`\mu` and
    standard deviation :math:`\sigma`, then :math:`X` follows a log-normal distribution.
    """

    def __init__(self, mu: DistributionParameter, sigma: DistributionParameter) -> None:
        """Initialize log-normal distribution.

        Args:
            mu: Mean of the logged variable.
            sigma: Standard deviation of the logged variable.
        """
        super().__init__(mu=mu, sigma=sigma)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        mu, sigma = self._param_values
        result = special.ndtr((np.log(x) - mu) / sigma)
        return result

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        mu, sigma = self._param_values
        return np.exp(special.ndtri(u) * sigma + mu)


class Gamma(DistributionBase):
    r"""Gamma Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \frac{1}{\Gamma(\alpha)} \gamma\left(\alpha, \frac{x-\mu}{\theta}\right), \quad x > \mu

    where :math:`\Gamma(\alpha)` is the gamma function, :math:`\gamma(\alpha, z)` is the
    lower incomplete gamma function, :math:`\alpha` is the shape parameter,
    :math:`\theta` is the scale parameter, and :math:`\mu` is the location parameter.
    """

    def __init__(
        self,
        alpha: DistributionParameter,
        theta: DistributionParameter,
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize gamma distribution.

        Args:
            alpha: Shape parameter.
            theta: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(alpha=alpha, theta=theta, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        alpha, theta, loc = self._param_values
        return special.gammainc(alpha, (x - loc) / theta)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, theta, loc = self._param_values
        result = special.gammaincinv(alpha, u) * theta + loc
        return result

    @override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        alpha, theta, loc = self._param_values
        result = StochasticScalar(rng.gamma(alpha, theta, size=n_sims) + loc)
        return result


class InverseGamma(DistributionBase):
    r"""Inverse Gamma Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = 1 - \frac{1}{\Gamma(\alpha)} \gamma\left(\alpha,
            \frac{\theta}{x-\mu}\right), \quad x > \mu

    where :math:`\Gamma(\alpha)` is the gamma function,
    :math:`\gamma(\alpha, z)` is the lower incomplete gamma function,
    :math:`\alpha > 0` is the shape parameter, :math:`\theta > 0` is the
    scale parameter, and :math:`\mu` is the location parameter.
    """

    def __init__(
        self,
        alpha: DistributionParameter,
        theta: DistributionParameter,
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize inverse gamma distribution.

        Args:
            alpha: Shape parameter.
            theta: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(alpha=alpha, theta=theta, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        alpha, theta, loc = self._param_values
        return special.gammaincc(alpha, np.divide(theta, (x - loc)))

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, theta, loc = self._param_values
        return np.divide(theta, special.gammainccinv(alpha, u)) + loc


class Pareto(DistributionBase):
    r"""Pareto Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = 1 - \left(\frac{x_m}{x}\right)^\alpha, \quad x \geq x_m

    where :math:`\alpha > 0` is the shape parameter (tail index) and
    :math:`x_m > 0` is the scale parameter (minimum value).

    The Pareto distribution is a power-law probability distribution often used
    to model heavy-tailed phenomena in actuarial science and economics.
    """

    def __init__(self, shape: DistributionParameter, scale: DistributionParameter) -> None:
        """Initialize Pareto distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
        """
        super().__init__(shape=shape, scale=scale)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        shape, scale = self._param_values
        return 1 - (x / scale) ** (-shape)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        shape, scale = self._param_values
        return (1 - u) ** (-1 / shape) * scale


class Paralogistic(DistributionBase):
    r"""ParaLogistic Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = 1 - \left[1 + \left(\frac{x-\mu}{\sigma}\right)^\alpha\right]^{-\alpha},
            \quad x > \mu

    where :math:`\alpha > 0` is the shape parameter, :math:`\sigma > 0` is the
    scale parameter, and :math:`\mu` is the location parameter.

    Parameters:
        shape: Shape parameter :math:`\alpha`.
        scale: Scale parameter :math:`\sigma`.
        loc: Location parameter :math:`\mu` (default 0).
    """

    def __init__(
        self,
        shape: DistributionParameter,
        scale: DistributionParameter,
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize paralogistic distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        shape, scale, loc = self._params.values()
        y = 1 / (1 + ((x - loc) / scale) ** shape)
        return 1 - y**shape

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        shape, scale, loc = self._params.values()
        return loc + scale * (((1 - u) ** (-1 / shape)) - 1) ** (1 / shape)


class InverseBurr(DistributionBase):
    r"""Inverse Burr Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \left[\frac{\left(\frac{x-\mu}{\sigma}\right)^\tau}
            {1 + \left(\frac{x-\mu}{\sigma}\right)^\tau}\right]^\alpha

    where :math:`\tau > 0` is the power parameter, :math:`\alpha > 0` is the shape
    parameter, :math:`\sigma > 0` is the scale parameter, and :math:`\mu` is the
    location parameter.

    Parameters:
        power: Power parameter :math:`\tau`.
        shape: Shape parameter :math:`\alpha`.
        scale: Scale parameter :math:`\sigma`.
        loc: Location parameter :math:`\mu`.
    """

    def __init__(
        self,
        power: DistributionParameter,
        shape: DistributionParameter,
        scale: DistributionParameter,
        loc: DistributionParameter,
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

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        y = ((x - self._loc) / self._scale) ** self._power
        return (y / (1 + y)) ** self._shape

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
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

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \left[\frac{\left(\frac{x-\mu}{\sigma}\right)^\alpha}
            {1 + \left(\frac{x-\mu}{\sigma}\right)^\alpha}\right]^\alpha,
            \quad x > \mu

    where :math:`\alpha > 0` is the shape parameter, :math:`\sigma > 0` is the
    scale parameter, and :math:`\mu` is the location parameter.
    """

    def __init__(
        self,
        shape: DistributionParameter,
        scale: DistributionParameter,
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize inverse paralogistic distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        # Unpack parameters with explicit type annotations
        params = tuple(self._params.values())
        shape_val = params[0]
        scale_val = params[1]
        loc_val = params[2]
        y = ((x - loc_val) / scale_val) ** shape_val
        return (y / (1 + y)) ** shape_val

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        params = tuple(self._params.values())
        shape_val = params[0]
        scale_val = params[1]
        loc_val = params[2]
        y = u ** (1 / shape_val)
        return loc_val + scale_val * (y / (1 - y)) ** (1 / shape_val)


class Weibull(DistributionBase):
    r"""Weibull Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = 1 - \exp\left[-\left(\frac{x-\mu}{\sigma}\right)^\alpha\right], \quad x > \mu

    where :math:`\alpha > 0` is the shape parameter, :math:`\sigma > 0` is the
    scale parameter, and :math:`\mu` is the location parameter.

    The Weibull distribution is widely used in reliability engineering and
    failure analysis.
    """

    def __init__(self, shape: float, scale: float, loc: float = 0) -> None:
        """Initialize Weibull distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** shape
        return -np.expm1(-y)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        shape, scale, loc = self._params.values()
        return loc + scale * (-np.log(1 - u)) ** (1 / shape)  # type: ignore[return-value]


class InverseWeibull(DistributionBase):
    r"""Inverse Weibull Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \exp\left[-\left(\frac{x-\mu}{\sigma}\right)^{-\alpha}\right],
            \quad x > \mu

    where :math:`\alpha > 0` is the shape parameter, :math:`\sigma > 0` is the
    scale parameter, and :math:`\mu` is the location parameter.

    Also known as the Fréchet distribution.

    Parameters:
        shape: Shape parameter :math:`\alpha`.
        scale: Scale parameter :math:`\sigma`.
        loc: Location parameter :math:`\mu`.
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

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        y = np.float_power((x - self._loc) / self._scale, -self._shape)
        return np.exp(-y)  # type: ignore[no-any-return]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        return self._loc + self._scale * (-1 / np.log(u)) ** (1 / self._shape)  # type: ignore[return-value]


class GEV(DistributionBase):
    r"""Generalized Extreme Value Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \begin{cases}
        \exp\left[-\left(1 + \xi\frac{x-\mu}{\sigma}\right)^{-1/\xi}\right]
            & \text{for } \xi \neq 0 \\
        \exp\left[-\exp\left(-\frac{x-\mu}{\sigma}\right)\right]
            & \text{for } \xi = 0
        \end{cases}

    where :math:`\xi` is the shape parameter, :math:`\sigma > 0` is the scale
    parameter, and :math:`\mu` is the location parameter.

    The GEV distribution unifies the Gumbel (:math:`\xi = 0`),
    Fréchet (:math:`\xi > 0`), and Weibull (:math:`\xi < 0`) families.
    Essential for extreme value analysis in catastrophe modeling.

    Parameters:
        shape: Shape parameter :math:`\xi`.
        scale: Scale parameter :math:`\sigma`.
        loc: Location parameter :math:`\mu` (default 0).
    """

    def __init__(
        self,
        shape: DistributionParameter,
        scale: DistributionParameter,
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize GEV distribution.

        Args:
            shape: Shape parameter.
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        shape, scale, loc = self._params.values()
        z = (x - loc) / scale
        if abs(shape) <= TOLERANCE:
            # Gumbel case (ξ = 0)
            return np.exp(-np.exp(-z))  # type: ignore[return-value]
        else:
            # Fréchet (ξ > 0) or Weibull (ξ < 0) case
            t = 1 + shape * z
            return np.exp(-np.power(t, -1 / shape))  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        shape, scale, loc = self._params.values()
        if abs(shape) <= TOLERANCE:
            # Gumbel case (ξ = 0)
            return loc - scale * np.log(-np.log(u))  # type: ignore[return-value]
        else:
            # Fréchet (ξ > 0) or Weibull (ξ < 0) case
            return loc + scale * (np.power(-np.log(u), -shape) - 1) / shape  # type: ignore[return-value]


class StudentsT(DistributionBase):
    r"""Student's t Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \frac{1}{2} + \frac{x\Gamma\left(\frac{\nu+1}{2}\right)}
            {\sqrt{\pi\nu}\Gamma\left(\frac{\nu}{2}\right)}
            \,_2F_1\left(\frac{1}{2}, \frac{\nu+1}{2}; \frac{3}{2};
            -\frac{x^2}{\nu}\right)

    where :math:`\nu > 0` is the degrees of freedom parameter,
    :math:`\Gamma` is the gamma function, and :math:`_2F_1` is the
    hypergeometric function.

    For the non-standardized version with location :math:`\mu` and
    scale :math:`\sigma`, substitute :math:`x \to (x-\mu)/\sigma`.

    The Student's t distribution has heavier tails than the normal distribution,
    making it useful for modeling extreme events in financial and operational risk.

    Parameters:
        nu: Degrees of freedom :math:`\nu`.
        mu: Location parameter :math:`\mu` (default 0).
        sigma: Scale parameter :math:`\sigma` (default 1).
    """

    def __init__(
        self,
        nu: DistributionParameter,
        mu: DistributionParameter = 0.0,
        sigma: DistributionParameter = 1.0,
    ) -> None:
        """Initialize Student's t distribution.

        Args:
            nu: Degrees of freedom.
            mu: Location parameter.
            sigma: Scale parameter.
        """
        super().__init__(nu=nu, mu=mu, sigma=sigma)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        nu, mu, sigma = self._params.values()
        z = (x - mu) / sigma
        # Use the relationship between t CDF and incomplete beta function
        # F(t; ν) = 1/2 + t * Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * 2F1(...)
        # Or equivalently: F(t; ν) = 1 - 1/2 * I_{ν/(ν+t²)}(ν/2, 1/2) for t > 0
        x_pos = np.abs(z)
        p = special.betainc(nu / 2, 0.5, nu / (nu + x_pos**2)) / 2
        result = np.where(z >= 0, 1 - p, p)
        return result  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        params = tuple(self._param_values)
        nu, mu, sigma = params[0], params[1], params[2]

        # Use the relationship between t-distribution and Beta distribution
        # to support GPU execution via betaincinv.
        # For X ~ t(nu), let Y = 2 * min(u, 1-u).
        # Then |X| = sqrt(nu * (1 / I^{-1}_Y(nu/2, 1/2) - 1))

        p_tilde = np.minimum(u, 1 - u)
        y = 2 * p_tilde
        x_beta = special.betaincinv(nu / 2, 0.5, y)
        x_sq = nu * (1 / x_beta - 1)
        x = np.sqrt(x_sq)
        sign = np.sign(u - 0.5)

        return mu + sigma * sign * x


class InverseGaussian(DistributionBase):
    r"""Inverse Gaussian (Wald) Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \Phi\left(\sqrt{\frac{\lambda}{x}}
            \left(\frac{x}{\mu}-1\right)\right)
            + \exp\left(\frac{2\lambda}{\mu}\right)
            \Phi\left(-\sqrt{\frac{\lambda}{x}}
            \left(\frac{x}{\mu}+1\right)\right)

    where :math:`\Phi` is the standard normal CDF, :math:`\mu > 0` is the mean
    parameter, and :math:`\lambda > 0` is the shape parameter.

    The inverse Gaussian distribution is widely used in operational risk modeling
    (Basel II) and for first passage time problems.

    Parameters:
        mu: Mean parameter :math:`\mu`.
        lambda_: Shape parameter :math:`\lambda`.
    """

    def __init__(
        self,
        mu: DistributionParameter,
        lambda_: DistributionParameter,
    ) -> None:
        """Initialize inverse Gaussian distribution.

        Args:
            mu: Mean parameter.
            lambda_: Shape parameter.
        """
        super().__init__(mu=mu, lambda_=lambda_)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        mu, lambda_ = self._param_values
        sqrt_lambda_x = np.sqrt(lambda_ / x)
        term1 = special.ndtr(sqrt_lambda_x * (x / mu - 1))
        term2 = np.exp(2 * lambda_ / mu) * special.ndtr(-sqrt_lambda_x * (x / mu + 1))
        return term1 + term2

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function.

        Uses numerical root finding since there is no closed form.
        """
        # For inverse Gaussian, there's no closed-form inverse CDF
        # We'll need to use numerical methods or approximations
        # This is a simplified implementation that may need scipy optimize
        raise NotImplementedError(
            "Inverse CDF for InverseGaussian requires numerical methods. "
            "Use the generate() method for sampling instead."
        )

    @override
    def _generate(self, n_sims: int, rng: np.random.Generator) -> StochasticScalar:
        """Generate samples using the algorithm from Michael, Schucany, and Haas.

        Reference:
            Michael, J. R., Schucany, W. R. and Haas, R. W. (1976).
            Generating random variates using transformations with multiple roots.
            The American Statistician 30, 88-90.
        """
        mu, lambda_ = self._param_values
        # Generate chi-squared(1) samples
        nu = rng.normal(0, 1, n_sims) ** 2
        y = mu + (mu**2 * nu) / (2 * lambda_) - (mu / (2 * lambda_)) * np.sqrt(4 * mu * lambda_ * nu + mu**2 * nu**2)

        # Random selection step
        u = rng.uniform(0, 1, n_sims)
        x = np.where(u <= mu / (mu + y), y, mu**2 / y)

        return StochasticScalar(x)


class Exponential(DistributionBase):
    r"""Exponential Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = 1 - \exp\left(-\frac{x-\mu}{\sigma}\right), \quad x > \mu

    where :math:`\sigma > 0` is the scale parameter (mean) and :math:`\mu` is the
    location parameter.

    The exponential distribution is memoryless and commonly used to model
    waiting times.

    Parameters:
        scale: Scale parameter :math:`\sigma`.
        loc: Location parameter :math:`\mu` (default 0).
    """

    def __init__(self, scale: DistributionParameter, loc: DistributionParameter = 0.0) -> None:
        """Initialize exponential distribution.

        Args:
            scale: Scale parameter.
            loc: Location parameter.
        """
        super().__init__(scale=scale, loc=loc)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        scale, loc = self._params.values()
        y = (x - loc) / scale
        return -np.expm1(-y)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        scale, loc = self._params.values()
        return loc + scale * (-np.log(1 - u))  # type: ignore[return-value]


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

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        a, b = self._params.values()
        return (x - a) / (b - a)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        a, b = self._params.values()
        return a + (b - a) * u


class InverseExponential(DistributionBase):
    r"""Inverse Exponential Distribution.

    The cumulative distribution function (CDF) is:

    .. math::

        F(x) = \exp\left(-\frac{\sigma}{x-\mu}\right), \quad x > \mu

    where :math:`\sigma > 0` is the scale parameter and :math:`\mu` is the
    location parameter.

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

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        scale, loc = self._params.values()
        y = scale * np.float_power((x - loc), -1)
        return np.exp(-y)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        scale, loc = self._params.values()
        return loc - scale / np.log(u)  # type: ignore[return-value]


# --- Distribution Generator Classes ---

AVAILABLE_DISCRETE_DISTRIBUTIONS: dict[str, t.Any] = {
    "bernoulli": Bernoulli,
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
    "gev": GEV,
    "gpd": GPD,
    "inversegaussian": InverseGaussian,
    "logistic": Logistic,
    "lognormal": LogNormal,
    "loglogistic": LogLogistic,
    "normal": Normal,
    "paralogistic": Paralogistic,
    "pareto": Pareto,
    "studentst": StudentsT,
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

    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Delegate to wrapped distribution."""
        return self.this_distribution.cdf(x)

    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Delegate to wrapped distribution."""
        return self.this_distribution.invcdf(u)

    def generate(
        self,
        n_sims: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> StochasticScalar:
        """Delegate to wrapped distribution.

        Args:
            n_sims: Number of simulations. Uses config.n_sims if None.
            rng: Random number generator. Uses config.rng if None.
        """
        return self.this_distribution.generate(n_sims, rng)


class DiscreteDistributionGenerator(DistributionGeneratorBase):
    """Discrete distribution generator instantiated by name."""

    def __init__(self, distribution_name: str, parameters: list[DistributionParameter]) -> None:
        """Initialize discrete distribution by name.

        Args:
            distribution_name: Name of the discrete distribution.
            parameters: Distribution parameters.
        """
        distribution_name = distribution_name.lower()
        if distribution_name not in AVAILABLE_DISCRETE_DISTRIBUTIONS:
            raise ValueError(
                f"Distribution {distribution_name} must be one of {list(AVAILABLE_DISCRETE_DISTRIBUTIONS.keys())}"
            )
        distribution_cls = AVAILABLE_DISCRETE_DISTRIBUTIONS[distribution_name]
        super().__init__(distribution_cls(*parameters))


class ContinuousDistributionGenerator(DistributionGeneratorBase):
    """Continuous distribution generator instantiated by name."""

    def __init__(self, distribution_name: str, parameters: list[DistributionParameter]) -> None:
        """Initialize continuous distribution by name.

        Args:
            distribution_name: Name of the continuous distribution.
            parameters: Distribution parameters.
        """
        distribution_name = distribution_name.lower()
        if distribution_name not in AVAILABLE_CONTINUOUS_DISTRIBUTIONS:
            raise ValueError(
                f"Distribution {distribution_name} must be one of {list(AVAILABLE_CONTINUOUS_DISTRIBUTIONS.keys())}"
            )
        distribution_cls = AVAILABLE_CONTINUOUS_DISTRIBUTIONS[distribution_name]
        super().__init__(distribution_cls(*parameters))
