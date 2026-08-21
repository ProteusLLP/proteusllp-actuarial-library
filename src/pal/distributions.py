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
import scipy.special as scipy_special
from scipy.stats import geninvgauss

# Local imports
from ._compat import override
from ._maths import asnumpy, scalar_or_array, special, to_backend, xp
from .config import config
from .stochastic_scalar import StochasticScalar
from .types import DistributionParameter, RandomGenerator

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
    if type(rng).__module__.startswith("numpy"):
        return asnumpy(value) if isinstance(value, xp.ndarray) else value
    return to_backend(value)


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

    def generate(self, n_sims: int | None = None, rng: RandomGenerator | None = None) -> StochasticScalar:
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
        if rng is None:
            raise RuntimeError("No random number generator is configured")

        result = self._generate(n_sims, rng)
        # Merge coupled variable groups from parameters if applicable.
        for param in self._params.values():
            if isinstance(param, StochasticScalar):
                result.coupled_variable_group.merge(param.coupled_variable_group)
        return result

    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
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
        uniform_samples = xp.asarray(rng.uniform(size=n_sims))
        result = self.invcdf(StochasticScalar(uniform_samples))
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
        return _special_call(special.pdtr, x, mean)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        # scipy.special functions support array inputs despite restrictive type stubs
        (mean,) = self._param_values
        # CuPy does not implement pdtrik. This uncommon inverse-CDF path uses
        # SciPy explicitly and transfers the result back to the active backend.
        u_values = u.values if isinstance(u, StochasticScalar) else u
        mean_values = asnumpy(mean) if hasattr(mean, "ndim") else mean
        result = xp.asarray(scipy_special.pdtrik(asnumpy(u_values), mean_values))
        if isinstance(u, StochasticScalar):
            wrapped = StochasticScalar(result)
            wrapped.coupled_variable_group.merge(u.coupled_variable_group)
            return wrapped
        return scalar_or_array(result)

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        (mean,) = self._param_values
        return StochasticScalar(rng.poisson(_rng_value(mean, rng), n_sims))


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
        return _special_call(special.nbdtr, x, n, p)  # type: ignore[misc, arg-type]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        n, p = self._param_values
        return _special_call(special.nbdtri, u, n, p)  # type: ignore[misc, arg-type]

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        n, p = self._param_values
        return StochasticScalar(rng.negative_binomial(_rng_value(n, rng), _rng_value(p, rng), size=n_sims))


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
        return _special_call(special.bdtr, x, n, p)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        n, p = self._param_values
        return _special_call(special.bdtri, u, n, p)  # type: ignore[return-value]

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        n, p = self._param_values
        return StochasticScalar(rng.binomial(_rng_value(n, rng), _rng_value(p, rng), n_sims))


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
        # Use scipy.stats because scipy.special does not expose hypergeom CDF directly
        from scipy.stats import hypergeom

        ngood, nbad, n_draws = self._param_values
        m = ngood + nbad
        n = ngood
        n_total = n_draws
        x_values = x.values if isinstance(x, StochasticScalar) else x
        result = xp.asarray(hypergeom.cdf(asnumpy(x_values), m, n, n_total))
        if isinstance(x, StochasticScalar):
            wrapped = StochasticScalar(result)
            wrapped.coupled_variable_group.merge(x.coupled_variable_group)
            return wrapped
        return scalar_or_array(result)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        from scipy.stats import hypergeom

        ngood, nbad, n_draws = self._param_values
        m = ngood + nbad
        n = ngood
        u_values = u.values if isinstance(u, StochasticScalar) else u
        result = xp.asarray(hypergeom.ppf(asnumpy(u_values), m, n, n_draws))
        if isinstance(u, StochasticScalar):
            wrapped = StochasticScalar(result)
            wrapped.coupled_variable_group.merge(u.coupled_variable_group)
            return wrapped
        return scalar_or_array(result)

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
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
        return _special_call(special.betainc, alpha, beta, (x - loc) / scale)  # type: ignore[return-type]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, beta, scale, loc = self._params.values()
        return _special_call(special.betaincinv, alpha, beta, u) * scale + loc  # type: ignore[return-type]

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        alpha, beta, scale, loc = self._param_values
        return StochasticScalar(
            rng.beta(_rng_value(alpha, rng), _rng_value(beta, rng), n_sims) * _rng_value(scale, rng)
            + _rng_value(loc, rng)
        )


class MBBEFD(DistributionBase):
    r"""Bernegger's MBBEFD distribution on the interval :math:`[0,1]`.

    This is the :math:`(g,b)` parameterisation of the Maxwell-Boltzmann,
    Bose-Einstein and Fermi-Dirac distribution class introduced by Bernegger
    for exposure rating. For finite :math:`g>1`, :math:`b>0`, :math:`b\ne1`
    and :math:`gb\ne1`, its cumulative distribution function is

    .. math::

        F(x) =
        \begin{cases}
        0, & x \leq 0, \\
        1 - \dfrac{(1-b)b^x}
        {(g-1)b + (1-gb)b^x}, & 0 < x < 1, \\
        1, & x \geq 1.
        \end{cases}

    The density of the continuous part on :math:`(0,1)` is

    .. math::

        f(x) = -\frac{(1-b)(g-1)\log(b)b^{1+x}}
        {\left((g-1)b+(1-gb)b^x\right)^2}.

    There is also an atom at total loss:

    .. math::

        \Pr(X=1)=\frac{1}{g},
        \qquad F(1^-)=1-\frac{1}{g}.

    Consequently, the quantile function in the main parameter region is

    .. math::

        F^{-1}(u) =
        \begin{cases}
        1 - \dfrac{\log\left(
        \dfrac{gb-1}{g-1}+
        \dfrac{1-b}{(1-u)(g-1)}
        \right)}{\log(b)}, & 0 \leq u < 1-1/g, \\
        1, & 1-1/g \leq u \leq 1.
        \end{cases}

    The limiting CDFs used when the main expression is indeterminate are

    .. math::

        F(x) = \frac{(g-1)x}{1+(g-1)x}
        \quad (b=1),
        \qquad
        F(x) = 1-b^x
        \quad (gb=1),

    for :math:`0<x<1`. If :math:`g=1` or :math:`b=0`, the distribution is a
    point mass at one.

    Its exposure curve is the normalized limited expected value

    .. math::

        G(x) = \frac{E[\min(X,x)]}{E[X]}
        = \frac{\log\left(
        \dfrac{(g-1)b+(1-gb)b^x}{1-b}
        \right)}{\log(gb)},
        \qquad 0 \leq x \leq 1.

    The corresponding mean is

    .. math::

        E[X] = \frac{(1-b)\log(gb)}{(1-gb)\log(b)}.

    The parameters satisfy :math:`g \geq 1` and :math:`b \geq 0` and are
    required to be finite by this implementation.

    Parameters:
        g: Reciprocal of the total-loss probability.
        b: Shape parameter.

    References:
        Bernegger, S. (1997). The Swiss Re Exposure Curves and the MBBEFD
        Distribution Class. ASTIN Bulletin 27(1), 99--111.
    """

    def __init__(
        self,
        g: DistributionParameter,
        b: DistributionParameter,
    ) -> None:
        """Initialize the MBBEFD distribution.

        Args:
            g: Reciprocal of the total-loss probability.
            b: Shape parameter.
        """
        super().__init__(g=g, b=b)

    @classmethod
    def from_c(cls, c: DistributionParameter) -> MBBEFD:
        r"""Construct the one-parameter Swiss Re curve associated with :math:`c`.

        The conversion is :math:`g=\exp((0.78+0.12c)c)` and
        :math:`b=\exp(3.1-0.15(1+c)c)`.

        Args:
            c: Swiss Re curve parameter.

        Returns:
            MBBEFD distribution with the corresponding ``g`` and ``b`` values.
        """
        g = t.cast(DistributionParameter, np.exp((0.78 + 0.12 * c) * c))
        b = t.cast(DistributionParameter, np.exp(3.1 - 0.15 * (1 + c) * c))
        return cls(g=g, b=b)

    def _validated_params(self) -> tuple[t.Any, t.Any]:
        """Return parameters after checking the admissible region."""
        g, b = self._param_values
        g = xp.asarray(g)
        b = xp.asarray(b)
        if bool(xp.any(~xp.isfinite(g))) or bool(xp.any(g < 1)):
            raise ValueError("g must be finite and greater than or equal to 1.")
        if bool(xp.any(~xp.isfinite(b))) or bool(xp.any(b < 0)):
            raise ValueError("b must be finite and greater than or equal to 0.")
        return g, b

    def _wrap_result(
        self,
        result: t.Any,
        argument: DistributionParameter,
    ) -> ReturnType:
        """Preserve stochastic coupling for array-valued results."""
        candidates = (argument, *self._params.values())
        stochastic_inputs = [value for value in candidates if isinstance(value, StochasticScalar)]
        if not stochastic_inputs:
            return float(result)

        wrapped = StochasticScalar(result)
        for value in stochastic_inputs:
            wrapped.coupled_variable_group.merge(value.coupled_variable_group)
        return wrapped

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute the cumulative distribution function, including the atom at one."""
        g, b = self._validated_params()
        values = xp.asarray(x.values if isinstance(x, StochasticScalar) else x)
        degenerate = (g == 1) | (b == 0)
        b_one = xp.isclose(b, 1, rtol=TOLERANCE, atol=TOLERANCE)
        bg_one = xp.isclose(b * g, 1, rtol=TOLERANCE, atol=TOLERANCE)

        safe_g = xp.where(degenerate, 2.0, g)
        safe_b = xp.where(degenerate | b_one | bg_one, 2.0, b)
        limit_b = xp.where(degenerate | b_one, 0.5, b)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            main = 1 - (1 - safe_b) * safe_b**values / ((safe_g - 1) * safe_b + (1 - safe_g * safe_b) * safe_b**values)
            b_limit = 1 - 1 / (1 + (g - 1) * values)
            bg_limit = 1 - limit_b**values

        interior = xp.where(
            degenerate,
            0.0,
            xp.where(b_one, b_limit, xp.where(bg_one, bg_limit, main)),
        )
        result = xp.where(values <= 0, 0.0, xp.where(values >= 1, 1.0, interior))
        return self._wrap_result(result, x)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute the inverse CDF, returning one across the total-loss atom."""
        g, b = self._validated_params()
        probabilities = xp.asarray(u.values if isinstance(u, StochasticScalar) else u)
        degenerate = (g == 1) | (b == 0)
        b_one = xp.isclose(b, 1, rtol=TOLERANCE, atol=TOLERANCE)
        bg_one = xp.isclose(b * g, 1, rtol=TOLERANCE, atol=TOLERANCE)

        safe_g = xp.where(degenerate, 2.0, g)
        safe_b = xp.where(degenerate | b_one | bg_one, 2.0, b)
        limit_b = xp.where(degenerate | b_one, 0.5, b)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            main_argument = (safe_g * safe_b - 1) / (safe_g - 1) + ((1 - safe_b) / ((1 - probabilities) * (safe_g - 1)))
            main = 1 - xp.log(main_argument) / xp.log(safe_b)
            b_limit = probabilities / ((1 - probabilities) * (safe_g - 1))
            bg_limit = xp.log1p(-probabilities) / xp.log(limit_b)

        below_atom = xp.where(
            degenerate,
            1.0,
            xp.where(b_one, b_limit, xp.where(bg_one, bg_limit, main)),
        )
        result = xp.where(
            (probabilities < 0) | (probabilities > 1),
            xp.nan,
            xp.where(
                degenerate | (probabilities >= 1 - 1 / g),
                1.0,
                xp.where(probabilities == 0, 0.0, below_atom),
            ),
        )
        return self._wrap_result(result, u)

    def exposure_curve(self, x: DistributionParameter) -> ReturnType:
        r"""Compute the normalized limited expected value curve.

        The exposure curve is :math:`G(x)=E[\min(X,x)]/E[X]`.

        Args:
            x: Policy limit as a proportion of the maximum loss.

        Returns:
            Proportion of expected loss below the policy limit.
        """
        g, b = self._validated_params()
        values = xp.asarray(x.values if isinstance(x, StochasticScalar) else x)
        degenerate = (g == 1) | (b == 0)
        b_one = xp.isclose(b, 1, rtol=TOLERANCE, atol=TOLERANCE)
        bg_one = xp.isclose(b * g, 1, rtol=TOLERANCE, atol=TOLERANCE)

        safe_g = xp.where(degenerate, 2.0, g)
        safe_b = xp.where(degenerate | b_one | bg_one, 2.0, b)
        limit_b = xp.where(degenerate | b_one, 0.5, b)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            numerator = (safe_g - 1) * safe_b + (1 - safe_g * safe_b) * safe_b**values
            main = xp.log(numerator / (1 - safe_b)) / xp.log(safe_g * safe_b)
            b_limit = xp.log1p((safe_g - 1) * values) / xp.log(safe_g)
            bg_limit = (1 - limit_b**values) / (1 - limit_b)

        interior = xp.where(
            degenerate,
            values,
            xp.where(b_one, b_limit, xp.where(bg_one, bg_limit, main)),
        )
        result = xp.where(values <= 0, 0.0, xp.where(values >= 1, 1.0, interior))
        return self._wrap_result(result, x)


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
        return _special_call(special.ndtr, arg)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        mu, sigma = self._param_values
        return _special_call(special.ndtri, u) * sigma + mu


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
        result = _special_call(special.ndtr, (np.log(x) - mu) / sigma)
        return result

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        mu, sigma = self._param_values
        return np.exp(_special_call(special.ndtri, u) * sigma + mu)


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
        return _special_call(special.gammainc, alpha, (x - loc) / theta)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, theta, loc = self._param_values
        result = _special_call(special.gammaincinv, alpha, u) * theta + loc
        return result

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        alpha, theta, loc = self._param_values
        result = StochasticScalar(
            rng.gamma(_rng_value(alpha, rng), _rng_value(theta, rng), size=n_sims) + _rng_value(loc, rng)
        )
        return result


class NonCentralChiSquared(DistributionBase):
    r"""Noncentral chi-squared distribution.

    The noncentral chi-squared distribution with degrees of freedom
    :math:`\nu>0` and noncentrality parameter :math:`\lambda\geq0` has density

    .. math::

        f(x) = \frac{1}{2}\exp\left(-\frac{x+\lambda}{2}\right)
        \left(\frac{x}{\lambda}\right)^{\nu/4-1/2}
        I_{\nu/2-1}\left(\sqrt{\lambda x}\right),
        \qquad x>0,

    for :math:`\lambda>0`, where :math:`I_a` is the modified Bessel function
    of the first kind. Equivalently, its CDF can be written as the Poisson
    mixture

    .. math::

        F(x) = \exp\left(-\frac{\lambda}{2}\right)
        \sum_{j=0}^{\infty}
        \frac{(\lambda/2)^j}{j!}
        P\left(\frac{\nu}{2}+j,\frac{x}{2}\right),

    where :math:`P(a,z)` is the regularized lower incomplete gamma function.
    When :math:`\lambda=0`, this reduces to the central chi-squared
    distribution. Its first two moments are

    .. math::

        E[X]=\nu+\lambda,
        \qquad
        \operatorname{Var}(X)=2(\nu+2\lambda).

    Parameters:
        df: Degrees of freedom :math:`\nu>0`.
        nonc: Noncentrality parameter :math:`\lambda\geq0`.

    Notes:
        Random generation supports both CPU and GPU execution. CDF and inverse
        CDF evaluation are currently supported on the CPU only.
    """

    def __init__(
        self,
        df: DistributionParameter,
        nonc: DistributionParameter,
    ) -> None:
        """Initialize a noncentral chi-squared distribution.

        Args:
            df: Positive degrees of freedom.
            nonc: Non-negative noncentrality parameter.
        """
        super().__init__(df=df, nonc=nonc)

    def _validated_params(self) -> tuple[t.Any, t.Any]:
        """Return parameters after checking the admissible region."""
        df, nonc = self._param_values
        if bool(xp.any(xp.asarray(df) <= 0)):
            raise ValueError("df must be strictly positive.")
        if bool(xp.any(xp.asarray(nonc) < 0)):
            raise ValueError("nonc must be greater than or equal to 0.")
        return df, nonc

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute the cumulative distribution function."""
        if xp.__name__ == "cupy":
            raise NotImplementedError("NonCentralChiSquared CDF is not supported on GPU.")
        df, nonc = self._validated_params()
        return _special_call(special.chndtr, x, df, nonc)  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute the inverse cumulative distribution function."""
        if xp.__name__ == "cupy":
            raise NotImplementedError("NonCentralChiSquared inverse CDF is not supported on GPU.")
        df, nonc = self._validated_params()
        return _special_call(special.chndtrix, u, df, nonc)  # type: ignore[return-value]

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        """Generate random samples on the backend used by the random generator."""
        df, nonc = self._validated_params()
        result = rng.noncentral_chisquare(_rng_value(df, rng), _rng_value(nonc, rng), size=n_sims)
        return StochasticScalar(result)


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
        return _special_call(special.gammaincc, alpha, np.divide(theta, (x - loc)))  # type: ignore[return-value]

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        alpha, theta, loc = self._param_values
        return np.divide(theta, _special_call(special.gammainccinv, alpha, u)) + loc


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
        p = _special_call(special.betainc, nu / 2, 0.5, nu / (nu + x_pos**2)) / 2
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
        x_beta = _special_call(special.betaincinv, nu / 2, 0.5, y)
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
        term1 = _special_call(special.ndtr, sqrt_lambda_x * (x / mu - 1))
        term2 = np.exp(2 * lambda_ / mu) * _special_call(special.ndtr, -sqrt_lambda_x * (x / mu + 1))
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
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        """Generate samples using the algorithm from Michael, Schucany, and Haas.

        Reference:
            Michael, J. R., Schucany, W. R. and Haas, R. W. (1976).
            Generating random variates using transformations with multiple roots.
            The American Statistician 30, 88-90.
        """
        mu, lambda_ = self._param_values
        # Generate chi-squared(1) samples
        nu = xp.asarray(rng.normal(0, 1, n_sims) ** 2)
        y = mu + (mu**2 * nu) / (2 * lambda_) - (mu / (2 * lambda_)) * np.sqrt(4 * mu * lambda_ * nu + mu**2 * nu**2)

        # Random selection step
        u = xp.asarray(rng.uniform(0, 1, n_sims))
        x = np.where(u <= mu / (mu + y), y, mu**2 / y)

        return StochasticScalar(x)


class GeneralizedInverseGaussian(DistributionBase):
    r"""Generalized inverse Gaussian distribution.

    Let :math:`Y=X-\mu`, where :math:`\mu` is the location parameter. The
    probability density function is

    .. math::

        f_X(x) =
        \begin{cases}
        \dfrac{(\psi / \chi)^{p / 2}}
        {2K_p(\sqrt{\chi\psi})}
        y^{p-1}
        \exp\left[-\dfrac{1}{2}\left(
        \dfrac{\chi}{y}+\psi y
        \right)\right], & y>0, \\
        0, & y\leq0,
        \end{cases}

    where :math:`p\in\mathbb{R}`, :math:`\chi>0`, :math:`\psi>0`, and
    :math:`K_\nu` is the modified Bessel function of the second kind. All
    power moments of :math:`Y` exist and satisfy

    .. math::

        E[Y^r] =
        \left(\frac{\chi}{\psi}\right)^{r/2}
        \frac{K_{p+r}(\sqrt{\chi\psi})}
        {K_p(\sqrt{\chi\psi})}.

    In particular,

    .. math::

        E[X] = \mu +
        \sqrt{\frac{\chi}{\psi}}
        \frac{K_{p+1}(\sqrt{\chi\psi})}
        {K_p(\sqrt{\chi\psi})},

    and

    .. math::

        \operatorname{Var}(X) = \frac{\chi}{\psi}
        \left[
        \frac{K_{p+2}(\sqrt{\chi\psi})}
        {K_p(\sqrt{\chi\psi})}
        -
        \left(
        \frac{K_{p+1}(\sqrt{\chi\psi})}
        {K_p(\sqrt{\chi\psi})}
        \right)^2
        \right].

    This parameterisation includes the inverse Gaussian distribution with
    mean :math:`m` and shape :math:`\lambda_{IG}` when

    .. math::

        p=-\frac{1}{2},
        \qquad \chi=\lambda_{IG},
        \qquad \psi=\frac{\lambda_{IG}}{m^2}.

    Parameters:
        p: Index parameter :math:`p`.
        chi: Reciprocal scale parameter :math:`\chi`.
        psi: Scale parameter :math:`\psi`.
        loc: Location parameter :math:`\mu` (default 0).

    Notes:
        CDF evaluation, inverse CDF evaluation, and random generation are
        currently supported on the CPU only.
    """

    def __init__(
        self,
        p: DistributionParameter,
        chi: DistributionParameter,
        psi: DistributionParameter,
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize generalized inverse Gaussian distribution.

        Args:
            p: Index parameter.
            chi: Positive reciprocal scale parameter.
            psi: Positive scale parameter.
            loc: Location parameter.
        """
        super().__init__(p=p, chi=chi, psi=psi, loc=loc)

    def _scipy_params(self) -> tuple[t.Any, t.Any, t.Any, t.Any]:
        """Convert the GIG parameters to SciPy's parameterisation."""
        if xp.__name__ == "cupy":
            raise NotImplementedError("GeneralizedInverseGaussian is not supported on GPU.")

        p, chi, psi, loc = self._param_values
        if bool(np.any(chi <= 0)):
            raise ValueError("chi must be strictly positive.")
        if bool(np.any(psi <= 0)):
            raise ValueError("psi must be strictly positive.")
        scipy_shape = np.sqrt(chi * psi)
        scipy_scale = np.sqrt(chi / psi)
        return p, scipy_shape, scipy_scale, loc

    def _wrap_result(
        self,
        result: t.Any,
        argument: DistributionParameter,
    ) -> ReturnType:
        """Preserve stochastic coupling when SciPy returns an array."""
        candidates = (argument, *self._params.values())
        stochastic_inputs = [value for value in candidates if isinstance(value, StochasticScalar)]
        if not stochastic_inputs:
            return float(result)

        wrapped = StochasticScalar(result)
        for value in stochastic_inputs:
            wrapped.coupled_variable_group.merge(value.coupled_variable_group)
        return wrapped

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute cumulative distribution function."""
        p, scipy_shape, scipy_scale, loc = self._scipy_params()
        values = x.values if isinstance(x, StochasticScalar) else x
        result = geninvgauss.cdf(
            values,
            p,
            scipy_shape,
            loc=loc,
            scale=scipy_scale,
        )
        return self._wrap_result(result, x)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute inverse cumulative distribution function."""
        p, scipy_shape, scipy_scale, loc = self._scipy_params()
        values = u.values if isinstance(u, StochasticScalar) else u
        result = geninvgauss.ppf(
            values,
            p,
            scipy_shape,
            loc=loc,
            scale=scipy_scale,
        )
        return self._wrap_result(result, u)

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        """Generate random samples using SciPy's GIG sampler."""
        p, scipy_shape, scipy_scale, loc = self._scipy_params()
        result = geninvgauss.rvs(
            p,
            scipy_shape,
            loc=loc,
            scale=scipy_scale,
            size=n_sims,
            random_state=rng,
        )
        return StochasticScalar(result)


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

    .. math::

        F(x) = \frac{x-a}{b-a}, \qquad a \leq x \leq b

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
    "generalizedinversegaussian": GeneralizedInverseGaussian,
    "inversegaussian": InverseGaussian,
    "logistic": Logistic,
    "lognormal": LogNormal,
    "loglogistic": LogLogistic,
    "mbbefd": MBBEFD,
    "noncentralchisquared": NonCentralChiSquared,
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
        rng: RandomGenerator | None = None,
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
