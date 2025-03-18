"""The Distributions module contains a number of classes for simulating from statistical distributions.
The distributions mainly follow the convention of Klugman"""

from .config import config, xp as np, _use_gpu
from .stochastic_scalar import StochasticScalar

if _use_gpu:
    import cupyx.scipy.special as special
else:
    import scipy.special as special

from abc import ABC, abstractmethod
from typing import TypeVar, Union, Dict, Any

NumberType = Union[float, int]
NumberOrStochasticScalar = TypeVar(
    "NumberOrStochasticScalar", NumberType, StochasticScalar
)


class DistributionBase(ABC):
    """An abstract base class for statistical distributions"""

    @property
    def _param_values(self):
        for param in self._params.values():
            if isinstance(param, StochasticScalar):
                yield param.values
            else:
                yield param

    def __init__(self, **params: Any):
        # Store all parameters in a private dictionary.
        self._params: Dict[str, Any] = params

    @abstractmethod
    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        pass

    @abstractmethod
    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        pass

    def generate(
        self, n_sims=None, rng: np.random.Generator = config.rng
    ) -> StochasticScalar:
        """
        Generate random samples from the distribution.

        Parameters:
            n_sims (int, optional): Number of simulations to generate. If not provided, the default value from the config will be used.
            rng (np.random.Generator, optional): Random number generator. Defaults to the value from the config.

        Returns:
            StochasticScalar: Array of random samples generated from the distribution.
        """
        if n_sims is None:
            n_sims = config.n_sims

        result = self._generate(n_sims, rng)
        # Merge the coupled variable groups
        for param in self._params.values():
            if isinstance(param, StochasticScalar):
                result.coupled_variable_group.merge(param.coupled_variable_group)

        return result

    def _generate(self, n_sims, rng: np.random.Generator):
        return StochasticScalar(self.invcdf(rng.uniform(size=n_sims)))


class DiscreteDistributionBase(DistributionBase, ABC):
    """An abstract base class for discrete distributions"""

    def __init__(self, **params: Any):
        # Store all parameters in a private dictionary.
        self._params: Dict[str, Any] = params

    @abstractmethod
    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        pass

    @abstractmethod
    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        pass


class Poisson(DiscreteDistributionBase):
    """Poisson Distribution

    The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space.

    Parameters:
    - mean (float): The mean or average number of events in the interval.

    Methods:
    - cdf(x): Calculates the cumulative distribution function of the Poisson distribution.
    - invcdf(u): Calculates the inverse cumulative distribution function of the Poisson distribution.
    - generate(n_sims=None, rng=config.rng): Generates random samples from the Poisson distribution.

    """

    def __init__(self, mean):
        super().__init__(mean=mean)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function (CDF) of the Poisson distribution.

        Parameters:
        - x: The value at which to evaluate the CDF.

        Returns:
        The probability that a random variable from the Poisson distribution is less than or equal to x.
        """
        return special.pdtr(x, self._params["mean"])

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the inverse cumulative distribution function of the Poisson distribution.

        Parameters:
            u (float or ndarray): The probability value(s) for which to calculate the inverse cumulative distribution.

        Returns:
            float or ndarray: The value(s) corresponding to the given probability value(s) in the Poisson distribution.
        """
        return special.pdtrik(u, self._params["mean"])

    def _generate(self, n_sims, rng: np.random.Generator) -> StochasticScalar:
        (mean,) = self._param_values
        return StochasticScalar(rng.poisson(mean, n_sims))


class NegBinomial(DiscreteDistributionBase):
    """NegBinomial Distribution

    This class represents the Negative Binomial distribution.

    Parameters:
    - n (float): The number of failures until the experiment is stopped.
    - p (float): The probability of success in each trial.

    Note that the parameter n, although having an interpretation as an integer can actually be a float

    Methods:
    - cdf(x): Calculates the cumulative distribution function of the Negative Binomial distribution.
    - invcdf(u): Calculates the inverse cumulative distribution function of the Negative Binomial distribution.
    - generate(n_sims=None, rng=config.rng): Generates random samples from the Negative Binomial distribution.
    """

    def __init__(self, n: float | StochasticScalar, p: float | StochasticScalar):
        """
        Create a new NegBinomial distribution with set parameters.

        Args:
            n (float): The number of failures until the experiment is stopped.
            p (float): The probability of success in each trial.

        Returns:
            None
        """
        super().__init__(n=n, p=p)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function of the Negative Binomial distribution.

        Parameters:
            x (float): The value at which to evaluate the cumulative distribution function.

        Returns:
            NumberOrStochasticScalar: The cumulative distribution function value at the given value.
        """
        n, p = self._param_values
        return special.nbdtr(x, n, p)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the inverse cumulative distribution function of the Negative Binomial distribution.

        Parameters:
            u (float or ndarray): The probability value(s) for which to calculate the inverse cumulative distribution.


        Returns:
            StochasticScalar: The inverse cumulative distribution function values corresponding to the given probability values.
        """
        n, p = self._param_values
        return special.nbdtri(u, n, p)

    def _generate(self, n_sims, rng: np.random.Generator) -> StochasticScalar:
        """
        Generates random samples from the Negative Binomial distribution.

        Parameters:
            n_sims (int, optional): Number of simulations to generate. If not provided, it uses the default value from the config.
            rng (numpy.random.Generator, optional): Random number generator. If not provided, it uses the default generator from the config.

        Returns:
            numpy.ndarray: Array of random samples from the Negative Binomial distribution.
        """
        n, p = self._param_values
        return StochasticScalar(rng.negative_binomial(n, p, n_sims))


class Binomial(DiscreteDistributionBase):
    """Binomial Distribution

    This class represents the Binomial distribution.

    Parameters:
    - n (int): The number of trials.
    - p (float): The probability of success in each trial.

    Methods:
    - cdf(x): Calculates the cumulative distribution function of the Binomial distribution.
    - invcdf(u): Calculates the inverse cumulative distribution function of the Binomial distribution.
    - generate(n_sims=None, rng=config.rng): Generates random samples from the Binomial distribution.
    """

    def __init__(self, n: int, p: float):
        """
        Create a new Binomial distribution with set parameters.

        Args:
            n (int): The number of trials.
            p (float): The probability of success in each trial.

        Returns:
            None
        """
        super().__init__(n=n, p=p)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function of the Binomial distribution.

        Parameters:
            x (float): The value at which to evaluate the cumulative distribution function.

        Returns:
            StochasticScalar: The cumulative distribution function value at the given value.
        """
        n, p = self._param_values
        return special.bdtr(x, n, p)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the inverse cumulative distribution function of the Binomial distribution.

        Parameters:
            u (float or ndarray): The probability value(s) for which to calculate the inverse cumulative distribution.


        Returns:
            StochasticScalar: The inverse cumulative distribution function values corresponding to the given probability values.
        """
        n, p = self._param_values
        return special.bdtri(u, n, p)

    def _generate(self, n_sims, rng: np.random.Generator) -> StochasticScalar:
        """
        Generates random samples from the Binomial distribution.

        Parameters:
            n_sims (int, optional): Number of simulations to generate. If not provided, it uses the default value from the config.
            rng (numpy.random.Generator, optional): Random number generator. If not provided, it uses the default generator from the config.

        Returns:
            numpy.ndarray: Array of random samples from the Binomial distribution.
        """
        n, p = self._param_values
        return StochasticScalar(rng.binomial(n, p, n_sims))


class HyperGeometric(DiscreteDistributionBase):
    """HyperGeometric Distribution

    This class represents the Hyper Geometric distribution. The hyper geometric distribution models the number of trials that must be run in order to achieve success.

    Parameters:
        n_good (int):
        n_bad (int):
        population_size (int):

    Methods:
    - cdf(x): Calculates the cumulative distribution function of the Hyper Geometric distribution.
    - invcdf(u): Calculates the inverse cumulative distribution function of the Hyper Geometric distribution.
    - generate(n_sims=None, rng=config.rng): Generates random samples from the Hyper Geometric distribution.
    """

    def __init__(self, ngood: int, nbad: int, population_size: int):
        """
        Create a new Hyper Geometric distribution with set parameters.

        Args:
            n_good (int):
            n_bad (int):
            population_size (int):

        Returns:
            None
        """
        super().__init__(ngood=ngood, nbad=nbad, n=population_size)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function of the Hyper Geometric distribution.

        Parameters:
            x (float): The value at which to evaluate the cumulative distribution function.

        Returns:
            StochasticScalar: The cumulative distribution function value at the given value.
        """
        return NotImplemented

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the inverse cumulative distribution function of the Hyper Geometric distribution.

        Parameters:
            u (float or ndarray): The probability value(s) for which to calculate the inverse cumulative distribution.


        Returns:
            StochasticScalar: The inverse cumulative distribution function values corresponding to the given
                probability values.
        """
        return NotImplemented

    def _generate(self, n_sims, rng: np.random.Generator) -> StochasticScalar:
        """
        Generates random samples from the hyper geometric distribution.

        Parameters:
            n_sims (int, optional): Number of simulations to generate. If not provided, it uses the default value from
                the config.
            rng (numpy.random.Generator, optional): Random number generator. If not provided, it uses the default
                generator from the config.

        Returns:
            numpy.ndarray: Array of random samples from the HyperGeometric distribution.
        """
        ngood, nbad, n = self._param_values
        return StochasticScalar(rng.hypergeometric(ngood, nbad, n, n_sims))


class GPD(DistributionBase):
    r"""The Generalised Pareto distribution is defined as through the cumulative distribution function:

        .. math::

            F(x) = 1 - (1+\frac{\xi(x-\mu)}{\sigma})^{-1/\xi}, \xi!=0 \\
            F(x) = 1 - e^{-(x-\mu)/\sigma}, \xi=0 \\
            
        where :math:`\xi` is the shape parameter, :math:`\sigma` is the scale parameter and :math:`\mu` is the location (or threshold) parameter. 

    """

    def __init__(
        self,
        shape: NumberOrStochasticScalar,
        scale: NumberOrStochasticScalar,
        loc: NumberOrStochasticScalar,
    ):
        """Initializes a new instance of the Generalised Pareto distribution with the specified scale, shape and location"""
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        r"""Calculates the cdf of the Generalized Pareto distribution.

        The cdf Generalised Pareto distribution is defined as

        .. math::

            F(x) = 1 - (1+\frac{\xi(x-\mu)}{\sigma})^{-1/\xi}, \xi!=0 \\
            F(x) = 1 - e^{-(x-\mu)/\sigma}, \xi=0 \\

        """
        shape, scale, loc = self._params.values()
        result = (
            1 - (1 + shape * (x - loc) / scale) ** (-1 / shape)
            if shape != 0
            else 1 - np.exp(-(x - loc) / scale)
        )
        return result

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cdf of the Generalized Pareto distribution"""
        shape, scale, loc = self._params.values()
        return (np.exp(np.log(1 - u) * (-shape)) - 1) * (scale / shape) + loc


class Burr(DistributionBase):
    r"""The Burr Distribution is defined through the cumulative distribution function:
    
    .. math::

        F(x) =    1 - \left[1 + \left(\frac{(x - \mu)}{\sigma}\right) ^k\right] ^
            {-c}, x>\mu \\

    where :math:`\mu` is the location parameter, :math:`\sigma` is the scale parameter, :math:`c` is the power parameter and :math:`k` is the shape parameter.
    
    """

    def __init__(
        self,
        power: NumberOrStochasticScalar,
        shape: NumberOrStochasticScalar,
        scale: NumberOrStochasticScalar,
        loc: NumberOrStochasticScalar,
    ):
        """
        Creates a new Burr distribution.

        Args:
            power (StochasticScalar|float): The power parameter.
            shape (StochasticScalar|float): The shape parameter.
            scale (StochasticScalar|float): The scale parameter.
            loc (StochasticScalar|float): The location parameter.
        """
        super().__init__(power=power, shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function (CDF) of the Burr distribution.

        Parameters:
            x: The value at which to evaluate the CDF.

        Returns:
            u: The CDF value at the given x.
        """
        power, shape, scale, loc = self._params.values()
        result = 1 - (1 + ((x - loc) / scale) ** power) ** (-shape)
        return result

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the inverse cumulative distribution function (CDF) of the Burr distribution.

        Parameters:
            u (float or np.ndarray): The probability value(s) for which to calculate the inverse CDF.

        Returns:
            float or np.ndarray: The corresponding quantile(s) for the given probability value(s).
        """
        power, shape, scale, loc = self._params.values()
        return scale * (((1 / (1 - u)) ** (1 / shape) - 1) ** (1 / power)) + loc


class Beta(DistributionBase):
    r"""Beta distribution
    
    The Beta Distribution is defined through the cumulative distribution function:
    
    .. math::

        F(x) =    \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \int_0^x u^{\alpha-1}(1-u)^{\beta-1} \\
        
    where :math:`u = \frac{x-\mu}{\sigma}`,:math:`\alpha` and :math:`\beta` are the shape parameters, :math:`\mu` is the location parameter, :math:`\sigma` is the scale parameter.

    Args:
            alpha (float): The alpha parameter.
            beta (float): The beta parameter.
            scale (float): The scale parameter.
            loc (float): The location parameter.
    
    """

    def __init__(self, alpha, beta, scale=1, loc=0):
        super().__init__(alpha=alpha, beta=beta, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function (CDF) of the Beta distribution.

        Parameters:
            x: The value at which to evaluate the CDF.

        Returns:
            u: The CDF value at the given x.
        """
        alpha, beta, scale, loc = self._params.values()
        return special.betainc(alpha, beta, (x - loc) / scale)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the inverse cumulative distribution function (CDF) of the Beta distribution.

        Parameters:
            u (float or np.ndarray): The probability value(s) for which to calculate the inverse CDF.

        Returns:
            float or np.ndarray: The corresponding quantile(s) for the given probability value(s).
        """
        alpha, beta, scale, loc = self._params.values()
        return special.betaincinv(alpha, beta, u) * scale + loc


class InverseBurr(DistributionBase):
    r"""Inverse Burr Distribution

    The Inverse Burr Distribution has cumulative distribution function:

    .. math::

        F(x) = l\eft(\frac{(\frac{x-\mu}{\sigma})^\tau}{1+(\frac{x-\mu}{\sigma})^\tau} \right)^\alpha

    where :math:`\mu` is the location parameter, :math:`\sigma` is the scale parameter, :math:`\tau` is the power parameter and :math:`\alpha` is the shape parameter.
    """

    def __init__(self, power, shape, scale, loc):
        """
        Creates a new Inverse Burr distribution.

        Args:
            power (float): The power parameter.
            shape (float): The shape parameter.
            scale (float): The scale parameter.
            loc (float): The location parameter.
        """
        super().__init__(power=power, shape=shape, scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function (CDF) of the Inverse Burr distribution.

        Parameters:
            x: The value at which to evaluate the CDF.

        Returns:
            u: The CDF value at the given x.
        """
        power, shape, scale, loc = self._params.values()

        y = ((x - loc) / scale) ** power

        return (y / (1 + y)) ** shape

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the inverse cumulative distribution function (CDF) of the Inverse Burr distribution.

        Parameters:
            u (float or np.ndarray): The probability value(s) for which to calculate the inverse CDF.

        Returns:
            float or np.ndarray: The corresponding quantile(s) for the given probability value(s).
        """
        power, shape, scale, loc = self._params.values()
        return (
            scale
            * (np.float_power((np.float_power(u, (-1 / shape)) - 1), (-1 / power)))
            + loc
        )


class LogLogistic(DistributionBase):
    r"""The Log Logistic Distribution is defined through the cumulative distribution function:

    .. math::

        F(x) =    1 - (1 + (\frac{(x - \mu)}{\sigma}) ^ {k}) ^ (
            -1
        ), x>\mu \\

    where :math:`\mu` is the location parameter, :math:`\sigma` is the scale parameter and :math:`k` is the shape parameter.
    
    """

    def __init__(self, shape, scale, loc=0):
        """
        Create a LogLogistic distribution.

        Args:
            shape (float): The shape parameter of the distribution.
            scale (float): The scale parameter of the distribution.
            loc (float): The location parameter of the distribution.
        """
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x):
        """
        Calculates the cumulative distribution function (CDF) of the Log Logistic distribution.

        Parameters:
            x (float): The input value.

        Returns:
            float: The CDF value at the given input.
        """
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** (shape)
        return y / (1 + y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the inverse cdf of the Log Logistic distribution.

        Parameters:
            u (float or StochasticScalar): The probability value(s) for which to calculate the inverse cdf.

        Returns:
            float or StochasticScalar: The corresponding inverse cdf value(s).
        """
        shape, scale, loc = self._params.values()
        return scale * ((u / (1 - u)) ** (1 / shape)) + loc


class Normal(DistributionBase):
    """Normal distribution"""

    def __init__(self, mu, sigma):
        """
        Create a Normal distribution.

        Parameters:
        - mu (float): The mean of the distribution.
        - sigma (float): The standard deviation of the distribution.
        """
        super().__init__(mu=mu, sigma=sigma)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function (CDF) of the Normal Distribution.

        Parameters:
        - x: The value at which to evaluate the CDF.

        Returns:
        The probability that a random variable from the Normal Distribution is less than or equal to x.
        """
        mu, sigma = self._param_values
        return special.ndtr((x - mu) / sigma)

    def invcdf(self, u) -> NumberOrStochasticScalar:
        """Calculates the inverse cdf of the Normal Distribution

        Parameters:
        u (float or ndarray): The probability value(s) for which to calculate the inverse cdf.

        Returns:
        float or ndarray: The corresponding value(s) from the inverse cdf of the Normal Distribution.
        """
        mu, sigma = self._param_values
        return special.ndtri(u) * sigma + mu


class Logistic(DistributionBase):
    """Logistic Distribution."""

    def __init__(self, mu: float, sigma: float):
        super().__init__(mu=mu, sigma=sigma)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        mu, sigma = self._param_values
        return 1 / (1 + np.exp(-(x - mu) / sigma))

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        mu, sigma = self._param_values
        return mu + sigma * np.log(u / (1 - u))


class LogNormal(DistributionBase):
    """Log Normal distribution

    Parameters:
        - mu (float): The mean of the logged distribution.
        - sigma (float): The standard deviation of the logged distribution.

    """

    def __init__(self, mu, sigma):
        super().__init__(mu=mu, sigma=sigma)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function (CDF) of the Log-Normal Distribution.

        Parameters:
        - x (float or ndarray): The value at which to evaluate the CDF.

        Returns:
        float or ndarray: The probability that a random variable from the Log-Normal Distribution is less than or equal to x.
        """
        mu, sigma = self._param_values
        return special.ndtr((np.log(x) - mu) / sigma)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cdf of the Log-Normal Distribution

        Parameters:
        u (float or ndarray): The probability value(s) for which to calculate the inverse cdf.

        Returns:
        float or ndarray: The corresponding value(s) from the inverse cdf of the Log-Normal Distribution.
        """
        mu, sigma = self._param_values
        return np.exp(special.ndtri(u) * sigma + mu)


class Gamma(DistributionBase):
    r"""Gamma distribution.

    The Gamma distribution has the following cumulative distribution function (CDF):

    .. math::

            F(x) = \frac{1}{\Gamma(\alpha)} \gamma(k, \frac{(\alpha-\mu)}{\theta}), x>\mu
        where :math:`\alpha` is the shape parameter, :math:`\theta` is the scale parameter, :math:`\mu` is the location parameter and :math:`\gamma(\alpha,z )` is the lower incomplete gamma function.

    Parameters:
        - alpha: The shape parameter :math:`\alpha`.
        - scale: The scale parameter :math:`\theta`.
        - loc: The location parameter :math:`\mu`.

    """

    def __init__(self, alpha, theta, loc=0):
        super().__init__(alpha=alpha, theta=theta, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function (CDF) of the Gamma Distribution.

        Parameters:
        - x (float or ndarray): The value at which to evaluate the CDF.

        Returns:
        float or ndarray: The probability that a random variable from the Gamma Distribution is less than or equal to x.
        """
        alpha, theta, loc = self._param_values
        return special.gammainc(alpha, (x - loc) / theta)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cdf of the Gamma Distribution

        Parameters:
        u (float or ndarray): The probability value(s) for which to calculate the inverse cdf.

        Returns:
        float or ndarray: The corresponding value(s) from the inverse cdf of the Gamma Distribution.
        """
        alpha, theta, loc = self._param_values
        return special.gammaincinv(alpha, u) * theta + loc

    def _generate(self, n_sims, rng=config.rng):
        alpha, theta, loc = self._param_values
        return StochasticScalar(rng.gamma(alpha, theta, size=n_sims) + loc)


class InverseGamma(DistributionBase):
    r"""Inverse Gamma distribution.

    The Inverse Gamma distribution has the following cumulative distribution function (CDF):

    .. math::

            F(x) = 1-\frac{1}{\Gamma(\alpha)} \gamma(\alpha, \frac{\theta}{(x-\mu)}), x>\mu
        where :math:`\alpha` is the shape parameter, :math:`\theta` is the scale parameter, :math:`\mu` is the location parameter and :math:`\gamma(\alpha,z )` is the lower incomplete gamma function.

    Parameters:
        - alpha: The shape parameter :math:`\alpha`.
        - scale: The scale parameter :math:`\theta`.
        - loc: The location parameter :math:`\mu`.

    """

    def __init__(self, alpha, theta, loc=0):
        super().__init__(alpha=alpha, theta=theta, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """
        Calculates the cumulative distribution function (CDF) of the Inverse Gamma Distribution.

        Parameters:
        - x (float or StochasticScalar): The value at which to evaluate the CDF.

        Returns:
        float or StochasticScalar: The probability that a random variable from the Inverse Gamma Distribution is less than or equal to x.
        """
        alpha, theta, loc = self._param_values
        return special.gammaincc(alpha, np.divide(theta, (x - loc)))

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cdf of the Inverse Gamma Distribution

        Parameters:
        u (float or ndarray): The probability value(s) for which to calculate the inverse cdf.

        Returns:
        float or ndarray: The corresponding value(s) from the inverse cdf of the Inverse Gamma Distribution.
        """
        alpha, theta, loc = self._param_values
        return np.divide(theta, special.gammainccinv(alpha, u)) + loc


class Pareto(DistributionBase):
    r"""Pareto Distribution

    Represents a Pareto distribution with given shape and scale parameters.

    The Pareto distribution is a power-law probability distribution that is frequently used to model the distribution of wealth, income, and other quantities. It is defined by the following probability density function (PDF):

    ..math ::

        f(x) = \frac{a * x_m^a}{x^{a+1}}

    where :math:`a` is the shape parameter and :math:`x_m` is the scale parameter.

    The cumulative distribution function (CDF) of the Pareto distribution is given by:

    ,,math ::
        F(x) = 1 - (\frac{x_m}{x})^a

    Args:
        shape (float): The shape parameter of the Pareto distribution.
        scale (float): The scale parameter of the Pareto distribution.
    """

    def __init__(self, shape, scale):
        super().__init__(shape=shape, scale=scale)

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) of the Pareto distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        shape, scale = self._params.values()
        return 1 - (x / scale) ** (-shape)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the Pareto distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            StochasticScalar or float: The inverse CDF value(s) at the given u.
        """
        shape, scale = self._params.values()
        return (1 - u) ** (-1 / shape) * scale


class Paralogistic(DistributionBase):
    r"""ParaLogistic Distribution

    Represents a ParaLogistic distribution with given shape and scale parameters.

    The cumulative distribution function (CDF) of the ParaLogistic distribution is given by:

    .. math::
        F(x) = 1 - \left[1+\left(\frac{x-\mu}{\sigma}\right)^\alpha\right]^{-\alpha}, x>\mu

    where the shape parameter :math:`\alpha` and the scale parameter :math:`\sigma` are both positive, and the location parameter :math:`\mu` is any real number.

    Args:
        shape (float): The shape parameter of the ParaLogistic distribution.
        scale (float): The scale parameter of the ParaLogistic distribution.
    """

    def __init__(self, shape, scale, loc=0):
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) of the ParaLogistic distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        shape, scale, loc = self._params.values()
        y = 1 / (1 + ((x - loc) / scale) ** (shape))
        return 1 - (y) ** shape

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the ParaLogistic distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            np.ndarray or float: The inverse CDF value(s) at the given u.
        """
        shape, scale, loc = self._params.values()
        return loc + scale * ((1 - u) ** (-1 / shape) - 1) ** (1 / shape)


class InverseParalogistic(DistributionBase):
    r"""Inverse ParaLogistic Distribution

    Represents an Inverse ParaLogistic distribution with given shape and scale parameters.

    The cumulative distribution function (CDF) of the ParaLogistic distribution is given by:

    .. math::
        F(x) = \left[\frac{\left(\frac{x-\mu}{\sigma}\right)^\alpha}
        {\left(1+\frac{x-\mu}{\sigma}\right)^\alpha}\right]^{-\alpha}, x>\mu

    where the shape parameter :math:`\alpha` and the scale parameter :math:`\sigma` are both positive, and the location
    parameter :math:`\mu` is any real number.

    Args:
        shape (float): The shape parameter of the Inverse Paralogistic distribution.
        scale (float): The scale parameter of the Inverse Paralogistic distribution.
    """

    def __init__(self, shape, scale, loc=0):
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) of the Inverse ParaLogistic distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** (shape)
        return (y / (1 + y)) ** shape

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the Inverse ParaLogistic
        distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            np.ndarray or float: The inverse CDF value(s) at the given u.
        """
        shape, scale, loc = self._params.values()
        y = (u) ** (1 / shape)
        return loc + scale * (y / (1 - y)) ** (1 / shape)


class Weibull(DistributionBase):
    r"""Weibull Distribution

    Represents a Weibull distribution with given shape and scale parameters.

    The cumulative distribution function (CDF) of the Weibull distribution is given by:

    .. math::
        F(x) = 1-e^{-((x-\mu)/\\sigma)^\alpha}, x>\mu

    where the shape parameter :math:`\alpha` and the scale parameter :math:`\sigma` are both positive, and the location
    parameter :math:`\mu` is any real number.

    Args:
        shape (float): The shape parameter of the Weibull distribution.
        scale (float): The scale parameter of the Weibull distribution.
        loc (float): The location parameter of the Weibull distribution.
    """

    def __init__(self, shape, scale, loc=0):
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) of the Weibull distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        shape, scale, loc = self._params.values()
        y = ((x - loc) / scale) ** (shape)
        return -np.expm1(-y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the Weibull distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            np.ndarray or float: The inverse CDF value(s) at the given u.
        """
        shape, scale, loc = self._params.values()
        return loc + scale * (-np.log(1 - u)) ** (1 / shape)


class InverseWeibull(DistributionBase):
    r"""Inverse Weibull Distribution

    Represents an Inverse Weibull distribution with given shape and scale parameters.

    The cumulative distribution function (CDF) of the Inverse Weibull distribution is given by:

    .. math::
        F(x) = e^{-((x-\mu)/\\sigma)^{-\alpha}}, x>\mu

    where the shape parameter :math:`\alpha` and the scale parameter :math:`\sigma` are both positive, and the location
    parameter :math:`\mu` is any real number.

    Args:
        shape (float): The shape parameter of the Inverse Weibull distribution.
        scale (float): The scale parameter of the Inverse Weibull distribution.
        loc (float): The location parameter of the Inverse Weibull distribution.
    """

    def __init__(self, shape, scale, loc=0):
        super().__init__(shape=shape, scale=scale, loc=loc)

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) of the Inverse Weibull distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        shape, scale, loc = self._params.values()
        y = np.float_power((x - loc) / scale, -shape)
        return np.exp(-y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the Weibull distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            np.ndarray or float: The inverse CDF value(s) at the given u.
        """
        shape, scale, loc = self._params.values()
        return loc + scale * (-1 / (np.log(u))) ** (1 / shape)


class Exponential(DistributionBase):
    r"""Exponential Distribution

    Represents a Exponential distribution with given shape parameters.

    The cumulative distribution function (CDF) of the Exponential distribution is given by:

    .. math::
        F(x) = 1-e^{-((x-\mu)/\\sigma)}, x>\mu

    where the scale parameter :math:`\sigma` is positive, and the location parameter :math:`\mu` is any real number.

    Args:
        scale (float): The scale parameter of the Exponential distribution.
        loc (float): The location parameter of the Exponential distribution.
    """

    def __init__(self, scale, loc=0):
        super().__init__(scale=scale, loc=loc)

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) of the Exponential distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        scale, loc = self._params.values()
        y = (x - loc) / scale
        return -np.expm1(-y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the Exponential distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            np.ndarray or float: The inverse CDF value(s) at the given u.
        """
        scale, loc = self._params.values()
        return loc + scale * (-np.log(1 - u))


class Uniform(DistributionBase):
    r"""Uniform Distribution

    Represents a Uniform distribution with given parameters.

    The cumulative distribution function (CDF) of the Uniform distribution is given by:

    .. math::
        F(x) = (x-a)/(b-a), a<=x<=b


    Args:
        a (float): The lower bound.
        b (float): The upper bound.
    """

    def __init__(self, a, b):
        super().__init__(a=a, b=b)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the cumulative distribution function (CDF) of the Uniform distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        a, b = self._params.values()
        y = (x - a) / (b - a)
        return y

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the Uniform distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
            np.ndarray or float: The inverse CDF value(s) at the given u.
        """
        a, b = self._params.values()
        return a + (b - a) * u


class InverseExponential(DistributionBase):
    r"""Inverse Exponential Distribution

    Represents an Inverse Exponential distribution with given shape and scale parameters.

    The cumulative distribution function (CDF) of the Inverse Exponential distribution is given by:

    .. math::
        F(x) = e^{-(\sigma/(x-\mu))}, x>\mu

    where the scale parameter :math:`\sigma` is positive, and the location parameter :math:`\mu` is any real number.

    Args:
        scale (float): The scale parameter of the Exponential distribution.
        loc (float): The location parameter of the Exponential distribution.
    """

    def __init__(self, scale, loc=0):
        super().__init__(scale=scale, loc=loc)

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the cumulative distribution function (CDF) of the Inverse Exponential distribution.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            float: The CDF value at the given x.
        """
        scale, loc = self._params.values()
        y = scale * np.float_power((x - loc), -1)
        return np.exp(-y)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        """Calculates the inverse cumulative distribution function (inverse CDF) of the Inverse Exponential distribution.

        Args:
            u (float): The probability value at which to evaluate the inverse CDF.

        Returns:
           StochasticScalar or float: The inverse CDF value(s) at the given u.
        """
        scale, loc = self._params.values()
        return loc - scale * 1 / (np.log(u))


AVAILABLE_DISCRETE_DISTRIBUTIONS = {
    "poisson": Poisson,
    "negbinomial": NegBinomial,
    "binomial": Binomial,
    "hypergeometric": HyperGeometric,
}


class DistributionGeneratorBase(ABC):
    """A base class for parameterised distribution generators."""

    this_distribution: DistributionBase

    def cdf(self, x: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        return self.this_distribution.cdf(x)

    def invcdf(self, u: NumberOrStochasticScalar) -> NumberOrStochasticScalar:
        return self.this_distribution.invcdf(u)

    def generate(self, n_sims=None, rng=config.rng) -> StochasticScalar:
        return self.this_distribution.generate(n_sims, rng)


class DiscreteDistribution(DistributionGeneratorBase):
    """A discrete distribution, created from a string name"""

    def __init__(self, distribution_name: str, parameters: list):
        if distribution_name.lower() in AVAILABLE_DISCRETE_DISTRIBUTIONS:
            cls: DistributionBase = AVAILABLE_DISCRETE_DISTRIBUTIONS[
                distribution_name.lower()
            ]
        else:
            raise (
                f"Distribution {distribution_name} must be one of {AVAILABLE_DISCRETE_DISTRIBUTIONS.keys()}"
            )

        this_distribution: DistributionBase = cls(*parameters)
        self.this_distribution = this_distribution


AVAILABLE_CONTINUOUS_DISTRIBUTIONS = {
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
    "uniform": Uniform,
    "weibull": Weibull,
}


class ContinuousDistribution(DistributionGeneratorBase):
    """A continuous distribution, created from a string name"""

    def __init__(self, distribution_name: str, parameters: list):
        if distribution_name.lower() in AVAILABLE_CONTINUOUS_DISTRIBUTIONS:
            cls: DistributionBase = AVAILABLE_CONTINUOUS_DISTRIBUTIONS[
                distribution_name.lower()
            ]
        else:
            raise (
                f"Distribution {distribution_name} must be one of {AVAILABLE_CONTINUOUS_DISTRIBUTIONS.keys()}"
            )

        this_distribution: DistributionBase = cls(*parameters)
        self.this_distribution = this_distribution
