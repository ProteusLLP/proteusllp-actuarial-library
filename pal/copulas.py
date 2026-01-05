"""Copula Module.

This module contains classes for representing and generating samples from various
copulas. It includes both elliptical (Gaussian and Student's T) and Archimedean
copulas.
"""

# Standard library imports
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

# Third-party imports
import numpy.typing as npt
import scipy.stats.distributions as distributions  # type: ignore [import-untyped]
from scipy.special import gamma
from scipy.stats import norm

from . import ProteusVariable, StochasticScalar
from ._maths import special
from ._maths import xp as np

# Local imports
from .config import config


class Copula(ABC):
    """Base class for copula implementations.

    A copula is a multivariate probability distribution that describes the
    dependence structure between random variables, separate from their individual
    marginal distributions. Copulas are used in risk modeling to simulate
    correlated stochastic variables.

    All copula implementations generate ProteusVariable containers with VectorLike
    values (typically StochasticScalar instances) that represent correlated
    uniform random samples on [0,1].
    """

    @abstractmethod
    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate correlated uniform samples from the copula.

        Args:
            n_sims: Number of simulations to generate. Uses config.n_sims if None.
            rng: Random number generator. Uses config.rng if None.

        Returns:
            ProteusVariable containing VectorLike values (typically StochasticScalar)
            with uniform marginal distributions on [0,1] and the copula's
            correlation structure.
        """
        pass

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the multivariate distribution underlying the copula.

        The marginal distribution of the samples will not necessarily be uniform.

        Args:
            n_sims: Number of simulations to generate.
            rng: Random number generator.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _transform_to_uniform(
        self, unnormalised_samples: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform unnormalised samples to uniform [0,1].

        Override this in subclasses that need custom transformations.
        Default implementation assumes samples are already uniform.

        Args:
            unnormalised_samples: Array of samples from the underlying distribution.

        Returns:
            Array of uniform samples on [0,1].
        """
        return unnormalised_samples

    def _create_result_from_uniform(
        self, uniform_samples: npt.NDArray[np.floating]
    ) -> ProteusVariable[StochasticScalar]:
        """Create ProteusVariable result from uniform samples with coupled groups.

        Args:
            uniform_samples: Array of uniform samples on [0,1].

        Returns:
            ProteusVariable with StochasticScalar values and merged coupling groups.
        """
        result = ProteusVariable[StochasticScalar](
            "dim1",
            {
                f"{type(self).__name__}_{i}": StochasticScalar(sample)
                for i, sample in enumerate(uniform_samples)
            },
        )
        # Merge all variables into the same coupled group
        first_scalar = result[0]
        for val in result:
            val.coupled_variable_group.merge(first_scalar.coupled_variable_group)
        return result

    def _generate_base(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Base implementation of generate with common boilerplate.

        Subclasses can call this from their generate() method to avoid repetition
        while maintaining individual docstrings.

        Args:
            n_sims: Number of simulations to generate. Uses config.n_sims if None.
            rng: Random number generator. Uses config.rng if None.

        Returns:
            ProteusVariable with StochasticScalar values representing copula samples.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if rng is None:
            rng = config.rng

        unnormalised = self._generate_unnormalised(n_sims, rng)
        uniform_samples = self._transform_to_uniform(unnormalised)
        return self._create_result_from_uniform(uniform_samples)

    def apply(
        self, variables: ProteusVariable[StochasticScalar] | list[StochasticScalar]
    ) -> None:
        """Apply the copula's correlation structure to existing variables.

        This method modifies the input variables in-place to exhibit the
        correlation structure defined by this copula while preserving their
        marginal distributions.

        Args:
            variables: Either a ProteusVariable containing VectorLike values or
                      a list of VectorLike instances. Only StochasticScalar
                      values are processed; other types are silently ignored
                      when passed in a ProteusVariable.

        Raises:
            TypeError: If list contains non-StochasticScalar values.
            ValueError: If variables have inconsistent simulation counts.
        """
        variables_list = list(variables)
        # Generate the copula samples
        # Check that n_sims is available
        n_sims = variables_list[0].n_sims
        copula_samples = [
            StochasticScalar(sample)
            for sample in self._generate_unnormalised(n_sims=n_sims, rng=config.rng)
        ]
        if len(variables) != len(copula_samples):
            raise ValueError("Number of variables and copula samples do not match.")
        # Apply the copula to the variables
        apply_copula(variables_list, copula_samples)


class EllipticalCopula(Copula, ABC):
    """A base class to represent an elliptical copula."""

    matrix: npt.NDArray[np.floating]
    chol: npt.NDArray[np.floating]

    def __init__(
        self,
        matrix: npt.NDArray[np.floating] | list[list[float]],
        *args: t.Any,
        matrix_type: str = "linear",
        **kwargs: t.Any,
    ) -> None:
        """Initialize an elliptical copula.

        Args:
            matrix: Correlation matrix or Cholesky decomposition.
            matrix_type: Type of matrix - "linear" or "chol".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _matrix = np.asarray(matrix)
        if _matrix.ndim != 2 or _matrix.shape[0] != _matrix.shape[1]:
            raise ValueError("Matrix must be square")
        if matrix_type == "linear":
            self.correlation_matrix = _matrix
            # Check that the correlation matrix is positive definite
            try:
                self.chol = np.linalg.cholesky(self.correlation_matrix)
            except np.linalg.LinAlgError as e:
                raise ValueError("Correlation matrix is not positive definite") from e
        elif matrix_type == "chol":
            self.chol = _matrix
        else:
            raise ValueError("matrix_type must be 'linear' or 'chol'")
        self.matrix = _matrix


class GaussianCopula(EllipticalCopula):
    """A class to represent a Gaussian copula."""

    def __init__(
        self,
        matrix: npt.NDArray[np.floating] | list[list[float]],
        matrix_type: str = "linear",
    ) -> None:
        """Initialize a Gaussian copula.

        Args:
            matrix: Correlation matrix.
            matrix_type: Type of matrix - "linear" or "chol".
        """
        super().__init__(matrix, matrix_type=matrix_type)

    def _transform_to_uniform(
        self, unnormalised_samples: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform normal samples to uniform using CDF."""
        return special.ndtr(unnormalised_samples)

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Gaussian copula."""
        return self._generate_base(n_sims, rng)

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        n_vars = self.correlation_matrix.shape[0]
        normal_samples = rng.standard_normal(size=(n_vars, n_sims))
        return self.chol.dot(normal_samples)


class StudentsTCopula(EllipticalCopula):
    """A class to represent a Student's T copula."""

    def __init__(
        self,
        matrix: npt.NDArray[np.float64] | list[list[float]],
        dof: float,
        matrix_type: str = "linear",
    ) -> None:
        """Initialize a Student's T copula.

        Args:
            matrix: Correlation matrix.
            dof: Degrees of freedom.
            matrix_type: Type of matrix - "linear" or "chol".
        """
        super().__init__(matrix, matrix_type=matrix_type)
        if dof <= 0:
            raise ValueError("Degrees of Freedom must be positive")
        self.dof = dof

    def _transform_to_uniform(
        self, unnormalised_samples: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform t-distributed samples to uniform using CDF."""
        return distributions.t(self.dof).cdf(unnormalised_samples)

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Student's T copula."""
        return self._generate_base(n_sims, rng)

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        n_vars = self.correlation_matrix.shape[0]
        normal_samples = self.chol.dot(rng.standard_normal(size=(n_vars, n_sims)))
        chi_samples = np.sqrt(rng.gamma(self.dof / 2, 2 / self.dof, size=n_sims))
        return normal_samples / chi_samples[np.newaxis, :]


class ArchimedeanCopula(Copula, ABC):
    """A base class to represent an Archimedean copula."""

    @abstractmethod
    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """The inverse generator function of the copula."""
        pass

    @abstractmethod
    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution of the copula."""
        pass

    def __init__(self, n: int) -> None:
        """Initialize an Archimedean copula.

        Args:
            n: Number of variables.
        """
        self.n = n

    def _transform_to_uniform(
        self, unnormalised_samples: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform using inverse generator function."""
        return self.generator_inv(-unnormalised_samples)

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Archimedean copula."""
        return self._generate_base(n_sims, rng)

    def _generate_unnormalised(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> npt.NDArray[np.floating]:
        if n_sims is None:
            n_sims = config.n_sims
        if rng is None:
            rng = config.rng
        n_vars = self.n
        # Generate samples from a uniform distribution
        u = rng.uniform(size=(n_vars, n_sims))
        # Generate samples from the latent distribution
        latent_samples = self.generate_latent_distribution(n_sims, rng)

        # Add shape validation
        if not (latent_samples.shape == (n_sims,)):
            raise AssertionError(
                f"Expected latent_samples shape ({n_sims},), got {latent_samples.shape}"
            )

        # Calculate the copula samples
        return np.log(u) / latent_samples[np.newaxis]


class ClaytonCopula(ArchimedeanCopula):
    """A class to represent a Clayton copula."""

    def __init__(self, theta: float, n: int) -> None:
        """Initialize a Clayton copula.

        Args:
            theta: Copula parameter (must be >= 0). When theta=0, represents
                   the independence copula.
            n: Number of variables.
        """
        if theta < 0:
            raise ValueError("Theta cannot be negative")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Inverse generator function for Clayton copula.

        Args:
            t: Input array values.

        Returns:
            Array of inverse generator values. When theta=0, returns exp(-t),
            corresponding to the independence copula.
        """
        if self.theta == 0:
            return np.exp(-t)
        return (1 + t) ** (-1 / self.theta)

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution.

        For Clayton copula, when theta=0, the copula reduces to the independence
        copula, and the latent distribution returns a constant value of 1.0 for
        all simulations.

        Returns:
            Array of shape (n_sims,) containing latent distribution samples.
        """
        if self.theta == 0:
            return np.ones(n_sims)
        return rng.gamma(1 / self.theta, size=n_sims)


def levy_stable(
    alpha: float,
    beta: float,
    size: int | tuple[int, ...],
    rng: np.random.Generator,
) -> npt.NDArray[np.floating]:
    """Simulate samples from a Lévy stable distribution using Chambers-Mallows-Stuck.

    Parameters:
        alpha (float): Stability parameter in (0, 2].
        beta (float): Skewness parameter in [-1, 1].
        size (int or tuple of ints): Output shape.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Samples from the Lévy stable distribution.
    """
    uniform_samples = rng.uniform(-np.pi / 2, np.pi / 2, size)
    exponential_samples = rng.exponential(1, size)

    if alpha != 1:
        theta = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
        factor = (1 + beta**2 * np.tan(np.pi * alpha / 2) ** 2) ** (1 / (2 * alpha))
        part1 = np.sin(alpha * (uniform_samples + theta)) / (
            np.cos(uniform_samples)
        ) ** (1 / alpha)
        part2 = (
            np.cos(uniform_samples - alpha * (uniform_samples + theta))
            / exponential_samples
        ) ** ((1 - alpha) / alpha)
        samples = factor * part1 * part2
    else:
        samples = (2 / np.pi) * (
            (np.pi / 2 + beta * uniform_samples) * np.tan(uniform_samples)
            - beta
            * np.log(
                (np.pi / 2 * exponential_samples * np.cos(uniform_samples))
                / (np.pi / 2 + beta * uniform_samples)
            )
        )
    return samples


class GumbelCopula(ArchimedeanCopula):
    """A class to represent a Gumbel copula."""

    def __init__(self, theta: float, n: int) -> None:
        """Initialize a Gumbel copula.

        Args:
            theta: Copula parameter (must be >= 1).
            n: Number of variables.
        """
        if theta < 1:
            raise ValueError("Theta must be at least 1")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Inverse generator function for Gumbel copula."""
        return np.exp(-(t ** (1 / self.theta)))

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution."""
        return levy_stable(1 / self.theta, 1, n_sims, rng) * (
            np.cos(np.pi / (2 * self.theta)) ** self.theta
        )


class FrankCopula(ArchimedeanCopula):
    """A class to represent a Frank copula."""

    def __init__(self, theta: float, n: int) -> None:
        """Initialize a Frank copula.

        Args:
            theta: Copula parameter.
            n: Number of variables.
        """
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Inverse generator function for Frank copula."""
        return -np.log1p(np.exp(-t) * (np.expm1(-self.theta))) / self.theta

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution."""
        return rng.logseries(1 - np.exp(-self.theta), size=n_sims).astype(np.float64)


class JoeCopula(ArchimedeanCopula):
    """A class to represent a Joe copula."""

    def __init__(self, theta: float, n: int) -> None:
        """Initialize a Joe copula.

        Args:
            theta: Copula parameter (must be >= 1).
            n: Number of variables.
        """
        if theta < 1:
            raise ValueError("Theta must be in the range [1, inf)")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Inverse generator function for Joe copula."""
        return 1 - (1 - np.exp(-t)) ** (1 / self.theta)

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution."""
        return _sibuya_gen(1 / self.theta, n_sims, rng)


class MM1Copula(Copula):
    """A multivariate max-mixture copula, denoted MM1 by Joe.

    The MM1 copula is a multivariate copula which allows for different upper tail
    dependence structures between each pair of dimensions. It can be regarded as an
    extension of the Gumbel copula to more flexible dependence.

    The upper tail dependence coefficient between any pair of variables i and j in the
    MM1 copula is given by

    2-(((2 ^ (1 / delta_ij)) / (d - 1) + 2 * (d - 2) / (d - 1)) ^ (1 / theta))

    where delta_ij is the pairwise parameter from the delta_matrix, d is the
    dimension of the copula, and theta is the overall mixing parameter.

    The simulation approach uses the max-mixture representation of the MM1 copula,
    detailed in Joe (2015, Chapter 6).

    References:
        Joe, H. (1997). Multivariate Models and Dependence Concepts. Chapman and Hall.
        Joe, H. (2015). Dependence Modeling with Copulas. Chapman and Hall.
    """

    delta_matrix: list[list[float]]
    """The matrix of pairwise parameters of the underlying Gumbel copulas"""
    theta: float
    """The theta parameter of the overall mixing variable"""

    def __init__(self, delta_matrix: list[list[float]], theta: float):
        """Initialise the MM1 Copula.

        Args:
            delta_matrix: A matrix of coefficients controlling the tail dependence
                between each pair of dimensions. Must be >= 1. Note that only
                the lower diagonal of this matrix is used.
            theta: Mixing parameter. Controls the overall dependence level.
                Must be greater than one.
        """
        self.n = len(delta_matrix)
        for i in range(1, self.n):
            if len(delta_matrix[i]) != self.n:
                raise ValueError("delta_matrix must be square")
            if min(delta_matrix[i][:i]) < 1:
                raise ValueError("delta_matrix must be greater than or equal to 1")
        self.delta_matrix = delta_matrix
        if theta < 1:
            raise ValueError("Theta must be in the range [1, inf)")
        self.theta = theta

    def _transform_to_uniform(
        self, unnormalised_samples: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform max-mixture samples to uniform."""
        return np.exp(-((-np.log(unnormalised_samples)) ** (1 / self.theta)))

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        n = self.n
        theta = self.theta
        delta_matrix = self.delta_matrix
        max_u = np.zeros((n, n_sims))
        mixing_variable = levy_stable(
            alpha=1 / theta, beta=1.0, size=n_sims, rng=rng
        ) * (np.cos(np.pi / (2 * theta)) ** theta)
        # generate the pairwise Gumbel copulas
        for j in range(n):
            for i in range(j + 1, n):
                uv = GumbelCopula(delta_matrix[i][j], 2).generate(n_sims, rng)
                u = uv[0].values
                v = uv[1].values
                max_u[i] = np.maximum(max_u[i], u)
                max_u[j] = np.maximum(max_u[j], v)
        v = max_u ** ((n - 1) / mixing_variable)
        return v

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from a multivariate max-mixture copula, denoted MM1 by Joe.

        The MM1 copula is a multivariate copula which allows for different tail
        dependence structures between each pair of dimensions. It can be regarded as an
        extension of the Gumbel copula to more general tail dependence.

        The simulation approach uses the max-mixture representation of the MM1 copula,
        detailed in Joe (2015, Chapter 6).

        References:
        Joe, H. (1997). Multivariate Models and Dependence Concepts. Chapman and Hall.
        Joe, H. (2015). Dependence Modeling with Copulas. Chapman and Hall.

        Args:
            n_sims: Number of simulations to generate. Uses config.n_sims if None.
            rng: Random number generator. Uses config.rng if None.

        Returns:
            ProteusVariable containing VectorLike values (typically StochasticScalar)
                with uniform marginal distributions on [0,1] and the copula's
                dependency structure.
        """
        return self._generate_base(n_sims, rng)


class GalambosCopula(Copula):
    """A class to represent a Galambos copula.

    The Galambos copula is an example of a multivariate extreme value copula,
    which is particularly suited for modeling upper tail dependence between
    random variables.

    Its dependence structure is characterized by a single parameter, theta>0,
    which controls the strength of the upper tail dependence.

    The tail dependence coefficient between any pair of variables in the
    Galambos copula is given by 2^(-1/theta),

    References:
        Galambos, János. The Asymptotic Theory of Extreme Order Statistics. New York:
        John Wiley & Sons, 1978.
    """

    def __init__(self, theta: float, d: int) -> None:
        """Initialize a Galambos copula.

        Args:
            theta: Copula parameter (must be > 0).
            d: Number of variables.
        """
        if theta <= 0:
            raise ValueError("Theta must be in the range (0, inf)")
        self.theta = theta
        self.d = d

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Vectorised simulation from the d-dimensional Galambos copula.

        Exact algorithm based on the max stable / reciprocal Archimedean
        representation in Mai (2018).

        References:
        Mai, Jan-Frederik. "Exact Simulation of Reciprocal Archimedean Copulas."
        Statistical Probability Letters (2018). arXiv preprint arXiv:1802.09996


        Args:
            n_sims: Number of samples.
            rng : np.random.Generator, optional
            Random generator.

        Returns:
        u : ndarray, shape (d, n)
            Samples on (0, 1)^d with Galambos copula.
        """
        d = self.d
        # Independence shortcut if needed
        if self.theta < 1e-4:
            return rng.uniform(0, 1, size=(self.d, n_sims))
        num = gamma(d) * gamma(1.0 / self.theta)
        den = gamma(d + 1.0 / self.theta) * self.theta
        # Compute c_theta for the Galambos copula in dimension d.
        # S^{-1}(t) = c_theta * t^{-theta}.
        c_theta = (num / den) ** (-self.theta)

        inv_theta = 1.0 / self.theta
        c_th = c_theta

        # Y holds the max stable representation for all samples
        y = np.zeros((self.d, n_sims))

        # Each row has its own Poisson process time T and radius R
        # First jump times T ~ Exp(1)
        t = rng.exponential(scale=1.0, size=n_sims)
        r = c_th * (t ** (-self.theta))

        # Loop over series terms, updating all active samples together
        while True:
            # For each sample, check whether the current radius still matters
            y_min = y.min(axis=0)
            active = r > y_min

            if not np.any(active):
                break  # all series truncated

            m = active.sum()

            # Directions on simplex for active samples: iid exponentials
            e = rng.exponential(scale=1.0, size=(d, m))
            e_sum = e.sum(axis=0, keepdims=True)

            # Candidate contribution for Y on active rows
            val = r[active] * e / e_sum

            # Update Y on active rows with elementwise max
            y_active = y[:, active]
            np.maximum(y_active, val, out=y_active)
            y[:, active] = y_active

            # Advance the Poisson times and radii for active rows only
            t[active] += rng.exponential(scale=1.0, size=m)
            r[active] = c_th * (t[active] ** (-self.theta))

        # Map max stable representation to uniforms:
        # U_i = exp( - Y_i^{-1/theta} )
        u = np.exp(-(y ** (-inv_theta)))
        return u

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Galambos copula.

        Exact algorithm based on the max stable / reciprocal Archimedean
        representation in Mai (2018).

        Mai, Jan-Frederik. "Exact Simulation of Reciprocal Archimedean Copulas."
        Statistical Probability Letters (2018). arXiv preprint arXiv:1802.09996

        Args:
            n_sims: Number of simulations to generate. Uses config.n_sims if None.
            rng: Random number generator. Uses config.rng if None.

        Returns:
            ProteusVariable with StochasticScalar values representing samples from the
            Galambos copula.
        """
        return self._generate_base(n_sims, rng)


class PlackettCopula(Copula):
    """A class to represent a Plackett copula.

    The Plackett copula is a bivariate copula that can model both positive and
    negative dependence between two random variables. It is characterized by a
    single parameter, delta>0, which controls the strength and direction of the
    dependence.

    References:
        Plackett, R. L. (1965). A class of bivariate distributions. Journal of the
        American Statistical Association, 60(310), 516-522.
    """

    def __init__(self, delta: float) -> None:
        """Initialize a Plackett copula.

        Args:
            delta: Copula parameter (must be > 0).
        """
        if delta <= 0:
            raise ValueError("Delta must be in the range (0, inf)")
        self.delta = delta

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the Plackett copula.

        Args:
            n_sims: Number of samples.
            rng : np.random.Generator, optional
            Random generator.

        Returns:
        u : ndarray, shape (2, n)
            Samples on (0, 1)^2 with Plackett copula.
        """
        u = rng.uniform(0, 1, size=(n_sims))
        w = rng.uniform(0, 1, size=(n_sims))
        if self.delta == 1:
            v = w
            return np.vstack((u, v))
        a = w * (1 - w)
        delta = self.delta
        b = delta + a * (delta - 1) ** 2
        c = 2 * a * (u * delta**2 + 1 - u) + delta * (1 - 2 * a)
        d = np.sqrt(delta * (delta + 4 * a * u * (1 - u) * (1 - delta) ** 2))
        v = (c - (1 - 2 * w) * d) / (2 * b)

        return np.vstack((u, v))

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Plackett copula.

        This uses the exact simulation algorithm for the Plackett copula from Johnson
        (1987).

        References:
        Johnson ME (1987) Multivariate Statistical Simulation. J. Wiley & Sons, New York

        Args:
            n_sims: Number of simulations to generate. Uses config.n_sims if None.
            rng: Random number generator. Uses config.rng if None.

        Returns:
            ProteusVariable with StochasticScalar values representing samples from the
            Plackett copula.
        """
        return self._generate_base(n_sims, rng)


def _sibuya_gen(
    alpha: float, size: int | tuple[int, ...], rng: np.random.Generator
) -> npt.NDArray[np.floating]:
    """Generate samples from a Sibuya distribution.

    Parameters:
        alpha (float): Parameter for the Sibuya distribution.
        size (int or tuple): Output shape.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Samples from a Sibuya distribution.
    """
    g1 = rng.gamma(alpha, 1, size=size)
    g2 = rng.gamma(1 - alpha, 1, size=size)
    r = g2 / g1
    e = rng.exponential(1, size=size)
    u = r * e
    return (1 + rng.poisson(u, size=size)).astype(np.float64)


class HuslerReissCopula(Copula):
    """A class to represent a Hüsler-Reiss copula.

    The Hüsler-Reiss copula is an example of a multivariate extreme value copula,
    which is suited for modeling upper tail dependence between
    random variables and allows for a flexible specification of tail dependency
    for each bivariate pair of variables.

    Its dependence structure is characterized by a matrix Lambda_ij which controls the
    strength of the upper tail dependence between each pair of variables. Lower values
    in the matrix correspond to stronger dependence.

    The upper tail dependence coefficient between any pair of variables i and j in the
    Hüsler-Reiss copula is given by:

    χ_ij = 2 * (1 - Phi( λ_ij  )),

    where Phi is the standard normal CDF.

    References:
        Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between
        independence and complete dependence. Statistics & Probability Letters,
        7(4), 283-286.
    """

    is_adjusted: bool = False
    """Indicates whether the provided lambda matrix was adjusted to ensure validity."""
    d: int
    """The dimension of the copula."""
    adjusted_lambda_matrix: npt.NDArray[np.floating]
    """The adjusted lambda matrix after ensuring validity."""

    def __init__(
        self,
        lambda_matrix: npt.NDArray[np.floating] | list[list[float]],
    ) -> None:
        """Initialize a Hüsler-Reiss copula.

        Its dependence structure is characterized by a matrix Lambda_ij which controls
        the strength of the upper tail dependence between each pair of variables. Lower
        values in the matrix correspond to stronger dependence.

        The upper tail dependence coefficient between any pair of variables i and j in
        the Hüsler-Reiss copula is given by:

        χ_ij = 2 * (1 - Phi( λ_ij  )),

        where Phi is the standard normal CDF.

        The parameters λ_ij must be non-negative, and the matrix must be symmetric.
        The diagonal elements λ_ij must always be zero. Values of λ_ij are capped
        at 100 to avoid numerical issues during simulation.

        The matrix λ_ij must satisfy certain conditions to ensure it corresponds
        to a valid Hüsler-Reiss copula. In particular, the matrix must be conditionally
        negative definite. That is, its square must correspond to a valid variogram of
        a random field Z_j:

        λ_ij^2 = 2 * ( Var(Z_i) + Var(Z_j) - 2 * Cov(Z_i, Z_j) )

        This is checked during initialization by attempting to construct a valid
        covariance matrix for the random process Z_i from the provided λ_ij matrix.

        If the provided matrix does not satisfy these conditions, it is adjusted
        to the nearest valid matrix by modifying the eigenvalues of the corresponding
        covariance matrix. The `is_adjusted` attribute will be set to True in this case.

        References:
            Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between
            independence and complete dependence. Statistics & Probability Letters,
            7(4), 283-286.

        Args:
            lambda_matrix: Symmetric matrix λ_ij determining the pairwise dependency
            between variables.
        """
        lambda_matrix = np.asarray(lambda_matrix)
        if lambda_matrix.ndim != 2 or lambda_matrix.shape[0] != lambda_matrix.shape[1]:
            raise ValueError("Parameter matrix must be square")
        if lambda_matrix.min() < 0.0:
            raise ValueError("Matrix values must be non-negative")
        if not np.allclose(lambda_matrix, lambda_matrix.T):
            raise ValueError("Matrix must be symmetric")
        if not np.allclose(np.diag(lambda_matrix), 0.0):
            raise ValueError("Matrix diagonal must be zero")

        # calculate the covariance matrix from the lambda matrix
        d = lambda_matrix.shape[0]
        np.clip(lambda_matrix, a_min=0, a_max=100, out=lambda_matrix)
        # pivot on the first variable to construct a covariance matrix
        # consistent with the variogram defined by lambda_matrix squared
        covariance_matrix = 2 * (
            lambda_matrix[0] ** 2 + lambda_matrix[:, 0, None] ** 2 - lambda_matrix**2
        )
        covariance_sub = covariance_matrix[1:, 1:]
        vals, vecs = np.linalg.eigh(covariance_sub)
        if vals.min() < 1e-6:
            covariance_sub = (
                vecs @ np.diag(np.clip(vals, a_min=1e-6, a_max=None)) @ vecs.T
            )
            self.is_adjusted = True
        try:
            chol_sub = np.linalg.cholesky(covariance_sub)
            self._chol = np.zeros((d, d))
            self._chol[1:, 1:] = chol_sub
        except np.linalg.LinAlgError as e:
            raise ValueError("Could not construct a valid correlation matrix") from e
        covariance_matrix = self._chol @ self._chol.T
        self.d = d
        self.adjusted_lambda_matrix = (
            np.sqrt(
                np.diag(covariance_matrix)
                + np.diag(covariance_matrix)[:, None]
                - 2 * covariance_matrix
            )
            / 2
        )

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Exact simulation from a d-dimensional Hüsler-Reiss copula.

        See Dombry-Engelke-Oesting (2016) Algorithm 2 (spectral measure on L1-sphere).

        References:
        Dombry, C., Engelke, S., & Oesting, M. (2016). Exact simulation of max-stable
        processes. Biometrika 103, no. 2 (2016): 303-17.

        Args:
            n_sims : int
                Number of samples to generate.
            rng : np.random.Generator.

        Returns:
            u : (d,n) ndarray
                Samples from the Hüsler–Reiss copula.
        """
        d = self.d

        # Cholesky for Gaussian simulation
        chol = self._chol
        # Precompute variogram appearing in Proposition 4 / Remark 2
        h = 2 * (self.adjusted_lambda_matrix**2)

        # Z will hold the unit Fréchet max-stable vector
        z = np.zeros((n_sims, d), dtype=float)

        # Poisson process in 1/zeta with rate = d (Alg. 2: Exp(N))
        # So scale = 1 / rate = 1/d
        zeta_inv = rng.exponential(scale=1.0 / d, size=n_sims)
        zeta = 1.0 / zeta_inv

        # Track which simulations are still active (have not met stopping criterion)
        active = np.ones(n_sims, dtype=bool)

        while np.any(active):
            # Active indices
            idx = np.where(active)[0]

            # Stopping rule: while zeta > min_j Z_j for each simulation
            min_z = z[idx].min(axis=1)
            still_active = zeta[idx] > min_z

            if not np.any(still_active):
                break

            idx = idx[still_active]
            na = idx.size

            # 1. Sample anchor indices T ~ Uniform{0,...,d-1}
            t = rng.integers(low=0, high=d, size=na)

            # 2. Sample Gaussian W ~ N(0, sigma^2 * R) for active sims
            g = rng.standard_normal(size=(na, d))
            w = g @ chol.T  # shape (na, d)

            # 3. Construct Y according to P_{x_T} (Proposition 4 / Remark 2):
            #  Y_j = exp( W_j - W_T - H_{j,T} ) with H_{j,T} = sigma^2 * (1 - R_{j,T}).
            w_anchor = w[np.arange(na), t]  # (na,)
            h_for_t = h[:, t].T  # (na, d), row r: H_{j, T_r}

            logy = w - w_anchor[:, None] - h_for_t
            y = np.exp(logy)  # shape (na, d)

            # 4. Normalize onto the L1-sphere
            y_sum = y.sum(axis=1, keepdims=True)
            y /= y_sum

            # 5. Update max-stable vector Z = max(Z, zeta * Y)
            contrib = zeta[idx][:, None] * y
            z[idx] = np.maximum(z[idx], contrib)

            # 6. Next Poisson point: zeta^{-1} += Exp(rate = d)
            e = rng.exponential(scale=1.0 / d, size=na)
            zeta_inv[idx] += e
            zeta[idx] = 1.0 / zeta_inv[idx]

            # Update active mask for next round
            min_z_new = z[idx].min(axis=1)
            active[idx] = zeta[idx] > min_z_new

        # Transform to unit uniform margins: U = exp(-1/Z)
        u = np.exp(-1.0 / z)
        return u.T

    @property
    def tail_dependence_matrix(self) -> npt.NDArray[np.floating]:
        """Calculate the upper tail dependence matrix for the Hüsler-Reiss copula.

        The upper tail dependence coefficient between any pair of variables i and j in
        the Hüsler-Reiss copula is given by:

        χ_ij = 2 * (1 - Phi( λ_ij  )),

        where Phi is the standard normal CDF.

        Returns:
            npt.NDArray[np.floating]: A 2D array representing the upper tail dependence
            coefficients between each pair of variables.
        """
        lambda_matrix = self.adjusted_lambda_matrix
        tail_dependence_matrix = 2.0 * (1.0 - norm.cdf(lambda_matrix))

        return tail_dependence_matrix

    @staticmethod
    def calculate_lambda_from_tail_dependence(
        tail_dependence_matrix: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Calculate the lambda matrix from a given upper tail dependence matrix.

        The upper tail dependence coefficient between any pair of variables i and j in
        the Hüsler-Reiss copula is given by:

        χ_ij = 2 * (1 - Phi( λ_ij  )),

        where Phi is the standard normal CDF.

        This method inverts the above relationship to compute the lambda matrix from
        the provided upper tail dependence coefficients.
        λ_ij = Phi^(-1)(1 - χ_ij / 2)

        where Phi^(-1) is the inverse standard normal CDF.

        Args:
            tail_dependence_matrix (npt.NDArray[np.floating]): A 2D array
                representing the upper tail dependence coefficients between
                each pair of variables.

        Returns:
            npt.NDArray[np.floating]: A 2D array representing the lambda matrix
            corresponding to the given upper tail dependence coefficients.
        """
        chi_ij = tail_dependence_matrix
        lambda_matrix = norm.ppf(1.0 - chi_ij / 2.0)

        return lambda_matrix

    @classmethod
    def from_tail_dependence_matrix(
        cls, tail_dependence_matrix: npt.NDArray[np.floating]
    ) -> HuslerReissCopula:
        """Create a Hüsler-Reiss copula from a given upper tail dependence matrix.

        The upper tail dependence coefficient between any pair of variables i and j in
        the Hüsler-Reiss copula is given by:

        χ_ij = 2 * (1 - Phi( λ_ij  )),

        where Phi is the standard normal CDF.

        Args:
            tail_dependence_matrix (npt.NDArray[np.floating]): A 2D array
                representing the upper tail dependence coefficients between
                each pair of variables.

        Returns:
            HuslerReissCopula: An instance of the Hüsler-Reiss copula initialized
            with the lambda matrix corresponding to the given upper tail
            dependence coefficients.
        """
        lambda_matrix = cls.calculate_lambda_from_tail_dependence(
            tail_dependence_matrix
        )
        return cls(lambda_matrix)

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Hüsler-Reiss copula.

        The simulation uses the exact algorithm from Dombry-Engelke-Oesting (2016)

        References:
        Dombry, C., Engelke, S., & Oesting, M. (2016). Exact simulation of max-stable
        processes. Biometrika 103, no. 2 (2016): 303-17.

        Args:
            n_sims: Number of simulations to generate. Uses config.n_sims if None.
            rng: Random number generator. Uses config.rng if None.

        Returns:
            ProteusVariable with StochasticScalar values representing samples from the
            Hüsler-Reiss copula.
        """
        return self._generate_base(n_sims, rng)


def apply_copula(
    variables: ProteusVariable[StochasticScalar] | list[StochasticScalar],
    copula_samples: ProteusVariable[StochasticScalar] | list[StochasticScalar],
) -> None:
    """Apply a reordering from a copula to a list of variables.

    Parameters:
        variables: List of StochasticScalar variables.
        copula_samples: List of StochasticScalar samples from the copula.
    """
    if len(variables) != len(copula_samples):
        raise ValueError("Number of variables and copula samples do not match.")
    variables_list = list(variables)

    # Check independence of variables
    for i, var1 in enumerate(variables_list):
        for j, var2 in enumerate(variables_list[i + 1 :]):
            if var1.coupled_variable_group is var2.coupled_variable_group:
                raise ValueError(
                    f"Cannot apply copula as the variables at positions {i} and "
                    f"{j + i + 1} are not independent"
                )

    # Get sort indices and ranks
    copula_sort_indices = np.argsort(
        np.array([cs.values for cs in copula_samples]), axis=1, kind="stable"
    )
    copula_ranks = np.argsort(copula_sort_indices, axis=1)
    variable_sort_indices = np.argsort(
        np.array([var.values for var in variables]), axis=1
    )
    first_variable_rank = np.argsort(variable_sort_indices[0])
    copula_ranks = copula_ranks[:, copula_sort_indices[0, first_variable_rank]]

    # Apply reordering
    for i, var in enumerate(variables):
        if i == 0:
            continue
        re_ordering = variable_sort_indices[i, copula_ranks[i]]
        for var2 in var.coupled_variable_group:
            # FIXME: _reorder_sims is a protected method but we need to access it here
            # for copula reordering functionality. Consider making this method public
            # or providing a public interface for reordering simulations across
            # coupled variable groups.
            var2._reorder_sims(re_ordering)  # type: ignore[misc]

    # Merge coupling groups
    for var in variables:
        var.coupled_variable_group.merge(variables[0].coupled_variable_group)
