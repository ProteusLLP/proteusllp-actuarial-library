"""Copula Module.

This module contains classes for representing and generating samples from various
copulas. It includes both elliptical (Gaussian and Student's T) and Archimedean
copulas.
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any

# Third-party imports
import numpy.typing as npt
import scipy.stats.distributions as distributions  # type: ignore [import-untyped]

from . import ProteusVariable, StochasticScalar
from ._maths import special
from ._maths import xp as np

# Local imports
from .config import config


class Copula(ABC):
    """A base class to represent a copula."""

    @abstractmethod
    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable:
        """Generate samples from the copula."""
        pass

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the multivariate distribution underlying the copula.

        The marginal distribution of the samples will not necessarily be uniform.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def apply(self, variables: ProteusVariable | list[StochasticScalar]) -> None:
        """Apply the copula to a list of variables.

        Parameters:
            variables: A ProteusVariable or list of StochasticScalar instances.
        """
        # Convert ProteusVariable to list of StochasticScalar if needed
        if isinstance(variables, ProteusVariable):
            variables_list = [
                val for val in variables if isinstance(val, StochasticScalar)
            ]
        else:
            variables_list = variables

        # Generate the copula samples
        rng_generator = config.rng
        copula_samples_pv = ProteusVariable(
            dim_name="dim1",
            values=[
                StochasticScalar(sample)
                for sample in self._generate_unnormalised(
                    n_sims=variables_list[0].n_sims, rng=rng_generator
                )
            ],
        )
        copula_samples_list = [
            val for val in copula_samples_pv if isinstance(val, StochasticScalar)
        ]

        if len(variables_list) != len(copula_samples_list):
            raise ValueError("Number of variables and copula samples do not match.")
        # Apply the copula to the variables
        apply_copula(variables_list, copula_samples_list)


class EllipticalCopula(Copula, ABC):
    """A base class to represent an elliptical copula."""

    matrix: npt.NDArray[np.floating]
    chol: npt.NDArray[np.floating]

    def __init__(
        self,
        matrix: npt.NDArray[np.floating] | list[list[float]],
        *args: Any,
        matrix_type: str = "linear",
        **kwargs: Any,
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

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable:
        """Generate samples from the Gaussian copula."""
        if n_sims is None:
            n_sims = config.n_sims
        if rng is None:
            rng = config.rng

        # Generate samples from a multivariate normal distribution
        samples = self._generate_unnormalised(n_sims, rng)
        uniform_samples = special.ndtr(samples)
        result = ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in uniform_samples]
        )
        for val in result:
            if isinstance(val, StochasticScalar):
                val.coupled_variable_group.merge(result[0].coupled_variable_group)
        return result

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        n_vars = self.correlation_matrix.shape[0]
        normal_samples = rng.multivariate_normal(
            mean=np.zeros(n_vars), cov=np.eye(n_vars), size=n_sims
        ).T
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

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable:
        """Generate samples from the Student's T copula."""
        if n_sims is None:
            n_sims = config.n_sims
        if rng is None:
            rng = config.rng
        t_samples = self._generate_unnormalised(n_sims, rng)
        uniform_samples = distributions.t(self.dof).cdf(t_samples)
        return ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in uniform_samples]
        )

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        n_vars = self.correlation_matrix.shape[0]
        normal_samples = self.chol.dot(
            rng.multivariate_normal(
                mean=np.zeros(n_vars), cov=np.eye(n_vars), size=n_sims
            ).T
        )
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

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusVariable:
        """Generate samples from the Archimedean copula."""
        if rng is None:
            rng = config.rng
        copula_samples = self.generator_inv(-self._generate_unnormalised(n_sims, rng))
        result = ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in copula_samples]
        )
        for val in result:
            if isinstance(val, StochasticScalar):
                val.coupled_variable_group.merge(result[0].coupled_variable_group)
        return result

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
        # Calculate the copula samples
        return np.log(u) / latent_samples[np.newaxis]


class ClaytonCopula(ArchimedeanCopula):
    """A class to represent a Clayton copula."""

    def __init__(self, theta: float, n: int) -> None:
        """Initialize a Clayton copula.

        Args:
            theta: Copula parameter.
            n: Number of variables.
        """
        if theta < 0:
            raise ValueError("Theta cannot be negative")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Inverse generator function for Clayton copula."""
        if self.theta == 0:
            return np.exp(-t)
        return (1 + t) ** (-1 / self.theta)

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution."""
        if self.theta == 0:
            return np.array([1])
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


def apply_copula(
    variables: list[StochasticScalar],
    copula_samples: list[StochasticScalar],
) -> None:
    """Apply a reordering from a copula to a list of variables.

    Parameters:
        variables: List of StochasticScalar variables.
        copula_samples: List of StochasticScalar samples from the copula.
    """
    if len(variables) != len(copula_samples):
        raise ValueError("Number of variables and copula samples do not match.")

    # Check independence of variables
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables[i + 1 :]):
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
        for var2 in var.coupled_variable_group.variables:
            var2._reorder_sims(re_ordering)

    # Merge coupling groups
    for var in variables:
        var.coupled_variable_group.merge(variables[0].coupled_variable_group)
