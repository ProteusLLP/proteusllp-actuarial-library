"""
Copula Module

This module contains classes for representing and generating samples from various
copulas. It includes both elliptical (Gaussian and Student's T) and Archimedean
copulas.
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

# Third-party imports
import numpy.typing as npt
import scipy.stats.distributions as distributions  # type: ignore

# Local imports
from .config import config, xp as np, _use_gpu
from .variables import ProteusVariable, StochasticScalar

if _use_gpu:
    import cupyx.scipy.special as special  # type: ignore
else:
    import scipy.special as special  # type: ignore


class Copula(ABC):
    """A base class to represent a copula."""

    @abstractmethod
    def generate(
        self, n_sims: Union[int, None] = None, rng: np.random.Generator = config.rng
    ) -> ProteusVariable:
        """Generate samples from the copula."""
        pass

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray:
        """
        Generate samples from the multivariate distribution underlying the copula.
        The marginal distribution of the samples will not necessarily be uniform.
        """
        return NotImplemented  # To be implemented by subclasses

    def apply(self, variables: Union[ProteusVariable, List[StochasticScalar]]) -> None:
        """
        Apply the copula to a list of variables.

        Parameters:
            variables: A ProteusVariable or list of StochasticScalar instances.
        """
        # Generate the copula samples
        copula_samples = ProteusVariable(
            dim_name="dim1",
            values=[
                StochasticScalar(sample)
                for sample in self._generate_unnormalised(
                    n_sims=variables[0].n_sims, rng=config.rng
                )
            ],
        )
        if len(variables) != len(copula_samples):
            raise ValueError("Number of variables and copula samples do not match.")
        # Apply the copula to the variables
        apply_copula(variables, copula_samples)


class EllipticalCopula(Copula, ABC):
    """A base class to represent an elliptical copula."""

    matrix: npt.NDArray
    chol: npt.NDArray

    def __init__(
        self,
        matrix: npt.NDArray | list[list[float]],
        *args: Any,
        matrix_type: str = "linear",
        **kwargs: Any,
    ) -> None:
        _matrix = np.asarray(matrix)
        if _matrix.ndim != 2 or _matrix.shape[0] != _matrix.shape[1]:
            raise ValueError("Matrix must be square")
        if matrix_type == "linear":
            self.correlation_matrix = _matrix
            # Check that the correlation matrix is positive definite
            try:
                self.chol = np.linalg.cholesky(self.correlation_matrix)
            except np.linalg.LinAlgError:
                raise ValueError("Correlation matrix is not positive definite")
        elif matrix_type == "chol":
            self.chol = _matrix
        else:
            raise ValueError("matrix_type must be 'linear' or 'chol'")
        self.matrix = _matrix


class GaussianCopula(EllipticalCopula):
    """A class to represent a Gaussian copula."""

    def __init__(self, matrix: npt.NDArray, matrix_type: str = "linear") -> None:
        super().__init__(matrix, matrix_type=matrix_type)

    def generate(
        self, n_sims: Union[int, None] = None, rng: np.random.Generator = config.rng
    ) -> ProteusVariable:
        """Generate samples from the Gaussian copula."""
        if n_sims is None:
            n_sims = config.n_sims

        # Generate samples from a multivariate normal distribution
        samples = self._generate_unnormalised(n_sims, rng)
        uniform_samples = special.ndtr(samples)
        result = ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in uniform_samples]
        )
        for val in result:
            val.coupled_variable_group.merge(result[0].coupled_variable_group)
        return result

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray:
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
        super().__init__(matrix, matrix_type=matrix_type)
        if dof <= 0:
            raise ValueError("Degrees of Freedom must be positive")
        self.dof = dof

    def generate(
        self, n_sims: Union[int, None] = None, rng: np.random.Generator = config.rng
    ) -> ProteusVariable:
        if n_sims is None:
            n_sims = config.n_sims
        t_samples = self._generate_unnormalised(n_sims, rng)
        uniform_samples = distributions.t(self.dof).cdf(t_samples)
        return ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in uniform_samples]
        )

    def _generate_unnormalised(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray:
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
    def generator_inv(self, t: npt.NDArray) -> npt.NDArray:
        """The inverse generator function of the copula."""
        pass

    @abstractmethod
    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray:
        """Generate samples from the latent distribution of the copula."""
        pass

    def __init__(self, n: int) -> None:
        self.n = n

    def generate(
        self, n_sims: Union[int, None] = None, rng: np.random.Generator = config.rng
    ) -> ProteusVariable:
        copula_samples = self.generator_inv(-self._generate_unnormalised(n_sims, rng))
        result = ProteusVariable(
            "dim1", [StochasticScalar(sample) for sample in copula_samples]
        )
        for val in result:
            val.coupled_variable_group.merge(result[0].coupled_variable_group)
        return result

    def _generate_unnormalised(
        self, n_sims: Union[int, None] = None, rng: np.random.Generator = config.rng
    ) -> npt.NDArray:
        if n_sims is None:
            n_sims = config.n_sims
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
        if theta < 0:
            raise ValueError("Theta cannot be negative")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray) -> npt.NDArray:
        if self.theta == 0:
            return np.exp(-t)
        return (1 + t) ** (-1 / self.theta)

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray:
        if self.theta == 0:
            return np.array([1])
        return rng.gamma(1 / self.theta, size=n_sims)


def levy_stable(
    alpha: float,
    beta: float,
    size: Union[int, Tuple[int, ...]],
    rng: np.random.Generator,
) -> npt.NDArray:
    """
    Simulate samples from a Lévy stable distribution using the Chambers-Mallows-Stuck algorithm.

    Parameters:
        alpha (float): Stability parameter in (0, 2].
        beta (float): Skewness parameter in [-1, 1].
        size (int or tuple of ints): Output shape.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Samples from the Lévy stable distribution.
    """
    U = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W = rng.exponential(1, size)

    if alpha != 1:
        theta = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
        factor = (1 + beta**2 * np.tan(np.pi * alpha / 2) ** 2) ** (1 / (2 * alpha))
        part1 = np.sin(alpha * (U + theta)) / (np.cos(U)) ** (1 / alpha)
        part2 = (np.cos(U - alpha * (U + theta)) / W) ** ((1 - alpha) / alpha)
        samples = factor * part1 * part2
    else:
        samples = (2 / np.pi) * (
            (np.pi / 2 + beta * U) * np.tan(U)
            - beta * np.log((np.pi / 2 * W * np.cos(U)) / (np.pi / 2 + beta * U))
        )
    return samples


class GumbelCopula(ArchimedeanCopula):
    """A class to represent a Gumbel copula."""

    def __init__(self, theta: float, n: int) -> None:
        if theta < 1:
            raise ValueError("Theta must be at least 1")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray) -> npt.NDArray:
        return np.exp(-(t ** (1 / self.theta)))

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray:
        return levy_stable(1 / self.theta, 1, n_sims, rng) * (
            np.cos(np.pi / (2 * self.theta)) ** self.theta
        )


class FrankCopula(ArchimedeanCopula):
    """A class to represent a Frank copula."""

    def __init__(self, theta: float, n: int) -> None:
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray) -> npt.NDArray:
        return -np.log1p(np.exp(-t) * (np.expm1(-self.theta))) / self.theta

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray:
        return rng.logseries(1 - np.exp(-self.theta), size=n_sims)


class JoeCopula(ArchimedeanCopula):
    """A class to represent a Joe copula."""

    def __init__(self, theta: float, n: int) -> None:
        if theta < 1:
            raise ValueError("Theta must be in the range [1, inf)")
        self.theta = theta
        self.n = n

    def generator_inv(self, t: npt.NDArray) -> npt.NDArray:
        return 1 - (1 - np.exp(-t)) ** (1 / self.theta)

    def generate_latent_distribution(
        self, n_sims: int, rng: np.random.Generator
    ) -> npt.NDArray:
        return _sibuya_gen(1 / self.theta, n_sims, rng)


def _sibuya_gen(
    alpha: float, size: Union[int, Tuple[int, ...]], rng: np.random.Generator
) -> npt.NDArray:
    """
    Generate samples from a Sibuya distribution.

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
    return 1 + rng.poisson(u, size=size)


def apply_copula(
    variables: Union[ProteusVariable, List[StochasticScalar]],
    copula_samples: Union[ProteusVariable, List[StochasticScalar]],
) -> None:
    """
    Apply a reordering from a copula to a list of variables.

    Parameters:
        variables: List of StochasticScalar variables.
        copula_samples: List of StochasticScalar samples from the copula.
    """
    if len(variables) != len(copula_samples):
        raise ValueError("Number of variables and copula samples do not match.")
    variables_list: list[StochasticScalar] = []
    for var in variables:
        assert isinstance(
            var, StochasticScalar
        ), "Variables must be StochasticScalar instances"
        variables_list.append(var)
    for i, var1 in enumerate(variables_list):
        for j, var2 in enumerate(variables_list[i + 1 :]):
            if var1.coupled_variable_group is var2.coupled_variable_group:
                raise ValueError(
                    f"Cannot apply copula as the variables at positions {i} and {j + i + 1} are not independent"
                )
    copula_sort_indices = np.argsort(
        np.array([cs.values for cs in copula_samples]), axis=1, kind="stable"
    )
    copula_ranks = np.argsort(copula_sort_indices, axis=1)
    variable_sort_indices = np.argsort(
        np.array([var.values for var in variables]), axis=1
    )
    first_variable_rank = np.argsort(variable_sort_indices[0])
    copula_ranks = copula_ranks[:, copula_sort_indices[0, first_variable_rank]]
    for i, var in enumerate(variables):
        if i == 0:
            continue
        re_ordering = variable_sort_indices[i, copula_ranks[i]]
        for var2 in var.coupled_variable_group.variables:
            var2._reorder_sims(re_ordering)
    for var in variables:
        var.coupled_variable_group.merge(variables[0].coupled_variable_group)
