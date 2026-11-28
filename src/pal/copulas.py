"""Copula Module.

This module contains classes for representing and generating samples from various
copulas. It includes both elliptical (Gaussian and Student's T) and Archimedean
copulas.
"""

# Standard library imports
from __future__ import annotations

import math
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Third-party imports
import numpy.typing as npt
import scipy.stats
from numpy import errstate
from scipy.special import gamma

from . import ProteusVariable, StochasticScalar

# Local imports
from ._correlation_preparation import RepairMode, prepare_correlation_matrix
from ._maths import asnumpy, special
from ._maths import xp as np
from .config import config
from .couplings import couple_stochastic_results
from .qmc import RandomizedQMCGenerator, _draw_active_scenarios, _draw_scenarios
from .types import RandomGenerator


def _host_rng_arg(value: t.Any) -> t.Any:
    """Copy backend array arguments to the host for NumPy random methods.

    Args:
        value: Value to convert, compare or assign.
    """
    if isinstance(value, np.ndarray):
        return asnumpy(value)
    if isinstance(value, tuple):
        return tuple(_host_rng_arg(item) for item in value)
    if isinstance(value, list):
        return [_host_rng_arg(item) for item in value]
    return value


class _BackendGenerator:
    """Adapt NumPy's seeded Generator so samples immediately move to the backend."""

    def __init__(self, generator: t.Any) -> None:
        """Initialize the backend-aware random generator wrapper.

        Args:
            generator: Random generator whose methods are wrapped for backend compatibility.
        """
        self._generator = generator

    def __getattr__(self, name: str) -> t.Any:
        """Delegate generator operations to the wrapped random generator.

        Args:
            name: Name of the delegated random-generator operation.
        """
        # CuPy's RandomState follows NumPy's legacy ``randint`` spelling.
        # Present the modern Generator spelling used by PAL's copula code.
        if name == "integers" and not hasattr(self._generator, name):
            attribute = self._generator.randint
        else:
            attribute = getattr(self._generator, name)
        if not callable(attribute):
            return attribute

        def sample(*args: t.Any, **kwargs: t.Any) -> t.Any:
            """Draw samples using the wrapped generator and return them on the active backend.

            Args:
                *args: Positional arguments supplied by the caller or NumPy dispatch.
                **kwargs: Keyword arguments supplied by the caller or NumPy dispatch.
            """
            if type(self._generator).__module__.startswith("numpy"):
                args = tuple(_host_rng_arg(arg) for arg in args)
                kwargs = {key: _host_rng_arg(value) for key, value in kwargs.items()}
            return np.asarray(attribute(*args, **kwargs))

        return sample


def _backend_rng(generator: RandomGenerator | _BackendGenerator) -> RandomGenerator | _BackendGenerator:
    """Return a backend-producing random generator adapter.

    Args:
        generator: Random generator whose methods are wrapped for backend compatibility.
    """
    if isinstance(generator, (_BackendGenerator, RandomizedQMCGenerator)):
        return generator
    if type(generator).__module__.startswith(np.__name__):
        return generator
    return _BackendGenerator(generator)


def _student_t_tail(values: t.Any, dof: float) -> t.Any:
    """Evaluate the smaller Student's t tail probability, min(F(x), 1 - F(x)).

    Args:
        values: Points at which to evaluate the tail probability.
        dof: Degrees of freedom of the Student's t distribution.
    """
    if np.__name__ == "numpy":
        return special.stdtr(dof, -np.abs(values))
    return _student_t_tail_from_beta(values, dof)


def _student_t_tail_from_beta(values: t.Any, dof: float) -> t.Any:
    """Evaluate the smaller Student's t tail probability from the incomplete beta function.

    This is the formulation used on backends without a Student's t CDF.

    Args:
        values: Points at which to evaluate the tail probability.
        dof: Degrees of freedom of the Student's t distribution.
    """
    # The tail beyond |x| is I_z(dof/2, 1/2)/2 with z = dof/(dof + x^2), but z
    # rounds towards one near the centre. There the complementary form
    # 1/2 - I_{1-z}(1/2, dof/2)/2 retains full precision, and the tail is at
    # least 0.15, so the subtraction is harmless.
    squared = values * values
    central = squared < 1
    beta = special.betainc(
        np.where(central, 0.5, dof / 2),
        np.where(central, dof / 2, 0.5),
        np.where(central, squared, dof) / (dof + squared),
    )
    return np.where(central, 0.5 - beta / 2, beta / 2)


def _student_t_cdf(values: t.Any, dof: float) -> t.Any:
    """Evaluate Student's t CDF without moving backend arrays to the host.

    Args:
        dof: Degrees of freedom of the Student's t distribution.
        values: Values aligned with the simulations or component axis.
    """
    if np.__name__ == "numpy":
        return special.stdtr(dof, values)

    tail_probability = _student_t_tail(values, dof)
    return np.where(values >= 0, 1 - tail_probability, tail_probability)


def _student_t_pdf(values: t.Any, dof: float) -> t.Any:
    """Evaluate Student's t density on the active array backend.

    Args:
        values: Points at which to evaluate the density.
        dof: Degrees of freedom of the Student's t distribution.
    """
    log_normaliser = math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2) - 0.5 * math.log(dof * math.pi)
    return np.exp(log_normaliser - (dof + 1) / 2 * np.log1p(values**2 / dof))


class Copula(ABC):
    """Base class for copula implementations.

    A copula is a multivariate probability distribution that describes the
    dependence structure between random variables, separate from their individual
    marginal distributions. Copulas are used in risk modeling to simulate
    correlated stochastic variables.

    All copula implementations generate ProteusVariable containers with VectorLike
    values (typically StochasticScalar instances) that represent correlated
    uniform random samples on [0,1].

    Attributes:
        dimension: Number of uniform margins in the copula.
    """

    dimension: int
    """The dimension of the copula."""

    @abstractmethod
    def generate(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
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

    def _generate_unnormalised(self, n_sims: int, rng: t.Any) -> npt.NDArray[np.floating]:
        """Generate samples from the multivariate distribution underlying the copula.

        The marginal distribution of the samples will not necessarily be uniform.

        Args:
            n_sims: Number of simulations to generate.
            rng: Random number generator.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _transform_to_uniform(self, unnormalised_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
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
            {f"{type(self).__name__}_{i}": StochasticScalar(sample) for i, sample in enumerate(uniform_samples)},
        )
        couple_stochastic_results(result.values.values(), n_sims=len(result[0]))

        return result

    def _generate_base(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
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

        unnormalised = self._generate_unnormalised(n_sims, _backend_rng(rng))
        uniform_samples = self._transform_to_uniform(unnormalised)
        return self._create_result_from_uniform(uniform_samples)

    def apply(self, variables: ProteusVariable[StochasticScalar] | list[StochasticScalar]) -> None:
        """Apply the copula's correlation structure to existing variables.

        This method modifies the input variables in-place to exhibit the
        correlation structure defined by this copula while preserving their
        marginal distributions.

        Args:
            variables: Either a ProteusVariable containing VectorLike values or
                      a list of VectorLike instances.

        Raises:
            TypeError: If list contains non-StochasticScalar values.
            ValueError: If variables have inconsistent simulation counts.
        """
        variables_list = list(variables)
        self.dimension = len(variables_list)
        # Generate the copula samples
        # Check that n_sims is available
        n_sims = variables_list[0].n_sims
        copula_samples = [
            StochasticScalar(sample)
            for sample in self._generate_unnormalised(n_sims=n_sims, rng=_backend_rng(config.rng))
        ]
        if len(variables) != len(copula_samples):
            raise ValueError("Number of variables and copula samples do not match.")
        # Apply the copula to the variables
        apply_copula(variables_list, copula_samples)


class EllipticalCopula(Copula, ABC):
    """A base class to represent an elliptical copula.

    Attributes:
        correlation_matrix: Correlation matrix specifying the elliptical or extremal-t dependence.
    """

    _matrix: npt.NDArray[np.floating]
    _chol: npt.NDArray[np.floating]

    def __init__(
        self,
        matrix: npt.NDArray[np.floating] | list[list[float]],
        *args: t.Any,
        matrix_type: str = "linear",
        repair: RepairMode = "higham",
        weights: npt.ArrayLike | None = None,
        pattern: t.Sequence[tuple[int, int]] | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialize an elliptical copula.

        Args:
            matrix: Correlation matrix or a factor when matrix_type="chol".
            matrix_type: "linear" for correlations or "chol" for a factor.
            repair: Higham repair by default; "none" rejects invalid inputs.
            weights: Optional positive vector of Higham row/column weights.
            pattern: Optional correlations to hold fixed during repair.
            *args: Retained for compatibility with subclasses.
            **kwargs: Retained for compatibility with subclasses.
        """
        _matrix = np.asarray(matrix, dtype=float)
        if _matrix.ndim != 2 or _matrix.shape[0] != _matrix.shape[1]:
            raise ValueError("Matrix must be square")
        if matrix_type == "linear":
            correlations = _matrix
        elif matrix_type == "chol":
            correlations = _matrix @ _matrix.T
        else:
            raise ValueError("matrix_type must be 'linear' or 'chol'")

        adjusted, self.was_repaired = prepare_correlation_matrix(
            correlations,
            repair=repair,
            weights=weights,
            pattern=pattern,
            name="Copula correlation matrix",
        )
        # Preserve the exact coefficients used by the factor independently
        # of the existing mutable correlation_matrix attribute.
        self._adjusted_correlation_matrix = adjusted.copy()
        self.correlation_matrix = adjusted
        if matrix_type == "chol" and not self.was_repaired:
            # A supplied factor already represents the requested correlations.
            self._chol = _matrix.copy()
        else:
            self._chol = np.linalg.cholesky(adjusted)
        self._matrix = _matrix.copy()

    @property
    def adjusted_correlation_matrix(self) -> t.Any:
        """The correlations actually used for simulation, as a copy."""
        return self._adjusted_correlation_matrix.copy()


class GaussianCopula(EllipticalCopula):
    r"""Gaussian (Normal) Copula.

    The Gaussian copula has the cumulative distribution function:

    .. math::

        C(u_1, \ldots, u_d) = \Phi_R\left(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_d)\right)

    where :math:`\Phi` is the standard normal CDF, :math:`\Phi^{-1}` is its inverse,
    and :math:`\Phi_R` is the multivariate normal CDF with correlation matrix :math:`R`.

    The Gaussian copula is an elliptical copula that models dependence through
    a multivariate normal distribution.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.floating] | list[list[float]],
        matrix_type: str = "linear",
        *,
        repair: RepairMode = "higham",
        weights: npt.ArrayLike | None = None,
        pattern: t.Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize a Gaussian copula with optional correlation repair.

        Higham repair is automatic when required and raises a UserWarning.
        Inspect adjusted_correlation_matrix to see the coefficients used.

        Args:
            matrix: Matrix supplied for correlation validation or repair.
            matrix_type: Whether the matrix contains correlations or rank correlations.
            pattern: Optional structural constraints on correlation matrix repair.
            repair: Whether and how to repair an invalid correlation matrix.
            weights: Optional weights controlling correlation matrix repair.
        """
        super().__init__(
            matrix,
            matrix_type=matrix_type,
            repair=repair,
            weights=weights,
            pattern=pattern,
        )

    def _transform_to_uniform(self, unnormalised_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform normal samples to uniform using CDF.

        Args:
            unnormalised_samples: Latent copula samples before transformation to uniform margins.
        """
        return special.ndtr(unnormalised_samples)

    def generate(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Gaussian copula.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        return self._generate_base(n_sims, rng)

    def _generate_unnormalised(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate the latent multivariate samples before uniform transformation.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        n_vars = self.correlation_matrix.shape[0]
        normal_samples = _draw_scenarios(rng, "standard_normal", size=(n_vars, n_sims), scenario_axis=1)
        return self._chol.dot(normal_samples)


class StudentsTCopula(EllipticalCopula):
    r"""Student's T Copula.

    The cumulative distribution function (CDF) is:

    .. math::

        C(u_1, \ldots, u_d) = t_{\nu,R}\left(t_{\nu}^{-1}(u_1), \ldots, t_{\nu}^{-1}(u_d)\right)

    where :math:`t_\nu` is the univariate Student's t CDF with :math:`\nu` degrees of freedom,
    :math:`t_\nu^{-1}` is its inverse, and :math:`t_{\nu,R}` is the multivariate Student's t
    CDF with :math:`\nu` degrees of freedom and correlation matrix :math:`R`.

    The Student's t copula exhibits symmetric tail dependence, making it useful for
    modeling joint extreme events. The upper and lower tail dependence coefficients are given by:

    .. math::

        \lambda_U = \lambda_L = 2\,t_{\nu+1}\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)

    where :math:`\nu` is the degrees of freedom and :math:`\rho` is the correlation parameter.

    Attributes:
        dof: Degrees of freedom of the Student's t copula.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.float64] | list[list[float]],
        dof: float,
        matrix_type: str = "linear",
        *,
        repair: RepairMode = "higham",
        weights: npt.ArrayLike | None = None,
        pattern: t.Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize a Student's t copula with automatic Higham repair.

        Raises UserWarning if correlation coefficients require adjustment.

        Args:
            dof: Degrees of freedom of the Student's t distribution.
            matrix: Matrix supplied for correlation validation or repair.
            matrix_type: Whether the matrix contains correlations or rank correlations.
            pattern: Optional structural constraints on correlation matrix repair.
            repair: Whether and how to repair an invalid correlation matrix.
            weights: Optional weights controlling correlation matrix repair.
        """
        super().__init__(
            matrix,
            matrix_type=matrix_type,
            repair=repair,
            weights=weights,
            pattern=pattern,
        )
        if dof <= 0:
            raise ValueError("Degrees of Freedom must be positive")
        self.dof = dof

    def _transform_to_uniform(self, unnormalised_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform t-distributed samples to uniform using CDF.

        Args:
            unnormalised_samples: Latent copula samples before transformation to uniform margins.
        """
        return _student_t_cdf(unnormalised_samples, self.dof)

    def generate(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Student's T copula.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        return self._generate_base(n_sims, rng)

    def _generate_unnormalised(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate the latent multivariate samples before uniform transformation.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        n_vars = self.correlation_matrix.shape[0]
        normal_samples = self._chol.dot(_draw_scenarios(rng, "standard_normal", size=(n_vars, n_sims), scenario_axis=1))
        chi_samples = np.sqrt(rng.gamma(self.dof / 2, 2 / self.dof, size=n_sims))
        return normal_samples / chi_samples[np.newaxis, :]


class ArchimedeanCopula(Copula, ABC):
    """A base class to represent an Archimedean copula."""

    @abstractmethod
    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """The inverse generator function of the copula.

        Args:
            t: Argument of the inverse Archimedean generator.
        """
        pass

    @abstractmethod
    def generate_latent_distribution(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution of the copula.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        pass

    def _transform_to_uniform(self, unnormalised_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform using inverse generator function.

        Args:
            unnormalised_samples: Latent copula samples before transformation to uniform margins.
        """
        return self.generator_inv(-unnormalised_samples)

    def generate(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Archimedean copula.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        return self._generate_base(n_sims, rng)

    def _generate_unnormalised(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
    ) -> npt.NDArray[np.floating]:
        """Generate the latent multivariate samples before uniform transformation.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if rng is None:
            rng = config.rng
        if rng is None:
            raise RuntimeError("No random number generator is configured")
        if self.dimension is None:
            raise RuntimeError("Subclasses of ArchimedeanCopula must set self.dimension to the number of variables")
        n_vars = self.dimension
        # Generate samples from a uniform distribution
        u = _draw_scenarios(rng, "uniform", size=(n_vars, n_sims), scenario_axis=1)
        # Generate samples from the latent distribution
        latent_samples = self.generate_latent_distribution(n_sims, rng)

        # Add shape validation
        if not (latent_samples.shape == (n_sims,)):
            raise AssertionError(f"Expected latent_samples shape ({n_sims},), got {latent_samples.shape}")

        # Calculate the copula samples
        return np.log(u) / latent_samples[np.newaxis]


class ClaytonCopula(ArchimedeanCopula):
    r"""Clayton Copula.

    The Clayton copula has the cumulative distribution function:

    .. math::

        C(u_1, \ldots, u_d) = \left(\sum_{i=1}^d u_i^{-\theta} - d + 1\right)^{-1/\theta}

    where :math:`\theta \geq 0` is the dependence parameter. The Clayton copula
    exhibits lower tail dependence and is part of the Archimedean family.

    The lower tail dependence coefficient between any pair of variables in the Clayton copula is given by:

    .. math::

        \lambda_L = 2^{-1/\theta}

    The upper tail dependence coefficient is zero for all :math:`\theta > 0`,
    and the copula reduces to the independence copula when :math:`\theta = 0`.

    The generator function is:

    .. math::

        \phi(t) = \frac{1}{\theta}(t^{-\theta} - 1)

    For :math:`\theta = 0`, the copula reduces to the independence copula.

    Attributes:
        dimension: Number of uniform margins in the copula.
        theta: Dependence parameter of the copula.
    """

    def __init__(self, theta: float, dimension: int | None = None) -> None:
        """Initialize a Clayton copula.

        Args:
            theta: Copula parameter (must be >= 0). When theta=0, represents
                   the independence copula.
            dimension: Number of variables. (Optional)
        """
        if theta < 0:
            raise ValueError("Theta cannot be negative")
        self.theta = theta
        if dimension is not None:
            self.dimension = dimension

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

    def generate_latent_distribution(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution.

        For Clayton copula, when theta=0, the copula reduces to the independence
        copula, and the latent distribution returns a constant value of 1.0 for
        all simulations.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.


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
    rng: RandomGenerator,
) -> npt.NDArray[np.floating]:
    """Simulate samples from a Lévy stable distribution using Chambers-Mallows-Stuck.

    Parameters:
        alpha (float): Stability parameter in (0, 2].
        beta (float): Skewness parameter in [-1, 1].
        size (int or tuple of ints): Output shape.
        rng (RandomGenerator): Random number generator.

    Returns:
        np.ndarray: Samples from the Lévy stable distribution.
    """
    uniform_samples = rng.uniform(-np.pi / 2, np.pi / 2, size)
    exponential_samples = rng.exponential(1, size)

    if alpha != 1:
        theta = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
        factor = (1 + beta**2 * np.tan(np.pi * alpha / 2) ** 2) ** (1 / (2 * alpha))
        part1 = np.sin(alpha * (uniform_samples + theta)) / (np.cos(uniform_samples)) ** (1 / alpha)
        part2 = (np.cos(uniform_samples - alpha * (uniform_samples + theta)) / exponential_samples) ** (
            (1 - alpha) / alpha
        )
        samples = factor * part1 * part2
    else:
        samples = (2 / np.pi) * (
            (np.pi / 2 + beta * uniform_samples) * np.tan(uniform_samples)
            - beta
            * np.log((np.pi / 2 * exponential_samples * np.cos(uniform_samples)) / (np.pi / 2 + beta * uniform_samples))
        )
    return samples


class GumbelCopula(ArchimedeanCopula):
    r"""Gumbel Copula.

    The Gumbel copula has the cumulative distribution function:

    .. math::

        C(u_1, \ldots, u_d) = \exp\left(-\left(\sum_{i=1}^d (-\ln u_i)^\theta\right)^{1/\theta}\right)

    where :math:`\theta \geq 1` is the dependence parameter. The Gumbel copula
    exhibits upper tail dependence and is part of the Archimedean family. The upper
    tail dependence coefficient between any pair of variables is given by:

    .. math::

        \lambda_U = 2 - 2^{1/\theta}

    The generator function is:

    .. math::

        \phi(t) = (-\ln t)^\theta

    Attributes:
        dimension: Number of uniform margins in the copula.
        theta: Dependence parameter of the copula.
    """

    def __init__(self, theta: float, dimension: int | None = None) -> None:
        """Initialize a Gumbel copula.

        Args:
            theta: Copula parameter (must be >= 1).
            dimension: Number of variables.
        """
        if theta < 1:
            raise ValueError("Theta must be at least 1")
        self.theta = theta
        if dimension is not None:
            self.dimension = dimension

    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Inverse generator function for Gumbel copula.

        Args:
            t: Argument of the inverse Archimedean generator.
        """
        return np.exp(-(t ** (1 / self.theta)))

    def generate_latent_distribution(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        if self.theta == 1:
            return np.ones(n_sims)
        return levy_stable(1 / self.theta, 1, n_sims, rng) * (np.cos(np.pi / (2 * self.theta)) ** self.theta)


class FrankCopula(ArchimedeanCopula):
    r"""Frank Copula.

    The Frank copula has the cumulative distribution function:

    .. math::

        C(u_1, \ldots, u_d) = -\frac{1}{\theta}\ln\left(
            1 + \frac{\prod_{i=1}^d (e^{-\theta u_i} - 1)}{(e^{-\theta} - 1)^{d-1}}
        \right)

    where :math:`\theta \in \mathbb{R} \setminus \{0\}` is the dependence parameter.
    The Frank copula is symmetric and does not exhibit tail dependence.

    The generator function is:

    .. math::

        \phi(t) = -\ln\left(\frac{e^{-\theta t} - 1}{e^{-\theta} - 1}\right)

    Attributes:
        dimension: Number of uniform margins in the copula.
        theta: Dependence parameter of the copula.
    """

    def __init__(self, theta: float, dimension: int | None = None) -> None:
        """Initialize a Frank copula.

        Args:
            theta: Copula parameter.
            dimension: Number of variables. (Optional)
        """
        self.theta = theta
        if dimension is not None:
            self.dimension = dimension

    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Inverse generator function for Frank copula.

        Args:
            t: Argument of the inverse Archimedean generator.
        """
        return -np.log1p(np.exp(-t) * (np.expm1(-self.theta))) / self.theta

    def generate_latent_distribution(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        return rng.logseries(1 - np.exp(-self.theta), size=n_sims).astype(np.float64)


class JoeCopula(ArchimedeanCopula):
    r"""A class to represent a Joe copula.

    The Joe copula has the cumulative distribution function:

    .. math::

        \begin{aligned}
        C(u_1, \ldots, u_d)
          &= 1 - \left\{
              1 - \prod_{i=1}^d \left[1 - (1-u_i)^\theta\right]
            \right\}^{1/\theta}
        \end{aligned}

    where :math:`\theta \geq 1` is the dependence parameter.

    The Joe copula is an Archimedean copula with generator function:

    .. math::

        \psi(t) = 1 - (1 - e^{-t})^{1/\theta}

    where :math:`\theta \geq 1` is the dependence parameter. The copula exhibits
    upper tail dependence with coefficient :math:`2 - 2^{1/\theta}`.

    Attributes:
        dimension: Number of uniform margins in the copula.
        theta: Dependence parameter of the copula.
    """

    def __init__(self, theta: float, dimension: int | None = None) -> None:
        """Initialize a Joe copula.

        Args:
            theta: Copula parameter (must be >= 1).
            dimension: Number of variables.
        """
        if theta < 1:
            raise ValueError("Theta must be in the range [1, inf)")
        self.theta = theta
        if dimension is not None:
            self.dimension = dimension

    def generator_inv(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Inverse generator function for Joe copula.

        Args:
            t: Argument of the inverse Archimedean generator.
        """
        return 1 - (1 - np.exp(-t)) ** (1 / self.theta)

    def generate_latent_distribution(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate samples from the latent distribution.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        if self.theta == 1:
            # Sibuya(1) is degenerate at one. The general sampler draws
            # Gamma(0) and is undefined at this independence boundary.
            return np.ones(n_sims)
        return _sibuya_gen(1 / self.theta, n_sims, rng)


class MM1Copula(Copula):
    r"""A multivariate max-mixture copula, denoted MM1 by Joe.

    The MM1 copula is a multivariate copula which allows for different upper tail
    dependence structures between each pair of dimensions. It can be regarded as an
    extension of the Gumbel copula to more flexible dependence.

    The cumulative distribution function of the MM1 copula is given by:

    .. math::

        \begin{aligned}
        C(u_1, \ldots, u_d)
          &= \exp\left\{
              -\left[
                  \sum_{i<j}\left\{
                      \begin{aligned}
                      &\left(\frac{(-\ln u_i)^\theta}{d-1}\right)^{\delta_{ij}} \\
                      &\quad+ \left(\frac{(-\ln u_j)^\theta}{d-1}\right)^{\delta_{ij}}
                      \end{aligned}
                  \right\}^{1/\delta_{ij}}
              \right]^{1/\theta}
            \right\}
        \end{aligned}

    for a symmetric matrix :math:`\delta_{ij} \geq 1` and
    :math:`\theta \geq 1`. The MM1 copula reduces to the Gumbel copula when all
    :math:`\delta_{ij} = 1`.

    The upper tail dependence coefficient between any pair of variables :math:`i`
    and :math:`j` in the MM1 copula is given by:

    .. math::

        \lambda_{ij} = 2 - \left(\frac{2^{1/\delta_{ij}}}{d-1} + \frac{2(d-2)}{d-1}\right)^{1/\theta}

    where :math:`\delta_{ij}` is the pairwise parameter from the delta_matrix,
    :math:`d` is the dimension of the copula, and :math:`\theta` is the overall
    mixing parameter.

    The simulation approach uses the max-mixture representation of the MM1 copula,
    detailed in Joe (2015, Chapter 6).

    References:
        Joe, H. (1997). Multivariate Models and Dependence Concepts. Chapman and Hall.
        Joe, H. (2015). Dependence Modeling with Copulas. Chapman and Hall.

    Attributes:
        delta_matrix: Pairwise delta parameters of the MM1 copula.
        dimension: Number of uniform margins in the copula.
        theta: Dependence parameter of the copula.
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
        self.dimension = len(delta_matrix)
        for i in range(1, self.dimension):
            if len(delta_matrix[i]) != self.dimension:
                raise ValueError("delta_matrix must be square")
            if min(delta_matrix[i][:i]) < 1:
                raise ValueError("delta_matrix must be greater than or equal to 1")
        self.delta_matrix = delta_matrix
        if theta < 1:
            raise ValueError("Theta must be in the range [1, inf)")
        self.theta = theta

    def _transform_to_uniform(self, unnormalised_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform max-mixture samples to uniform.

        Args:
            unnormalised_samples: Latent copula samples before transformation to uniform margins.
        """
        return np.exp(-((-np.log(unnormalised_samples)) ** (1 / self.theta)))

    def _generate_unnormalised(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate the latent multivariate samples before uniform transformation.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        n = self.dimension
        theta = self.theta
        delta_matrix = self.delta_matrix
        max_u = np.zeros((n, n_sims))
        mixing_variable = levy_stable(alpha=1 / theta, beta=1.0, size=n_sims, rng=rng) * (
            np.cos(np.pi / (2 * theta)) ** theta
        )
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
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
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
    r"""A class to represent a Galambos copula.

    The Galambos copula is an example of a multivariate extreme value copula,
    which is particularly suited for modeling upper tail dependence between
    random variables.

    The bivariate cumulative distribution function (CDF) of the Galambos copula is given by:

    .. math::

        \begin{aligned}
        C(u, v)
          &= uv\exp\!\left\{
              \left[(-\ln u)^{-\theta} + (-\ln v)^{-\theta}\right]^{-1/\theta}
            \right\}
        \end{aligned}

    Its dependence structure is characterized by a single parameter,
    :math:`\theta > 0`, which controls the strength of the upper tail dependence.

    The upper tail dependence coefficient between any pair of variables in the
    Galambos copula is given by:

    .. math::

        \lambda_U = 2^{-1/\theta}

    where :math:`\theta > 0` is the copula parameter.

    The multivariate extension to :math:`d` dimensions is a complicated expression
    given in Joe (1997, Chapter 5), but it can be simulated using the max stable / reciprocal
    Archimedean representation detailed in Mai (2018).


    References:
        Galambos, János. The Asymptotic Theory of Extreme Order Statistics. New York:
        John Wiley & Sons, 1978.
        Joe, H. (1997). Multivariate Models and Dependence Concepts. Chapman and Hall.
        Mai, Jan-Frederik. "Exact Simulation of Reciprocal Archimedean Copulas."
        Statistical Probability Letters (2018). arXiv preprint arXiv:1802.09996

    Attributes:
        dimension: Number of uniform margins in the copula.
        theta: Dependence parameter of the copula.
    """

    def __init__(self, theta: float, dimension: int | None = None) -> None:
        """Initialize a Galambos copula.

        Args:
            theta: Copula parameter (must be > 0).
            dimension: Number of variables.
        """
        if theta <= 0:
            raise ValueError("Theta must be in the range (0, inf)")
        self.theta = theta
        if dimension is not None:
            self.dimension = dimension

    def _generate_unnormalised(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Vectorised simulation from the d-dimensional Galambos copula.

        Exact algorithm based on the max stable / reciprocal Archimedean
        representation in Mai (2018).

        References:
            Mai, Jan-Frederik. "Exact Simulation of Reciprocal Archimedean Copulas."
            Statistical Probability Letters (2018). arXiv preprint arXiv:1802.09996

        Args:
            n_sims: Number of samples.
            rng: Random generator.

        Returns:
            Samples on (0, 1)^d from the Galambos copula, with one row per
            dimension and one column per simulation.
        """
        d = self.dimension
        # Independence shortcut if needed
        if self.theta < 1e-4:
            return _draw_scenarios(rng, "uniform", 0, 1, size=(d, n_sims), scenario_axis=1)
        num = gamma(d) * gamma(1.0 / self.theta)
        den = gamma(d + 1.0 / self.theta) * self.theta
        # Compute c_theta for the Galambos copula in dimension d.
        # S^{-1}(t) = c_theta * t^{-theta}.
        c_theta = (num / den) ** (-self.theta)

        inv_theta = 1.0 / self.theta
        c_th = c_theta

        # Y holds the max stable representation for all samples
        y = np.zeros((d, n_sims))

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

            # ``size`` arguments require a host integer. Transfer only this
            # scalar count; all simulation arrays remain on the device.
            active_indices = np.flatnonzero(active)
            m = int(active_indices.size)

            # Every iteration consumes coordinates of the original n_sims
            # design, even when the number of active paths has decreased.
            e = _draw_active_scenarios(
                rng, "exponential", n_sims, active_indices, scale=1.0, size=(d, m), scenario_axis=1
            )
            e_sum = e.sum(axis=0, keepdims=True)

            # Candidate contribution for Y on active rows
            val = r[active] * e / e_sum

            # Update Y on active rows with elementwise max
            y_active = y[:, active]
            np.maximum(y_active, val, out=y_active)
            y[:, active] = y_active

            # Advance the Poisson times and radii for active rows only
            t[active] += _draw_active_scenarios(rng, "exponential", n_sims, active_indices, scale=1.0, size=m)
            r[active] = c_th * (t[active] ** (-self.theta))

        # Map max stable representation to uniforms:
        # U_i = exp( - Y_i^{-1/theta} )
        u = np.exp(-(y ** (-inv_theta)))
        return u

    def generate(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
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
    r"""A class to represent a Plackett copula.

    The Plackett copula is a bivariate copula that can model both positive and
    negative dependence between two random variables. It is characterized by a
    single parameter, :math:`\delta > 0`, which controls the strength and
    direction of the dependence.

    The bivariate cumulative distribution function is:

    .. math::

        C(u, v) = \frac{S - \sqrt{S^2 - 4uv\delta(\delta-1)}}{2(\delta-1)}

    where :math:`S = 1 + (u+v)(\delta-1)` and :math:`\delta \neq 1`. When
    :math:`\delta = 1`, the copula reduces to the independence copula.

    Currently, only the bivariate case is implemented for the Plackett copula.

    References:
        Plackett, R. L. (1965). A class of bivariate distributions. Journal of the
        American Statistical Association, 60(310), 516-522.

    Attributes:
        delta: Bivariate dependence parameter of the Plackett copula.
        dimension: Number of uniform margins in the copula.
    """

    dimension: int = 2

    def __init__(self, delta: float) -> None:
        """Initialize a Plackett copula.

        Args:
            delta: Copula parameter (must be > 0).
        """
        if delta <= 0:
            raise ValueError("Delta must be in the range (0, inf)")
        self.delta = delta

    def _generate_unnormalised(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Generate samples from the Plackett copula.

        Args:
            n_sims: Number of samples.
            rng: Random generator.

        Returns:
            Samples on (0, 1)^2 from the Plackett copula, with one row per
            dimension and one column per simulation.
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
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
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


def _sibuya_gen(alpha: float, size: int | tuple[int, ...], rng: RandomGenerator) -> npt.NDArray[np.floating]:
    """Generate samples from a Sibuya distribution.

    Parameters:
        alpha (float): Parameter for the Sibuya distribution.
        size (int or tuple): Output shape.
        rng (RandomGenerator): Random number generator.

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
    r"""A class to represent a Hüsler-Reiss copula.

    The Hüsler-Reiss copula is an example of a multivariate extreme value copula,
    which is suited for modeling upper tail dependence between
    random variables and allows for a flexible specification of tail dependency
    for each bivariate pair of variables.

    The bivariate cumulative distribution function (CDF) of the Hüsler-Reiss
    copula is given by:

    .. math::

        \begin{aligned}
        C(u_i, u_j)
          &= \exp\Bigg\{
              \ln u_i \, \Phi\!\left(
                  \lambda_{ij}
                  + \frac{1}{2\lambda_{ij}}
                    \ln\!\left(\frac{-\ln u_i}{-\ln u_j}\right)
              \right) \\
          &\qquad\quad +
              \ln u_j \, \Phi\!\left(
                  \lambda_{ij}
                  + \frac{1}{2\lambda_{ij}}
                    \ln\!\left(\frac{-\ln u_j}{-\ln u_i}\right)
              \right)
            \Bigg\}
        \end{aligned}

    where :math:`\Phi` is the standard normal CDF and :math:`\lambda_{ij}` is the parameter
    controlling the dependence between the two variables.

    Its dependence structure is characterized by a matrix :math:`\lambda_{ij}`
    which controls the strength of the upper tail dependence between each pair
    of variables. Lower values in the matrix correspond to stronger dependence.
    Hüsler and Reiss (1989) parameterise the same family by the reciprocal of
    :math:`\lambda_{ij}`, so that larger values give stronger dependence there.

    The upper tail dependence coefficient between any pair of variables
    :math:`i` and :math:`j` in the Hüsler-Reiss copula is given by:

    .. math::

        \chi_{ij} = 2\left(1 - \Phi(\lambda_{ij})\right)

    where :math:`\Phi` is the standard normal CDF and :math:`\lambda_{ij}` are
    the elements of the lambda matrix.

    References:
        Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between
        independence and complete dependence. Statistics & Probability Letters,
        7(4), 283-286.

    Attributes:
        adjusted_lambda_matrix: Valid Hüsler-Reiss dependence matrix after any necessary adjustment.
        dimension: Number of uniform margins in the copula.
        is_adjusted: Whether the supplied Hüsler-Reiss matrix was adjusted.
    """

    is_adjusted: bool = False
    """Indicates whether the provided lambda matrix was adjusted to ensure validity."""
    dimension: int
    """The dimension of the copula."""
    adjusted_lambda_matrix: npt.NDArray[np.floating]
    """The adjusted lambda matrix after ensuring validity."""

    def __init__(
        self,
        lambda_matrix: npt.NDArray[np.floating] | list[list[float]],
    ) -> None:
        r"""Initialize a Hüsler-Reiss copula.

        Its dependence structure is characterized by a matrix :math:`\Lambda_{ij}`
        which controls the strength of the upper tail dependence between each pair
        of variables. Lower values in the matrix correspond to stronger dependence.

        The upper tail dependence coefficient between any pair of variables
        :math:`i` and :math:`j` in the Hüsler-Reiss copula is given by:

        .. math::

            \chi_{ij} = 2\left(1 - \Phi(\lambda_{ij})\right)

        where :math:`\Phi` is the standard normal CDF.

        The parameters :math:`\lambda_{ij}` must be non-negative, and the matrix
        must be symmetric. The diagonal elements :math:`\lambda_{ij}` must always
        be zero. Values of :math:`\lambda_{ij}` are capped at 100 to avoid
        numerical issues during simulation.

        The matrix :math:`\lambda_{ij}` must satisfy certain conditions to ensure
        it corresponds to a valid Hüsler-Reiss copula. In particular, the matrix
        must be conditionally negative definite. That is, its square must correspond
        to a valid variogram of a random field :math:`Z_j`:

        .. math::

            \lambda_{ij}^2 = \frac{1}{4}\left(
                \operatorname{Var}(Z_i) + \operatorname{Var}(Z_j)
                - 2\operatorname{Cov}(Z_i, Z_j)
            \right)

        This is checked during initialization by attempting to construct a valid
        covariance matrix for the random process :math:`Z_i` from the provided
        :math:`\lambda_{ij}` matrix.

        If the provided matrix does not satisfy these conditions, it is adjusted
        to the nearest valid matrix by modifying the eigenvalues of the corresponding
        covariance matrix. The `is_adjusted` attribute will be set to True in this
        case.

        References:
            Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between
            independence and complete dependence. Statistics & Probability Letters,
            7(4), 283-286.

        Args:
            lambda_matrix: Symmetric matrix :math:`\lambda_{ij}` determining the
                pairwise dependency between variables.
        """
        lambda_matrix = t.cast(npt.NDArray[np.floating], np.asarray(lambda_matrix))
        if lambda_matrix.ndim != 2 or lambda_matrix.shape[0] != lambda_matrix.shape[1]:  # type: ignore[union-attr]
            raise ValueError("Parameter matrix must be square")
        if lambda_matrix.min() < 0.0:  # type: ignore[union-attr]
            raise ValueError("Matrix values must be non-negative")
        if not np.allclose(lambda_matrix, lambda_matrix.T):  # type: ignore[union-attr]
            raise ValueError("Matrix must be symmetric")
        if not np.allclose(np.diag(lambda_matrix), 0.0):  # type: ignore[union-attr]
            raise ValueError("Matrix diagonal must be zero")

        # calculate the covariance matrix from the lambda matrix
        d = lambda_matrix.shape[0]  # type: ignore[union-attr]
        np.clip(lambda_matrix, a_min=0, a_max=100, out=lambda_matrix)  # type: ignore[arg-type]
        # pivot on the first variable to construct a covariance matrix
        # consistent with the variogram defined by lambda_matrix squared
        covariance_matrix = 2 * (lambda_matrix[0] ** 2 + lambda_matrix[:, 0, None] ** 2 - lambda_matrix**2)  # type: ignore[index,operator]
        covariance_sub = covariance_matrix[1:, 1:]
        vals, vecs = np.linalg.eigh(covariance_sub)
        if vals.min() < 1e-6:
            covariance_sub = vecs @ np.diag(np.clip(vals, a_min=1e-6, a_max=None)) @ vecs.T
            self.is_adjusted = True
        try:
            chol_sub = np.linalg.cholesky(covariance_sub)
            self._chol = np.zeros((d, d))
            self._chol[1:, 1:] = chol_sub
        except np.linalg.LinAlgError as e:
            raise ValueError("Could not construct a valid correlation matrix") from e
        covariance_matrix = self._chol @ self._chol.T
        self.dimension = d
        self.adjusted_lambda_matrix = (
            np.sqrt(np.diag(covariance_matrix) + np.diag(covariance_matrix)[:, None] - 2 * covariance_matrix) / 2
        )

    def _generate_unnormalised(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Exact simulation from a d-dimensional Hüsler-Reiss copula.

        See Dombry-Engelke-Oesting (2016), Algorithm 2 (spectral measure on the
        L1-sphere; numbered Algorithm 1 in arXiv:1506.04430v1).

        References:
            Dombry, C., Engelke, S., & Oesting, M. (2016). Exact simulation of max-stable
            processes. Biometrika 103, no. 2 (2016): 303-17.

        Args:
            n_sims: Number of samples to generate.
            rng: Random generator.

        Returns:
            Samples from the Hüsler-Reiss copula, with one row per dimension and
            one column per simulation.
        """
        d = self.dimension

        # Cholesky for Gaussian simulation
        chol = self._chol
        # Precompute variogram appearing in Proposition 4 / Remark 2
        h = 2 * (self.adjusted_lambda_matrix**2)

        # Z will hold the unit Fréchet max-stable vector
        z = np.zeros((n_sims, d), dtype=float)

        # Poisson process in 1/zeta with rate = d (Alg. 2 in Biometrika: Exp(N))
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
            t = _draw_active_scenarios(rng, "integers", n_sims, idx, low=0, high=d, size=na)

            # 2. Sample Gaussian W ~ N(0, sigma^2 * R) for active sims
            g = _draw_active_scenarios(rng, "standard_normal", n_sims, idx, size=(na, d), scenario_axis=0)
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
            e = _draw_active_scenarios(rng, "exponential", n_sims, idx, scale=1.0 / d, size=na)
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
        r"""Calculate the upper tail dependence matrix for the Hüsler-Reiss copula.

        The upper tail dependence coefficient between any pair of variables
        :math:`i` and :math:`j` in the Hüsler-Reiss copula is given by:

        .. math::

            \chi_{ij} = 2\left(1 - \Phi(\lambda_{ij})\right)

        where :math:`\Phi` is the standard normal CDF.

        Returns:
            npt.NDArray[np.floating]: A 2D array representing the upper tail dependence
            coefficients between each pair of variables.
        """
        lambda_matrix = self.adjusted_lambda_matrix
        tail_dependence_matrix = np.asarray(2.0 * (1.0 - scipy.stats.norm.cdf(asnumpy(lambda_matrix))))

        return tail_dependence_matrix

    @staticmethod
    def calculate_lambda_from_tail_dependence(
        tail_dependence_matrix: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        r"""Calculate the lambda matrix from a given upper tail dependence matrix.

        The upper tail dependence coefficient between any pair of variables
        :math:`i` and :math:`j` in the Hüsler-Reiss copula is given by:

        .. math::

            \chi_{ij} = 2\left(1 - \Phi(\lambda_{ij})\right)

        where :math:`\Phi` is the standard normal CDF.

        This method inverts the above relationship to compute the lambda matrix from
        the provided upper tail dependence coefficients.

        .. math::

            \lambda_{ij} = \Phi^{-1}\left(1 - \frac{\chi_{ij}}{2}\right)

        where :math:`\Phi^{-1}` is the inverse standard normal CDF.

        Args:
            tail_dependence_matrix (npt.NDArray[np.floating]): A 2D array
                representing the upper tail dependence coefficients between
                each pair of variables.

        Returns:
            npt.NDArray[np.floating]: A 2D array representing the lambda matrix
            corresponding to the given upper tail dependence coefficients.
        """
        chi_ij = tail_dependence_matrix
        lambda_matrix = np.asarray(scipy.stats.norm.ppf(1.0 - asnumpy(chi_ij) / 2.0))

        return lambda_matrix

    @classmethod
    def from_tail_dependence_matrix(cls, tail_dependence_matrix: npt.NDArray[np.floating]) -> HuslerReissCopula:
        r"""Create a Hüsler-Reiss copula from a given upper tail dependence matrix.

        The upper tail dependence coefficient between any pair of variables
        :math:`i` and :math:`j` in the Hüsler-Reiss copula is given by:

        .. math::

            \chi_{ij} = 2\left(1 - \Phi(\lambda_{ij})\right)

        where :math:`\Phi` is the standard normal CDF.

        Args:
            tail_dependence_matrix (npt.NDArray[np.floating]): A 2D array
                representing the upper tail dependence coefficients between
                each pair of variables.

        Returns:
            HuslerReissCopula: An instance of the Hüsler-Reiss copula initialized
            with the lambda matrix corresponding to the given upper tail
            dependence coefficients.
        """
        lambda_matrix = cls.calculate_lambda_from_tail_dependence(tail_dependence_matrix)
        return cls(lambda_matrix)

    def generate(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
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


class ExtremalTCopula(Copula):
    r"""A class to represent an Extremal-t copula.

    The Extremal-t copula, also known as the t-EV copula, is an example of a
    multivariate extreme value copula, which is suited for modeling upper tail
    dependence between random variables.

    The bivariate cumulative distribution function (CDF) of the Extremal-t copula is
    given by:

    .. math::

        \begin{aligned}
        C(u_i, u_j)
          &= \exp\Bigg\{
              \ln u_i \, t_{\nu+1}\!\left(
                  \sqrt{\frac{\nu+1}{1-\rho_{ij}^2}}
                  \left[
                      \left(\frac{-\ln u_i}{-\ln u_j}\right)^{1/\nu}
                      - \rho_{ij}
                  \right]
              \right) \\
          &\qquad\quad +
              \ln u_j \, t_{\nu+1}\!\left(
                  \sqrt{\frac{\nu+1}{1-\rho_{ij}^2}}
                  \left[
                      \left(\frac{-\ln u_j}{-\ln u_i}\right)^{1/\nu}
                      - \rho_{ij}
                  \right]
              \right)
            \Bigg\}.
        \end{aligned}

    Its dependence structure is characterized by a correlation matrix and a degrees
    of freedom parameter :math:`\nu > 0`, which controls the strength of the
    upper tail dependence.

    The upper tail dependence coefficient between any pair of variables :math:`i`
    and :math:`j` in the Extremal-t copula is given by:

    .. math::

        \lambda_{ij} = 2\, t_{\nu+1}\left(-\sqrt{\frac{(\nu+1)(1-\rho_{ij})}{1+\rho_{ij}}}\right)

    where :math:`t_{\nu+1}` is the CDF of a univariate t-distribution with
    :math:`\nu+1` degrees of freedom and :math:`\rho_{ij}` is the correlation
    between variables :math:`i` and :math:`j`.

    The lower tail dependence coefficient is zero for all pairs of variables.

    References:
        Nikoloulopoulos, A. K., Joe, H., & Li, H. (2009). Extreme value properties of
        multivariate t copulas. Extremes, 12(2), 129-148.

    Attributes:
        correlation_matrix: Correlation matrix specifying the elliptical or extremal-t dependence.
        dimension: Number of uniform margins in the copula.
        nu: Degrees of freedom controlling extremal-t dependence.
    """

    def __init__(
        self,
        correlation_matrix: npt.NDArray[np.floating],
        nu: float,
    ) -> None:
        """Initialize an Extremal-t copula.

        Args:
            correlation_matrix: Correlation matrix determining the pairwise
                dependency between variables. Must be positive
                semi-definite. Note that only the lower diagonal of this matrix is used.
            nu: Degrees of freedom parameter (must be > 0).
        """
        if nu <= 0:
            raise ValueError("Degrees of freedom nu must be in the range (0, inf)")
        self.correlation_matrix = np.asarray(correlation_matrix)
        # validate correlation matrix
        if self.correlation_matrix.ndim != 2 or (self.correlation_matrix.shape[0] != self.correlation_matrix.shape[1]):
            raise ValueError("Correlation matrix must be square")
        if self.correlation_matrix.min() < -1.0 or self.correlation_matrix.max() > 1.0:
            raise ValueError("Correlation matrix values must be in the range [-1, 1]")
        if not np.allclose(np.diag(self.correlation_matrix), 1.0):
            raise ValueError("Correlation matrix diagonal must be all ones")
        self.nu = nu
        self.dimension = correlation_matrix.shape[0]

    def _generate_unnormalised(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Exact simulation of the t-EV copula.

        See Dombry-Engle-Oesting (2016).

        References:
            Dombry, C., Engelke, S., & Oesting, M. (2016). Exact simulation of max-stable
            processes. Biometrika 103, no. 2 (2016): 303-17.

        Args:
            n_sims: Number of simulations on the scenario axis.
            rng: Random number generator supplying the draws.
        """
        d = self.correlation_matrix.shape[0]

        conditional_cholesky: list[npt.NDArray[np.floating]] = []
        conditional_mu: list[npt.NDArray[np.floating]] = []
        conditional_mask: list[npt.NDArray[np.bool_]] = []
        rho = self.correlation_matrix
        nu = self.nu

        for j in range(d):
            rho_j = rho[:, j]

            # Schur Complement: Sigma_cond = rho - rho_j * rho_j^T
            sigma_cond_full = rho - np.outer(rho_j, rho_j)

            # Drop j-th row/col to avoid singularity
            mask = np.ones(d, dtype=bool)
            mask[j] = False

            sigma_cond_sub = sigma_cond_full[mask][:, mask]

            # Scale by (nu + 1)
            covar_scaled = sigma_cond_sub / (nu + 1.0)

            try:
                l_sub = np.linalg.cholesky(covar_scaled)
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    "Could not compute Cholesky decomposition of "
                    "conditional covariance matrix. "
                    "Check that the correlation matrix is valid."
                ) from e

            conditional_cholesky.append(l_sub)
            conditional_mu.append(rho_j)
            conditional_mask.append(mask)

        zeta = rng.exponential(1.0, size=n_sims)
        chi_sq = rng.chisquare(df=nu + 1.0, size=(1, n_sims))

        # Gaussian: Shape (d-1, n)
        z_sub = _draw_scenarios(rng, "normal", size=(d - 1, n_sims), scenario_axis=1)
        normal_dev_sub = conditional_cholesky[0] @ z_sub

        # Scale deviations
        scale_factor = np.sqrt((nu + 1.0) / chi_sq)
        x_sub = normal_dev_sub * scale_factor
        x = np.zeros((d, n_sims))
        x[0, :] = 1.0

        # Broadcasting mu (d-1, 1) against x_sub (d-1, n)
        mu_sub_0 = conditional_mu[0][conditional_mask[0]][:, np.newaxis]
        x[conditional_mask[0], :] = mu_sub_0 + x_sub

        y = np.maximum(0, x) ** nu
        samp = y / zeta
        for j in range(1, d):
            # Boolean mask for active samples
            active_mask = np.ones(n_sims, dtype=bool)
            current_zeta = rng.exponential(1.0, size=n_sims)

            # Unpack parameters for speed
            l_sub = conditional_cholesky[j]
            mask_j = conditional_mask[j]
            # Prepare mu as column vector (d-1, 1)
            mu_sub = conditional_mu[j][mask_j][:, np.newaxis]

            while True:  # 'active_mask' check is done inside via indices
                # Find running samples: (1/zeta > samp[j])
                # Subset active arrays
                # We work with indices to avoid copying the whole big mask
                active_idx = np.flatnonzero(active_mask)

                if active_idx.size == 0:
                    break

                # Check thresholds
                thresholds = samp[j, active_idx]
                intensities = 1.0 / current_zeta[active_idx]

                still_running_mask = intensities > thresholds

                # Identify samples to process in this batch
                process_idx = active_idx[still_running_mask]

                # Samples that stopped are removed from active_mask
                finished_idx = active_idx[~still_running_mask]
                active_mask[finished_idx] = False

                n_proc = process_idx.size
                if n_proc == 0:
                    break

                # --- Generate Y anchored at j ---

                # 1. Chi-Square & Normal
                chi_sq_j = _draw_active_scenarios(
                    rng, "chisquare", n_sims, process_idx, df=nu + 1.0, size=(1, n_proc), scenario_axis=1
                )
                z_sub_j = _draw_active_scenarios(
                    rng, "normal", n_sims, process_idx, size=(d - 1, n_proc), scenario_axis=1
                )

                # 2. Cholesky & Scale
                normal_dev = l_sub @ z_sub_j
                scale_j = np.sqrt((nu + 1.0) / chi_sq_j)
                x_sub_j = normal_dev * scale_j

                # 3. Construct X_j (d, n_proc)
                x_j = np.zeros((d, n_proc))
                x_j[j, :] = 1.0
                x_j[mask_j, :] = mu_sub + x_sub_j

                # 4. Spectral Function Y
                y_j = np.maximum(0, x_j) ** nu

                # --- Acceptance Check (0...j-1) ---

                # Candidates: Y_j / zeta
                zeta_proc = current_zeta[process_idx]
                candidates = y_j / zeta_proc

                # Current values for indices 0 to j-1
                currents = samp[0:j, process_idx]
                cand_check = candidates[0:j, :]

                # Violation: any(candidate > current) along dim 0
                violation = np.any(cand_check >= currents, axis=0)

                accepted_batch_mask = ~violation

                if np.any(accepted_batch_mask):
                    # Indices in the full array
                    final_acc_idx = process_idx[accepted_batch_mask]

                    # Accepted values
                    acc_cand = candidates[:, accepted_batch_mask]
                    acc_curr = samp[:, final_acc_idx]

                    # Update Maxima (all dimensions 0..d)
                    samp[:, final_acc_idx] = np.maximum(acc_curr, acc_cand)

                # Increment Poisson Process
                current_zeta[process_idx] += _draw_active_scenarios(
                    rng, "exponential", n_sims, process_idx, scale=1.0, size=n_proc
                )

        # --- 5. Final Transform ---
        # Transpose back to (n_samples, d) for standard format
        # exp(-1/z)
        return np.exp(-1.0 / samp)

    def generate(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate samples from the Extremal-t copula.

        The simulation uses the exact algorithm from Dombry et al. (2016).

        References:
            Nikoloulopoulos, A. K., Joe, H., & Li, H. (2009). Extreme value properties of
            multivariate t copulas. Extremes, 12(2), 129-148.
            Dombry, C., Engelke, S., & Oesting, M. (2016). Exact simulation of max-stable
            processes. Biometrika 103, no. 2 (2016): 303-17.


        Args:
            n_sims: Number of simulations to generate. Uses config.n_sims if None.
            rng: Random number generator. Uses config.rng if None.

        Returns:
            ProteusVariable with StochasticScalar values representing samples from the
            Extremal-t copula.
        """
        return self._generate_base(n_sims, rng)

    @classmethod
    def from_tail_dependence_matrix(
        cls,
        tail_dependence_matrix: npt.NDArray[np.floating],
        nu: float,
    ) -> ExtremalTCopula:
        r"""Create an Extremal-t copula from a given upper tail dependence matrix.

        The upper tail dependence coefficient between any pair of variables
        :math:`i` and :math:`j` in the Extremal-t copula is given by:

        .. math::

            \chi_{ij} = 2 \, t_{\nu+1}\left(-\sqrt{
            \frac{(\nu+1)(1-\rho_{ij})}{1+\rho_{ij}}}\right)

        where :math:`t_{\nu+1}` is the CDF of a univariate t-distribution with
        :math:`\nu+1` degrees of freedom and :math:`\rho_{ij}` is the correlation
        between variables :math:`i` and :math:`j`.

        This method inverts the above relationship to compute the correlation matrix
        from the provided upper tail dependence coefficients.

        Args:
            tail_dependence_matrix (npt.NDArray[np.floating]): A 2D array
                representing the upper tail dependence coefficients between
                each pair of variables.
            nu: Degrees of freedom parameter (must be > 0).

        Returns:
            ExtremalTCopula: An instance of the Extremal-t copula initialized
            with the correlation matrix corresponding to the given upper tail
            dependence coefficients.
        """
        chi_ij = tail_dependence_matrix
        inv_t = np.asarray(scipy.stats.t.ppf(1 - asnumpy(chi_ij) / 2.0, df=nu + 1.0))
        rho_matrix = -((inv_t**2) / (nu + 1.0) - 1) / ((inv_t**2) / (nu + 1.0) + 1)
        np.fill_diagonal(rho_matrix, 1.0)
        return cls(rho_matrix, nu)


def apply_copula(
    variables: ProteusVariable[StochasticScalar] | list[StochasticScalar],
    copula_samples: ProteusVariable[StochasticScalar] | list[StochasticScalar],
) -> None:
    """Apply a reordering from a copula to a list of variables.

    Parameters:
        variables: List of StochasticScalar variables. Their coupling groups
            must be independent of one another.
        copula_samples: List of StochasticScalar samples from the copula.
            They may be coupled to one another, but none may share a coupling
            group with a variable being reordered.

    Raises:
        ValueError: If the number of dimensions or simulations differs, the
            marginal variables are coupled, or a copula sample is coupled to
            a marginal variable.
    """
    if len(variables) != len(copula_samples):
        raise ValueError("Number of variables and copula samples do not match.")
    variables_list = list(variables)
    copula_list = list(copula_samples)
    if not variables_list:
        return

    n_sims = variables_list[0].n_sims
    if any(var.n_sims != n_sims for var in variables_list) or any(sample.n_sims != n_sims for sample in copula_list):
        raise ValueError("Variables and copula samples must have the same number of simulations.")

    # Check independence in linear time, before mutating coupled variables.
    group_positions: dict[int, int] = {}
    for i, var in enumerate(variables_list):
        group_id = id(var.coupled_variable_group)
        if group_id in group_positions:
            previous = group_positions[group_id]
            raise ValueError(
                f"Cannot apply copula as the variables at positions {previous} and {i} are not independent"
            )
        group_positions[group_id] = i

    # The copula supplies the dependence ordering and must not depend on any
    # marginal that this operation might reorder. Copula samples may remain
    # coupled to *one another*, as they normally are for a generated copula.
    for i, sample in enumerate(copula_list):
        marginal_position = group_positions.get(id(sample.coupled_variable_group))
        if marginal_position is not None:
            raise ValueError(
                f"Copula sample at position {i} is coupled to marginal variable at position {marginal_position}"
            )

    if len(variables_list) == 1:
        return

    # Align the first variable's sorted observations to the first copula margin.
    # Keep ranks and permutations on the active array backend. Constructing a
    # NumPy stack here copies every GPU margin to the host (or raises on CuPy).
    first_order = np.argsort(variables_list[0].values, kind="stable")
    first_ranks = np.empty(n_sims, dtype=first_order.dtype)
    first_ranks[first_order] = np.arange(n_sims)
    copula_first_order = np.argsort(copula_list[0].values, kind="stable")
    sample_positions = copula_first_order[first_ranks]
    ranks = np.empty(n_sims, dtype=first_order.dtype)
    rank_values = np.arange(n_sims)

    # Process one margin at a time instead of allocating (dimension, n_sims)
    # matrices of ranks and sort indices. The first variable is unchanged.
    for var, sample in zip(variables_list[1:], copula_list[1:], strict=False):
        sample_order = np.argsort(sample.values, kind="stable")
        ranks[sample_order] = rank_values
        variable_order = np.argsort(var.values, kind="stable")
        re_ordering = variable_order[ranks[sample_positions]]
        for coupled in var.coupled_variable_group:
            coupled._reorder_sims(re_ordering)  # type: ignore[misc]

    # Merge coupling groups only after all permutations have been applied.
    first_group = variables_list[0].coupled_variable_group
    for var in variables_list[1:]:
        first_group.merge(var.coupled_variable_group)


_VineArray = npt.NDArray[np.float64]

# Conditional quantiles without a closed form are found by Newton iteration.
# Quadratic convergence means a step below the tolerance leaves an error far
# below double precision. Most points converge within a few steps and only
# those still unconverged are iterated further; the limit guards against
# pathological inputs.
_INVERSION_MAX_ITERATIONS = 100
_NEWTON_TOLERANCE = 1e-12
_BISECTION_TOLERANCE = 1e-14
_LOGIT_LOWER = math.log(np.finfo(float).tiny)
_LOGIT_UPPER = -math.log(np.finfo(float).eps / 2)


def _open_unit(value: _VineArray) -> _VineArray:
    """Keep probabilities inside the unit interval for conditional transforms.

    Args:
        value: Uniform probabilities.
    """
    return np.clip(value, np.finfo(float).tiny, 1.0 - np.finfo(float).eps / 2)


def _log_expm1(value: _VineArray) -> _VineArray:
    """Evaluate log(expm1(x)) for non-negative x without overflow.

    Args:
        value: Non-negative arguments.
    """
    return value + np.log(-np.expm1(-value))


_PairFamily = t.Literal[
    "independence",
    "gaussian",
    "students_t",
    "clayton",
    "gumbel",
    "frank",
    "joe",
    "galambos",
    "plackett",
    "husler_reiss",
    "extremal_t",
    "mm1",
]
_PairRotation = t.Literal[0, 90, 180, 270]


def _student_t_ppf(probability: _VineArray, dof: float) -> _VineArray:
    """Evaluate Student's t quantiles on the active array backend.

    Args:
        probability: Probabilities in the open unit interval.
        dof: Positive degrees of freedom.
    """
    if np.__name__ == "numpy":
        return t.cast(_VineArray, special.stdtrit(dof, probability))
    tail = 2 * np.minimum(probability, 1 - probability)
    z = special.betaincinv(dof / 2, 0.5, tail)
    return np.sign(probability - 0.5) * np.sqrt(dof * ((1 - z) / z))


@dataclass(frozen=True)
class PairCopula:
    """Bivariate copula specifying one edge of a simplified vine.

    Each edge can have its own family, dependence parameter and rotation.
    The rotation convention is C90(u, v) = v - C(1-u, v) and
    C270(u, v) = u - C(u, 1-v).

    References:
        Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). Pair-copula
        constructions of multiple dependence. Insurance: Mathematics and
        Economics, 44(2), 182-198.
        Czado, C. (2019). Analyzing Dependent Data with Vine Copulas: A Practical
        Guide With R. Lecture Notes in Statistics 222. Springer.
        Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between
        independence and complete dependence. Statistics & Probability Letters,
        7(4), 283-286.
        Joe, H. (2015). Dependence Modeling with Copulas. Chapman and Hall.
        Johnson, M. E. (1987). Multivariate Statistical Simulation. Wiley.
        Nikoloulopoulos, A. K., Joe, H., & Li, H. (2009). Extreme value properties of
        multivariate t copulas. Extremes, 12(2), 129-148.

    Attributes:
        family: Bivariate copula family. Named constructors are available for every family.
        parameter: Primary dependence parameter in the chosen family's parameterisation.
        parameter2: Degrees of freedom for Student's t and extremal-t, or delta for MM1.
            Unused for single-parameter families.
        rotation: 0, 90, 180 or 270 degrees. A 180-degree rotation interchanges
            upper and lower tail dependence; 90 and 270 allow negative association.
    """

    family: _PairFamily = "independence"
    parameter: float = 0.0
    rotation: _PairRotation = 0
    parameter2: float | None = None

    def __post_init__(self) -> None:
        """Validate the family, dependence parameter and rotation."""
        theta = self.parameter
        if not math.isfinite(theta):
            raise ValueError("Pair copula parameter must be finite")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("Pair copula rotation must be 0, 90, 180 or 270")
        second = self.parameter2
        if self.family in ("students_t", "extremal_t", "mm1"):
            if second is None or not math.isfinite(second) or second <= 0:
                raise ValueError("Pair copula requires a finite positive second parameter")
        elif second is not None:
            raise ValueError("This pair copula does not take a second parameter")
        if self.family in ("gaussian", "students_t", "extremal_t"):
            valid = -1 < theta < 1
        elif self.family in ("clayton", "frank"):
            valid = theta >= 0
        elif self.family in ("gumbel", "joe"):
            valid = theta >= 1
        elif self.family in ("galambos", "plackett", "husler_reiss"):
            valid = theta > 0
        elif self.family == "mm1":
            valid = second is not None and theta >= 1 and second >= 1 and math.isfinite(theta * second)
        elif self.family == "independence":
            valid = theta == 0
        else:
            raise ValueError(f"Unknown pair copula family: {self.family}")
        if not valid:
            raise ValueError(f"Invalid parameter for {self.family} pair copula: {theta}")

    @classmethod
    def Independence(cls) -> PairCopula:  # noqa: N802
        """Create an independent pair."""
        return cls()

    @classmethod
    def Gaussian(cls, rho: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Gaussian pair.

        Args:
            rho: Linear correlation, strictly between -1 and 1.
            rotation: Pair copula rotation in degrees.
        """
        return cls("gaussian", rho, rotation)

    @classmethod
    def StudentsT(cls, rho: float, dof: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Student's t pair.

        Args:
            rho: Linear correlation, strictly between -1 and 1.
            dof: Positive degrees of freedom.
            rotation: Pair copula rotation in degrees.
        """
        return cls("students_t", rho, rotation, dof)

    @classmethod
    def Clayton(cls, theta: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Clayton pair.

        Args:
            theta: Nonnegative dependence parameter.
            rotation: Pair copula rotation in degrees.
        """
        return cls("clayton", theta, rotation)

    @classmethod
    def Gumbel(cls, theta: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Gumbel pair.

        Args:
            theta: Dependence parameter, at least one.
            rotation: Pair copula rotation in degrees.
        """
        return cls("gumbel", theta, rotation)

    @classmethod
    def Frank(cls, theta: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Frank pair.

        Args:
            theta: Nonnegative dependence parameter.
            rotation: Pair copula rotation in degrees.
        """
        return cls("frank", theta, rotation)

    @classmethod
    def Joe(cls, theta: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Joe pair.

        Args:
            theta: Dependence parameter, at least one.
            rotation: Pair copula rotation in degrees.
        """
        return cls("joe", theta, rotation)

    @classmethod
    def Galambos(cls, theta: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Galambos pair.

        Args:
            theta: Positive dependence parameter.
            rotation: Pair copula rotation in degrees.
        """
        return cls("galambos", theta, rotation)

    @classmethod
    def Plackett(cls, delta: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Plackett pair.

        Args:
            delta: Positive odds-ratio parameter; one gives independence.
            rotation: Pair copula rotation in degrees.
        """
        return cls("plackett", delta, rotation)

    @classmethod
    def HuslerReiss(cls, lam: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create a Hüsler-Reiss pair.

        The copula is C(u, v) = exp{-a Φ(λ + log(a/b)/(2λ)) - b Φ(λ + log(b/a)/(2λ))}
        with a = -log(u) and b = -log(v), as for HuslerReissCopula. Smaller
        values of λ give stronger dependence: the upper tail dependence
        coefficient is 2(1 - Φ(λ)), approaching comonotonicity as λ tends to
        zero and independence as λ grows. Hüsler and Reiss (1989) parameterise
        the same family by the reciprocal of λ.

        Args:
            lam: Positive Hüsler-Reiss parameter λ; smaller values give stronger dependence.
            rotation: Pair copula rotation in degrees.
        """
        return cls("husler_reiss", lam, rotation)

    @classmethod
    def ExtremalT(cls, rho: float, nu: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create an extremal-t pair.

        Args:
            rho: Correlation parameter, strictly between -1 and 1.
            nu: Positive degrees of freedom.
            rotation: Pair copula rotation in degrees.
        """
        return cls("extremal_t", rho, rotation, nu)

    @classmethod
    def MM1(cls, theta: float, delta: float, *, rotation: _PairRotation = 0) -> PairCopula:  # noqa: N802
        """Create an MM1 pair, equivalent to Gumbel with parameter theta times delta.

        Args:
            theta: Mixing parameter, at least one.
            delta: Pair dependence parameter, at least one.
            rotation: Pair copula rotation in degrees.
        """
        return cls("mm1", theta, rotation, delta)

    @property
    def _independent(self) -> bool:
        """Whether the pair copula is at its independence limit."""
        if self.family == "mm1":
            return self.parameter == 1 and self.parameter2 == 1
        if self.family in ("gumbel", "joe", "plackett"):
            return self.parameter == 1
        return self.family in ("independence", "gaussian", "clayton", "frank") and self.parameter == 0

    def _reflections(self, first: bool) -> tuple[bool, bool]:
        """Return reflection flags for the target and conditioning margin.

        Args:
            first: Whether the first coordinate is the target.
        """
        reflect_first = self.rotation in (90, 180)
        reflect_second = self.rotation in (180, 270)
        return (reflect_first, reflect_second) if first else (reflect_second, reflect_first)

    def _conditional(self, value: _VineArray, given: _VineArray, *, first: bool) -> _VineArray:
        """Evaluate the conditional CDF of a margin given the other margin.

        Args:
            value: Target uniform probabilities.
            given: Conditioning uniform probabilities.
            first: Whether the first coordinate is the target.
        """
        if self._independent:
            return value
        reflect_target, reflect_given = self._reflections(first)
        u = _open_unit(1 - value if reflect_target else value)
        v = _open_unit(1 - given if reflect_given else given)
        h = self._conditional_unrotated(u, v)
        return _open_unit(1 - h if reflect_target else h)

    def _conditional_unrotated(self, u: _VineArray, v: _VineArray) -> _VineArray:
        """Evaluate the unrotated conditional distribution of u given v.

        Args:
            u: Target uniform probabilities.
            v: Conditioning uniform probabilities.
        """
        theta = self.parameter
        if self.family == "gaussian":
            h = special.ndtr((special.ndtri(u) - theta * special.ndtri(v)) / math.sqrt((1 - theta) * (1 + theta)))
        elif self.family == "clayton":
            log_term = _log_expm1(-theta * np.log(u)) + theta * np.log(v)
            h = np.exp(-(1 + 1 / theta) * np.logaddexp(0, log_term))
        elif self.family == "students_t":
            dof = self._second_parameter()
            x, y = _student_t_ppf(u, dof), _student_t_ppf(v, dof)
            scale = np.hypot(math.sqrt(dof), y) * math.sqrt((1 - theta) * (1 + theta) / (dof + 1))
            h = _student_t_cdf((x - theta * y) / scale, dof + 1)
        elif self.family in ("gumbel", "mm1"):
            if self.family == "mm1":
                theta *= self._second_parameter()
            a, b = -np.log(u), -np.log(v)
            delta = np.logaddexp(0, theta * (np.log(a) - np.log(b))) / theta
            h = np.exp(-b * np.expm1(delta) - (theta - 1) * delta)
        elif self.family == "joe":
            log_a = theta * np.log1p(-u)
            log_b = theta * np.log1p(-v)
            log_one_minus_a = np.log(-np.expm1(log_a))
            log_ratio = np.logaddexp(log_a - log_b, log_one_minus_a)
            h = np.exp(log_one_minus_a - (1 - 1 / theta) * log_ratio)
        elif self.family == "plackett":
            # D = (S-2*delta*u)^2 + 4*delta*u*(1-u).
            # Rationalise the lower-tail expression to avoid cancellation.
            diff = (1 - v) + theta * v - (1 + theta) * u
            term = 4 * theta * u * (1 - u)
            root = np.hypot(diff, np.sqrt(term))
            h = np.where(diff > 0, (term / root) / (2 * (root + diff)), (1 - diff / root) / 2)
        elif self.family in ("galambos", "husler_reiss", "extremal_t"):
            log_scale, _, partial_b, _ = self._extreme_value_terms(u, v)
            h = np.exp(log_scale) * partial_b
        else:  # Frank
            log_num = -theta * v + np.log(-np.expm1(-theta * u))
            log_den = np.logaddexp(
                -theta * u + np.log(-np.expm1(-theta * v)),
                -theta * v + np.log(-np.expm1(-theta * (1 - v))),
            )
            h = np.exp(log_num - log_den)
        return h

    def _extreme_value_terms(self, u: _VineArray, v: _VineArray) -> tuple[_VineArray, ...]:
        """Evaluate the exponent-function terms of an extreme-value pair copula.

        Writing C(u, v) = exp(-V(a, b)) with a = -log(u) and b = -log(v), the
        conditional CDF of u given v is exp(b - V) V_b and the copula density
        is exp(b - V) (V_a V_b - V_ab) / u.

        Args:
            u: Target uniform probabilities.
            v: Conditioning uniform probabilities.

        Returns:
            The terms b - V, V_a, V_b and -V_ab.
        """
        theta = self.parameter
        a, b = -np.log(u), -np.log(v)
        log_ratio = np.log(a) - np.log(b)
        if self.family == "galambos":
            # V = a + b - m with m = (a^-theta + b^-theta)^(-1/theta).
            log_m_over_a = -np.logaddexp(0, theta * log_ratio) / theta
            log_m_over_b = -np.logaddexp(0, -theta * log_ratio) / theta
            log_scale = a * np.expm1(log_m_over_a)
            partial_a = -np.expm1((theta + 1) * log_m_over_a)
            partial_b = -np.expm1((theta + 1) * log_m_over_b)
            cross = (theta + 1) * np.exp(theta * log_m_over_a + (theta + 1) * log_m_over_b - np.log(a))
        elif self.family == "husler_reiss":
            z = log_ratio / (2 * theta)
            partial_a = special.ndtr(theta + z)
            partial_b = special.ndtr(theta - z)
            log_scale = np.minimum(0, -a * partial_a + b * special.ndtr(z - theta))
            cross = np.exp(-((theta - z) ** 2) / 2) / (2 * theta * math.sqrt(2 * math.pi) * a)
        else:  # Extremal-t
            nu = self._second_parameter()
            scale = math.sqrt((nu + 1) / ((1 - theta) * (1 + theta)))
            b_over_a = np.exp(np.minimum(-log_ratio / nu, 700))
            z1 = scale * (np.exp(np.minimum(log_ratio / nu, 700)) - theta)
            z2 = scale * (b_over_a - theta)
            partial_a = _student_t_cdf(z1, nu + 1)
            # One tail evaluation gives V_b and its complement, each accurately.
            tail = _student_t_tail(z2, nu + 1)
            partial_b = np.where(z2 >= 0, 1 - tail, tail)
            log_scale = np.minimum(0, -a * partial_a + b * np.where(z2 >= 0, tail, 1 - tail))
            cross = _student_t_pdf(z2, nu + 1) * scale * b_over_a / (nu * a)
        return log_scale, partial_a, partial_b, cross

    def _conditional_and_density(self, u: _VineArray, v: _VineArray) -> tuple[_VineArray, _VineArray]:
        """Evaluate the unrotated conditional CDF and copula density together.

        Args:
            u: Target uniform probabilities.
            v: Conditioning uniform probabilities.
        """
        log_scale, partial_a, partial_b, cross = self._extreme_value_terms(u, v)
        scale = np.exp(log_scale)
        return scale * partial_b, scale * (partial_a * partial_b + cross) / u

    def _solve_conditional(self, probability: _VineArray, given: _VineArray) -> _VineArray:
        """Invert the unrotated conditional CDF by safeguarded Newton iteration.

        Newton's method is applied to logit(h) - logit(p) as a function of
        logit(u). In both tails this is close to linear, so convergence is fast
        and relative resolution is retained. Steps that leave the bracket known
        to contain the root, or that have no usable slope, are replaced by
        bisection. Points are frozen once converged, so later iterations only
        evaluate the copula where a root is still being refined.

        Args:
            probability: Conditional uniform probabilities.
            given: Conditioning uniform probabilities.
        """
        p = probability.ravel()
        v = given.ravel()
        target = special.logit(p)
        x = np.clip(target, _LOGIT_LOWER, _LOGIT_UPPER)
        lo = np.full_like(x, _LOGIT_LOWER)
        hi = np.full_like(x, _LOGIT_UPPER)
        active = np.arange(x.size)
        for _ in range(_INVERSION_MAX_ITERATIONS):
            x_active, p_active = x[active], p[active]
            u = special.expit(x_active)
            h, density = self._conditional_and_density(u, v[active])
            below = h < p_active
            lo_active = np.where(below, x_active, lo[active])
            hi_active = np.where(below, hi[active], x_active)
            spread = h * (1 - h)
            usable = spread > 0
            slope = density * u * special.expit(-x_active) / np.where(usable, spread, 1.0)
            usable = usable & np.isfinite(slope) & (slope > 0)
            newton = x_active - (special.logit(h) - target[active]) / np.where(usable, slope, 1.0)
            accepted = usable & (newton >= lo_active) & (newton <= hi_active)
            update = np.where(accepted, newton, (lo_active + hi_active) / 2)
            # Near one, h only resolves steps of one unit in the last place, so
            # a residual at that resolution is also accepted as convergence.
            # Requiring a bracket on both sides keeps the result at the
            # crossing point where h is flat and rounds to p over a wide range.
            resolved = (
                (np.abs(h - p_active) <= np.finfo(float).eps * p_active)
                & (lo_active > _LOGIT_LOWER)
                & (hi_active < _LOGIT_UPPER)
            )
            converged = resolved | np.where(
                accepted,
                np.abs(update - x_active) <= _NEWTON_TOLERANCE,
                hi_active - lo_active <= _BISECTION_TOLERANCE,
            )
            x[active] = np.where(resolved, x_active, update)
            lo[active] = lo_active
            hi[active] = hi_active
            active = active[~converged]
            if active.size == 0:
                break
        return special.expit(x).reshape(probability.shape)

    def _joe_inverse(self, probability: _VineArray, given: _VineArray) -> _VineArray:
        """Invert the unrotated Joe conditional CDF by monotone Newton iteration.

        With A = (1-u)^theta, B = (1-v)^theta and c = (1-B)/B, the conditional
        CDF is (1-A)(1+cA)^(-k), where k = 1 - 1/theta. In s = log(A), the
        equation f(s) = log(1 - e^s) - k log1p(c e^s) - log(p) = 0 has a
        concave, decreasing left side. A Newton step from any point therefore
        lands at or above the root, and iteration from there decreases
        monotonically to it. Each of the first two terms is at most zero, so
        dropping either one gives an upper bound on the root. When A is close
        to one, -s is approximately p (1+c)^k, and a Newton step from that
        approximation gives a third, much closer, upper bound. The iteration
        starts from the smallest bound. Working in s keeps relative precision
        in u near zero, where -s is small, and in 1-u near one, where s is
        large and negative.

        Args:
            probability: Conditional uniform probabilities.
            given: Conditioning uniform probabilities.
        """
        theta = self.parameter
        k = 1 - 1 / theta
        log_p = np.log(probability).ravel()
        log_c = _log_expm1(-theta * np.log1p(-given)).ravel()

        def newton_step(s: _VineArray, log_p: _VineArray, log_c: _VineArray) -> _VineArray:
            """Return f(s)/f'(s) for the Joe quantile equation.

            Args:
                s: Logarithms of A = (1-u)^theta.
                log_p: Logarithms of the conditional probabilities.
                log_c: Logarithms of c = (1-B)/B.
            """
            # Share exponentials between f and f': log1p(e^z) and expit(z) are
            # both formed from exp(-|z|). log(1 - e^s) switches from log(-expm1)
            # to log1p(-exp) below -log(2) to keep relative precision (Mächler,
            # 2012, Accurately Computing log(1 - exp(-|a|)), Rmpfr vignette).
            z = s + log_c
            exp_s, expm1_s, tail = np.exp(s), np.expm1(s), np.exp(-np.abs(z))
            log_one_minus = np.where(s > -math.log(2), np.log(-expm1_s), np.log1p(-exp_s))
            residual = log_one_minus - k * (np.maximum(z, 0) + np.log1p(tail)) - log_p
            logistic = np.where(z >= 0, 1, tail) / (1 + tail)
            return residual / (exp_s / expm1_s - k * logistic)

        near_one = -np.exp(np.minimum(log_p + k * np.logaddexp(0, log_c), 0))
        with errstate(divide="ignore", invalid="ignore", over="ignore"):
            refined = near_one - newton_step(near_one, log_p, log_c)
        flat = np.minimum(np.log1p(-probability.ravel()), _log_expm1(-log_p / k) - log_c)
        flat = np.where(np.isfinite(refined) & (refined < 0), np.minimum(flat, refined), flat)
        active = np.arange(flat.size)
        for _ in range(_INVERSION_MAX_ITERATIONS):
            s_active = flat[active]
            step = newton_step(s_active, log_p[active], log_c[active])
            flat[active] = s_active - step
            active = active[np.abs(step) > _NEWTON_TOLERANCE * np.abs(s_active)]
            if active.size == 0:
                break
        return -np.expm1(flat / theta).reshape(probability.shape)

    def _second_parameter(self) -> float:
        """Return the validated second parameter of a two-parameter family."""
        if self.parameter2 is None:
            raise ValueError("Pair copula requires a second parameter")
        return self.parameter2

    def _inverse(self, probability: _VineArray, given: _VineArray, *, first: bool) -> _VineArray:
        """Invert the conditional CDF of a margin given the other margin.

        Args:
            probability: Conditional uniform probabilities.
            given: Conditioning uniform probabilities.
            first: Whether the first coordinate is the target.
        """
        if self._independent:
            return probability
        reflect_target, reflect_given = self._reflections(first)
        p = _open_unit(1 - probability if reflect_target else probability)
        v = _open_unit(1 - given if reflect_given else given)
        theta = self.parameter
        if self.family == "gaussian":
            u = special.ndtr(theta * special.ndtri(v) + math.sqrt((1 - theta) * (1 + theta)) * special.ndtri(p))
        elif self.family == "clayton":
            log_term = _log_expm1(-(theta / (1 + theta)) * np.log(p)) - theta * np.log(v)
            u = np.exp(-np.logaddexp(0, log_term) / theta)
        elif self.family == "students_t":
            dof = self._second_parameter()
            y = _student_t_ppf(v, dof)
            scale = np.hypot(math.sqrt(dof), y) * math.sqrt((1 - theta) * (1 + theta) / (dof + 1))
            u = _student_t_cdf(theta * y + scale * _student_t_ppf(p, dof + 1), dof)
        elif self.family in ("gumbel", "mm1"):
            if self.family == "mm1":
                theta *= self._second_parameter()
            # Solve b * expm1(delta) + (theta-1)*delta = -log(p).
            # Both terms are increasing; either term alone bounds the root.
            # The left side is also convex, so Newton iteration from this
            # upper bound decreases monotonically to the root.
            # Converged points are frozen so later iterations only refine the rest.
            b, target = -np.log(v).ravel(), -np.log(p).ravel()
            delta = np.minimum(np.log1p(target / b), target / (theta - 1))
            active = np.arange(delta.size)
            for _ in range(_INVERSION_MAX_ITERATIONS):
                b_active, delta_active = b[active], delta[active]
                step = (b_active * np.expm1(delta_active) + (theta - 1) * delta_active - target[active]) / (
                    b_active * np.exp(delta_active) + theta - 1
                )
                delta[active] = delta_active - step
                active = active[np.abs(step) > _NEWTON_TOLERANCE * delta_active]
                if active.size == 0:
                    break
            log_a = (np.log(b) + _log_expm1(theta * delta) / theta).reshape(p.shape)
            u = np.exp(-np.exp(log_a))
        elif self.family == "joe":
            u = self._joe_inverse(p, v)
        elif self.family == "frank":
            log_den = np.logaddexp(-theta * v + np.log1p(-p), -theta + np.log(p))
            log_ratio = np.log(p) + math.log(-math.expm1(-theta)) - log_den
            u = np.logaddexp(0, log_ratio) / theta
        elif self.family == "plackett":
            # Quadratic solution of Johnson (1987). The two roots of the
            # quadratic are rationalised differently so that neither branch
            # subtracts nearly equal terms.
            a = p * (1 - p)
            q = 1 - 2 * p
            s = 1 - v + theta * v
            c = 2 * a * (v * theta**2 + 1 - v) + theta * (1 - 2 * a)
            d = math.sqrt(theta) * np.sqrt(theta + 4 * a * v * (1 - v) * (1 - theta) ** 2)
            u = np.where(q >= 0, 2 * a * s**2 / (c + q * d), (c - q * d) / (2 * (theta + a * (theta - 1) ** 2)))
        else:
            u = self._solve_conditional(p, v)
        return _open_unit(1 - u if reflect_target else u)


@dataclass(frozen=True)
class VineEdge:
    """An ordered pair copula and its conditioning variables in a regular vine.

    Variable indices are zero-based. The order of the two variables matters
    for rotated pair copulas; the order of the conditioning variables does not.

    The two variables form the conditioned set and the conditioning variables
    the conditioning set of the edge, in the terminology of Bedford and Cooke
    (2002).

    References:
        Bedford, T., & Cooke, R. M. (2002). Vines - a new graphical model for
        dependent random variables. The Annals of Statistics, 30(4), 1031-1068.

    Attributes:
        first: First variable index.
        second: Second variable index.
        pair: Pair copula connecting the variables.
        conditioning: Indices of the conditioning variables, empty in tree one.
    """

    first: int
    second: int
    pair: PairCopula
    conditioning: tuple[int, ...] = ()

    @property
    def _nodes(self) -> tuple[frozenset[int], frozenset[int]]:
        """The complete variable sets of the two nodes in this tree."""
        given = frozenset(self.conditioning)
        return given | {self.first}, given | {self.second}


_VineKey = tuple[int, frozenset[int]]
_VineStep = tuple[PairCopula, bool, bool, _VineKey, _VineKey, _VineKey]
_VineStage = tuple[_VineKey, tuple[_VineStep, ...]]


def _validate_vine_trees(trees: tuple[tuple[VineEdge, ...], ...], dimension: int) -> None:
    """Validate regular-vine trees and the proximity condition.

    Args:
        trees: Edges in increasing tree order, possibly truncated.
        dimension: Number of margins.
    """
    # Each node in tree k is an edge in tree k-1, represented by its complete
    # variable set and the two original endpoint sets.
    previous: dict[frozenset[int], frozenset[frozenset[int]]] = {frozenset({i}): frozenset() for i in range(dimension)}
    for level, tree in enumerate(trees):
        adjacency: dict[frozenset[int], set[frozenset[int]]] = {node: set() for node in previous}
        following: dict[frozenset[int], frozenset[frozenset[int]]] = {}
        for edge in tree:
            given = frozenset(edge.conditioning)
            if len(edge.conditioning) != level or len(given) != level:
                raise ValueError("Each vine edge must have one conditioning variable per preceding tree")
            if edge.first == edge.second or edge.first in given or edge.second in given:
                raise ValueError("Conditioned variables must be distinct and outside the conditioning set")
            if not isinstance(edge.pair, PairCopula):
                raise TypeError("Every vine edge must contain a PairCopula")
            left, right = edge._nodes
            if left not in previous or right not in previous:
                raise ValueError("Vine edge endpoints must be nodes of the preceding tree")
            if level and not previous[left].intersection(previous[right]):
                raise ValueError("Vine edges must satisfy the proximity condition")
            combined = left | right
            if right in adjacency[left] or combined in following:
                raise ValueError("Duplicate vine edge or complete variable set")
            adjacency[left].add(right)
            adjacency[right].add(left)
            following[combined] = frozenset((left, right))
        visited: set[frozenset[int]] = set()
        stack = [next(iter(previous))]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(adjacency[node] - visited)
        if len(visited) != len(previous) or len(tree) != len(previous) - 1:
            raise ValueError("Each vine tree must be connected and acyclic")
        previous = following


def _vine_elimination_order(
    trees: tuple[tuple[VineEdge, ...], ...], dimension: int
) -> list[tuple[int, tuple[VineEdge, ...]]]:
    """Find a sampling order by removing leaves from the highest retained tree.

    Args:
        trees: Validated regular-vine trees.
        dimension: Number of margins.

    Returns:
        Margins in sampling order with the incident edges in increasing tree order.
    """
    remaining = set(range(dimension))
    active = [list(tree) for tree in trees]
    removed: list[tuple[int, tuple[VineEdge, ...]]] = []
    while len(remaining) > 1:
        highest = next(tree for tree in reversed(active) if tree)
        degree: dict[frozenset[int], int] = {}
        for edge in highest:
            for node in edge._nodes:
                degree[node] = degree.get(node, 0) + 1
        candidates: set[int] = set()
        for edge in highest:
            left, right = edge._nodes
            if degree[left] == 1:
                candidates.add(edge.first)
            if degree[right] == 1:
                candidates.add(edge.second)
        # Prefer the original reverse variable order whenever it is valid.
        # A leaf's unique variable must be an endpoint of exactly one edge
        # at every retained level of the reduced vine.
        variable = max(candidates)
        incident: list[VineEdge] = []
        for tree in active:
            matching = [edge for edge in tree if variable in (edge.first, edge.second) or variable in edge.conditioning]
            if not tree:
                continue
            if len(matching) != 1 or variable not in (matching[0].first, matching[0].second):
                raise ValueError("Vine tree does not admit a valid leaf elimination")
            edge = matching[0]
            incident.append(edge)
            tree.remove(edge)
        removed.append((variable, tuple(incident)))
        remaining.remove(variable)
    removed.append((next(iter(remaining)), ()))
    removed.reverse()
    return removed


def _compile_vine(
    trees: tuple[tuple[VineEdge, ...], ...], dimension: int
) -> tuple[tuple[_VineStage, ...], dict[_VineKey, int]]:
    """Compile conditional transformations before any simulation draws.

    Args:
        trees: Validated vine trees.
        dimension: Number of margins.

    Returns:
        Sampling stages and the number of uses of each conditional probability.
    """
    definitions: dict[_VineKey, tuple[PairCopula, bool, _VineKey, _VineKey]] = {}
    for tree in trees:
        for edge in tree:
            given = frozenset(edge.conditioning)
            for target, other, first in ((edge.first, edge.second, True), (edge.second, edge.first, False)):
                definitions[target, given | {other}] = (edge.pair, first, (target, given), (other, given))

    stages: list[_VineStage] = []
    available: set[_VineKey] = set()
    uses: dict[_VineKey, int] = {}

    def ensure(key: _VineKey, steps: list[_VineStep]) -> None:
        """Add the forward transformations needed to obtain a conditional probability.

        Args:
            key: Target variable and conditioning set.
            steps: Operations for the current sampling stage.
        """
        if key in available:
            return
        if key not in definitions:
            raise ValueError("Vine does not define a required conditional distribution")
        pair, first, source, given = definitions[key]
        ensure(source, steps)
        ensure(given, steps)
        steps.append((pair, first, False, source, given, key))
        available.add(key)

    for variable, incident in _vine_elimination_order(trees, dimension):
        steps: list[_VineStep] = []
        if incident:
            edge = incident[-1]
            other = edge.second if variable == edge.first else edge.first
            draw_key = (variable, frozenset(edge.conditioning) | {other})
        else:
            draw_key = (variable, frozenset())
        available.add(draw_key)
        current = draw_key
        for edge in reversed(incident):
            first = variable == edge.first
            other = edge.second if first else edge.first
            conditioning = frozenset(edge.conditioning)
            given = (other, conditioning)
            output = (variable, conditioning)
            if current != (variable, conditioning | {other}):
                raise ValueError("Vine does not define a nested sequence of conditional distributions")
            ensure(given, steps)
            steps.append((edge.pair, first, True, current, given, output))
            available.add(output)
            current = output
        if current[1]:
            raise ValueError("Vine sampling stage did not reach the unconditional margin")
        # The last value of each stage is also copied into the output matrix.
        uses[current] = uses.get(current, 0) + 1
        for _, _, _, source, given, _ in steps:
            uses[source] = uses.get(source, 0) + 1
            uses[given] = uses.get(given, 0) + 1
        stages.append((draw_key, tuple(steps)))
    return tuple(stages), uses


class VineCopula(Copula):
    """Simplified regular vine (R-vine), including canonical and drawable vines.

    C-vines and D-vines can be specified compactly with rows of pair copulas.
    In a C-vine the edge at row k, column i connects k to k+i+1,
    conditional on variables 0 through k-1. In a D-vine it connects i to
    i+k+1, conditional on the intervening variables. General regular vines
    use rows of VineEdge objects specifying the two variables and their
    conditioning set. All variable indices are zero-based.

    The first tree contains d-1 edges and each subsequent tree contains one
    fewer. Higher trees may be omitted, representing conditional independence.
    Conditional copula parameters are constant across conditioning values
    (the simplifying assumption).

    Samples are generated by the inverse Rosenblatt transform, applying the
    conditional quantile functions of the pair copulas tree by tree, as
    described for regular vines by Dissmann et al. (2013) and Czado (2019).

    References:
        Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). Pair-copula
        constructions of multiple dependence. Insurance: Mathematics and
        Economics, 44(2), 182-198.
        Bedford, T., & Cooke, R. M. (2001). Probability density decomposition for
        conditionally dependent random variables modeled by vines. Annals of
        Mathematics and Artificial Intelligence, 32, 245-268.
        Bedford, T., & Cooke, R. M. (2002). Vines - a new graphical model for
        dependent random variables. The Annals of Statistics, 30(4), 1031-1068.
        Czado, C. (2019). Analyzing Dependent Data with Vine Copulas: A Practical
        Guide With R. Lecture Notes in Statistics 222. Springer.
        Dissmann, J., Brechmann, E. C., Czado, C., & Kurowicka, D. (2013). Selecting
        and estimating regular vine copulae and application to financial returns.
        Computational Statistics & Data Analysis, 59, 52-69.

    Attributes:
        dimension: Number of uniform margins.
        structure: "c", "d" or "r".
        pair_copulas: Immutable rows of pair copulas in tree order.
        trees: Immutable rows of ordered edges in tree order.
        truncation_level: Number of supplied trees; omitted trees are independent.
    """

    def __init__(
        self,
        pair_copulas: t.Sequence[t.Sequence[PairCopula | VineEdge]],
        *,
        structure: t.Literal["c", "d", "r"] = "c",
    ) -> None:
        """Construct a vine from successive rows of edges or pair copulas.

        Args:
            pair_copulas: Rows of lengths d-1, d-2, and so on. For structure
                "r", provide VineEdge objects; for "c" or "d", provide PairCopula
                objects. The number of margins is inferred from the first row.
            structure: Canonical ("c"), drawable ("d") or regular ("r") vine.

        Raises:
            ValueError: If the structure, row lengths or vine constraints are invalid.
            TypeError: If a compact C- or D-vine contains an invalid pair copula.
        """
        if structure not in ("c", "d", "r"):
            raise ValueError("Vine structure must be 'c', 'd' or 'r'")
        rows = tuple(tuple(row) for row in pair_copulas)
        if not rows or not rows[0] or len(rows) > len(rows[0]):
            raise ValueError("Vine requires nonempty rows of lengths d-1, d-2, ..., 1")
        if any(len(row) != len(rows[0]) - k for k, row in enumerate(rows)):
            raise ValueError("Vine requires rows of lengths d-1, d-2, ..., 1")
        self.dimension = len(rows[0]) + 1
        self.structure = structure
        if structure == "r":
            if any(not isinstance(edge, VineEdge) for row in rows for edge in row):
                raise ValueError("R-vine structure requires VineEdge objects")
            trees = t.cast(tuple[tuple[VineEdge, ...], ...], rows)
        else:
            if any(not isinstance(pair, PairCopula) for row in rows for pair in row):
                raise TypeError("Every vine edge must be a PairCopula")
            pairs = t.cast(tuple[tuple[PairCopula, ...], ...], rows)
            if structure == "c":
                trees = tuple(
                    tuple(VineEdge(k, k + i + 1, pair, tuple(range(k))) for i, pair in enumerate(row))
                    for k, row in enumerate(pairs)
                )
            else:
                trees = tuple(
                    tuple(VineEdge(i, i + k + 1, pair, tuple(range(i + 1, i + k + 1))) for i, pair in enumerate(row))
                    for k, row in enumerate(pairs)
                )
        _validate_vine_trees(trees, self.dimension)
        self.trees = trees
        self.pair_copulas = tuple(tuple(edge.pair for edge in tree) for tree in trees)
        self.truncation_level = len(trees)
        self._stages, self._uses = _compile_vine(trees, self.dimension)

    @classmethod
    def from_edges(cls, trees: t.Sequence[t.Sequence[VineEdge]]) -> VineCopula:
        """Construct a regular vine from its edges.

        Args:
            trees: Successive trees of ordered edges, with zero-based variable
                indices. Higher trees may be omitted to truncate the vine.

        Returns:
            Regular vine with the given edges.

        Raises:
            ValueError: If the trees are not connected and acyclic, violate the
                proximity condition or do not define a valid simulation order.
            TypeError: If an edge does not contain a PairCopula.
        """
        return cls(trees, structure="r")

    def generate(
        self, n_sims: int | None = None, rng: RandomGenerator | None = None
    ) -> ProteusVariable[StochasticScalar]:
        """Generate coupled uniform margins using the inverse Rosenblatt transform.

        Args:
            n_sims: Number of simulations; defaults to config.n_sims.
            rng: Random generator; defaults to config.rng.
        """
        return self._generate_base(n_sims, rng)

    def _generate_unnormalised(self, n_sims: int, rng: RandomGenerator) -> npt.NDArray[np.floating]:
        """Draw one independent coordinate per margin and invert the vine.

        Args:
            n_sims: Number of simulations.
            rng: Random generator supplying scenario-aligned coordinates.
        """
        if n_sims <= 0:
            raise ValueError("Number of simulations must be positive")
        draws = _open_unit(
            np.asarray(
                _draw_scenarios(rng, "uniform", size=(self.dimension, n_sims), scenario_axis=1),
                dtype=float,
            )
        )
        result = np.empty_like(draws)
        values: dict[_VineKey, _VineArray] = {}
        uses = self._uses.copy()

        def release(key: _VineKey) -> None:
            """Free a conditional probability once its last consumer finishes.

            Args:
                key: Variable and conditioning set.
            """
            uses[key] -= 1
            if not uses[key]:
                del values[key]

        for draw_key, steps in self._stages:
            variable = draw_key[0]
            values[draw_key] = draws[variable]
            for pair, first, inverse, source, given, output in steps:
                transform = pair._inverse if inverse else pair._conditional
                values[output] = transform(values[source], values[given], first=first)
                release(source)
                release(given)
            key = (variable, frozenset())
            result[variable] = values[key]
            release(key)
        return result

    def apply(self, variables: ProteusVariable[StochasticScalar] | list[StochasticScalar]) -> None:
        """Apply the vine to independent marginal simulations in variable order.

        Args:
            variables: Marginal variables to reorder in place. Their number must
                equal the fixed vine dimension and their simulation counts must match.
        """
        if len(variables) != self.dimension:
            raise ValueError("Number of variables and copula samples do not match.")
        apply_copula(variables, self.generate(n_sims=list(variables)[0].n_sims))
