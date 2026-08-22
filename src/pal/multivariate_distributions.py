"""Multivariate probability distributions for simulation models.

The distributions in this module return a :class:`ProteusVariable` containing
one :class:`StochasticScalar` per component. This preserves PAL's convention
that each stochastic scalar is indexed only by simulation while giving the
components a named dimension.
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from ._maths import asnumpy, special, to_backend, xp
from .config import config
from .stochastic_scalar import StochasticScalar
from .types import DistributionParameter, RandomGenerator
from .variables import ProteusVariable

MultivariateParameter = t.Union[t.Sequence[DistributionParameter], ProteusVariable[t.Any]]
MultivariateInput = t.Union[npt.ArrayLike, ProteusVariable[t.Any]]
MultivariateResult = ProteusVariable[StochasticScalar]
DensityResult = t.Union[float, StochasticScalar]


def _is_numpy_rng(rng: RandomGenerator) -> bool:
    """Return whether a generator produces NumPy arrays."""
    return type(rng).__module__.startswith("numpy")


def _rng_value(value: t.Any, rng: RandomGenerator) -> t.Any:
    """Place a value on the array backend used by the supplied generator."""
    if _is_numpy_rng(rng):
        return asnumpy(value) if isinstance(value, xp.ndarray) else value
    return to_backend(value)


def _sequence_values(parameter: MultivariateParameter) -> tuple[list[DistributionParameter], list[str] | None]:
    """Return the values and optional names from a vector parameter."""
    if isinstance(parameter, ProteusVariable):
        return list(parameter.values.values()), list(parameter.values)
    values = list(parameter)
    return values, None


def _validate_parameter_vector(
    parameter: MultivariateParameter,
    name: str,
    *,
    minimum_length: int = 1,
) -> tuple[list[DistributionParameter], list[str] | None]:
    """Validate and unpack a vector of scalar or stochastic parameters."""
    values, names = _sequence_values(parameter)
    if len(values) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} value(s).")
    for value in values:
        raw = value.values if isinstance(value, StochasticScalar) else value
        backend_value = xp.asarray(to_backend(raw))
        if bool(xp.any(~xp.isfinite(backend_value))):
            raise ValueError(f"{name} values must be finite.")
    return values, names


def _validate_positive_parameter_vector(
    parameter: MultivariateParameter,
    name: str,
    *,
    minimum_length: int = 1,
) -> tuple[list[DistributionParameter], list[str] | None]:
    """Validate and unpack a strictly positive parameter vector."""
    values, names = _validate_parameter_vector(parameter, name, minimum_length=minimum_length)
    for value in values:
        raw = value.values if isinstance(value, StochasticScalar) else value
        backend_value = xp.asarray(to_backend(raw))
        if bool(xp.any(backend_value <= 0)):
            raise ValueError(f"{name} values must be positive.")
    return values, names


def _parameter_n_sims(parameters: t.Iterable[DistributionParameter]) -> int | None:
    """Return the common simulation count of stochastic parameters."""
    result: int | None = None
    for parameter in parameters:
        if not isinstance(parameter, StochasticScalar):
            continue
        if result is None:
            result = parameter.n_sims
        elif parameter.n_sims != result:
            raise ValueError("Stochastic parameters must have the same number of simulations.")
    return result


def _parameter_matrix(
    parameters: t.Sequence[DistributionParameter],
    n_sims: int,
    rng: RandomGenerator,
) -> t.Any:
    """Broadcast vector parameters across simulations on the generator backend."""
    rows: list[t.Any] = []
    for parameter in parameters:
        if isinstance(parameter, StochasticScalar):
            if parameter.n_sims != n_sims:
                raise ValueError(
                    "The number of simulations in each stochastic parameter must "
                    "match the requested number of simulations."
                )
            rows.append(_rng_value(parameter.values, rng))
        else:
            backend = np if _is_numpy_rng(rng) else xp
            rows.append(backend.full(n_sims, parameter, dtype=float))
    backend = np if _is_numpy_rng(rng) else xp
    return backend.stack(rows, axis=0)


def _active_parameter_matrix(parameters: t.Sequence[DistributionParameter], n_sims: int) -> t.Any:
    """Broadcast vector parameters across simulations on PAL's active backend."""
    rows: list[t.Any] = []
    for parameter in parameters:
        if isinstance(parameter, StochasticScalar):
            if parameter.n_sims != n_sims:
                raise ValueError("Stochastic parameters and observations must have the same number of simulations.")
            rows.append(parameter.values)
        else:
            rows.append(xp.full(n_sims, parameter, dtype=float))
    return xp.stack(rows, axis=0)


def _validate_matrix(matrix: npt.ArrayLike, dimension: int, name: str) -> t.Any:
    """Return the Cholesky factor of a symmetric positive-definite matrix."""
    values = xp.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape != (dimension, dimension):
        raise ValueError(f"{name} must be a {dimension} by {dimension} matrix.")
    if bool(xp.any(~xp.isfinite(values))):
        raise ValueError(f"{name} values must be finite.")
    if not bool(xp.allclose(values, values.T)):
        raise ValueError(f"{name} must be symmetric.")
    if float(xp.linalg.eigvalsh(values).min().item()) <= 0:
        raise ValueError(f"{name} must be positive definite.")
    try:
        return xp.linalg.cholesky(values)
    except xp.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite.") from error


def _observation_matrix(x: MultivariateInput, dimension: int) -> tuple[t.Any, list[StochasticScalar]]:
    """Return observations with components in rows and simulations in columns."""
    stochastic_inputs: list[StochasticScalar] = []
    if isinstance(x, ProteusVariable):
        if len(x) != dimension:
            raise ValueError(f"x must contain {dimension} components.")
        values = list(x.values.values())
        stochastic_inputs = [value for value in values if isinstance(value, StochasticScalar)]
        n_sims = _parameter_n_sims(stochastic_inputs) or 1
        rows = [
            value.values if isinstance(value, StochasticScalar) else xp.full(n_sims, value, dtype=float)
            for value in values
        ]
        return xp.stack(rows, axis=0), stochastic_inputs

    values = xp.asarray(x, dtype=float)
    if values.ndim == 1:
        if values.shape[0] != dimension:
            raise ValueError(f"x must contain {dimension} components.")
        return values[:, xp.newaxis], stochastic_inputs
    if values.ndim == 2 and values.shape[0] == dimension:
        return values, stochastic_inputs
    raise ValueError(f"x must have shape ({dimension},) or ({dimension}, n_sims).")


def _density_result(values: t.Any, stochastic_inputs: t.Iterable[DistributionParameter]) -> DensityResult:
    """Wrap simulation-varying density values and preserve their coupling."""
    inputs = [value for value in stochastic_inputs if isinstance(value, StochasticScalar)]
    result = xp.asarray(values)
    if result.size == 1 and not inputs:
        return float(result.item())
    wrapped = StochasticScalar(result.reshape(-1))
    for value in inputs:
        wrapped.coupled_variable_group.merge(value.coupled_variable_group)
    return wrapped


class MultivariateDistributionBase(ABC):
    """Base class for distributions whose samples contain several components."""

    dimension: int

    def __init__(
        self,
        dimension: int,
        *,
        component_names: t.Sequence[str] | None = None,
        dim_name: str = "component",
    ) -> None:
        """Initialize a multivariate distribution.

        Args:
            dimension: Number of components in each generated sample.
            component_names: Optional names for the components.
            dim_name: Name of the component dimension.
        """
        self.dimension = dimension
        self.dim_name = dim_name
        if component_names is None:
            self.component_names = [f"component_{index + 1}" for index in range(dimension)]
        else:
            self.component_names = list(component_names)
            if len(self.component_names) != dimension:
                raise ValueError(f"component_names must contain {dimension} names.")
            if len(set(self.component_names)) != dimension:
                raise ValueError("component_names must be unique.")

    @property
    @abstractmethod
    def _parameters(self) -> tuple[DistributionParameter, ...]:
        """Return scalar and stochastic parameters used by the distribution."""

    @abstractmethod
    def _generate_matrix(self, n_sims: int, rng: RandomGenerator) -> t.Any:
        """Generate samples with components in rows and simulations in columns."""

    @abstractmethod
    def logpdf(self, x: MultivariateInput) -> DensityResult:
        """Compute the joint log probability density."""

    def pdf(self, x: MultivariateInput) -> DensityResult:
        """Compute the joint probability density."""
        result = self.logpdf(x)
        if isinstance(result, StochasticScalar):
            return t.cast(StochasticScalar, np.exp(result))
        return float(np.exp(result))

    def generate(
        self,
        n_sims: int | None = None,
        rng: RandomGenerator | None = None,
    ) -> MultivariateResult:
        """Generate random samples from the distribution.

        Args:
            n_sims: Number of simulations. Uses the configured value when omitted.
            rng: Random number generator. Uses the configured generator when omitted.

        Returns:
            A named variable containing one stochastic scalar per component.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if n_sims < 1:
            raise ValueError(f"n_sims must be >= 1, got {n_sims}")
        parameter_n_sims = _parameter_n_sims(self._parameters)
        if parameter_n_sims is not None and parameter_n_sims != n_sims:
            raise ValueError(
                "The number of simulations in stochastic parameters must match the requested number of simulations."
            )
        if rng is None:
            rng = config.rng

        samples = self._generate_matrix(n_sims, rng)
        active_samples = to_backend(samples)
        if active_samples.shape != (self.dimension, n_sims):
            raise RuntimeError("A multivariate sampler returned an invalid sample shape.")
        components = {name: StochasticScalar(active_samples[index]) for index, name in enumerate(self.component_names)}
        result = ProteusVariable[StochasticScalar](self.dim_name, components)
        first = result[0]
        for component in result:
            first.coupled_variable_group.merge(component.coupled_variable_group)
        for parameter in self._parameters:
            if isinstance(parameter, StochasticScalar):
                first.coupled_variable_group.merge(parameter.coupled_variable_group)
        return result


class MultivariateNormal(MultivariateDistributionBase):
    r"""Multivariate normal distribution.

    For a :math:`d`-dimensional random vector :math:`X`, the density is

    .. math::

        f(x) = \frac{\exp\left(-\tfrac12(x-\mu)^\mathsf{T}
        \Sigma^{-1}(x-\mu)\right)}
        {(2\pi)^{d/2}|\Sigma|^{1/2}},

    where :math:`\mu` is the mean vector and :math:`\Sigma` is a symmetric
    positive-definite covariance matrix.

    Parameters:
        mean: Mean of each component.
        covariance: Covariance matrix :math:`\Sigma`.
        component_names: Optional names for the components.
        dim_name: Name of the component dimension.
    """

    def __init__(
        self,
        mean: MultivariateParameter,
        covariance: npt.ArrayLike,
        *,
        component_names: t.Sequence[str] | None = None,
        dim_name: str = "component",
    ) -> None:
        """Initialize a multivariate normal distribution."""
        self.mean, parameter_names = _validate_parameter_vector(mean, "mean")
        super().__init__(
            len(self.mean),
            component_names=parameter_names if component_names is None else component_names,
            dim_name=dim_name,
        )
        self._chol = _validate_matrix(covariance, self.dimension, "covariance")

    @property
    def _parameters(self) -> tuple[DistributionParameter, ...]:
        return tuple(self.mean)

    def _generate_matrix(self, n_sims: int, rng: RandomGenerator) -> t.Any:
        mean = _parameter_matrix(self.mean, n_sims, rng)
        chol = _rng_value(self._chol, rng)
        normal = rng.standard_normal(size=(self.dimension, n_sims))
        return mean + chol @ normal

    def logpdf(self, x: MultivariateInput) -> DensityResult:
        """Compute the joint log probability density."""
        observations, stochastic_inputs = _observation_matrix(x, self.dimension)
        n_sims = max(observations.shape[1], _parameter_n_sims(self.mean) or 1)
        if observations.shape[1] not in (1, n_sims):
            raise ValueError("Observations and stochastic parameters must have compatible simulation counts.")
        observations = xp.broadcast_to(observations, (self.dimension, n_sims))
        mean = _active_parameter_matrix(self.mean, n_sims)
        centred = observations - mean
        solved = xp.linalg.solve(self._chol, centred)
        quadratic = xp.sum(solved**2, axis=0)
        log_determinant = 2 * xp.sum(xp.log(xp.diag(self._chol)))
        result = -0.5 * (self.dimension * np.log(2 * np.pi) + log_determinant + quadratic)
        return _density_result(result, (*stochastic_inputs, *self._parameters))


class MultivariateStudentsT(MultivariateDistributionBase):
    r"""Multivariate Student's t distribution.

    The density of a :math:`d`-dimensional random vector is

    .. math::

        f(x) = \frac{\Gamma((\nu+d)/2)}
        {\Gamma(\nu/2)(\nu\pi)^{d/2}|\Sigma|^{1/2}}
        \left(1+\frac{(x-\mu)^\mathsf{T}\Sigma^{-1}(x-\mu)}{\nu}
        \right)^{-(\nu+d)/2},

    where :math:`\nu>0` is the degrees of freedom and :math:`\Sigma` is the
    scale matrix. When :math:`\nu>2`, the covariance matrix is
    :math:`\nu\Sigma/(\nu-2)`.

    Parameters:
        nu: Degrees of freedom :math:`\nu`.
        mean: Location of each component.
        scale: Symmetric positive-definite scale matrix :math:`\Sigma`.
        component_names: Optional names for the components.
        dim_name: Name of the component dimension.
    """

    def __init__(
        self,
        nu: DistributionParameter,
        mean: MultivariateParameter,
        scale: npt.ArrayLike,
        *,
        component_names: t.Sequence[str] | None = None,
        dim_name: str = "component",
    ) -> None:
        """Initialize a multivariate Student's t distribution."""
        raw_nu = nu.values if isinstance(nu, StochasticScalar) else nu
        backend_nu = xp.asarray(to_backend(raw_nu))
        if bool(xp.any(~xp.isfinite(backend_nu))) or bool(xp.any(backend_nu <= 0)):
            raise ValueError("nu must be finite and positive.")
        self.nu = nu
        self.mean, parameter_names = _validate_parameter_vector(mean, "mean")
        super().__init__(
            len(self.mean),
            component_names=parameter_names if component_names is None else component_names,
            dim_name=dim_name,
        )
        self._chol = _validate_matrix(scale, self.dimension, "scale")

    @property
    def _parameters(self) -> tuple[DistributionParameter, ...]:
        return (self.nu, *self.mean)

    def _generate_matrix(self, n_sims: int, rng: RandomGenerator) -> t.Any:
        mean = _parameter_matrix(self.mean, n_sims, rng)
        nu = _rng_value(self.nu.values if isinstance(self.nu, StochasticScalar) else self.nu, rng)
        chol = _rng_value(self._chol, rng)
        normal = chol @ rng.standard_normal(size=(self.dimension, n_sims))
        chi_scale = rng.gamma(nu / 2, 2 / nu, size=n_sims)
        return mean + normal / chi_scale[None, :] ** 0.5

    def logpdf(self, x: MultivariateInput) -> DensityResult:
        """Compute the joint log probability density."""
        observations, stochastic_inputs = _observation_matrix(x, self.dimension)
        parameter_n_sims = _parameter_n_sims(self._parameters) or 1
        n_sims = max(observations.shape[1], parameter_n_sims)
        if observations.shape[1] not in (1, n_sims):
            raise ValueError("Observations and stochastic parameters must have compatible simulation counts.")
        observations = xp.broadcast_to(observations, (self.dimension, n_sims))
        mean = _active_parameter_matrix(self.mean, n_sims)
        nu_values = self.nu.values if isinstance(self.nu, StochasticScalar) else xp.full(n_sims, self.nu)
        centred = observations - mean
        solved = xp.linalg.solve(self._chol, centred)
        quadratic = xp.sum(solved**2, axis=0)
        log_determinant = 2 * xp.sum(xp.log(xp.diag(self._chol)))
        result = (
            special.gammaln((nu_values + self.dimension) / 2)
            - special.gammaln(nu_values / 2)
            - 0.5 * (self.dimension * xp.log(nu_values * np.pi) + log_determinant)
            - 0.5 * (nu_values + self.dimension) * xp.log1p(quadratic / nu_values)
        )
        return _density_result(result, (*stochastic_inputs, *self._parameters))


class Dirichlet(MultivariateDistributionBase):
    r"""Dirichlet distribution on the probability simplex.

    Its density is

    .. math::

        f(x_1,\ldots,x_d) = \frac{\Gamma(\alpha_0)}
        {\prod_{i=1}^d\Gamma(\alpha_i)}
        \prod_{i=1}^d x_i^{\alpha_i-1},
        \qquad \alpha_0=\sum_{i=1}^d\alpha_i,

    for positive components satisfying :math:`\sum_i x_i=1`.

    Parameters:
        alpha: Positive concentration parameter for each component.
        component_names: Optional names for the components.
        dim_name: Name of the component dimension.
    """

    def __init__(
        self,
        alpha: MultivariateParameter,
        *,
        component_names: t.Sequence[str] | None = None,
        dim_name: str = "component",
    ) -> None:
        """Initialize a Dirichlet distribution."""
        self.alpha, parameter_names = _validate_positive_parameter_vector(alpha, "alpha", minimum_length=2)
        super().__init__(
            len(self.alpha),
            component_names=parameter_names if component_names is None else component_names,
            dim_name=dim_name,
        )

    @property
    def _parameters(self) -> tuple[DistributionParameter, ...]:
        return tuple(self.alpha)

    def _generate_matrix(self, n_sims: int, rng: RandomGenerator) -> t.Any:
        alpha = _parameter_matrix(self.alpha, n_sims, rng)
        gamma = rng.gamma(alpha, 1.0)
        return gamma / gamma.sum(axis=0, keepdims=True)

    def logpdf(self, x: MultivariateInput) -> DensityResult:
        """Compute the joint log probability density."""
        observations, stochastic_inputs = _observation_matrix(x, self.dimension)
        n_sims = max(observations.shape[1], _parameter_n_sims(self.alpha) or 1)
        if observations.shape[1] not in (1, n_sims):
            raise ValueError("Observations and stochastic parameters must have compatible simulation counts.")
        observations = xp.broadcast_to(observations, (self.dimension, n_sims))
        alpha = _active_parameter_matrix(self.alpha, n_sims)
        valid = xp.all(observations > 0, axis=0) & xp.isclose(xp.sum(observations, axis=0), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = special.gammaln(xp.sum(alpha, axis=0)) - xp.sum(special.gammaln(alpha), axis=0)
            result += xp.sum((alpha - 1) * xp.log(observations), axis=0)
        result = xp.where(valid, result, -xp.inf)
        return _density_result(result, (*stochastic_inputs, *self._parameters))


class InvertedDirichlet(MultivariateDistributionBase):
    r"""Inverted Dirichlet distribution on the positive orthant.

    For :math:`d` positive components and :math:`d+1` positive parameters,

    .. math::

        f(x) = \frac{\Gamma(\alpha_0)}{\prod_{i=1}^{d+1}\Gamma(\alpha_i)}
        \frac{\prod_{i=1}^d x_i^{\alpha_i-1}}
        {(1+\sum_{i=1}^d x_i)^{\alpha_0}},
        \qquad \alpha_0=\sum_{i=1}^{d+1}\alpha_i.

    It can be generated by dividing the first :math:`d` independent gamma
    variables by the final gamma variable.

    Parameters:
        alpha: Positive parameters. The final value controls the common denominator.
        component_names: Optional names for the generated components.
        dim_name: Name of the component dimension.

    References:
        Tiao, G. G. and Guttman, I. (1965). The Inverted Dirichlet
        Distribution with Applications. Journal of the American Statistical
        Association 60(311), 793--805.
    """

    def __init__(
        self,
        alpha: MultivariateParameter,
        *,
        component_names: t.Sequence[str] | None = None,
        dim_name: str = "component",
    ) -> None:
        """Initialize an inverted Dirichlet distribution."""
        self.alpha, parameter_names = _validate_positive_parameter_vector(alpha, "alpha", minimum_length=2)
        names = parameter_names[:-1] if parameter_names is not None else None
        super().__init__(
            len(self.alpha) - 1,
            component_names=names if component_names is None else component_names,
            dim_name=dim_name,
        )

    @property
    def _parameters(self) -> tuple[DistributionParameter, ...]:
        return tuple(self.alpha)

    def _generate_matrix(self, n_sims: int, rng: RandomGenerator) -> t.Any:
        alpha = _parameter_matrix(self.alpha, n_sims, rng)
        gamma = rng.gamma(alpha, 1.0)
        return gamma[:-1] / gamma[-1]

    def logpdf(self, x: MultivariateInput) -> DensityResult:
        """Compute the joint log probability density."""
        observations, stochastic_inputs = _observation_matrix(x, self.dimension)
        n_sims = max(observations.shape[1], _parameter_n_sims(self.alpha) or 1)
        if observations.shape[1] not in (1, n_sims):
            raise ValueError("Observations and stochastic parameters must have compatible simulation counts.")
        observations = xp.broadcast_to(observations, (self.dimension, n_sims))
        alpha = _active_parameter_matrix(self.alpha, n_sims)
        valid = xp.all(observations > 0, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = special.gammaln(xp.sum(alpha, axis=0)) - xp.sum(special.gammaln(alpha), axis=0)
            result += xp.sum((alpha[:-1] - 1) * xp.log(observations), axis=0)
            result -= xp.sum(alpha, axis=0) * xp.log1p(xp.sum(observations, axis=0))
        result = xp.where(valid, result, -xp.inf)
        return _density_result(result, (*stochastic_inputs, *self._parameters))


class GeneralizedDirichlet(MultivariateDistributionBase):
    r"""Connor--Mosimann generalized Dirichlet distribution.

    Let :math:`V_i` be independent beta variables with parameters
    :math:`(\alpha_i,\beta_i)`. The simplex components are constructed by

    .. math::

        X_i = V_i\prod_{j<i}(1-V_j), \qquad
        X_{d+1}=\prod_{j=1}^d(1-V_j).

    This parameterisation permits a richer covariance structure than the
    Dirichlet distribution. The generated result includes the final remainder,
    so its :math:`d+1` components sum to one.

    Parameters:
        alpha: First positive beta shape for each stick-breaking step.
        beta: Second positive beta shape for each stick-breaking step.
        component_names: Optional names for all generated components, including the remainder.
        dim_name: Name of the component dimension.

    References:
        Connor, R. J. and Mosimann, J. E. (1969). Concepts of Independence for
        Proportions with a Generalization of the Dirichlet Distribution.
        Journal of the American Statistical Association 64(325), 194--206.
    """

    def __init__(
        self,
        alpha: MultivariateParameter,
        beta: MultivariateParameter,
        *,
        component_names: t.Sequence[str] | None = None,
        dim_name: str = "component",
    ) -> None:
        """Initialize a generalized Dirichlet distribution."""
        self.alpha, _ = _validate_positive_parameter_vector(alpha, "alpha")
        self.beta, _ = _validate_positive_parameter_vector(beta, "beta")
        if len(self.alpha) != len(self.beta):
            raise ValueError("alpha and beta must have the same length.")
        super().__init__(len(self.alpha) + 1, component_names=component_names, dim_name=dim_name)

    @property
    def _parameters(self) -> tuple[DistributionParameter, ...]:
        return (*self.alpha, *self.beta)

    def _generate_matrix(self, n_sims: int, rng: RandomGenerator) -> t.Any:
        alpha = _parameter_matrix(self.alpha, n_sims, rng)
        beta = _parameter_matrix(self.beta, n_sims, rng)
        breaks = rng.beta(alpha, beta)
        components = []
        backend = np if _is_numpy_rng(rng) else xp
        remainder = backend.ones(n_sims)
        for index in range(len(self.alpha)):
            component = remainder * breaks[index]
            components.append(component)
            remainder = remainder - component
        components.append(remainder)
        return backend.stack(components, axis=0)

    def logpdf(self, x: MultivariateInput) -> DensityResult:
        """Compute the joint log probability density."""
        observations, stochastic_inputs = _observation_matrix(x, self.dimension)
        parameter_n_sims = _parameter_n_sims(self._parameters) or 1
        n_sims = max(observations.shape[1], parameter_n_sims)
        if observations.shape[1] not in (1, n_sims):
            raise ValueError("Observations and stochastic parameters must have compatible simulation counts.")
        observations = xp.broadcast_to(observations, (self.dimension, n_sims))
        alpha = _active_parameter_matrix(self.alpha, n_sims)
        beta = _active_parameter_matrix(self.beta, n_sims)
        valid = xp.all(observations > 0, axis=0) & xp.isclose(xp.sum(observations, axis=0), 1.0)
        remainder = xp.ones(n_sims)
        result = xp.zeros(n_sims)
        with np.errstate(divide="ignore", invalid="ignore"):
            for index in range(len(self.alpha)):
                proportion = observations[index] / remainder
                result += (alpha[index] - 1) * xp.log(proportion)
                result += (beta[index] - 1) * xp.log1p(-proportion)
                result -= special.betaln(alpha[index], beta[index]) + xp.log(remainder)
                remainder = remainder - observations[index]
        result = xp.where(valid, result, -xp.inf)
        return _density_result(result, (*stochastic_inputs, *self._parameters))


class InvertedGeneralizedDirichlet(MultivariateDistributionBase):
    r"""Inverted generalized Dirichlet distribution.

    Let :math:`R_i` be independent beta-prime variables with parameters
    :math:`(\alpha_i,\beta_i)`. The positive components are constructed by

    .. math::

        X_1=R_1, \qquad
        X_i=R_i\left(1+\sum_{j<i}X_j\right), \quad i=2,\ldots,d.

    Equivalently, its density is

    .. math::

        f(x)=\prod_{i=1}^d\frac{x_i^{\alpha_i-1}}{B(\alpha_i,\beta_i)}
        \left(1+\sum_{j=1}^i x_j\right)^{-\gamma_i},
        \qquad \gamma_i=\alpha_i+\beta_i-\beta_{i+1},

    with :math:`\beta_{d+1}=0`. It reduces to an inverted Dirichlet when
    :math:`\gamma_i=0` for :math:`i<d`.

    Parameters:
        alpha: First positive beta-prime shape for each component.
        beta: Second positive beta-prime shape for each component.
        component_names: Optional names for the generated components.
        dim_name: Name of the component dimension.

    References:
        Lingappaiah, G. S. (1976). On the Generalized Inverted Dirichlet
        Distribution. Demonstratio Mathematica 9(2), 423--433.
    """

    def __init__(
        self,
        alpha: MultivariateParameter,
        beta: MultivariateParameter,
        *,
        component_names: t.Sequence[str] | None = None,
        dim_name: str = "component",
    ) -> None:
        """Initialize an inverted generalized Dirichlet distribution."""
        self.alpha, parameter_names = _validate_positive_parameter_vector(alpha, "alpha")
        self.beta, _ = _validate_positive_parameter_vector(beta, "beta")
        if len(self.alpha) != len(self.beta):
            raise ValueError("alpha and beta must have the same length.")
        super().__init__(
            len(self.alpha),
            component_names=parameter_names if component_names is None else component_names,
            dim_name=dim_name,
        )

    @property
    def _parameters(self) -> tuple[DistributionParameter, ...]:
        return (*self.alpha, *self.beta)

    def _generate_matrix(self, n_sims: int, rng: RandomGenerator) -> t.Any:
        alpha = _parameter_matrix(self.alpha, n_sims, rng)
        beta = _parameter_matrix(self.beta, n_sims, rng)
        ratios = rng.gamma(alpha, 1.0) / rng.gamma(beta, 1.0)
        components = []
        backend = np if _is_numpy_rng(rng) else xp
        cumulative = backend.ones(n_sims)
        for index in range(self.dimension):
            component = ratios[index] * cumulative
            components.append(component)
            cumulative = cumulative + component
        return backend.stack(components, axis=0)

    def logpdf(self, x: MultivariateInput) -> DensityResult:
        """Compute the joint log probability density."""
        observations, stochastic_inputs = _observation_matrix(x, self.dimension)
        parameter_n_sims = _parameter_n_sims(self._parameters) or 1
        n_sims = max(observations.shape[1], parameter_n_sims)
        if observations.shape[1] not in (1, n_sims):
            raise ValueError("Observations and stochastic parameters must have compatible simulation counts.")
        observations = xp.broadcast_to(observations, (self.dimension, n_sims))
        alpha = _active_parameter_matrix(self.alpha, n_sims)
        beta = _active_parameter_matrix(self.beta, n_sims)
        valid = xp.all(observations > 0, axis=0)
        cumulative = xp.ones(n_sims)
        result = xp.zeros(n_sims)
        with np.errstate(divide="ignore", invalid="ignore"):
            for index in range(self.dimension):
                ratio = observations[index] / cumulative
                result += (alpha[index] - 1) * xp.log(ratio)
                result -= (alpha[index] + beta[index]) * xp.log1p(ratio)
                result -= special.betaln(alpha[index], beta[index]) + xp.log(cumulative)
                cumulative = cumulative + observations[index]
        result = xp.where(valid, result, -xp.inf)
        return _density_result(result, (*stochastic_inputs, *self._parameters))


__all__ = [
    "Dirichlet",
    "GeneralizedDirichlet",
    "InvertedDirichlet",
    "InvertedGeneralizedDirichlet",
    "MultivariateDistributionBase",
    "MultivariateNormal",
    "MultivariateStudentsT",
]
