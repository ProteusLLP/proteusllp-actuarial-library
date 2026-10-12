"""Hyperexponential distribution.

This module contains the finite-mixture hyperexponential distribution. It lives
separately from :mod:`pal.distributions` because its component weights and rates
are vector-valued parameters rather than single ``DistributionParameter`` values.
The public class is exposed as ``pal.HyperExponential`` and
``pal.distributions.HyperExponential`` by :mod:`pal`.
"""

from __future__ import annotations

import typing as t

from ._compat import override
from ._maths import scalar_or_array, xp
from .distributions import DistributionBase, ReturnType
from .stochastic_scalar import StochasticScalar
from .types import DistributionParameter, RandomGenerator


class HyperExponential(DistributionBase):
    r"""Finite mixture of exponential distributions.

    Let :math:`p_i \geq 0`, :math:`\sum_{i=1}^m p_i=1`, and
    :math:`\lambda_i>0`. With location parameter :math:`\mu`, the density is

    .. math::

        f(x) = \sum_{i=1}^m p_i \lambda_i
        \exp\left[-\lambda_i(x-\mu)\right], \qquad x>\mu,

    and the cumulative distribution function is

    .. math::

        F(x) = \begin{cases}
        0, & x \leq \mu, \\
        1-\displaystyle\sum_{i=1}^m p_i
        \exp\left[-\lambda_i(x-\mu)\right], & x>\mu.
        \end{cases}

    The first two moments are

    .. math::

        E[X] = \mu + \sum_{i=1}^m \frac{p_i}{\lambda_i},

    .. math::

        \operatorname{Var}(X) =
        2\sum_{i=1}^m \frac{p_i}{\lambda_i^2}
        - \left(\sum_{i=1}^m \frac{p_i}{\lambda_i}\right)^2.

    A one-component hyperexponential distribution is an ordinary exponential
    distribution. Mixtures with distinct rates have a decreasing hazard rate
    and can represent substantially more heterogeneity than a single
    exponential distribution.

    ``weights`` and ``rates`` may contain ``StochasticScalar`` values. In that
    case the parameter constraints are checked for every simulation and the
    generated result is placed in the same coupled variable groups.

    Parameters:
        weights: Component probabilities :math:`p_i`. They must be non-negative
            and sum to one.
        rates: Positive component rates :math:`\lambda_i`.
        loc: Location parameter :math:`\mu` (default 0).
    """

    def __init__(
        self,
        weights: t.Sequence[DistributionParameter],
        rates: t.Sequence[DistributionParameter],
        loc: DistributionParameter = 0.0,
    ) -> None:
        """Initialize a hyperexponential distribution.

        Args:
            weights: Mixture probabilities, one per component.
            rates: Positive exponential rates, one per component.
            loc: Location parameter.
        """
        if len(weights) == 0:
            raise ValueError("weights and rates must contain at least one component.")
        if len(weights) != len(rates):
            raise ValueError("weights and rates must have the same number of components.")

        self._weights = tuple(weights)
        self._rates = tuple(rates)
        super().__init__(loc=loc)
        self._validated_component_parameters()

    @classmethod
    def from_scales(
        cls,
        weights: t.Sequence[DistributionParameter],
        scales: t.Sequence[DistributionParameter],
        loc: DistributionParameter = 0.0,
    ) -> HyperExponential:
        r"""Construct the distribution from exponential scale parameters.

        PAL's :class:`pal.distributions.Exponential` uses the scale (mean)
        parameterisation. This constructor accepts corresponding component
        scales :math:`\theta_i=1/\lambda_i`.

        Args:
            weights: Mixture probabilities, one per component.
            scales: Positive exponential scales, one per component.
            loc: Location parameter.

        Returns:
            Hyperexponential distribution with rates equal to reciprocal scales.
        """
        rates = [t.cast(DistributionParameter, 1 / scale) for scale in scales]
        return cls(weights=weights, rates=rates, loc=loc)

    def _component_inputs(self) -> tuple[DistributionParameter, ...]:
        """Return all component parameters in coupling order."""
        return (*self._weights, *self._rates)

    def _component_arrays(self) -> tuple[t.Any, t.Any]:
        """Broadcast and stack component weights and rates on the active backend."""
        values = [
            xp.asarray(value.values if isinstance(value, StochasticScalar) else value)
            for value in self._component_inputs()
        ]
        try:
            broadcast = xp.broadcast_arrays(*values)
        except ValueError as exc:
            raise ValueError("All stochastic component parameters must have compatible simulation dimensions.") from exc

        n_components = len(self._weights)
        weights = xp.stack(broadcast[:n_components], axis=-1)
        rates = xp.stack(broadcast[n_components:], axis=-1)
        return weights, rates

    def _validated_component_parameters(self) -> tuple[t.Any, t.Any]:
        """Return component arrays after checking the admissible parameter region."""
        weights, rates = self._component_arrays()
        if bool(xp.any(~xp.isfinite(weights))):
            raise ValueError("weights must be finite.")
        if bool(xp.any(weights < 0)):
            raise ValueError("weights must be non-negative.")
        if bool(xp.any(~xp.isfinite(rates))) or bool(xp.any(rates <= 0)):
            raise ValueError("rates must be finite and strictly positive.")

        weight_sums = xp.sum(weights, axis=-1)
        if not bool(xp.all(xp.isclose(weight_sums, 1.0, rtol=1e-10, atol=1e-12))):
            raise ValueError("weights must sum to 1 for every simulation.")
        return weights, rates

    def _wrap_result(self, result: t.Any, argument: DistributionParameter) -> ReturnType:
        """Preserve coupling from the argument and all stochastic parameters."""
        candidates = (argument, self._params["loc"], *self._component_inputs())
        stochastic_inputs = [value for value in candidates if isinstance(value, StochasticScalar)]
        if not stochastic_inputs:
            return scalar_or_array(result)

        wrapped = StochasticScalar(result)
        for value in stochastic_inputs:
            wrapped.coupled_variable_group.merge(value.coupled_variable_group)
        return wrapped

    @staticmethod
    def _mixture_cdf(values: t.Any, weights: t.Any, rates: t.Any) -> t.Any:
        """Evaluate the zero-location CDF for non-negative values."""
        component_cdfs = -xp.expm1(-values[..., None] * rates)
        return xp.sum(weights * component_cdfs, axis=-1)

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute the cumulative distribution function."""
        weights, rates = self._validated_component_parameters()
        values = xp.asarray(x.values if isinstance(x, StochasticScalar) else x)
        loc = self._params["loc"]
        loc_values = loc.values if isinstance(loc, StochasticScalar) else loc
        shifted = values - loc_values
        non_negative = xp.maximum(shifted, 0.0)
        result = self._mixture_cdf(non_negative, weights, rates)
        result = xp.where(shifted <= 0, 0.0, result)
        return self._wrap_result(result, x)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute the inverse CDF using monotone bisection.

        A general hyperexponential quantile has no closed form. The upper
        bracket follows from the slowest exponential component, so no heuristic
        root-search bound is required.
        """
        weights, rates = self._validated_component_parameters()
        probabilities = xp.asarray(u.values if isinstance(u, StochasticScalar) else u)
        interior = (probabilities > 0) & (probabilities < 1)
        safe_probabilities = xp.where(interior, probabilities, 0.5)

        min_rate = xp.min(rates, axis=-1)
        safe_probabilities, min_rate = xp.broadcast_arrays(safe_probabilities, min_rate)
        lower = xp.zeros_like(safe_probabilities, dtype=float)
        upper = -xp.log1p(-safe_probabilities) / min_rate

        for _ in range(64):
            midpoint = (lower + upper) / 2
            midpoint_cdf = self._mixture_cdf(midpoint, weights, rates)
            below = midpoint_cdf < safe_probabilities
            lower = xp.where(below, midpoint, lower)
            upper = xp.where(below, upper, midpoint)

        quantile = (lower + upper) / 2
        quantile = xp.where(
            (probabilities < 0) | (probabilities > 1),
            xp.nan,
            xp.where(probabilities == 0, 0.0, xp.where(probabilities == 1, xp.inf, quantile)),
        )

        loc = self._params["loc"]
        loc_values = loc.values if isinstance(loc, StochasticScalar) else loc
        return self._wrap_result(quantile + loc_values, u)

    @override
    def generate(
        self,
        n_sims: int | None = None,
        rng: RandomGenerator | None = None,
    ) -> StochasticScalar:
        """Generate samples and preserve component-parameter coupling."""
        result = super().generate(n_sims=n_sims, rng=rng)
        for value in self._component_inputs():
            if isinstance(value, StochasticScalar):
                result.coupled_variable_group.merge(value.coupled_variable_group)
        return result

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        """Generate directly by first drawing the mixture component."""
        if n_sims < 1:
            raise ValueError(f"n_sims must be >= 1, got {n_sims}")

        weights, rates = self._validated_component_parameters()
        n_components = len(self._weights)
        if weights.ndim == 1:
            weights = xp.broadcast_to(weights, (n_sims, n_components))
            rates = xp.broadcast_to(rates, (n_sims, n_components))
        elif weights.ndim != 2 or weights.shape[0] != n_sims:
            raise ValueError("Stochastic component parameters must contain exactly n_sims values.")

        component_uniforms = xp.asarray(rng.uniform(size=n_sims))
        cumulative_weights = xp.cumsum(weights, axis=-1)
        components = xp.sum(component_uniforms[:, None] > cumulative_weights, axis=-1).astype(int)
        selected_rates = xp.take_along_axis(rates, components[:, None], axis=-1)[:, 0]

        exponential_uniforms = xp.asarray(rng.uniform(size=n_sims))
        samples = -xp.log1p(-exponential_uniforms) / selected_rates

        loc = self._params["loc"]
        loc_values = loc.values if isinstance(loc, StochasticScalar) else loc
        return StochasticScalar(samples + loc_values)
