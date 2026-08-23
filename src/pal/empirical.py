"""Empirical distribution.

This module contains a finite empirical distribution defined by observed samples
and optional observation weights. It is separate from :mod:`pal.distributions`
because its support and weights are vector-valued inputs rather than scalar
distribution parameters.
"""

from __future__ import annotations

import typing as t

from ._compat import override
from ._maths import scalar_or_array, xp
from .distributions import DiscreteDistributionBase, ReturnType
from .stochastic_scalar import StochasticScalar
from .types import DistributionParameter, RandomGenerator


class Empirical(DiscreteDistributionBase):
    r"""Empirical distribution defined by observed samples.

    For observations :math:`x_1,\ldots,x_n` with non-negative weights
    :math:`w_1,\ldots,w_n`, let

    .. math::

        p_i = \frac{w_i}{\sum_{j=1}^n w_j}.

    The empirical cumulative distribution function is

    .. math::

        F(x) = \sum_{i=1}^n p_i\,\mathbf{1}\{x_i \leq x\}.

    If ``weights`` is omitted, all observations receive equal probability
    :math:`1/n`. The weights need not already sum to one; PAL normalises them
    internally. Zero-weight observations are ignored.

    The inverse CDF is the generalized inverse

    .. math::

        F^{-1}(u) = \inf\{x : F(x) \geq u\}, \qquad 0 \leq u \leq 1.

    Generation resamples the observed values with replacement using the
    empirical probabilities. The observed samples may be supplied directly as
    a :class:`~pal.stochastic_scalar.StochasticScalar`; they are treated as the
    fixed empirical support rather than as scenario-varying distribution
    parameters, so resampled values are independent of the source coupling
    group.

    Parameters:
        samples: One-dimensional finite numeric observations.
        weights: Optional non-negative observation weights. Must have the same
            length as ``samples`` and contain at least one positive value.
    """

    def __init__(self, samples: t.Any, weights: t.Any | None = None) -> None:
        """Initialize an empirical distribution.

        Args:
            samples: One-dimensional finite numeric observations.
            weights: Optional non-negative observation weights.
        """
        sample_values = samples.values if isinstance(samples, StochasticScalar) else samples
        try:
            sample_array = xp.asarray(sample_values)
        except (TypeError, ValueError) as exc:
            raise TypeError("samples must be a one-dimensional numeric sequence.") from exc

        if sample_array.ndim != 1:
            raise ValueError("samples must be one-dimensional.")
        if sample_array.size == 0:
            raise ValueError("samples must contain at least one observation.")
        if sample_array.dtype.kind not in "iuf":
            raise TypeError("samples must contain numeric real values.")
        if bool(xp.any(~xp.isfinite(sample_array))):
            raise ValueError("samples must be finite.")

        if weights is None:
            weight_array = xp.ones(sample_array.shape, dtype=float)
        else:
            weight_values = weights.values if isinstance(weights, StochasticScalar) else weights
            try:
                weight_array = xp.asarray(weight_values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError("weights must be a one-dimensional numeric sequence.") from exc

            if weight_array.ndim != 1:
                raise ValueError("weights must be one-dimensional.")
            if weight_array.shape != sample_array.shape:
                raise ValueError("weights must have the same length as samples.")
            if bool(xp.any(~xp.isfinite(weight_array))):
                raise ValueError("weights must be finite.")
            if bool(xp.any(weight_array < 0)):
                raise ValueError("weights must be non-negative.")

        positive = weight_array > 0
        if not bool(xp.any(positive)):
            raise ValueError("weights must contain at least one positive value.")

        sample_array = sample_array[positive]
        weight_array = weight_array[positive]
        order = xp.argsort(sample_array)
        self._samples = sample_array[order]
        normalized_weights = weight_array[order] / xp.sum(weight_array)
        self._cumulative_weights = xp.cumsum(normalized_weights)
        self._cumulative_weights[-1] = 1.0
        super().__init__()

    def _wrap_result(self, result: t.Any, argument: DistributionParameter) -> ReturnType:
        """Preserve coupling from a stochastic CDF or inverse-CDF argument."""
        if not isinstance(argument, StochasticScalar):
            return scalar_or_array(result)

        wrapped = StochasticScalar(result)
        wrapped.coupled_variable_group.merge(argument.coupled_variable_group)
        return wrapped

    @override
    def cdf(self, x: DistributionParameter) -> ReturnType:
        """Compute the weighted empirical cumulative distribution function."""
        values = xp.asarray(x.values if isinstance(x, StochasticScalar) else x)
        indices = xp.searchsorted(self._samples, values, side="right")
        cumulative = xp.concatenate((xp.zeros(1, dtype=float), self._cumulative_weights))
        result = cumulative[indices]
        result = xp.where(xp.isnan(values), xp.nan, result)
        return self._wrap_result(result, x)

    @override
    def invcdf(self, u: DistributionParameter) -> ReturnType:
        """Compute the generalized inverse empirical CDF."""
        probabilities = xp.asarray(u.values if isinstance(u, StochasticScalar) else u)
        invalid = (probabilities < 0) | (probabilities > 1) | xp.isnan(probabilities)
        safe_probabilities = xp.clip(probabilities, 0.0, 1.0)
        indices = xp.searchsorted(self._cumulative_weights, safe_probabilities, side="left")
        quantiles = self._samples[indices].astype(float, copy=False)
        result = xp.where(invalid, xp.nan, quantiles)
        return self._wrap_result(result, u)

    @override
    def _generate(self, n_sims: int, rng: RandomGenerator) -> StochasticScalar:
        """Resample observations with replacement using the empirical weights."""
        if n_sims < 1:
            raise ValueError(f"n_sims must be >= 1, got {n_sims}")

        uniforms = xp.asarray(rng.uniform(size=n_sims))
        indices = xp.searchsorted(self._cumulative_weights, uniforms, side="left")
        return StochasticScalar(self._samples[indices])
