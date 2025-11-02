"""Statistical utilities for actuarial loss analysis.

Provides functions for generating loss summaries, percentile calculations,
and statistical analysis of frequency-severity simulation results.
"""

from __future__ import annotations

import math
import typing

import numpy.typing as npt

from ._maths import xp as np
from .types import Numeric

percentiles = np.array([1, 2, 5, 10, 20, 50, 70, 80, 90, 95, 99, 99.5, 99.8, 99.9])

NumberOrList = Numeric | list[Numeric]


if typing.TYPE_CHECKING:
    from .frequency_severity import FreqSevSims


def tvar(values: npt.ArrayLike, p: NumberOrList) -> NumberOrList:
    """Calculate Tail Value at Risk (TVAR) for given percentiles.

    TVAR represents the expected loss above a given percentile threshold.
    Also known as Conditional Value at Risk (CVaR) or Expected Shortfall.

    Args:
        values: Array of loss values to analyze (accepts StochasticScalar via __array__)
        p: Percentile(s) as number or list (e.g., 95 for 95th percentile)

    Returns:
        TVAR value(s) corresponding to the input percentile(s)

    Example:
        >>> import numpy as np
        >>> from pal.stats import tvar
        >>> losses = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> tvar(losses, 80)  # Expected loss above 80th percentile
        9.0
        >>> tvar(losses, [80, 90])  # Multiple percentiles
        [9.0, 9.5]
        >>>
        >>> # Works with StochasticScalar too
        >>> from pal.variables import StochasticScalar
        >>> ss = StochasticScalar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> tvar(ss, 80)  # Automatic conversion via __array__()
        9.0
    """
    values_array = np.asarray(values)
    n_sims = len(values_array)
    if n_sims == 0:
        raise ValueError("Cannot compute TVAR for empty array.")

    # Get the rank of the variable
    rank_positions = np.argsort(values_array)
    if isinstance(p, list):
        result: list[Numeric] = []
        for perc in p:
            idx = math.ceil(perc / 100 * n_sims)
            if idx >= n_sims:
                # For high percentiles with small datasets, return the maximum value
                result.append(float(values_array[rank_positions[-1]]))
                continue
            result.append(float(values_array[rank_positions[idx:]].mean()))
        return result

    idx = math.ceil(p / 100 * n_sims)
    if idx > n_sims:
        raise ValueError(
            f"Percentile {p}% requires more data points than available ({n_sims})"
        )
    # Handle edge case where idx == n_sims (e.g., single value at 50th percentile)
    if idx >= n_sims:
        return float(values_array[rank_positions[-1]])  # Return the maximum value
    return float(values_array[rank_positions[idx:]].mean())


def loss_summary(losses: FreqSevSims) -> dict[str, npt.NDArray[np.floating]]:
    """Generate summary statistics for frequency-severity losses.

    Args:
        losses: Frequency-severity simulation results to summarize.

    Returns:
        Dictionary containing occurrence and aggregate loss percentiles.
    """
    occurrence_losses = losses.occurrence()
    occurrence_statistics = np.percentile(occurrence_losses, percentiles)
    aggregate_losses = losses.aggregate()
    aggregate_statistics = np.percentile(aggregate_losses, percentiles)
    result = {"Occurrence": occurrence_statistics, "Aggregate": aggregate_statistics}
    return result
