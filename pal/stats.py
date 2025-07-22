"""Statistical utilities for actuarial loss analysis.

Provides functions for generating loss summaries, percentile calculations,
and statistical analysis of frequency-severity simulation results.
"""
import numpy.typing as npt

from ._maths import xp as np
from .frequency_severity import FreqSevSims

percentiles = np.array([1, 2, 5, 10, 20, 50, 70, 80, 90, 95, 99, 99.5, 99.8, 99.9])


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
