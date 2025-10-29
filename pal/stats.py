from pal import FreqSevSims
from .config import xp as np

percentiles = np.array([1, 2, 5, 10, 20, 50, 70, 80, 90, 95, 99, 99.5, 99.8, 99.9])


def loss_summary(losses: FreqSevSims):
    occurrence_losses = losses.occurrence()
    occurrence_statistics = np.percentile(occurrence_losses, percentiles)
    aggregate_losses = losses.aggregate()
    aggregate_statistics = np.percentile(aggregate_losses, percentiles)
    result = {"Occurrence": occurrence_statistics, "Aggregate": aggregate_statistics}
    return result
