import numpy as np


def lognormal_params(mean, sd) -> tuple:
    """Calculates the parameters of a lognormal distribution given the mean and
    standard deviation.

    """
    mu = np.log(mean**2 / np.sqrt(sd**2 + mean**2))
    sigma = np.sqrt(np.log(1 + sd**2 / mean**2))
    return mu, sigma
