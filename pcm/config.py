import os

_use_gpu = os.environ.get("RIPPY_USE_GPU") == "1"

if _use_gpu:
    import cupy as xp

    print("Using GPU")
else:
    import numpy as xp

    xp.seterr(divide="ignore")


class config:
    """
    Configuration class for Rippy.
    """

    n_sims = 10000
    seed = 123456789
    rng = xp.random.default_rng(seed)


def set_default_n_sims(n):
    """
    Sets the default number of simulations.

    Args:
        n (int): The number of simulations.
    """
    config.n_sims = n


def set_random_seed(seed):
    """
    Sets the random seed for the simulation.

    Args:
        seed (int): The random seed.
    """
    config.rng.bit_generator.state = type(config.rng.bit_generator)(seed).state
