"""Configuration utilities for the PAL library.

Provides configuration management for random seeding, simulation parameters,
and global library settings.
"""

from pal._maths import create_random_generator
from pal.types import Config

config = Config()  # config is assumed to be a singleton


def set_default_n_sims(n: int) -> None:
    """Sets the default number of simulations.

    Args:
        n (int): The number of simulations.
    """
    config.n_sims = n


def set_random_seed(seed: int) -> None:
    """Sets the random seed for the simulation.

    Args:
        seed (int): The random seed.
    """
    config.seed = seed
    config.rng = create_random_generator(seed)
