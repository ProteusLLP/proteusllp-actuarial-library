"""Configuration utilities for the PAL library.

Provides configuration management for random seeding, simulation parameters,
logging configuration, and global library settings.
"""

import logging
import logging.config
import os

from pal.types import Config

logging.config.fileConfig(
    os.path.join(os.path.dirname(__file__), "..", "logging.ini"),
    disable_existing_loggers=False,
)

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
    config.rng.bit_generator.state = type(config.rng.bit_generator)(seed).state
