"""Configuration utilities for the PAL library.

Provides configuration management for random seeding, simulation parameters,
and global library settings.
"""

import typing as t

from pal._maths import create_random_generator
from pal.qmc import RandomizedQMCGenerator
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
    if config.sampling_method == "rqmc":
        config.rng = RandomizedQMCGenerator(seed, config.qmc_dimensions)
    else:
        config.rng = create_random_generator(seed)


def set_sampling_method(
    method: t.Literal["pseudo", "rqmc"], *, dimensions: int = 32
) -> None:
    """Choose ordinary pseudorandom or randomized Sobol sampling.

    In RQMC mode, PAL allocates Sobol coordinates automatically and recycles
    them in independently scrambled, shuffled blocks. Existing distribution
    and copula generation APIs do not change.
    """
    if method == "rqmc":
        generator = RandomizedQMCGenerator(config.seed, dimensions)
    elif method == "pseudo":
        generator = create_random_generator(config.seed)
    else:
        raise ValueError("method must be 'pseudo' or 'rqmc'")
    config.sampling_method = method
    config.qmc_dimensions = dimensions
    config.rng = generator


def start_new_sampling_block() -> None:
    """Force the next QMC draw to begin a freshly randomized Sobol block.

    Ordinary pseudorandom sampling is unaffected.
    """
    if isinstance(config.rng, RandomizedQMCGenerator):
        config.rng.new_block()
