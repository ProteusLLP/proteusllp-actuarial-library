"""Configuration utilities for the PAL library.

Provides configuration management for random seeding, simulation parameters,
and global library settings.
"""

from pal._maths import create_random_generator
from pal.qmc import RandomizedQMCGenerator
from pal.types import Config
from pal.types import SamplingMethod as SamplingMethod

__all__ = [
    "SamplingMethod",
    "config",
    "set_default_n_sims",
    "set_random_seed",
    "set_sampling_method",
    "start_new_sampling_block",
]

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
    if config.sampling_method == SamplingMethod.RQMC:
        config.rng = RandomizedQMCGenerator(seed, config.qmc_dimensions)
    else:
        config.rng = create_random_generator(seed)


def set_sampling_method(method: SamplingMethod, *, dimensions: int = 32) -> None:
    """Select Monte Carlo or randomized Sobol sampling.

    A `SamplingMethod` member is preferred, for example
    `set_sampling_method(SamplingMethod.RQMC)`. Previously accepted string
    values remain valid for compatibility.

    In RQMC mode, PAL allocates Sobol coordinates automatically and recycles
    them in independently scrambled, shuffled blocks.
    """
    try:
        selected = SamplingMethod(method)
    except ValueError as exc:
        raise ValueError("method must be SamplingMethod.MC or SamplingMethod.RQMC") from exc

    if selected is SamplingMethod.RQMC:
        generator = RandomizedQMCGenerator(config.seed, dimensions)
    else:
        generator = create_random_generator(config.seed)
    config.sampling_method = selected
    config.qmc_dimensions = dimensions
    config.rng = generator


def start_new_sampling_block() -> None:
    """Force the next QMC draw to begin a freshly randomized Sobol block.

    Ordinary pseudorandom sampling is unaffected.
    """
    if isinstance(config.rng, RandomizedQMCGenerator):
        config.rng.new_block()
