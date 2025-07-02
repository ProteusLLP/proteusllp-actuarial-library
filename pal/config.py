import os

_use_gpu = os.environ.get("PAL_USE_GPU") == "1"

if _use_gpu:
    import cupy as xp

    print("Using GPU")
else:
    import numpy as xp

    xp.seterr(divide="ignore")


class Config:
    """Configuration class for PAL."""

    n_sims = 10000
    seed = 123456789
    rng = xp.random.default_rng(seed)


# Create an instance for backwards compatibility
config = Config()


def set_default_n_sims(n: int) -> None:
    """Sets the default number of simulations.

    Args:
        n (int): The number of simulations.
    """
    Config.n_sims = n


def set_random_seed(seed: int) -> None:
    """Sets the random seed for the simulation.

    Args:
        seed (int): The random seed.
    """
    Config.rng.bit_generator.state = type(Config.rng.bit_generator)(seed).state
