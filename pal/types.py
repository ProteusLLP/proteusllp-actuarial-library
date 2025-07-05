import dataclasses

from ._maths import xp


@dataclasses.dataclass
class Config:
    """Configuration class for PAL."""

    n_sims: int = 10000
    seed: int = 123456789
    rng: xp.random.Generator = xp.random.default_rng(seed)
