"""Hardware specific math functions for PAL."""

import logging
import os
import typing as t

_USE_GPU_ENV_VAR = "PAL_USE_GPU"
_USE_GPU = os.environ.get(_USE_GPU_ENV_VAR) == "1"
LOGGER = logging.getLogger(__file__)


if t.TYPE_CHECKING:
    # For type checking, we need to ensure that xp and special are defined
    # even if we don't use them at runtime.
    import numpy as xp
    import scipy.special as special
else:
    if _USE_GPU:
        LOGGER.info("Using GPU")
        import cupy as xp
        import cupyx.scipy.special as special
    else:
        LOGGER.info("No GPU hardware detected. Using CPU.")
        import numpy as xp
        import scipy.special as special

        xp.seterr(divide="ignore")

# export the numpy/cupy and scipy/cupyx special functions/modules for the current
# execution environment.
__all__ = [
    "xp",
    "special",
    "generate_upsample_indices",
]


def generate_upsample_indices(
    target_n_sims: int, current_n_sims: int, seed: int | None = None
) -> xp.ndarray:  # type: ignore[name-defined]
    """Generate indices for upsampling or downsampling simulations.

    Creates an array of indices that map from current simulations to target simulations.
    For downsampling (target < current), randomly selects without replacement.
    For upsampling (target > current), creates random permutations plus a remainder.

    Args:
        target_n_sims: The target number of simulations.
        current_n_sims: The current number of simulations.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of indices with length target_n_sims, containing values in
        [0, current_n_sims).

    Examples:
        >>> # Downsample: 10 -> 3 (random selection)
        >>> indices = generate_upsample_indices(3, 10, seed=42)
        >>> # Returns 3 unique indices from [0, 9]

        >>> # Upsample: 3 -> 10 (3 full rounds + 1 partial)
        >>> indices = generate_upsample_indices(10, 3, seed=42)
        >>> # Returns 10 indices: 3 permutations of [0,1,2] + 1 random selection
    """
    # Set random seed if provided
    if seed is not None:
        xp.random.seed(seed)  # type: ignore[attr-defined]

    # Calculate full chunks and remainder
    full_chunks = target_n_sims // current_n_sims
    remainder = target_n_sims % current_n_sims

    # Determine structure upfront
    has_ordered_chunk = full_chunks >= 1
    random_full_chunks = full_chunks - 1 if has_ordered_chunk else 0

    # Build indices array efficiently
    chunks: list[xp.ndarray] = []  # type: ignore[name-defined]

    # 1. Add ordered chunk if needed
    if has_ordered_chunk:
        chunks.append(xp.arange(current_n_sims))  # type: ignore[attr-defined]

    # 2. Add random full chunks
    for _ in range(random_full_chunks):
        chunks.append(xp.random.permutation(current_n_sims))  # type: ignore[attr-defined]

    # 3. Add remainder (random selection without replacement)
    if remainder > 0:
        chunks.append(
            xp.random.choice(current_n_sims, size=remainder, replace=False)  # type: ignore[attr-defined]
        )

    indices: xp.ndarray = xp.concatenate(chunks) if chunks else xp.array([], dtype=int)  # type: ignore[attr-defined,name-defined]
    return indices
