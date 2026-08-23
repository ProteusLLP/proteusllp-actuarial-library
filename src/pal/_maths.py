"""Hardware specific math functions for PAL."""

import logging
import os
import typing as t

import numpy as np
import numpy.typing as npt

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
        xp = np
        import scipy.special as special

        xp.seterr(divide="ignore")


def asnumpy(value: t.Any, dtype: t.Any = None) -> npt.NDArray[t.Any]:
    """Return a host NumPy array, copying from the GPU when required.

    This is the explicit device-to-host boundary used by ``__array__`` and by
    integrations which do not support CuPy arrays.
    """
    if _USE_GPU and isinstance(value, xp.ndarray):
        result = t.cast(t.Any, xp).asnumpy(value)
    else:
        result = np.asarray(value)
    return result if dtype is None else result.astype(dtype, copy=False)


def to_backend(value: t.Any) -> t.Any:
    """Move array-like operands to the active NumPy-compatible backend.

    NumPy dispatches PAL operations through ``__array_ufunc__`` and
    ``__array_function__``. Before calling the original NumPy operation, PAL
    uses this helper to ensure that mixed NumPy/CuPy operands all live on the
    same device.
    """
    if isinstance(value, np.ndarray):
        return xp.asarray(value)
    if isinstance(value, list):
        if not value or all(np.isscalar(item) for item in value):
            return xp.asarray(value)
        return [to_backend(item) for item in value]
    if isinstance(value, tuple):
        if value and all(np.isscalar(item) for item in value):
            return xp.asarray(value)
        return tuple(to_backend(item) for item in value)
    return value


def scalar_or_array(value: t.Any) -> t.Any:
    """Convert zero-dimensional backend results to Python scalar values."""
    if getattr(value, "ndim", None) == 0:
        return value.item()
    return value


def create_random_generator(seed: int) -> t.Any:
    """Create a seeded generator that produces arrays on the active backend.

    CuPy's newer ``Generator`` API does not yet expose every distribution used
    by PAL, whereas ``RandomState`` does. NumPy continues to use its modern
    ``Generator`` API on the CPU.
    """
    if _USE_GPU:
        return xp.random.RandomState(seed)
    return np.random.default_rng(seed)


def generate_upsample_indices(n_sims: int, source_n_sims: int, rng: t.Any) -> t.Any:
    """Generate balanced random indices for resizing a simulation set.

    Complete copies of the source simulations are sampled without replacement.
    The first complete copy remains in its original order; later copies are
    independently permuted. A partial final copy is also sampled without
    replacement.
    """
    if n_sims < 0:
        raise ValueError("n_sims must be non-negative")
    if source_n_sims <= 0:
        raise ValueError("source_n_sims must be positive")
    if n_sims == 0:
        return xp.empty(0, dtype=int)

    full_copies, remainder = divmod(n_sims, source_n_sims)
    parts: list[t.Any] = []

    if full_copies:
        parts.append(xp.arange(source_n_sims, dtype=int))
        for _ in range(full_copies - 1):
            parts.append(xp.asarray(rng.permutation(source_n_sims), dtype=int))

    if remainder:
        permutation = xp.asarray(rng.permutation(source_n_sims), dtype=int)
        parts.append(permutation[:remainder])

    if len(parts) == 1:
        return parts[0]
    return xp.concatenate(parts)


# export the numpy/cupy and scipy/cupyx special functions/modules for the current
# execution environment.
__all__ = [
    "asnumpy",
    "create_random_generator",
    "generate_upsample_indices",
    "scalar_or_array",
    "to_backend",
    "xp",
    "special",
]
