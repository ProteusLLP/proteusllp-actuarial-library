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
        result = xp.asnumpy(value)
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

# export the numpy/cupy and scipy/cupyx special functions/modules for the current
# execution environment.
__all__ = [
    "asnumpy",
    "scalar_or_array",
    "to_backend",
    "xp",
    "special",
]
