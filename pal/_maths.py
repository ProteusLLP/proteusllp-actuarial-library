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
]
