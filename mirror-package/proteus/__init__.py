"""Mirror package for backward compatibility.

This package (proteus-actuarial-library) is now a mirror of the renamed
proteusllp-actuarial-library package. All functionality is re-exported
from the main package.

For new projects, please use proteusllp-actuarial-library directly.
"""

__version__ = "0.2.8"

from pal import *  # noqa: F401, F403

__all__ = ["__version__"]
