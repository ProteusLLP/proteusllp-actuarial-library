"""Additional analytical methods for the MBBEFD distribution."""

from __future__ import annotations

import typing as t

import numpy as np

from ._maths import xp
from .distributions import MBBEFD as _MBBEFD
from .distributions import TOLERANCE, ReturnType
from .types import DistributionParameter


class MBBEFD(_MBBEFD):
    """MBBEFD distribution with analytical moment methods."""

    # Keep the public wrapper's documentation identical to the canonical
    # mathematical description defined on the base implementation.
    __doc__ = _MBBEFD.__doc__

    @classmethod
    def from_c(cls, c: DistributionParameter) -> MBBEFD:
        """Construct a Swiss Re MBBEFD curve while preserving this class type."""
        return t.cast(MBBEFD, super().from_c(c))

    def mean(self) -> ReturnType:
        r"""Return the expected damage ratio :math:`E[X]`.

        For the main parameter region,

        .. math::

            E[X] = \frac{(1-b)\log(gb)}{(1-gb)\log(b)}.

        The corresponding continuous limits are used for ``b=1`` and
        ``gb=1``. The degenerate boundary ``g=1`` or ``b=0`` has mean one.
        """
        g, b = self._validated_params()
        degenerate = (g == 1) | (b == 0)
        b_one = xp.isclose(b, 1, rtol=TOLERANCE, atol=TOLERANCE)
        bg_one = xp.isclose(b * g, 1, rtol=TOLERANCE, atol=TOLERANCE)

        safe_g = xp.where(degenerate, 2.0, g)
        safe_b = xp.where(degenerate | b_one | bg_one, 2.0, b)
        limit_g = xp.where(degenerate, 2.0, g)
        limit_b = xp.where(degenerate | b_one, 0.5, b)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            main = (1 - safe_b) * xp.log(safe_g * safe_b) / ((1 - safe_g * safe_b) * xp.log(safe_b))
            b_limit = xp.log(limit_g) / (limit_g - 1)
            bg_limit = (1 - limit_b) / (-xp.log(limit_b))

        result = xp.where(
            degenerate,
            1.0,
            xp.where(b_one, b_limit, xp.where(bg_one, bg_limit, main)),
        )
        return self._wrap_result(result, 0.0)
