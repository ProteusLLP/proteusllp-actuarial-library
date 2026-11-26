"""Backend-specific incomplete beta functions."""

from __future__ import annotations

import typing as t

from ._maths import special, xp

if t.TYPE_CHECKING or xp.__name__ == "numpy":
    betainc = special.betainc
    betaincinv = special.betaincinv
else:
    from ._gpu_beta import betainc, betaincinv

__all__ = ["betainc", "betaincinv"]
