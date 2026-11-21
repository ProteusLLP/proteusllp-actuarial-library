"""GPU-native exact sampling for the half-order GIG distribution.

Only the p=+/-1/2 family is accelerated: the remaining GIG indices use
SciPy through the established host fallback. No custom CUDA kernels.
"""

from __future__ import annotations

import typing as t

from ._maths import xp


def generalized_inverse_gaussian_mc(
    rng: t.Any, n_sims: int, p: t.Any, shape: t.Any, scale: t.Any, loc: t.Any
) -> t.Any:
    """Sample half-order GIG exactly by a two-draw inverse-Gaussian transform.

    Standard GIG(-1/2,b) equals IG(mean=1,shape=b). Reciprocal symmetry
    gives GIG(+1/2,b). All random draws and array operations are on the
    active backend, with only a scalar condition check for the index family.
    """
    index = xp.broadcast_to(xp.asarray(p, dtype=xp.float64), (n_sims,))
    if not bool(xp.all(xp.abs(index) == 0.5)):
        raise NotImplementedError("GPU-native GIG sampling supports only p=+/-1/2")
    b = xp.broadcast_to(xp.asarray(shape, dtype=xp.float64), (n_sims,))
    factor = xp.broadcast_to(xp.asarray(scale, dtype=xp.float64), (n_sims,))
    location = xp.broadcast_to(xp.asarray(loc, dtype=xp.float64), (n_sims,))
    squared_normal = rng.normal(size=n_sims) ** 2
    # Rationalised Michael-Schucany-Haas root avoids cancellation.
    root = 2 * b / (2 * b + squared_normal + xp.sqrt(squared_normal * (4 * b + squared_normal)))
    choice = rng.uniform(size=n_sims)
    inverse_gaussian = xp.where(choice <= 1 / (1 + root), root, 1 / root)
    sampled = xp.where(index == 0.5, 1 / inverse_gaussian, inverse_gaussian)
    return location + factor * sampled
