"""Randomized Sobol sampling with automatic dimension recycling.

Each call consumes one Sobol coordinate per requested stochastic variate. A new,
independently scrambled and row-permuted Sobol block is used when the configured
number of coordinates is exhausted. Rows are simulation paths; permutations
apply to whole rows so the joint low-discrepancy design within a block survives.
"""

from __future__ import annotations

import dataclasses
import math
import typing as t
from collections import OrderedDict

import numpy as np
import scipy.special as scipy_special
import scipy.stats as scipy_stats
from scipy.stats import qmc

from ._maths import asnumpy, xp


@dataclasses.dataclass
class _Block:
    """Current Sobol point set and next unused coordinate."""

    points: np.ndarray
    dimension: int = 0


class RandomizedQMCGenerator:
    """A generator-compatible randomized Sobol sampler for PAL.

    An independently Owen-scrambled Sobol design is created for each block.
    Paths are shuffled together between blocks, preserving all coordinates'
    within-block low-discrepancy structure. Each marginal draw consumes a
    coordinate. Calls producing several components consume one coordinate
    per component, including correlated normals used in copula simulation.

    Different simulation counts use separate designs. The small bounded cache
    avoids retaining large event-level designs after frequency-severity draws.

    Args:
        seed: Seed for reproducible scrambling and permutations.
        dimensions: Number of Sobol coordinates before starting a new block.
    """

    def __init__(self, seed: int = 123456789, dimensions: int = 32) -> None:
        """Create an independently randomized sequence of Sobol blocks."""
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 1 <= dimensions <= 21201:
            raise ValueError("dimensions must be an integer between 1 and 21201")
        self.seed = seed
        self.dimensions = dimensions
        self._random = np.random.default_rng(seed)
        self._streams: OrderedDict[int, _Block] = OrderedDict()

    def new_block(self) -> None:
        """Start independent scrambled, shuffled blocks at coordinate one."""
        self._streams.clear()

    def _next_coordinate(self, n: int) -> np.ndarray:
        if n < 1:
            raise ValueError("Simulation count must be positive")
        block = self._streams.get(n)
        if block is None or block.dimension == self.dimensions:
            if n > 2**30:
                raise ValueError("Sobol sampling supports at most 2**30 simulation paths")
            seed = int(self._random.integers(0, 2**32, dtype=np.uint64))
            engine = qmc.Sobol(d=self.dimensions, scramble=True, seed=seed)
            # Drawing a power of two first gives a balanced design for such n
            # and a well-defined prefix for arbitrary simulation counts.
            points = engine.random_base2(m=(n - 1).bit_length())[:n]
            points = points[self._random.permutation(n)]
            block = _Block(points)
            self._streams[n] = block
        self._streams.move_to_end(n)
        while len(self._streams) > 2:
            self._streams.popitem(last=False)
        result = block.points[:, block.dimension].copy()
        block.dimension += 1
        return result

    @staticmethod
    def _host(value: t.Any) -> np.ndarray:
        """Convert distribution parameters to SciPy-compatible arrays."""
        return np.asarray(asnumpy(value))

    def _shape(self, size: t.Any, *parameters: t.Any) -> tuple[int, ...]:
        if size is not None:
            shape = (int(np.asarray(size).item()),) if np.isscalar(size) else tuple(int(part) for part in size)
        else:
            shape = np.broadcast_shapes(*(self._host(parameter).shape for parameter in parameters))
        if any(part < 0 for part in shape):
            raise ValueError("Negative sample dimensions are not allowed")
        return shape

    def _uniform_host(self, shape: tuple[int, ...]) -> np.ndarray:
        if not shape:
            return np.asarray(self._next_coordinate(1)[0])
        if 0 in shape:
            return np.empty(shape, dtype=float)
        # PAL's multivariate generators conventionally have a simulation axis
        # and one or more component axes. The largest axis is the simulation
        # axis (last for (dimensions, n_sims), first for (n_sims, d, d)).
        axis = int(np.argmax(shape))
        n = shape[axis]
        component_shape = shape[:axis] + shape[axis + 1 :]
        count = math.prod(component_shape)
        data = np.stack([self._next_coordinate(n) for _ in range(count)], axis=1)
        data = data.reshape((n, *component_shape))
        return np.moveaxis(data, 0, axis)

    def _quantiles(
        self,
        func: t.Callable[..., t.Any],
        size: t.Any,
        *parameters: t.Any,
        discrete: bool = False,
    ) -> t.Any:
        shape = self._shape(size, *parameters)
        uniforms = self._uniform_host(shape)
        # Avoid infinities in inverse CDFs of unbounded distributions.
        uniforms = np.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        result = np.asarray(func(uniforms, *(self._host(param) for param in parameters)))
        if discrete:
            result = result.astype(np.int64)
        return xp.asarray(result)

    def random(self, size: t.Any = None) -> t.Any:
        """Draw U(0,1) values, using one coordinate per component."""
        return xp.asarray(self._uniform_host(self._shape(size)))

    def uniform(self, low: t.Any = 0.0, high: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw continuous uniform values using the active Sobol coordinate."""
        shape = self._shape(size, low, high)
        lo, hi = self._host(low), self._host(high)
        return xp.asarray(lo + (hi - lo) * self._uniform_host(shape))

    def standard_normal(self, size: t.Any = None) -> t.Any:
        """Draw standard normal quantiles."""
        return self._quantiles(scipy_special.ndtri, size)

    def normal(self, loc: t.Any = 0.0, scale: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw normal quantiles."""
        return self._quantiles(lambda u, mean, std: mean + std * scipy_special.ndtri(u), size, loc, scale)

    def exponential(self, scale: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw exponential quantiles."""
        return self._quantiles(lambda u, s: -s * np.log1p(-u), size, scale)

    def gamma(self, shape: t.Any, scale: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw gamma quantiles."""
        return self._quantiles(lambda u, a, s: scipy_special.gammaincinv(a, u) * s, size, shape, scale)

    def beta(self, a: t.Any, b: t.Any, size: t.Any = None) -> t.Any:
        """Draw beta quantiles."""
        return self._quantiles(scipy_special.betaincinv, size, a, b)

    def poisson(self, lam: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw Poisson quantiles."""
        return self._quantiles(scipy_stats.poisson.ppf, size, lam, discrete=True)

    def negative_binomial(self, n: t.Any, p: t.Any, size: t.Any = None) -> t.Any:
        """Draw negative-binomial quantiles."""
        return self._quantiles(scipy_stats.nbinom.ppf, size, n, p, discrete=True)

    def binomial(self, n: t.Any, p: t.Any, size: t.Any = None) -> t.Any:
        """Draw binomial quantiles."""
        return self._quantiles(scipy_stats.binom.ppf, size, n, p, discrete=True)

    def hypergeometric(self, ngood: t.Any, nbad: t.Any, nsample: t.Any, size: t.Any = None) -> t.Any:
        """Draw hypergeometric quantiles."""
        return self._quantiles(
            lambda u, good, bad, sample: scipy_stats.hypergeom.ppf(u, good + bad, good, sample),
            size,
            ngood,
            nbad,
            nsample,
            discrete=True,
        )

    def standard_t(self, df: t.Any, size: t.Any = None) -> t.Any:
        """Draw Student's t quantiles."""
        return self._quantiles(scipy_stats.t.ppf, size, df)

    def chisquare(self, df: t.Any, size: t.Any = None) -> t.Any:
        """Draw chi-squared quantiles."""
        return self._quantiles(lambda u, degrees: 2 * scipy_special.gammaincinv(degrees / 2, u), size, df)

    def noncentral_chisquare(self, df: t.Any, nonc: t.Any, size: t.Any = None) -> t.Any:
        """Draw noncentral chi-squared quantiles."""
        return self._quantiles(scipy_stats.ncx2.ppf, size, df, nonc)

    def logseries(self, p: t.Any, size: t.Any = None) -> t.Any:
        """Draw logarithmic-series quantiles."""
        return self._quantiles(scipy_stats.logser.ppf, size, p, discrete=True)

    def integers(self, low: t.Any, high: t.Any = None, size: t.Any = None, endpoint: bool = False) -> t.Any:
        """Draw uniformly distributed integers using quantile mapping."""
        if high is None:
            low, high = 0, low
        if endpoint:
            high = self._host(high) + 1
        return self._quantiles(
            lambda u, minimum, maximum: np.floor(minimum + (maximum - minimum) * u), size, low, high, discrete=True
        )
