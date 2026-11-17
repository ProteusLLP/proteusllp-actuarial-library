"""Randomized Sobol sampling with automatic dimension recycling.

Each call consumes one Sobol coordinate per requested stochastic variate. A new,
independently scrambled and row-permuted Sobol block is used when the configured
number of coordinates is exhausted. Rows are simulation paths; permutations
apply to whole rows so the joint low-discrepancy design within a block survives.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import typing as t
from collections import OrderedDict

import numpy as np
import scipy.special as scipy_special
import scipy.stats as scipy_stats
from scipy.stats import qmc

from ._maths import asnumpy, special, xp


@dataclasses.dataclass
class _Block:
    """Current Sobol point set and next unused coordinate."""

    points: t.Any
    permutation: t.Any
    shifts: t.Any = None
    dimension: int = 0

    @property
    def nbytes(self) -> int:
        """Memory retained by the design, scrambling and path permutation."""
        return self.points.nbytes + self.permutation.nbytes + (
            self.shifts.nbytes if self.shifts is not None else 0
        )


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
        # Retain only a small memory budget across different path counts.
        # The active block is kept even if its design alone exceeds the cap.
        self.max_cache_bytes = 128 * 1024 * 1024

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
            if xp is np:
                seed = int(self._random.integers(0, 2**32, dtype=np.uint64))
                engine = qmc.Sobol(d=self.dimensions, scramble=True, seed=seed)
                # Preserve the base-two Sobol balance where n is a power of two.
                points = engine.random_base2(m=(n - 1).bit_length())[:n]
                # Retain one row index instead of copying the full d-column
                # matrix solely to permute paths.
                block = _Block(points, self._random.permutation(n))
            else:
                block = self._gpu_points(n)
            self._streams[n] = block
        self._streams.move_to_end(n)
        while len(self._streams) > 1 and (
            len(self._streams) > 2 or sum(b.nbytes for b in self._streams.values()) > self.max_cache_bytes
        ):
            self._streams.popitem(last=False)
        if block.shifts is None:
            result = block.points[block.permutation, block.dimension]
        else:
            # cuRAND's Sobol32 output stays in compact uint32 form until a
            # coordinate is requested. This halves retained GPU memory versus
            # caching float64 designs and avoids full-matrix permutation copies.
            values = xp.bitwise_xor(block.points[block.dimension, block.permutation], block.shifts[block.dimension])
            result = (values.astype(xp.float64) + 0.5) * (2.0**-32)
        block.dimension += 1
        return result

    def _gpu_points(self, n: int) -> t.Any:
        """Generate scrambled Sobol points in device memory using NVIDIA cuRAND.

        CuPy does not expose the multidimensional cuRAND Sobol generator at
        the random.Generator level. The NVIDIA nvmath bindings expose its host
        API without requiring PAL to compile or maintain CUDA kernels.

        cuRAND's built-in scramble is fixed rather than controlled by a seed.
        An independent seeded 32-bit digital shift per coordinate gives
        randomized, reproducible Sobol blocks, followed by a common row
        permutation to decorrelate successive blocks. CPU and GPU streams are
        reproducible separately, but are not bitwise identical.
        """
        if self.dimensions > 20000:
            raise ValueError("cuRAND supports at most 20000 Sobol dimensions")
        try:
            curand: t.Any = importlib.import_module("nvmath.bindings.curand")
        except ImportError as exc:
            raise RuntimeError("GPU RQMC requires nvmath-python; install PAL with the gpu extra.") from exc

        gpu_xp: t.Any = xp
        padded = 1 << (n - 1).bit_length()
        generator = curand.create_generator(curand.RngType.QUASI_SCRAMBLED_SOBOL32)
        try:
            curand.set_quasi_random_generator_dimensions(generator, self.dimensions)
            curand.set_stream(generator, gpu_xp.cuda.get_current_stream().ptr)
            # cuRAND's QUASI_DEFAULT ordering stores all paths of one
            # coordinate contiguously. It requires an exact multiple of d.
            raw = gpu_xp.empty((self.dimensions, padded), dtype=gpu_xp.uint32)
            curand.generate(generator, raw.data.ptr, raw.size)
        finally:
            curand.destroy_generator(generator)

        # Keep the digital shift and permutation separate from the raw Sobol
        # matrix, applying them to each coordinate as it is consumed.
        shifts = gpu_xp.asarray(self._random.integers(0, 2**32, size=self.dimensions, dtype=np.uint32))
        permutation_seed = int(self._random.integers(0, 2**32, dtype=np.uint64))
        permutation = gpu_xp.random.RandomState(permutation_seed).permutation(n)
        return _Block(raw, permutation, shifts)

    @staticmethod
    def _host(value: t.Any) -> np.ndarray:
        """Convert distribution parameters to SciPy-compatible arrays."""
        return np.asarray(asnumpy(value))

    def _shape(self, size: t.Any, *parameters: t.Any) -> tuple[int, ...]:
        if size is not None:
            shape = (int(np.asarray(size).item()),) if np.isscalar(size) else tuple(int(part) for part in size)
        else:
            # Inspect shapes without transferring stochastic GPU parameters.
            shape = np.broadcast_shapes(
                *(
                    parameter.shape if hasattr(parameter, "shape") else np.asarray(parameter).shape
                    for parameter in parameters
                )
            )
        if any(part < 0 for part in shape):
            raise ValueError("Negative sample dimensions are not allowed")
        return shape

    def _uniform_values(self, shape: tuple[int, ...]) -> t.Any:
        if not shape:
            return xp.asarray(self._next_coordinate(1)[0])
        if 0 in shape:
            return xp.empty(shape, dtype=float)
        # PAL's multivariate generators conventionally have a simulation axis
        # and one or more component axes. The largest axis is the simulation
        # axis (last for (dimensions, n_sims), first for (n_sims, d, d)).
        axis = int(np.argmax(shape))
        n = shape[axis]
        component_shape = shape[:axis] + shape[axis + 1 :]
        count = math.prod(component_shape)
        if count == 1:
            return self._next_coordinate(n).reshape(shape)
        data = xp.empty((n, count), dtype=xp.float64)
        for component in range(count):
            data[:, component] = self._next_coordinate(n)
        data = data.reshape((n, *component_shape))
        return xp.moveaxis(data, 0, axis)

    def _quantiles(
        self,
        func: t.Callable[..., t.Any],
        size: t.Any,
        *parameters: t.Any,
        discrete: bool = False,
        gpu_func: t.Callable[..., t.Any] | None = None,
    ) -> t.Any:
        shape = self._shape(size, *parameters)
        uniforms = self._uniform_values(shape)
        # Avoid infinities in inverse CDFs of unbounded distributions.
        uniforms = xp.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        if xp is not np and gpu_func is not None:
            # Keep the Sobol points, parameters and quantiles on the device.
            return xp.asarray(gpu_func(uniforms, *(xp.asarray(param) for param in parameters)))
        # Discrete PPFs and unsupported special functions still use SciPy.
        result = np.asarray(func(asnumpy(uniforms), *(self._host(param) for param in parameters)))
        if discrete:
            result = result.astype(np.int64)
        return xp.asarray(result)

    def random(self, size: t.Any = None) -> t.Any:
        """Draw U(0,1) values, using one coordinate per component."""
        return self._uniform_values(self._shape(size))

    def uniform(self, low: t.Any = 0.0, high: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw continuous uniform values using the active Sobol coordinate."""
        shape = self._shape(size, low, high)
        lo, hi = xp.asarray(low), xp.asarray(high)
        return lo + (hi - lo) * self._uniform_values(shape)

    def standard_normal(self, size: t.Any = None) -> t.Any:
        """Draw standard normal quantiles."""
        return self._quantiles(scipy_special.ndtri, size, gpu_func=special.ndtri)

    def normal(self, loc: t.Any = 0.0, scale: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw normal quantiles."""
        return self._quantiles(
            lambda u, mean, std: mean + std * scipy_special.ndtri(u),
            size,
            loc,
            scale,
            gpu_func=lambda u, mean, std: mean + std * special.ndtri(u),
        )

    def exponential(self, scale: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw exponential quantiles."""
        return self._quantiles(
            lambda u, s: -s * np.log1p(-u),
            size,
            scale,
            gpu_func=lambda u, s: -s * xp.log1p(-u),
        )

    def gamma(self, shape: t.Any, scale: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw gamma quantiles."""
        return self._quantiles(
            lambda u, a, s: scipy_special.gammaincinv(a, u) * s,
            size,
            shape,
            scale,
            gpu_func=lambda u, a, s: special.gammaincinv(a, u) * s,
        )

    def beta(self, a: t.Any, b: t.Any, size: t.Any = None) -> t.Any:
        """Draw beta quantiles."""
        return self._quantiles(
            lambda u, alpha, beta: scipy_special.betaincinv(alpha, beta, u),
            size,
            a,
            b,
            gpu_func=lambda u, alpha, beta: special.betaincinv(alpha, beta, u),
        )

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
        return self._quantiles(
            lambda u, degrees: 2 * scipy_special.gammaincinv(degrees / 2, u),
            size,
            df,
            gpu_func=lambda u, degrees: 2 * special.gammaincinv(degrees / 2, u),
        )

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
