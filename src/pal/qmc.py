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

from ._gpu_distributions import generalized_inverse_gaussian_mc
from ._maths import asnumpy, special, to_backend, xp


@dataclasses.dataclass
class _Block:
    """Scrambled direction tables and one Gray-code simulation-index vector.

    No path-by-dimension point matrix is retained. Each coordinate is
    reconstructed only when requested using the same multidimensional Sobol
    direction vectors and one shared ordering of simulation indices.

    Attributes:
        dimension: Number of Sobol coordinates allocated to this block.
        gray: Gray-code indices for the Sobol points.
        shifts: Random digital shifts for the block.
        tables: Precomputed lookup tables of direction-number contributions.
    """

    gray: t.Any
    tables: t.Any
    shifts: t.Any
    dimension: int = 0

    @property
    def nbytes(self) -> int:
        """Memory retained by direction tables, shifts and path indices."""
        return self.gray.nbytes + self.tables.nbytes + self.shifts.nbytes


class RandomizedQMCGenerator:
    """A generator-compatible randomized Sobol sampler for PAL.

    Independently randomized Sobol designs are created for each block.
    Paths are shuffled together between blocks, preserving all coordinates'
    within-block low-discrepancy structure. Each marginal draw consumes a
    coordinate. Calls producing several components consume one coordinate
    per component, including correlated normals used in copula simulation.

    The generator retains scrambled direction numbers, digital shifts and
    one Gray-code path-index vector. It reconstructs each coordinate on demand
    instead of allocating a full paths-by-dimensions point matrix. Different
    simulation counts use separate bounded-cache designs.

    Args:
        seed: Seed for reproducible scrambling and permutations.
        dimensions: Number of Sobol coordinates before starting a new block.

    Attributes:
        dimensions: Number of Sobol coordinates available per randomized block.
        max_cache_bytes: Maximum memory budget for cached Sobol blocks.
        seed: Seed used to initialize the random number generator.
    """

    def __init__(self, seed: int = 123456789, dimensions: int = 32) -> None:
        """Create an independently randomized sequence of Sobol blocks.

        Args:
            dimensions: Number of Sobol coordinates available in each randomized block.
            seed: Seed controlling the reproducible randomisation of the generator.
        """
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 1 <= dimensions <= 21201:
            raise ValueError("dimensions must be an integer between 1 and 21201")
        self.seed = seed
        self.dimensions = dimensions
        self._random = np.random.default_rng(seed)
        self._streams: OrderedDict[int, _Block] = OrderedDict()
        # A small bounded cache avoids keeping path-index arrays of many sizes.
        self.max_cache_bytes = 128 * 1024 * 1024

    def new_block(self) -> None:
        """Start independent scrambled, shuffled blocks at coordinate one."""
        self._streams.clear()

    @staticmethod
    def _lookup_tables(vectors: np.ndarray) -> np.ndarray:
        """Build compact 8-bit XOR tables for 32-bit Sobol direction vectors.

        Args:
            vectors: Sobol direction numbers used to construct lookup tables.
        """
        dimensions = vectors.shape[0]
        tables = np.zeros((dimensions, 4, 256), dtype=np.uint32)
        byte_values = np.arange(256, dtype=np.uint32)
        for bit in range(32):
            selected = (byte_values >> (bit % 8)) & 1
            tables[:, bit // 8, :] ^= vectors[:, bit, None] * selected[None, :]
        return tables

    def _new_direction_block(self, n: int) -> _Block:
        """Create one scrambled multidimensional Sobol block without its points.

        Args:
            n: Number of points or trials required by the operation.
        """
        if xp is np:
            seed = int(self._random.integers(0, 2**32, dtype=np.uint64))
            engine: t.Any = qmc.Sobol(d=self.dimensions, scramble=True, bits=32, seed=seed)
            # SciPy's scrambled direction numbers and digital shifts are
            # small, but these attributes are private. Guard their layout
            # and test the generated coordinates against its public API.
            vectors = np.asarray(engine._sv, dtype=np.uint32)
            shifts = np.asarray(engine._shift, dtype=np.uint32)
            if vectors.shape != (self.dimensions, 32) or shifts.shape != (self.dimensions,):
                raise RuntimeError("Unexpected SciPy Sobol direction-vector layout")
            permutation = self._random.permutation(n).astype(np.uint32)
            gray = permutation ^ (permutation >> 1)
            tables = self._lookup_tables(vectors)
            return _Block(gray, tables, shifts)

        if self.dimensions > 20000:
            raise ValueError("cuRAND supports at most 20000 Sobol dimensions")
        try:
            curand: t.Any = importlib.import_module("nvmath.bindings.curand")
        except ImportError as exc:
            raise RuntimeError("GPU RQMC requires nvmath-python; install PAL with the gpu extra.") from exc

        vectors = np.asarray(
            curand.get_direction_vectors32(
                curand.DirectionVectorSet.SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
                self.dimensions,
            ),
            dtype=np.uint32,
        )
        if vectors.shape != (self.dimensions, 32):
            raise RuntimeError("Unexpected cuRAND Sobol direction-vector layout")
        shifts_host = np.array(curand.get_scramble_constants32(self.dimensions), dtype=np.uint32, copy=True)
        # The bundled NVIDIA scrambled directions are fixed. Applying a
        # separately seeded digital shift per block gives independent RQMC
        # replications and preserves Sobol balance in each coordinate.
        shifts_host ^= self._random.integers(0, 2**32, size=self.dimensions, dtype=np.uint32)
        gpu_xp: t.Any = xp
        permutation_seed = int(self._random.integers(0, 2**32, dtype=np.uint64))
        permutation = gpu_xp.random.RandomState(permutation_seed).permutation(n).astype(gpu_xp.uint32)
        gray = permutation ^ (permutation >> 1)
        return _Block(gpu_xp.asarray(gray), gpu_xp.asarray(self._lookup_tables(vectors)), gpu_xp.asarray(shifts_host))

    def _active_block(self, n: int) -> _Block:
        """Return the next available Sobol block for a simulation count.

        Args:
            n: Number of points or trials required by the operation.
        """
        if n < 1:
            raise ValueError("Simulation count must be positive")
        if n > 2**30:
            raise ValueError("Sobol sampling supports at most 2**30 simulation paths")
        block = self._streams.get(n)
        if block is None or block.dimension == self.dimensions:
            block = self._new_direction_block(n)
            self._streams[n] = block
        self._streams.move_to_end(n)
        while len(self._streams) > 1 and (
            len(self._streams) > 2 or sum(b.nbytes for b in self._streams.values()) > self.max_cache_bytes
        ):
            self._streams.popitem(last=False)
        return block

    def _next_coordinate(self, n: int) -> t.Any:
        """Construct one scrambled Sobol coordinate, retaining O(n + d) state.

        Args:
            n: Number of points or trials required by the operation.
        """
        block = self._active_block(n)
        j = block.dimension
        values = xp.full((n,), block.shifts[j], dtype=xp.uint32)
        for byte in range(4):
            values ^= block.tables[j, byte, (block.gray >> (byte * 8)) & 255]
        block.dimension += 1
        if xp is np:
            return values.astype(np.float64) * (2.0**-32)
        # GPU Sobol32 midpoints avoid the extremes of inverse CDF transforms.
        return (values.astype(xp.float64) + 0.5) * (2.0**-32)

    def _active_uniforms(self, n_sims: int, path_indices: t.Any, size: tuple[int, ...], *, scenario_axis: int) -> t.Any:
        """Take Sobol coordinates for selected paths of the original design.

        Each component advances the *n_sims* stream once, even when only a
        subset of paths remains active in an iterative simulation algorithm.
        The direction lookup evaluates only those active rows. Consequently,
        shrinking active sets do not start unrelated Sobol sequences, and all
        paths in one iteration share the same coordinate.

        Args:
            n_sims: Number of simulations on the scenario axis.
            path_indices: Indices of simulations that need draws from the active Sobol block.
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        if not size:
            raise ValueError("Active draws require at least one scenario axis")
        if not -len(size) <= scenario_axis < len(size):
            raise ValueError("scenario_axis must index size")
        axis = scenario_axis % len(size)
        indices = xp.asarray(path_indices, dtype=xp.int64)
        if len(indices) != size[axis]:
            raise ValueError("Active path count must agree with the scenario axis")
        if len(indices) == 0:
            return xp.empty(size, dtype=xp.float64)
        if bool(xp.any((indices < 0) | (indices >= n_sims))):
            raise ValueError("Active simulation indices must be within the original design")

        component_shape = size[:axis] + size[axis + 1 :]
        count = math.prod(component_shape)
        data = xp.empty((len(indices), count), dtype=xp.float64)
        for component in range(count):
            block = self._active_block(n_sims)
            j = block.dimension
            gray = block.gray[indices]
            raw = xp.full((len(indices),), block.shifts[j], dtype=xp.uint32)
            for byte in range(4):
                raw ^= block.tables[j, byte, (gray >> (byte * 8)) & 255]
            block.dimension += 1
            if xp is np:
                data[:, component] = raw.astype(np.float64) * (2.0**-32)
            else:
                data[:, component] = (raw.astype(xp.float64) + 0.5) * (2.0**-32)
        return xp.moveaxis(data.reshape((len(indices), *component_shape)), 0, axis)

    def _event_uniforms(self, n_sims: int, event_paths: t.Any, event_ordinals: t.Any, max_count: int) -> t.Any:
        """Generate exactly the Sobol coordinates used by each active claim.

        A claim belongs to path i and has ordinal k within that path. It
        consumes the k-th sequential Sobol dimension, sharing the same
        multidimensional design as a full-path draw for every claim ordinal.
        Only observed claims are reconstructed; no n_sims-by-max_count matrix
        or per-ordinal inverse-CDF call is required. Dimension recycling uses
        exactly the same sequence of independently randomized blocks.

        Args:
            event_ordinals: Ordinal of each event within its own simulation.
            event_paths: Simulation indices of the events requiring draws.
            max_count: Maximum number of events in any simulation.
            n_sims: Number of simulations on the scenario axis.
        """
        total_events = len(event_paths)
        if total_events != len(event_ordinals) or max_count < 0:
            raise ValueError("Event paths, ordinals or maximum claim count are invalid")
        if total_events == 0:
            return xp.empty((0,), dtype=xp.float64)
        if max_count < 1:
            raise ValueError("Positive claim counts are required for event uniforms")

        uniforms = xp.empty(total_events, dtype=xp.float64)
        consumed = 0
        while consumed < max_count:
            block = self._active_block(n_sims)
            width = min(self.dimensions - block.dimension, max_count - consumed)
            # Most compound models fit within one Sobol block. Avoid masks,
            # sorting and scatter operations in that common case.
            if consumed == 0 and width == max_count:
                paths = event_paths
                dimensions = event_ordinals + block.dimension
                selected = None
            else:
                selected = xp.nonzero((event_ordinals >= consumed) & (event_ordinals < consumed + width))[0]
                paths = event_paths[selected]
                dimensions = event_ordinals[selected] - consumed + block.dimension

            gray = block.gray[paths]
            raw = block.shifts[dimensions].copy()
            for byte in range(4):
                raw ^= block.tables[dimensions, byte, (gray >> (byte * 8)) & 255]
            if xp is np:
                values = raw.astype(np.float64) * (2.0**-32)
            else:
                values = (raw.astype(xp.float64) + 0.5) * (2.0**-32)
            if selected is None:
                uniforms[:] = values
            else:
                uniforms[selected] = values
            block.dimension += width
            consumed += width
        return uniforms

    @staticmethod
    def _host(value: t.Any) -> np.ndarray:
        """Convert distribution parameters to SciPy-compatible arrays.

        Args:
            value: Scalar or array to convert to a host NumPy array.
        """
        return np.asarray(asnumpy(value))

    def _shape(self, size: t.Any, *parameters: t.Any) -> tuple[int, ...]:
        """Determine the broadcast output shape of the requested random draws.

        Args:
            *parameters: Distribution parameters broadcast to the requested sample shape.
            size: Requested output shape or number of draws.
        """
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

    def _uniform_values(self, shape: tuple[int, ...], scenario_axis: int | None = None) -> t.Any:
        """Generate randomized uniform values on the requested scenario axis.

        Args:
            scenario_axis: Axis of the requested output corresponding to simulations.
            shape: Requested output array shape.
        """
        if not shape:
            return xp.asarray(self._next_coordinate(1)[0])
        if 0 in shape:
            return xp.empty(shape, dtype=float)
        # Callers with multidimensional draws should pass the known scenario
        # axis. A shape heuristic remains for NumPy Generator API compatibility.
        if scenario_axis is None:
            axis = int(np.argmax(shape))
        else:
            if not -len(shape) <= scenario_axis < len(shape):
                raise ValueError("scenario_axis must index a dimension of size")
            axis = scenario_axis % len(shape)
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
        scenario_axis: int | None = None,
    ) -> t.Any:
        """Transform randomized Sobol uniforms to the requested quantiles.

        Args:
            discrete: Whether the inverse CDF returns integer-valued counts.
            func: Inverse CDF or quantile function used to transform uniforms.
            gpu_func: Device-compatible inverse CDF to use when available.
            *parameters: Distribution parameters broadcast to the requested sample shape.
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        shape = self._shape(size, *parameters)
        uniforms = self._uniform_values(shape, scenario_axis=scenario_axis)
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

    def random(self, size: t.Any = None, *, scenario_axis: int | None = None) -> t.Any:
        """Draw U(0,1) values, using one coordinate per component.

        Args:
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        return self._uniform_values(self._shape(size), scenario_axis=scenario_axis)

    def uniform(
        self, low: t.Any = 0.0, high: t.Any = 1.0, size: t.Any = None, *, scenario_axis: int | None = None
    ) -> t.Any:
        """Draw continuous uniform values using the active Sobol coordinate.

        Args:
            high: Upper endpoint of the requested random values.
            low: Lower endpoint of the requested random values.
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        shape = self._shape(size, low, high)
        lo, hi = xp.asarray(low), xp.asarray(high)
        return lo + (hi - lo) * self._uniform_values(shape, scenario_axis=scenario_axis)

    def standard_normal(self, size: t.Any = None, *, scenario_axis: int | None = None) -> t.Any:
        """Draw standard normal quantiles.

        Args:
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(scipy_special.ndtri, size, gpu_func=special.ndtri, scenario_axis=scenario_axis)

    def normal(
        self, loc: t.Any = 0.0, scale: t.Any = 1.0, size: t.Any = None, *, scenario_axis: int | None = None
    ) -> t.Any:
        """Draw normal quantiles.

        Args:
            loc: Location shift of the sampled distribution.
            scale: Positive scale of the sampled distribution.
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(
            lambda u, mean, std: mean + std * scipy_special.ndtri(u),
            size,
            loc,
            scale,
            gpu_func=lambda u, mean, std: mean + std * special.ndtri(u),
            scenario_axis=scenario_axis,
        )

    def exponential(self, scale: t.Any = 1.0, size: t.Any = None, *, scenario_axis: int | None = None) -> t.Any:
        """Draw exponential quantiles.

        Args:
            scale: Positive scale of the sampled distribution.
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(
            lambda u, s: -s * np.log1p(-u),
            size,
            scale,
            gpu_func=lambda u, s: -s * xp.log1p(-u),
            scenario_axis=scenario_axis,
        )

    def gamma(self, shape: t.Any, scale: t.Any = 1.0, size: t.Any = None, *, scenario_axis: int | None = None) -> t.Any:
        """Draw gamma quantiles.

        Args:
            scale: Positive scale of the sampled distribution.
            scenario_axis: Axis of the requested output corresponding to simulations.
            shape: Distribution shape parameter or requested output shape.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(
            lambda u, a, s: scipy_special.gammaincinv(a, u) * s,
            size,
            shape,
            scale,
            gpu_func=lambda u, a, s: special.gammaincinv(a, u) * s,
            scenario_axis=scenario_axis,
        )

    def beta(self, a: t.Any, b: t.Any, size: t.Any = None, *, scenario_axis: int | None = None) -> t.Any:
        """Draw beta quantiles.

        Args:
            a: First shape parameter of the distribution.
            b: Second shape parameter of the distribution.
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(
            lambda u, alpha, beta: scipy_special.betaincinv(alpha, beta, u),
            size,
            a,
            b,
            gpu_func=lambda u, alpha, beta: special.betaincinv(alpha, beta, u),
            scenario_axis=scenario_axis,
        )

    def poisson(self, lam: t.Any = 1.0, size: t.Any = None) -> t.Any:
        """Draw Poisson quantiles, with a GPU CDF lookup for fixed intensity.

        For constant moderate intensities, evaluate the CDF on the device
        once and invert all Sobol coordinates with CuPy searchsorted.
        A stochastic or very large intensity retains the SciPy PPF fallback.
        The inverse transform preserves RQMC dependence between coordinates.

        Args:
            lam: Expected Poisson event count.
            size: Requested output shape or number of draws.
        """
        intensity = float(lam) if isinstance(lam, (int, float, np.integer, np.floating)) else math.nan
        if xp is not np and math.isfinite(intensity) and intensity >= 0:
            # GPU Sobol32 midpoints never approach 1 more closely than
            # 2**-33. The more extreme (1 - 1e-14) cutoff safely bounds them.
            bound = scipy_stats.poisson.ppf(1 - 1e-14, intensity)
            if np.isfinite(bound) and bound <= 8192:
                count = int(bound) + 5
                # P(X <= k) = gammaincc(k + 1, lambda).
                cdf = special.gammaincc(xp.arange(count, dtype=xp.float64) + 1, intensity)
                uniforms = self._uniform_values(self._shape(size, lam))
                return xp.searchsorted(cdf, uniforms, side="left").astype(xp.int64)
        return self._quantiles(scipy_stats.poisson.ppf, size, lam, discrete=True)

    def negative_binomial(self, n: t.Any, p: t.Any, size: t.Any = None) -> t.Any:
        """Draw negative-binomial quantiles.

        Args:
            n: Negative binomial size parameter.
            p: Probability of success or distribution shape parameter.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(scipy_stats.nbinom.ppf, size, n, p, discrete=True)

    def binomial(self, n: t.Any, p: t.Any, size: t.Any = None) -> t.Any:
        """Draw binomial quantiles.

        Args:
            n: Number of Bernoulli trials.
            p: Probability of success or distribution shape parameter.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(scipy_stats.binom.ppf, size, n, p, discrete=True)

    def hypergeometric(self, ngood: t.Any, nbad: t.Any, nsample: t.Any, size: t.Any = None) -> t.Any:
        """Draw hypergeometric quantiles.

        Args:
            nbad: Number of unsuccessful items in the finite population.
            ngood: Number of successful items in the finite population.
            nsample: Number of items drawn without replacement.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(
            lambda u, good, bad, sample: scipy_stats.hypergeom.ppf(u, good + bad, good, sample),
            size,
            ngood,
            nbad,
            nsample,
            discrete=True,
        )

    def standard_t(self, df: t.Any, size: t.Any = None) -> t.Any:
        """Draw Student's t quantiles.

        Args:
            df: Degrees of freedom of the distribution.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(scipy_stats.t.ppf, size, df)

    def chisquare(self, df: t.Any, size: t.Any = None, *, scenario_axis: int | None = None) -> t.Any:
        """Draw chi-squared quantiles.

        Args:
            df: Degrees of freedom of the distribution.
            scenario_axis: Axis of the requested output corresponding to simulations.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(
            lambda u, degrees: 2 * scipy_special.gammaincinv(degrees / 2, u),
            size,
            df,
            gpu_func=lambda u, degrees: 2 * special.gammaincinv(degrees / 2, u),
            scenario_axis=scenario_axis,
        )

    def noncentral_chisquare(self, df: t.Any, nonc: t.Any, size: t.Any = None) -> t.Any:
        """Draw noncentral chi-squared samples, using a two-coordinate GPU mixture when supported.

        Args:
            df: Degrees of freedom of the distribution.
            nonc: Noncentrality parameter of the chi-squared distribution.
            size: Requested output shape or number of draws.
        """
        if xp is not np and isinstance(nonc, (int, float, np.integer, np.floating)) and 0 <= nonc <= 16000:
            shape = self._shape(size, df, nonc)
            count = self.poisson(float(nonc) / 2, size=shape)
            return self.chisquare(xp.asarray(df) + 2 * count, size=shape)
        return self._quantiles(scipy_stats.ncx2.ppf, size, df, nonc)

    def logseries(self, p: t.Any, size: t.Any = None) -> t.Any:
        """Draw logarithmic-series quantiles.

        Args:
            p: Probability of success or distribution shape parameter.
            size: Requested output shape or number of draws.
        """
        return self._quantiles(scipy_stats.logser.ppf, size, p, discrete=True)

    def integers(self, low: t.Any, high: t.Any = None, size: t.Any = None, endpoint: bool = False) -> t.Any:
        """Draw uniformly distributed integers using quantile mapping.

        Args:
            endpoint: Whether the integer upper endpoint is inclusive.
            high: Upper endpoint of the requested random values.
            low: Lower endpoint of the requested random values.
            size: Requested output shape or number of draws.
        """
        if high is None:
            low, high = 0, low
        if endpoint:
            high = self._host(high) + 1
        return self._quantiles(
            lambda u, minimum, maximum: np.floor(minimum + (maximum - minimum) * u), size, low, high, discrete=True
        )


def _draw_active_scenarios(
    rng: t.Any,
    method: str,
    n_sims: int,
    path_indices: t.Any,
    *,
    size: int | tuple[int, ...],
    scenario_axis: int = 0,
    **kwargs: t.Any,
) -> t.Any:
    """Sample active paths without losing their original Sobol row identity.

    For ordinary Monte Carlo, keep the existing draw behaviour. Under RQMC,
    produce the requested marginal transform using coordinates selected from
    the n_sims-path design. No new stream is allocated for the active count.

    Args:
        **kwargs: Keyword arguments supplied by the caller or NumPy dispatch.
        method: Name of the random-generator sampling method to invoke.
        n_sims: Number of simulations on the scenario axis.
        path_indices: Indices of simulations that need draws from the active Sobol block.
        rng: Random number generator supplying the draws.
        scenario_axis: Axis of the requested output corresponding to simulations.
        size: Requested output shape or number of draws.
    """
    if not isinstance(rng, RandomizedQMCGenerator):
        # NumPy Generator uses integers(), but CuPy's legacy RandomState
        # exposes randint() instead. Preserve ordinary MC sampling on both.
        name = "randint" if method == "integers" and not hasattr(rng, "integers") else method
        return _draw_scenarios(rng, name, size=size, scenario_axis=scenario_axis, **kwargs)

    shape = (size,) if isinstance(size, int) else size
    uniforms = rng._active_uniforms(n_sims, path_indices, shape, scenario_axis=scenario_axis)
    uniforms = xp.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    if method == "standard_normal":
        return special.ndtri(uniforms)
    if method == "normal":
        return kwargs.get("loc", 0.0) + kwargs.get("scale", 1.0) * special.ndtri(uniforms)
    if method == "exponential":
        return -kwargs.get("scale", 1.0) * xp.log1p(-uniforms)
    if method == "chisquare":
        return 2 * special.gammaincinv(kwargs["df"] / 2, uniforms)
    if method == "integers":
        low = kwargs["low"]
        high = kwargs["high"]
        return xp.floor(low + (high - low) * uniforms).astype(xp.int64)
    raise NotImplementedError(f"Active RQMC draws do not support {method!r}")


def _draw_scenarios(rng: t.Any, method: str, *args: t.Any, scenario_axis: int, **kwargs: t.Any) -> t.Any:
    """Tell RQMC which axis holds simulation paths without changing NumPy's API.

    Args:
        *args: Positional arguments supplied by the caller or NumPy dispatch.
        **kwargs: Keyword arguments supplied by the caller or NumPy dispatch.
        method: Name of the random-generator sampling method to invoke.
        rng: Random number generator supplying the draws.
        scenario_axis: Axis of the requested output corresponding to simulations.
    """
    draw = getattr(rng, method)
    if isinstance(rng, RandomizedQMCGenerator):
        return draw(*args, scenario_axis=scenario_axis, **kwargs)
    return draw(*args, **kwargs)


# Sampling support for PAL distribution classes. Keeping backend selection here
# means distributions.py can express their sampling parameters without branching
# on NumPy, CuPy or the randomized Sobol generator.


def _rng_value(value: t.Any, rng: t.Any) -> t.Any:
    """Convert parameters to the backend expected by a random generator.

    Args:
        rng: Random number generator supplying the draws.
        value: Value to convert, compare or assign.
    """
    if type(rng).__module__.startswith("numpy"):
        return asnumpy(value) if isinstance(value, xp.ndarray) else value
    return to_backend(value)


def _sample_rng(rng: t.Any, method: str, n_sims: int, *parameters: t.Any) -> t.Any:
    """Sample one PAL distribution on the generator's own backend.

    Args:
        method: Name of the random-generator sampling method to invoke.
        n_sims: Number of simulations on the scenario axis.
        *parameters: Distribution parameters broadcast to the requested sample shape.
        rng: Random number generator supplying the draws.
    """
    converted = (_rng_value(value, rng) for value in parameters)
    return getattr(rng, method)(*converted, size=n_sims)


def _inverse_transform_uniforms(rng: t.Any, n_sims: int) -> t.Any:
    """Generate probability inputs on PAL's active backend for inverse CDFs.

    Args:
        n_sims: Number of simulations on the scenario axis.
        rng: Random number generator supplying the draws.
    """
    return xp.asarray(_sample_rng(rng, "uniform", n_sims))


def _sample_generalized_beta_prime(rng: t.Any, n_sims: int, a: t.Any, b: t.Any, p: t.Any, q: t.Any) -> t.Any:
    """Sample GB2 as the ratio of two independent unit-scale Gamma variables.

    Args:
        a: Positive power parameter of the GB2 distribution.
        b: Positive GB2 scale parameter.
        n_sims: Number of simulations on the scenario axis.
        p: Positive lower-tail beta-prime shape parameter.
        q: Upper-tail beta-prime shape parameter.
        rng: Random number generator supplying the draws.
    """
    numerator = _sample_rng(rng, "gamma", n_sims, p, 1.0)
    denominator = _sample_rng(rng, "gamma", n_sims, q, 1.0)
    return _rng_value(b, rng) * (numerator / denominator) ** (1 / _rng_value(a, rng))


def _sample_scaled_beta(rng: t.Any, n_sims: int, alpha: t.Any, beta: t.Any, scale: t.Any, loc: t.Any) -> t.Any:
    """Sample a shifted and scaled beta distribution.

    Args:
        alpha: First positive shape parameter of the sampled distribution.
        beta: Second positive shape parameter of the sampled distribution.
        loc: Location shift of the sampled distribution.
        n_sims: Number of simulations on the scenario axis.
        rng: Random number generator supplying the draws.
        scale: Positive scale of the sampled distribution.
    """
    return _sample_rng(rng, "beta", n_sims, alpha, beta) * _rng_value(scale, rng) + _rng_value(loc, rng)


def _sample_shifted_gamma(rng: t.Any, n_sims: int, alpha: t.Any, theta: t.Any, loc: t.Any) -> t.Any:
    """Sample a shifted gamma distribution.

    Args:
        alpha: First positive shape parameter of the sampled distribution.
        loc: Location shift of the sampled distribution.
        n_sims: Number of simulations on the scenario axis.
        rng: Random number generator supplying the draws.
        theta: Positive scale of the gamma distribution.
    """
    return _sample_rng(rng, "gamma", n_sims, alpha, theta) + _rng_value(loc, rng)


def _sample_students_t(rng: t.Any, n_sims: int, nu: t.Any, mu: t.Any, sigma: t.Any) -> t.Any:
    """Sample a shifted and scaled Student's t distribution.

    Args:
        mu: Location or mean of the sampled distribution.
        n_sims: Number of simulations on the scenario axis.
        nu: Degrees of freedom of the Student's t distribution.
        rng: Random number generator supplying the draws.
        sigma: Positive dispersion of the sampled distribution.
    """
    return _rng_value(mu, rng) + _rng_value(sigma, rng) * _sample_rng(rng, "standard_t", n_sims, nu)


def _sample_inverse_gaussian(rng: t.Any, n_sims: int, mu: t.Any, lambda_: t.Any) -> t.Any:
    """Sample the inverse Gaussian using the Michael–Schucany–Haas method.

    This construction uses two draws, and therefore two Sobol coordinates when
    RQMC is active. It retains the ordinary generator semantics under MC.

    Args:
        lambda_: Positive inverse-Gaussian shape parameter.
        mu: Location or mean of the sampled distribution.
        n_sims: Number of simulations on the scenario axis.
        rng: Random number generator supplying the draws.
    """
    nu = xp.asarray(_sample_rng(rng, "normal", n_sims, 0, 1) ** 2)
    y = mu + (mu**2 * nu) / (2 * lambda_) - (mu / (2 * lambda_)) * xp.sqrt(4 * mu * lambda_ * nu + mu**2 * nu**2)
    u = xp.asarray(_sample_rng(rng, "uniform", n_sims, 0, 1))
    return xp.where(u <= mu / (mu + y), y, mu**2 / y)


def _sample_generalized_inverse_gaussian(
    rng: t.Any, n_sims: int, p: t.Any, shape: t.Any, scale: t.Any, loc: t.Any
) -> t.Any:
    """Use a native half-order GPU sampler or the established SciPy fallback.

    Args:
        loc: Location shift of the sampled distribution.
        n_sims: Number of simulations on the scenario axis.
        p: GIG index controlling the power of loss in the density.
        rng: Random number generator supplying the draws.
        scale: Positive scale of the sampled distribution.
        shape: Distribution shape parameter or requested output shape.
    """
    half_order_gpu = xp is not np and bool(xp.all(xp.abs(xp.asarray(p)) == 0.5))
    if half_order_gpu:
        return generalized_inverse_gaussian_mc(rng, n_sims, p, shape, scale, loc)
    if isinstance(rng, RandomizedQMCGenerator):
        return _scipy_on_backend(
            lambda u, index, b, location, factor: scipy_stats.geninvgauss.ppf(u, index, b, loc=location, scale=factor),
            rng.uniform(size=n_sims),
            p,
            shape,
            loc,
            scale,
        )
    if xp is not np:
        seed = int(asnumpy(rng.randint(0, 2**31 - 1)))
        host_rng = np.random.default_rng(seed)
    else:
        host_rng = rng
    result = scipy_stats.geninvgauss.rvs(
        asnumpy(p),
        asnumpy(shape),
        loc=asnumpy(loc),
        scale=asnumpy(scale),
        size=n_sims,
        random_state=host_rng,
    )
    return xp.asarray(result)


def _scipy_on_backend(function: t.Callable[..., t.Any], *parameters: t.Any) -> t.Any:
    """Evaluate a SciPy function on host arrays and restore PAL's backend.

    Args:
        function: Probability or special function to evaluate on the appropriate backend.
        *parameters: Distribution parameters broadcast to the requested sample shape.
    """
    return xp.asarray(function(*(asnumpy(value) for value in parameters)))
