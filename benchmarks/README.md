# PAL performance benchmarks

Run the benchmarks from an editable installation of PAL:

```bash
python -m benchmarks.benchmark_core --sizes 10000,100000 --repeat 5 --warmup 2
```

Run a larger portfolio with `--sizes 1000000`, or focus on a case with
`--cases aggregate,occurrence`. To benchmark a CUDA device, use an environment
with CuPy and set `PAL_USE_GPU=1` **before** starting Python.

The runner writes JSON (stable schema version 1) and a Markdown report.
Use `--baseline previous.json` to compare with an earlier run **only when the
CPU/GPU and numerical software stack match**. The corresponding workflow
records results as downloadable GitHub Actions artifacts, including the commit,
runtime, hardware and dependency versions. The workflow runs weekly on CPU;
GPU benchmarks are opt-in through workflow dispatch to avoid occupying the
dedicated GPU runner unnecessarily.

Workloads cover event indexing, stochastic arithmetic, aggregate and occurrence
reductions, a 99.5th percentile, compound Poisson/lognormal generation,
an excess-of-loss layer, Gaussian copula generation and a full compound-to-layer
to-quantile calculation. Inputs and model setup are outside timed sections.
Times are medians of warmed repeated wall-clock runs, with median absolute
deviation (MAD) shown as a measure of timing variability. GPU measurements
synchronise before and after each run and exclude first-time compilation.
The full-model benchmark includes stochastic generation and contract application.

Results are **descriptive**, not performance gates. GitHub-hosted runner variance,
changes in hardware, allocator caching and dependency versions can move timings
independently of PAL. Investigate persistent changes with repeated, same-stack
measurements and a profiler before setting regression thresholds. For memory, run `python -m benchmarks.profile_memory --sizes 10000,100000`.
Each case runs in a separate Python process to isolate the OS peak process
RSS (including native NumPy allocations). The JSON records after-import,
after-setup and after-run high-water marks. On GPU, it also records CuPy's
caching-pool reservation (which is not the same as total VRAM use or a
precise temporary-allocation peak). The optional Python `tracemalloc` reading
does not capture native array buffers. These measurements are observational
rather than regression gates.

For portfolio scaling, run `python -m benchmarks.profile_portfolio
--simulations 100000 --lobs 100 --layers 500`. The workload generates
multiple compound loss streams, combines them, and processes an adjacent,
non-overlapping XoL tower. It is a synthetic computational stress test, not a
calibrated capital model; large settings may need substantial memory and time.
To profile memory for it, use `--cases portfolio --portfolio-lobs 100
--portfolio-layers 500` with `profile_memory`. Start small, then scale.



Run normal correctness, numerical and stochastic-coupling tests before accepting
a performance improvement. Faster results alone do not establish correctness.

## Variable I/O: compact versus row-form Parquet

To compare the new Arrow-native columnar representation against the earlier
long-form DataFrame representation, run:

<!--pytest-codeblocks:skip-->

```bash
python -m benchmarks.benchmark_variable_io --simulations 20000 --variables 20
```

For a larger synthetic model, skip the memory-intensive row-form baseline:

<!--pytest-codeblocks:skip-->

```bash
python -m benchmarks.benchmark_variable_io --simulations 100000 --variables 100 --compact-only
```

Requires PAL's optional `parquet` dependency. The script reports read and
write wall times, Parquet file sizes and traced Python memory peaks, together
with runtime and dependency versions. Python tracing does not capture all
NumPy/Arrow native allocations, so do not treat the figures as total peak RSS.
The benchmark validates round-trip numeric values and is not a performance
regression gate.

## Mean excess diagnostics

<!--pytest-codeblocks:skip-->

```bash
python -m benchmarks.benchmark_mean_excess --sizes 1000,10000 --repeat 3
```

The CPU benchmark reports warmed median wall times for the public distribution
mean-excess methods and empirical mean excess on 100,000 observations. It includes
ordinary and stressed parameterisations. At up to 1,000 thresholds it compares
batch evaluation with repeated scalar calls (or repeated filtering for empirical
losses), checking numerical agreement. This is a same-version comparison, not a
comparison with a previous PAL release. Results are descriptive, not timing gates.

To reproduce the independent 50-digit parameter sweep (961 cases):

<!--pytest-codeblocks:skip-->

```bash
pip install mpmath
python -m benchmarks.check_mean_excess_accuracy
```

The sweep covers GB2, Beta, Gamma, inverse Gaussian, GEV and LogNormal,
including difficult shapes and thresholds; it does not require mpmath at PAL
runtime. The performance workflow also runs the sweep. The GIG timing reports
batched values only because repeated adaptive scalar integration is costly.

## Fitting optimizers

<!--pytest-codeblocks:skip-->

```bash
python -m benchmarks.benchmark_fitting --size 10000 --output fitting.json
```

Compare the default safeguarded L-BFGS-B with Nelder-Mead on identical
quantile-spaced samples, including a common 10,000-loss sample, known-family
samples, weighted truncated and censored tail samples, and bounded-support
cases. The report records log-likelihood differences, parameters, convergence,
boundary warnings, actual optimizer, fallback use and elapsed time. Use
`--case Gamma` to select a subset. Timings are descriptive; a worse likelihood
or unsuccessful fit is a numerical investigation target, not a speedup.

## Limited expected value

<!--pytest-codeblocks:skip-->

```bash
python -m benchmarks.benchmark_lev --sizes 1000,10000,100000 --repeat 3
```

Reports median batch times for analytical and adaptive LEV paths, sampled
scalar agreement, and scalar-loop comparisons at 1,000 limits. GIG is
benchmarked through 10,000 limits because it still integrates its density.
The accuracy tests include extreme parameters, infinite means and the
finite-mean identity relating LEV to mean excess. Timings are descriptive.
