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

## CUDA event-indexing comparison

For issue #133, on a CUDA runner, compare all three event-index algorithms
with identical integer outputs and separate pool-cold and warmed measurements:

<!--pytest-codeblocks:skip-->

```bash
PAL_USE_GPU=1 python -m benchmarks.benchmark_gpu_event_indexing \
  --sizes 10000,100000,1000000
```

The dedicated "GPU event-indexing comparison" workflow runs when its benchmark
or tests change in a same-repository pull request. It saves JSON, an Actions
summary, and an optional Nsight Systems CUDA/NVTX trace when nsys is installed.
The experiment checks the existing cumulative-search implementation, CuPy
repeat with device-resident counts, and a custom prefix-fill kernel.

Times include output allocation and a CUDA synchronisation. The input transfer,
correctness copies and kernel compilation are excluded. Pool occupancy in JSON
is not peak device-memory consumption. A faster single run is insufficient for
a production change: repeat the experiment on the same GPU/software stack and
inspect traces for host synchronisations and temporary allocations. In
particular, output arrays of data-dependent length require resolving the total
event count on the host before allocation. The report highlights candidates
whose speedup exceeds 10% and three summed median absolute deviations; these
are screening criteria, not automatic selection rules.
