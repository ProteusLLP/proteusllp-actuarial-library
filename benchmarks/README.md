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
measurements and a profiler before setting regression thresholds. Current
measurements do not provide a reliable per-workload peak memory measurement;
dedicated CPU RSS and GPU allocated-memory benchmarks are follow-up work.

Run normal correctness, numerical and stochastic-coupling tests before accepting
a performance improvement. Faster results alone do not establish correctness.
