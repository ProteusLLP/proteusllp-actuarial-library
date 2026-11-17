# Monte Carlo and randomized quasi-Monte Carlo

Monte Carlo (MC) sampling uses pseudorandom numbers. Randomized quasi-Monte
Carlo (RQMC) uses independently scrambled Sobol designs to cover the sampling
space more evenly. RQMC can reduce the error in actuarial estimates without
changing the model or the number of simulations. The improvement depends on
the statistic and the effective dimension of the model.

PAL allocates Sobol dimensions as random draws are requested. It starts a new,
independently scrambled and shuffled block when the configured dimension
limit is reached. Users do not need to assign dimensions to distributions,
copulas or compound models.

## Switching sampling methods

The usual simulation calls stay the same:

```python
import numpy as np

from pal import set_random_seed
from pal.config import SamplingMethod, set_sampling_method
from pal.distributions import Gamma, Normal

set_sampling_method(SamplingMethod.RQMC, dimensions=32)
set_random_seed(42)

inflation = Normal(mu=0.03, sigma=0.015).generate(4096)
claims = Gamma(alpha=2.5, theta=100_000).generate(4096)
total = claims * (1 + inflation)

assert len(total) == 4096

set_sampling_method(SamplingMethod.MC)
```

Both methods use exactly the same distribution and stochastic-variable API.
The seed controls reproducibility, including the Sobol scrambling. For
independent uncertainty estimates under RQMC, run the model repeatedly with
different seeds. **Individual paths of one Sobol scramble are not IID**.

## Copula density comparison

A useful visual check is to simulate the same bivariate copula under each
sampling method and plot the empirical copula density in uniform rank space.
The 2×2 comparison below uses a Gaussian copula and a Student-t copula with
four degrees of freedom, both with correlation 0.7.

The runnable example is
[examples/example_qmc_vs_mc.py](https://github.com/ProteusLLP/proteusllp-actuarial-library/blob/main/examples/example_qmc_vs_mc.py).
It produces two interactive Plotly charts:

- **Four density heatmaps:** MC and RQMC for both copulas, with identical bins,
  axes and colour scale, and 8,192 simulations per panel.
- **Convergence:** root-mean-squared error of a Gaussian copula joint
  exceedance probability, evaluated against an independent numerical integral
  over repeated random seeds.

Run the example from the repository root:

<!--pytest-codeblocks:skip-->

```bash
python -m examples.example_qmc_vs_mc --output qmc-plots
```

Open `qmc-plots/qmc_copula_densities.html` and
`qmc-plots/qmc_vs_mc_convergence.html` to explore the figures.
The figures are Plotly objects and can be modified directly in Python:

```python
from pal.examples.example_qmc_vs_mc import copula_density_chart

fig = copula_density_chart(n_sims=512)
assert len(fig.data) == 4
```

The heatmaps estimate **empirical density**, not theoretical density.
Differences in bin counts are useful to inspect, but lower histogram noise
does not by itself establish a lower pricing or capital-estimation error.

## Accuracy and performance

The benchmark compares PAL implementations directly, rather than comparing a
raw Sobol sequence with a NumPy random generator. It times the entire
generation step, including scrambling, distribution transforms and, on GPU,
device synchronization. Each repetition uses a fresh seed.

```python
from benchmarks.benchmark_sampling import run_comparison

comparison = run_comparison(
    sizes=[512],
    replicates=3,
    cases=("normal_mean", "gaussian_joint_tail"),
)
assert len(comparison["results"]) == 4
```

For a more stable comparison across simulation counts:

<!--pytest-codeblocks:skip-->

```bash
python -m benchmarks.benchmark_sampling --sizes 1024,4096,16384 --replicates 16
```

The benchmark writes a JSON report and a Markdown table for normal means,
Poisson means, Gaussian-copula joint tail probabilities, and compound Poisson
loss means. It reports median runtime, RMSE against the theoretical value,
and variation across *independent* runs.

### Practical qualifications

- Powers of two, such as 4,096 or 16,384, preserve the strongest Sobol
  balance properties. Non-power-of-two sizes still work.
- RQMC typically helps smooth, low-effective-dimensional integrands the most.
  Discontinuities (attachment points, quantiles and rare-event indicators)
  can reduce the improvement, so we benchmark these separately.
- Blocks are independently scrambled and permuted. They avoid an unbounded
  Sobol dimension count, but do not reproduce a single high-dimensional
  low-discrepancy design across blocks.
- On GPU the current implementation generates Sobol coordinates and inverse
  CDFs on the CPU before transferring values to CuPy. It may be slower than
  native GPU pseudorandom draws for some models.
- A separate scramble is required for each statistically independent
  replication when assessing error or uncertainty.

The original [copula tutorial](coupling_groups_and_copulas.md) explains
how PAL also preserves dependence when copulas reorder existing stochastic
variables.
