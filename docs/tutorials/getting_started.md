# Getting Started with PAL

This tutorial introduces the core concepts of the Proteus Actuarial
Library through a short, end-to-end example.

## Setup

```python
import numpy as np

from pal import config, copulas, distributions, set_random_seed

config.n_sims = 10_000
set_random_seed(42)
```

`config.n_sims` controls how many Monte Carlo simulations are generated
globally. `set_random_seed` makes results reproducible.

## Generating Stochastic Variables

Create a loss variable from a LogNormal distribution:

<!--pytest-codeblocks:cont-->

```python
loss = distributions.LogNormal(mu=14, sigma=0.5).generate()
```

This returns a `StochasticScalar` — a vector of 10,000 simulated values.
You can inspect it immediately:

<!--pytest-codeblocks:cont-->

```python
loss.mean()       # => 1,358,389
loss.std()        # =>   732,520
np.median(loss.values)               # => 1,200,047
np.percentile(loss.values, 99.5)     # => 4,443,841
```

## Arithmetic on Stochastic Variables

Standard arithmetic works element-wise across simulations:

<!--pytest-codeblocks:cont-->

```python
expenses = loss * 0.10         # 10% expense loading
total = loss + expenses        # gross = loss + expenses

total.mean()                   # => 1,494,228
```

Because `expenses` and `total` are derived from `loss`, they
automatically join the same **coupling group** — if `loss` is later
reordered by a copula, `expenses` and `total` are reordered in
lockstep.

## Combining Multiple Lines of Business

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)
motor = distributions.LogNormal(mu=14, sigma=0.5).generate()
prop = distributions.LogNormal(mu=15, sigma=0.8).generate()

combined = motor + prop
```

```
Motor mean:      1,358,389
Property mean:   4,545,084
Combined mean:   5,903,473
Combined std:    4,399,880
```

The combined standard deviation reflects the fact that `motor` and
`prop` are **independent** — their peaks and troughs don't coincide.

## Adding Dependencies with Copulas

In reality, lines of business are often correlated (e.g. through
economic conditions). Use a copula to introduce dependence:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)
motor = distributions.LogNormal(mu=14, sigma=0.5).generate()
prop = distributions.LogNormal(mu=15, sigma=0.8).generate()

copulas.GaussianCopula([[1, 0.5], [0.5, 1]]).apply([motor, prop])

combined = motor + prop
```

```
Combined mean:        5,903,473    (unchanged)
Combined std:         4,694,549    (higher — dependence adds volatility)
Combined 99.5th:     29,435,135
```

The **mean is unchanged** because copulas only reorder simulations —
they don't change the individual values. But the **standard deviation
increases** because bad years now tend to coincide across both lines.

See the [Coupling Groups, Copulas and Variable Reordering](coupling_groups_and_copulas.md)
tutorial for a deeper treatment.

## Frequency-Severity Models

For compound distributions (random number of random-sized claims):

<!--pytest-codeblocks:cont-->

```python
from pal.frequency_severity import FrequencySeverityModel

set_random_seed(42)
model = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=100),
    sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
)
events = model.generate()
```

This generates ~1,000,000 individual events across 10,000 simulations.
Aggregate them to get total loss per simulation:

<!--pytest-codeblocks:cont-->

```python
agg = events.aggregate()
```

```
Events generated:   1,000,131
Aggregate mean:     6,783,564
Aggregate 99.5th:  15,098,023
```

See the [Frequency-Severity Modelling](frequency_severity_modelling.md)
tutorial for more detail.

## Visualisation

Every `StochasticScalar` has a `show_cdf()` method that displays an
interactive CDF plot:

<!--pytest.mark.skip-->

```python
agg.show_cdf("Aggregate Loss")
```

Set the environment variable `PAL_SUPPRESS_PLOTS=true` to disable
plot display in headless environments.

## Configuration Summary

| Setting | Default | Description |
|---------|---------|-------------|
| `config.n_sims` | 100,000 | Number of Monte Carlo simulations |
| `config.rng` | `numpy.random.default_rng()` | Random number generator |
| `set_random_seed(n)` | — | Set seed for reproducibility |
| `PAL_SUPPRESS_PLOTS` | unset | Set to `true` to suppress plots |
| `PAL_USE_GPU` | unset | Set to `1` for CuPy GPU acceleration |

## Next Steps

- [Distributions Guide](distributions_guide.md) — choosing and fitting
  distributions
- [Frequency-Severity Modelling](frequency_severity_modelling.md) —
  compound models and event-level analysis
- [Coupling Groups, Copulas and Variable Reordering](coupling_groups_and_copulas.md) —
  dependency structures
- [Pricing an XoL Reinsurance Program](xol_reinsurance.md) — layers,
  towers and reinstatements
