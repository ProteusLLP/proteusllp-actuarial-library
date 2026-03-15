# Frequency-Severity Modelling

This tutorial covers compound distribution modelling in PAL, where
total losses are the sum of a random number (frequency) of random
amounts (severity).

## Setup

```python
import numpy as np

from pal import config, distributions, set_random_seed
from pal.frequency_severity import FrequencySeverityModel

config.n_sims = 10_000
set_random_seed(42)
```

## 1. Creating a Frequency-Severity Model

A `FrequencySeverityModel` combines a frequency distribution (how many
events) with a severity distribution (how large each event is):

<!--pytest-codeblocks:cont-->

```python
model = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=50),
    sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
)
events = model.generate()
```

This generates individual events across all simulations. With 10,000
simulations and an average of 50 events each, you get roughly 500,000
individual events:

```
Total events:            499,780
Events per sim (first 5): [56, 59, 56, 41, 47]
```

The result is a `FreqSevSims` object that stores both event values
and simulation indices — it knows which events belong to which
simulation.

## 2. Aggregation and Occurrence

### Aggregate Loss

Sum all events within each simulation to get the total loss per year:

<!--pytest-codeblocks:cont-->

```python
agg = events.aggregate()
```

```
Aggregate mean:       3,373,050
Aggregate std:        1,420,195
Aggregate 99.5th:     9,390,666
```

`aggregate()` returns a `StochasticScalar` — the same type as any
other simulated variable in PAL. You can use it for statistics,
copulas, arithmetic, and plotting.

### Occurrence (Maximum Event)

Find the largest single event in each simulation:

<!--pytest-codeblocks:cont-->

```python
occ = events.occurrence()
```

```
Occurrence mean:        837,677
Occurrence std:         875,484
Occurrence 99.5th:    5,533,691
```

This is useful for pricing per-occurrence reinsurance layers.

### Coupling Groups

Both `aggregate()` and `occurrence()` are automatically coupled with
the underlying `FreqSevSims` events. If the events are reordered by a
copula, the aggregate and occurrence are reordered together:

<!--pytest-codeblocks:cont-->

```python
len(agg.coupled_variable_group)
# => 7  (events, freq, sev, agg, occ, and intermediates)
```

## 3. Arithmetic on Events

`FreqSevSims` objects support standard arithmetic and numpy operations.
Each operation applies element-wise to the individual event values.

### Capping Individual Losses

<!--pytest-codeblocks:cont-->

```python
capped_events = np.minimum(events, 500_000)
capped_agg = capped_events.aggregate()
```

```
Capped at 500k:
  Aggregate mean:     2,902,038
  Aggregate 99.5th:   5,202,702
```

Capping reduces both the mean and tail because large events are
truncated.

### Stochastic Inflation

Multiply by a simulation-level inflation factor. PAL automatically
broadcasts the `StochasticScalar` (one value per simulation) across
all events in that simulation:

<!--pytest-codeblocks:cont-->

```python
inflation = distributions.Normal(0.05, 0.02).generate()
inflated = events * (1 + inflation)
```

```
With 5% stochastic inflation:
  Aggregate mean:     3,541,639
```

### Other Operations

<!--pytest-codeblocks:cont-->

```python
# Deductible per event
excess = np.maximum(events - 100_000, 0)

# Scale by a factor
scaled = events * 1.10

# Conditional logic
large_only = np.where(events > 200_000, events, 0)
```

All of these return new `FreqSevSims` objects that are automatically
coupled with the originals.

## 4. Choosing the Frequency Distribution

### Poisson

The standard choice for claim counts. Variance equals the mean:

<!--pytest-codeblocks:cont-->

```python
model = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=25),
    sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
)
```

```
Aggregate mean:     1,689,715
Aggregate std:      1,001,221
Aggregate 99.5th:   6,020,593
```

### Negative Binomial

Use when claim counts are over-dispersed (variance > mean). This
happens when there is uncertainty in the underlying rate, or
heterogeneity across risk units:

<!--pytest-codeblocks:cont-->

```python
model = FrequencySeverityModel(
    freq_dist=distributions.NegBinomial(n=25, p=0.5),
    sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
)
```

```
Aggregate mean:     1,689,863
Aggregate std:      1,067,447    (higher than Poisson!)
Aggregate 99.5th:   6,243,797    (fatter tail)
```

Both models have the same mean (~25 events), but the Negative Binomial
produces a wider distribution of aggregate losses because the claim
count itself is more variable.

## 5. Common Modelling Patterns

### Attritional + Large Loss Split

Model small frequent losses separately from rare large losses:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)

# Small frequent claims
attritional = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=500),
    sev_dist=distributions.LogNormal(mu=8, sigma=0.5),
).generate()

# Rare large claims
large = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=3),
    sev_dist=distributions.Pareto(shape=1.5, scale=1_000_000),
).generate()

total = attritional.aggregate() + large.aggregate()
```

### Loss Development / Expense Loading

Apply multiplicative loadings after generating events:

<!--pytest-codeblocks:cont-->

```python
events = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=50),
    sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
).generate()

# Development factor
developed = events * 1.05

# Loss adjustment expenses
with_lae = developed * 1.08

# Final aggregate
net_agg = with_lae.aggregate()
```

### Feeding into Reinsurance Contracts

`FreqSevSims` objects are the natural input to `XoL` and `XoLTower`:

<!--pytest-codeblocks:cont-->

```python
from pal.contracts import XoL

layer = XoL(
    name="1m xs 500k",
    limit=1_000_000,
    excess=500_000,
    premium=30_000,
)
result = layer.apply(events)
recoveries = result.recoveries.aggregate()
```

See the [XoL Reinsurance](xol_reinsurance.md) tutorial for full
details.

## 6. Applying Copulas to FreqSev Results

After aggregation, you can link frequency-severity results with other
variables using copulas:

<!--pytest-codeblocks:cont-->

```python
from pal import copulas

set_random_seed(42)
motor = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=50),
    sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
).generate().aggregate()

property_loss = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=20),
    sev_dist=distributions.Pareto(shape=2, scale=50_000),
).generate().aggregate()

copulas.GaussianCopula(
    [[1, 0.6], [0.6, 1]]
).apply([motor, property_loss])

combined = motor + property_loss
```

See the [Coupling Groups and Copulas](coupling_groups_and_copulas.md)
tutorial for worked examples including how coupling groups ensure
consistency across derived variables.

## Key Classes

| Class | Description |
|-------|-------------|
| `FrequencySeverityModel` | Creates compound models from freq + sev distributions |
| `FreqSevSims` | Container for event-level simulations with sim indices |
| `StochasticScalar` | Simulation-level vector returned by `aggregate()` / `occurrence()` |

## See Also

- [Getting Started](getting_started.md) — basic PAL concepts
- [Distributions Guide](distributions_guide.md) — choosing frequency
  and severity distributions
- [Pricing an XoL Reinsurance Program](xol_reinsurance.md) — applying
  reinsurance structures to FreqSev losses
