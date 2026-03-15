# Pricing an Excess-of-Loss Reinsurance Program

This tutorial shows how to model and price an excess-of-loss (XoL)
reinsurance program using PAL. We build up from a single layer to a
full tower, including reinstatements and aggregate limits.

## Prerequisites

```python
import numpy as np

from pal import XoLTower, config, distributions, set_random_seed
from pal.contracts import XoL
from pal.frequency_severity import FrequencySeverityModel

config.n_sims = 100_000
set_random_seed(42)
```

## 1. Generating Gross Losses

First, model the underlying claims using a frequency-severity approach.
Here we use a GPD (Generalised Pareto Distribution) for severities
above a threshold of 1,000,000, with roughly 2 large losses per year:

<!--pytest-codeblocks:cont-->

```python
sev_dist = distributions.GPD(
    shape=0.33, scale=100_000, loc=1_000_000
)
freq_dist = distributions.Poisson(mean=2)

losses = FrequencySeverityModel(freq_dist, sev_dist).generate()
```

```
Total events:       200,194
Aggregate mean:     2,299,070
Max event mean:     1,078,879
```

Each simulation contains a random number of events (Poisson with
mean 2). The `FreqSevSims` object tracks both the event values and
which simulation each event belongs to.

## 2. A Single XoL Layer

An XoL layer pays the portion of each individual loss that falls within
a band defined by an **excess** (attachment point) and a **limit**:

```
Recovery = min(max(loss - excess, 0), limit)
```

### Example: £1m xs £1m

<!--pytest-codeblocks:cont-->

```python
layer = XoL(
    name="1m xs 1m",
    limit=1_000_000,
    excess=1_000_000,
    premium=50_000,
    reinstatement_cost=[1.0, 1.0],
    aggregate_limit=3_000_000,
)

result = layer.apply(losses)
```

```
Layer Name : 1m xs 1m
Mean Recoveries:   282,124
SD Recoveries:     325,378
Probability of Attachment:           86.5%
Probability of Vertical Exhaustion:  43.2%
Probability of Horizontal Exhaustion: 0.0%
```

**Key metrics:**

- **Probability of attachment** — fraction of simulations with at least
  one recovery (86.5% — most years see at least one loss above 1m)
- **Probability of vertical exhaustion** — fraction of simulations where
  at least one event pierces through the layer (loss > 2m)
- **Probability of horizontal exhaustion** — fraction of simulations
  where the aggregate limit is fully used

### Reinstatements

The `reinstatement_cost` parameter defines how many times the layer can
be reinstated and at what cost (as a fraction of the base premium).
Here `[1.0, 1.0]` means 2 reinstatements, each costing 100% of the
base premium.

With 2 reinstatements and a limit of 1m, the **aggregate limit** is
3 × 1m = 3m (original + 2 reinstatements).

### Aggregate Limits and Deductibles

- `aggregate_limit` — maximum total recovery across all events in a
  year. Once reached, the layer is exhausted horizontally.
- `aggregate_deductible` — the cedant retains the first N of aggregate
  recoveries before the reinsurer starts paying.
- `franchise` / `reverse_franchise` — individual loss must be within
  `[franchise, reverse_franchise)` for the layer to respond.

## 3. Building an XoL Tower

A tower stacks multiple layers to cover a range of loss severity.
`XoLTower` creates all layers in one call:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)
losses = FrequencySeverityModel(freq_dist, sev_dist).generate()

tower = XoLTower(
    limit=   [1_000_000, 1_000_000, 2_000_000, 5_000_000],
    excess=  [1_000_000, 2_000_000, 3_000_000, 5_000_000],
    premium= [   80_000,    40_000,    25_000,    10_000],
    reinstatement_cost=[
        [1.0, 1.0, 1.0],   # 3 reinstatements at 100%
        [1.0, 1.0],         # 2 reinstatements at 100%
        [1.0],              # 1 reinstatement at 100%
        None,               # unlimited (no reinstatements)
    ],
    aggregate_limit=[
        4_000_000,   # 1m × 4
        3_000_000,   # 1m × 3
        4_000_000,   # 2m × 2
        None,        # unlimited
    ],
)
```

This creates a 4-layer tower:

| Layer | Band | Premium | Reinstatements | Agg Limit |
|-------|------|---------|----------------|-----------|
| 1 | 1m xs 1m | 80,000 | 3 @ 100% | 4m |
| 2 | 1m xs 2m | 40,000 | 2 @ 100% | 3m |
| 3 | 2m xs 3m | 25,000 | 1 @ 100% | 4m |
| 4 | 5m xs 5m | 10,000 | unlimited | unlimited |

### Applying the Tower

<!--pytest-codeblocks:cont-->

```python
tower_result = tower.apply(losses)
tower.print_summary()
```

```
Layer Name : Layer 1
Mean Recoveries:   282,124
Probability of Attachment:  86.5%

Layer Name : Layer 2
Mean Recoveries:    10,565
Probability of Attachment:   2.4%

Layer Name : Layer 3
Mean Recoveries:     3,392
Probability of Attachment:   0.4%

Layer Name : Layer 4
Mean Recoveries:       944
Probability of Attachment:   0.1%
```

Notice how mean recoveries drop sharply as the attachment point
increases — Layer 4 (5m xs 5m) attaches in only 0.1% of simulations.

### Net Loss Calculation

<!--pytest-codeblocks:cont-->

```python
total_recoveries = tower_result.recoveries.aggregate()
gross_agg = losses.aggregate()
net_agg = gross_agg - total_recoveries
```

```
Gross mean:                2,299,070
Total tower recoveries:      297,025
Net mean:                  2,002,045
Cession rate:                  12.9%
```

### Reinstatement Premiums

The tower also calculates reinstatement premiums — additional premium
the cedant owes when reinstatements are used:

<!--pytest-codeblocks:cont-->

```python
tower_result.reinstatement_premium.mean()
# => 23,035
```

This can be incorporated into the overall cost of the program.

## 4. Adding Stochastic Inflation

Real-world losses are subject to inflation. You can multiply
frequency-severity losses by a stochastic inflation factor before
applying the tower:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)
losses = FrequencySeverityModel(freq_dist, sev_dist).generate()

# Stochastic inflation: mean 5%, sd 2%
inflation = distributions.Normal(0.05, 0.02).generate()
inflated_losses = losses * (1 + inflation)

tower_result = tower.apply(inflated_losses)
```

Because `FreqSevSims` supports arithmetic with `StochasticScalar`
objects, each event is multiplied by the inflation factor for its
simulation. The coupling group system ensures consistency.

## 5. Adding Expense Loadings

Loss adjustment expenses (LAE) can be applied before the tower:

<!--pytest-codeblocks:cont-->

```python
losses_with_lae = losses * 1.05   # 5% LAE loading
tower_result = tower.apply(losses_with_lae)
```

This increases recoveries because more losses now pierce the
attachment point.

## 6. Combining with Copulas

To model a scenario where reinsurance losses are correlated with
catastrophe events, use a copula on the aggregate results:

<!--pytest-codeblocks:cont-->

```python
from pal import copulas

set_random_seed(42)
losses = FrequencySeverityModel(freq_dist, sev_dist).generate()
tower_result = tower.apply(losses)

cat_loss = distributions.LogNormal(mu=16, sigma=1.2).generate()

# Correlate tower recoveries with cat losses
tower_recoveries = tower_result.recoveries.aggregate()
copulas.GumbelCopula(theta=1.5, n=2).apply(
    [tower_recoveries, cat_loss]
)

# Total reinsurance cost to cedant
total_cost = tower_recoveries + cat_loss
```

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Excess** | Attachment point — losses below this are retained |
| **Limit** | Maximum recovery per event |
| **Aggregate limit** | Maximum total recovery per year |
| **Aggregate deductible** | Annual deductible before layer responds |
| **Reinstatement** | Restoring coverage after a loss; may cost additional premium |
| **Franchise** | Minimum loss size for layer to respond |
| **Reverse franchise** | Maximum loss size for layer to respond |

## See Also

- [Getting Started](getting_started.md) — basic PAL concepts
- [Frequency-Severity Modelling](frequency_severity_modelling.md) —
  generating the underlying losses
- [Coupling Groups, Copulas and Variable Reordering](coupling_groups_and_copulas.md) —
  adding dependencies between reinsurance and other variables
