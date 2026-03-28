# XoL Reinstatement Pricing — Mata (2000)

This tutorial reproduces key results from a classic paper on pricing
excess-of-loss reinsurance with reinstatements:

> Mata, A.J. (2000). "Pricing Excess of Loss Reinsurance with
> Reinstatements." *ASTIN Bulletin*, 30(2), 349-368.

We focus on **free reinstatements** (c = 0) and compare pure premiums
and PH transform risk-adjusted premiums for individual layers versus
a combined layer.

## Background

In excess-of-loss reinsurance, the reinsurer's liability for each
event is capped at the layer limit. **Reinstatements** cap the
*aggregate* number of layer limits the reinsurer will pay per year.
With K reinstatements the aggregate limit is (K + 1) × m, where m is
the occurrence limit.

- **Free reinstatements** (c = 0): no additional premium is charged
  when the layer is reinstated.
- **Paid reinstatements**: each reinstatement triggers an extra
  premium pro rata to the claim size.

The paper studies how reinstatements affect the premium under
different premium principles (pure premium, standard deviation,
PH transform) and proves several inequalities relating the premium
for individual layers versus a combined layer.

## Setup

The paper models an insurance portfolio with:

- **Frequency**: Poisson(λ = 10) — expected 10 claims per year
- **Severity**: Pareto Type II (α = 3, β = 10)

```python
import pandas as pd

from pal.config import set_default_n_sims, set_random_seed
from pal.contracts import XoLTower
from pal.distributions import Pareto, Poisson
from pal.frequency_severity import FrequencySeverityModel
from pal.risk_measures import proportional_hazards_transform

set_random_seed(42)
set_default_n_sims(100_000)
```

PAL's `Pareto` distribution is Type I (support [scale, ∞)). The
paper's Pareto Type II (Lomax) with α = 3, β = 10 is obtained by
subtracting the scale parameter:

<!--pytest-codeblocks:cont-->

```python
model = FrequencySeverityModel(
    Poisson(10.0), Pareto(shape=3, scale=10)
)
claims = model.generate() - 10.0
```

## Contracts

The paper compares two separate layers (10 xs 10 and 10 xs 20)
against a single combined layer (20 xs 10). We model all three
as layers in an `XoLTower`.

With K = 1 free reinstatement, the aggregate limit is 2 × the
occurrence limit and the reinstatement cost is zero:

<!--pytest-codeblocks:cont-->

```python
tower_k1 = XoLTower(
    name=["10 xs 10", "10 xs 20", "20 xs 10"],
    limit=[10, 10, 20],
    excess=[10, 20, 10],
    premium=[1, 1, 1],
    reinstatement_cost=[[0.0], [0.0], [0.0]],
    aggregate_limit=[20, 20, 40],
)
```

For unlimited reinstatements we omit the aggregate limit:

<!--pytest-codeblocks:cont-->

```python
tower_inf = XoLTower(
    name=["10 xs 10", "10 xs 20", "20 xs 10"],
    limit=[10, 10, 20],
    excess=[10, 20, 10],
    premium=[1, 1, 1],
)
```

## Table 2 — Pure Premiums

For **free** reinstatements (c = 0) the pure premium is simply the
expected aggregate recovery E[R_K]. We compare K = 1 (one free
reinstatement) with K = ∞ (unlimited):

<!--pytest-codeblocks:cont-->

```python
for tower in [tower_k1, tower_inf]:
    for layer in tower.layers:
        layer.apply(claims)


def mean(tower, i):
    return tower.layers[i].summary["mean"]


df2 = pd.DataFrame([
    {
        "Layer": name,
        "K=1 (c=0)": mean(tower_k1, i),
        "K=inf": mean(tower_inf, i),
    }
    for i, name in enumerate(["10 xs 10", "10 xs 20"])
] + [
    {
        "Layer": "SUM",
        "K=1 (c=0)": mean(tower_k1, 0)
        + mean(tower_k1, 1),
        "K=inf": mean(tower_inf, 0)
        + mean(tower_inf, 1),
    },
    {
        "Layer": "20 xs 10",
        "K=1 (c=0)": mean(tower_k1, 2),
        "K=inf": mean(tower_inf, 2),
    },
]).set_index("Layer")
print(df2)
```

Expected values from the paper (Table 2, λ = 10):

| Layer | K=1 (c=0) | K=∞ |
|-------|-----------|-----|
| 10 xs 10 | 6.6128 | 6.9444 |
| 10 xs 20 | 2.4103 | 2.4305 |
| SUM | 9.0231 | 9.3749 |
| 20 xs 10 | 9.2173 | 9.3749 |

Key observations:

- With **unlimited** reinstatements the sum of the layer premiums
  equals the combined layer premium — since the aggregate losses
  are the same when there is no aggregate cap.
- With **K = 1** the sum of the layer premiums is **less than** the
  combined layer premium. The combined layer provides more cover
  because hitting the aggregate limit in one sub-layer doesn't
  consume capacity in the other.

## Table 5 — PH Transform Premiums

The Proportional Hazard (PH) transform is a risk-adjusted premium
principle introduced by Wang (1995). For a loss X with survival
function S(x), the PH premium is:

$$
\pi_p(X) = \int_0^\infty [S(x)]^{1/p} \, dx
$$

For p = 1 this is the pure premium (expected value). Higher values
of p place more weight on the tail, producing a larger loaded
premium.

Mata (2000) shows that for free reinstatements with λ = 10:

1. **Sub-additivity**: π_p(S₁* + S₂*) ≤ π_p(S₁*) + π_p(S₂*)
2. **Combined ≥ sum**: π_p(S₁*) + π_p(S₂*) ≤ π_p(S_c*)

PAL's `proportional_hazards_transform` uses alpha = 1/p:

<!--pytest-codeblocks:cont-->

```python
agg1 = tower_k1.layers[0].apply(claims).recoveries.aggregate()
agg2 = tower_k1.layers[1].apply(claims).recoveries.aggregate()
aggc = tower_k1.layers[2].apply(claims).recoveries.aggregate()

rows = []
for p in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    alpha = 1.0 / p
    ph1 = proportional_hazards_transform(agg1, alpha=alpha)
    ph2 = proportional_hazards_transform(agg2, alpha=alpha)
    ph_sum = proportional_hazards_transform(
        agg1 + agg2, alpha=alpha
    )
    phc = proportional_hazards_transform(aggc, alpha=alpha)
    rows.append({
        "p": p,
        "pi_p(S1*+S2*)": ph_sum.value,
        "pi_p(S1*)": ph1.value,
        "pi_p(S2*)": ph2.value,
        "pi_p(S1*)+pi_p(S2*)": ph1.value + ph2.value,
        "pi_p(Sc*)": phc.value,
    })

df5 = pd.DataFrame(rows).set_index("p")
print(df5)
```

Expected values from the paper (Table 5, λ = 10, K = 1 free):

| p | π_p(S₁\*+S₂\*) | π_p(S₁\*) | π_p(S₂\*) | π_p(S₁\*)+π_p(S₂\*) | π_p(S_c\*) |
|---|------|------|------|-------|------|
| 1.0 | 9.0232 | 6.6128 | 2.4103 | 9.0232 | 9.2173 |
| 1.2 | 10.9094 | 7.7403 | 3.2344 | 10.9747 | 11.1852 |
| 1.4 | 12.6235 | 8.7116 | 4.0313 | 12.8183 | 12.9715 |
| 1.6 | 14.1775 | 9.5518 | 4.7875 | 14.3393 | 14.5856 |
| 1.8 | 15.5862 | 10.2828 | 5.4971 | 15.7799 | 16.0425 |
| 2.0 | 16.8645 | 10.9230 | 6.1590 | 17.0820 | 17.3585 |

## Summary

- **Reinstatements** are modelled via `aggregate_limit` and
  `reinstatement_cost` on `XoL` layers (or `XoLTower`).
- For **free** reinstatements, `reinstatement_cost=[0.0]` with
  `aggregate_limit = (K+1) * limit`.
- The **pure premium** with limited reinstatements is simply the
  mean aggregate recovery.
- The **PH transform** premium is computed via
  `proportional_hazards_transform` with `alpha = 1/p`.

## References

- Mata, A.J. (2000). "Pricing Excess of Loss Reinsurance with
  Reinstatements." *ASTIN Bulletin*, 30(2), 349-368.
- Sundt, B. (1991). "On excess of loss reinsurance with
  reinstatements." *Bulletin of the Swiss Association of
  Actuaries*, 1991(1), 51-65.
- Wang, S.S. (1995). "Insurance pricing and increased limits
  ratemaking by proportional hazards transforms." *Insurance:
  Mathematics and Economics*, 17(1), 43-54.
