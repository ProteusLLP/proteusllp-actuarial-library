# Risk Measures and Capital Allocation

This tutorial shows how to compute risk measures and allocate capital
across lines of business using the `pal.risk_measures` module.

## What is a Risk Measure?

A risk measure is a function ρ that maps a random loss variable X to a
real number ρ(X), representing the capital needed to make the risk
acceptable. The simplest example is the expected value E[X], but this
ignores variability. More useful measures account for the tail of the
distribution - the rare but severe scenarios that drive solvency
requirements.

One of the most commonly used risk measures is **Value at Risk** (VaR). VaR at level α is the α-quantile of the loss
distribution: the loss exceeded with probability 1 − α. VaR is
intuitive and widely used (for example, in Solvency II), but it has a serious
flaw: it ignores what happens *beyond* the quantile. Two portfolios
can have the same VaR but very different tail behaviour.

### Coherent Risk Measures

Artzner, Delbaen, Eber and Heath (1999) introduced four axioms that
a "coherent" risk measure should satisfy:

1. **Monotonicity** — if X ≤ Y in every scenario, then ρ(X) ≤ ρ(Y).
   Larger losses require more capital.
2. **Sub-additivity** — ρ(X + Y) ≤ ρ(X) + ρ(Y). Diversification
   cannot increase risk. This is the property that VaR famously
   violates.
3. **Positive homogeneity** — ρ(λX) = λρ(X) for λ > 0. Doubling the
   exposure doubles the capital.
4. **Translation invariance** — ρ(X + c) = ρ(X) + c for constant c.
   Adding a certain loss increases capital by the same amount.

TVaR (Tail Value at Risk, also known as Expected Shortfall) is the simplest coherent alternative to
VaR. Other risk measures include distortion risk measures. All distortion risk measures with a concave distortion function
are also coherent: this includes the proportional hazards, Wang, and
dual-power transforms implemented in this module.

### Spectral Risk Measures

Acerbi (2002) showed that within the class of coherent risk measures,
spectral risk measures are those that can be written as a weighted
average of quantiles:

$$
ρ(X) = E[φ(F(X))  X]
$$

where φ is a non-negative, non-decreasing weight function (the "risk
spectrum") that integrates to 1. The non-decreasing condition ensures
coherence: larger losses must receive at least as much weight as
smaller ones. TVaR is the special case where φ is a step function.

### Capital Allocation

Once a risk measure determines the total capital, it must be allocated
to individual lines of business. The Euler (gradient) allocation
splits the total so that each line's share equals its marginal
contribution:

$$
C_k = E[X_k · w]
$$

where w are the per-simulation weights derived from the risk measure.
The key property is that allocations are additive: Σ C_k = ρ(X).
This is guaranteed for any positively homogeneous, differentiable
risk measure by Euler's theorem on homogeneous functions (Tasche,
1999).

### References

- Artzner, P., Delbaen, F., Eber, J.-M. and Heath, D. (1999).
  "Coherent measures of risk." *Mathematical Finance*, 9(3), 203-228.
- Acerbi, C. (2002). "Spectral measures of risk: a coherent
  representation of subjective risk aversion." *Journal of Banking &
  Finance*, 26(7), 1505-1518.
- Tasche, D. (1999). "Risk contributions and performance measurement."
  Working paper, TU München.
- McNeil, A.J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk
  Management*, 2nd ed. Princeton University Press.

## The `RiskMeasureResult` API

Every risk measure function returns a `RiskMeasureResult` with:

- `.value` — the scalar risk measure of the aggregate loss
- `.weights` — per-simulation weights
- `.allocate(component)` — weighted expectation for allocation

When you allocate each component of a sum, the allocated amounts add
up to the total `.value`. This is the Euler property described above.

## Setup

```python
import numpy as np

from pal import ProteusVariable, config, copulas, distributions, set_random_seed
from pal.risk_measures import (
    percentile_layer,
    proportional_hazards_transform,
    standard_deviation_principle,
    tvar,
    wang_transform,
)

config.n_sims = 100_000
set_random_seed(42)
```

## 1. Generating a Multi-Line Portfolio

We model three lines of business with different loss distributions and
introduce dependence via a copula:

<!--pytest-codeblocks:cont-->

```python
portfolio = ProteusVariable(
    dim_name="lob",
    values={
        "property": distributions.LogNormal(mu=14, sigma=0.8).generate(),
        "casualty": distributions.LogNormal(mu=13, sigma=0.5).generate(),
        "marine": distributions.LogNormal(mu=12, sigma=0.6).generate(),
    },
)

copulas.GalambosCopula(2).apply(portfolio)

total = portfolio.sum()
```

```
Property mean:  4,545,084
Casualty mean:  1,358,389
Marine mean:      536,735
Total mean:     6,440,208
```

The copula introduces tail dependence — large losses across lines
tend to occur together, which matters for risk measurement.

## 2. TVaR (Tail Value at Risk)

TVaR at level α is the average of all losses above the α-quantile.
It is the most widely used coherent risk measure in insurance
regulation:

<!--pytest-codeblocks:cont-->

```python
rm_tvar = tvar(total, alpha=0.99)
print(f"TVaR 99%:  {rm_tvar.value:,.0f}")
```

```
TVaR 99%:  30,536,849
```

### Allocating TVaR

To see how much each line contributes to the 99% TVaR, pass
individual components to `.allocate()`:

<!--pytest-codeblocks:cont-->

```python
alloc = rm_tvar.allocate(portfolio)
print(alloc)
```

```
ProteusVariable (lob):
  property:  20,517,649
  casualty:   5,416,037
  marine:     4,603,163
```

Property dominates because it has the heaviest tail (σ = 0.8). The
allocated amounts sum to the total TVaR:

<!--pytest-codeblocks:cont-->

```python
print(f"Sum of allocations: {float(alloc.sum()):,.0f}")
print(f"Total TVaR:         {rm_tvar.value:,.0f}")
```

```
Sum of allocations: 30,536,849
Total TVaR:         30,536,849
```

## 3. Proportional Hazards Transform

The proportional hazards transform with parameter α distorts the
survival function: S(x) → S(x)^α. Lower α gives more weight to
the tail:

<!--pytest-codeblocks:cont-->

```python
rm_ph = proportional_hazards_transform(total, alpha=0.5)
print(f"PH (α=0.5): {rm_ph.value:,.0f}")
print(rm_ph.allocate(portfolio))
```

```
PH (α=0.5): 10,363,913
ProteusVariable (lob):
  property:   6,496,135
  casualty:   2,109,345
  marine:     1,758,433
```

With α = 1 the measure reduces to the expected value — no risk
loading at all.

## 4. Wang Transform

The Wang transform shifts the loss distribution through the normal
CDF. Parameter α controls the risk loading — higher values load
more heavily on the tail:

<!--pytest-codeblocks:cont-->

```python
rm_wang = wang_transform(total, alpha=1.0)
print(f"Wang (α=1): {rm_wang.value:,.0f}")
print(rm_wang.allocate(portfolio))
```

```
Wang (α=1): 11,046,037
ProteusVariable (lob):
  property:   7,073,001
  casualty:   2,165,019
  marine:     1,808,017
```

For normally distributed losses, the Wang transform recovers the
CAPM: W(X) = E[X] + α·Std(X).

## 5. Standard Deviation Principle

The standard deviation principle prices risk as ρ(X) = E[X] + k·σ(X).
Unlike the spectral measures above, it is **not** coherent (it can
violate monotonicity), but it is simple and widely used:

<!--pytest-codeblocks:cont-->

```python
rm_sd = standard_deviation_principle(total, k=2.0)
print(f"Std dev (k=2): {rm_sd.value:,.0f}")
print(rm_sd.allocate(portfolio))
```

```
Std dev (k=2): 18,036,513
ProteusVariable (lob):
  property:  11,684,483
  casualty:   3,336,282
  marine:     3,015,748
```

The Euler weights for the standard deviation principle are
w_j = 1 + k·(X_j − E[X]) / σ(X), so simulations above the mean
get weights greater than 1, and those below get less.

## 6. Capital Allocation by Percentile Layer (CAPL)

Unlike the measures above, `percentile_layer` does not determine the
capital amount — you supply it. This is useful when capital has
already been set (e.g. by a regulator) and you need to allocate it
across lines.

The method works by slicing the sorted loss distribution into
horizontal layers. Each layer's capital is shared equally among
the simulations that reach it:

<!--pytest-codeblocks:cont-->

```python
capital = float(np.percentile(total.values, 99.5))
rm_pl = percentile_layer(total, capital)
print(f"Capital (VaR 99.5%): {capital:,.0f}")
print(f"Allocated total:     {rm_pl.value:,.0f}")
print(rm_pl.allocate(portfolio))
```

```
Capital (VaR 99.5%): 35,839,251
Allocated total:     35,839,251
ProteusVariable (lob):
  property:  26,010,802
  casualty:   5,320,499
  marine:     4,507,950
```

The total allocated equals the capital as long as the maximum
simulated loss exceeds the capital (which it does with 100,000
simulations and a 99.5th percentile).

## 7. Comparing Allocations

Different risk measures lead to different capital allocations. Here
is a side-by-side comparison normalised to percentages:

<!--pytest-codeblocks:cont-->

```python
import pandas as pd

measures = {
    "TVaR 99%": rm_tvar,
    "PH (α=0.5)": rm_ph,
    "Wang (α=1)": rm_wang,
    "Std dev (k=2)": rm_sd,
    "CAPL (VaR 99.5%)": rm_pl,
}

rows = []
for name, rm in measures.items():
    a = rm.allocate(portfolio)
    rows.append({
        "Measure": name,
        "Property": float(a["property"]) / rm.value,
        "Casualty": float(a["casualty"]) / rm.value,
        "Marine": float(a["marine"]) / rm.value,
    })

df = pd.DataFrame(rows).set_index("Measure")
print(df)
```

```
                  Property  Casualty    Marine
Measure
TVaR 99%          0.672168  0.177221  0.150611
PH (α=0.5)        0.627253  0.203114  0.169633
Wang (α=1)        0.640697  0.195603  0.163700
Std dev (k=2)     0.648085  0.185421  0.166494
CAPL (VaR 99.5%)  0.726285  0.147909  0.125806
```

Property receives the largest share under every measure because of
its heavy tail. The more tail-focused the measure (TVaR, CAPL), the
larger property's share.

## 8. Inspecting Weights

Each `RiskMeasureResult` carries per-simulation weights. You can
inspect them to understand how the measure treats different parts
of the distribution:

<!--pytest-codeblocks:cont-->

```python
w = rm_tvar.weights.values
print(f"TVaR weights — min: {w.min():.2f}, max: {w.max():.2f}, "
      f"mean: {w.mean():.2f}")
print(f"  Fraction that are zero: {(w == 0).mean():.1%}")
```

```
TVaR weights — min: 0.00, max: 100.00, mean: 1.00
  Fraction that are zero: 99.0%
```

TVaR at 99% puts all weight on the top 1% of simulations (weight =
1/(1−0.99) = 100) and zero on the rest. Spectral measures like
the proportional hazards or Wang transforms spread weight more
smoothly across the distribution.

## 9. Visualising Weights

Plotting the weight function against percentile rank makes the
difference between risk measures tangible. We sort simulations by
total loss and plot the weight each receives:

<!--pytest.mark.skip-->

```python
import plotly.graph_objects as go

# Sort by total loss
order = np.argsort(total.values)
percentiles = np.arange(len(order)) / len(order) * 100

fig = go.Figure()

for name, rm in [
    ("TVaR 99%", rm_tvar),
    ("PH (α=0.5)", rm_ph),
    ("Wang (α=1)", rm_wang),
    ("Std dev (k=2)", rm_sd),
]:
    w_sorted = rm.weights.values[order]
    fig.add_trace(go.Scatter(
        x=percentiles, y=w_sorted, mode="lines", name=name
    ))

fig.update_layout(
    title="Risk Measure Weights by Percentile",
    xaxis_title="Percentile of Total Loss",
    yaxis_title="Weight",
    yaxis=dict(range=[0, 10]),
)
fig.show()
```

The chart shows how each risk measure distributes emphasis across
the loss distribution:

- **TVaR** is a step function — zero below the 99th percentile,
  then a flat 100 in the top 1%.
- **PH transform** rises smoothly, giving increasing weight to
  worse outcomes without the hard cutoff.
- **Wang transform** has a similar smooth shape, curving upward
  more steeply in the far tail.
- **Standard deviation** is linear in the loss value (not the
  percentile), so its shape depends on the distribution.

## 10. Pricing an XoL Tower with a Distortion Risk Measure

Risk measures provide a principled way to price reinsurance layers.
An XoL (excess-of-loss) layer pays the portion of each loss between
an attachment point and a limit. The distortion-based price is the
risk measure applied to the layer's aggregate recoveries.

A key feature of distortion pricing is that higher-attaching layers
attract a greater risk loading — they respond only in severe
scenarios, so the distortion weights push the price further above
the expected loss.

### Generating Ground-Up Losses

<!--pytest-codeblocks:cont-->

```python
from pal import XoLTower
from pal.frequency_severity import FrequencySeverityModel

set_random_seed(42)

losses = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=2),
    sev_dist=distributions.LogNormal(mu=12, sigma=1.5),
).generate()

print(f"Mean aggregate loss: {losses.aggregate().mean():,.0f}")
```

```
Mean aggregate loss: 2,744,823
```

### Defining a Tower of Layers

We use `XoLTower` to define five layers, each with a £2m limit,
stacked from £1m up to £11m:

<!--pytest-codeblocks:cont-->

```python
tower = XoLTower(
    name=["2m xs 1m", "2m xs 3m", "2m xs 5m", "2m xs 7m", "2m xs 9m"],
    limit=  [2_000_000] * 5,
    excess= [1_000_000, 3_000_000, 5_000_000, 7_000_000, 9_000_000],
    premium=[0] * 5,
)
```

### Pricing Each Layer

We apply the proportional hazards transform (α = 0.5) to each
layer's aggregate recoveries. For each layer we show:

- **Loss on line** — expected loss as a fraction of the limit
- **Rate on line** — PH price as a fraction of the limit
- **Loading** — how much the PH price exceeds the expected loss, as a percentage of the expected loss

<!--pytest-codeblocks:cont-->

```python
rows = []
for layer in tower.layers:
    recoveries = layer.apply(losses).recoveries.aggregate()
    el = recoveries.mean()
    rm = proportional_hazards_transform(recoveries, alpha=0.5)
    rows.append({
        "Layer": layer.name,
        "Expected": el,
        "PH Price": rm.value,
        "Loss/Line": el / layer.limit,
        "Rate/Line": rm.value / layer.limit,
        "Loading": rm.value / el - 1,
    })

df = pd.DataFrame(rows).set_index("Layer")
print(df)
```

```
              Expected      PH Price  Loss/Line  Rate/Line   Loading
Layer
2m xs 1m  621827.3125  797618.9375   0.310916   0.398810  0.282729
2m xs 3m  223455.8125  443695.8125   0.111728   0.221848  0.985656
2m xs 5m   91261.0000  269310.0625   0.045631   0.134655  1.950308
2m xs 7m   42084.4375  170485.0625   0.021042   0.085243  3.051515
2m xs 9m   21300.1250  113502.0000   0.010650   0.056751  4.328703
```

The pattern is clear: as the attachment point increases, the loss
on line drops (fewer events reach higher layers), but the loading
increases sharply. The £2m xs £9m layer has a loss on line of only
1%, but the PH transform loads it at over 400% — more than five
times the expected loss.

This is a fundamental property of coherent distortion pricing:
the distortion function places increasing weight on tail
percentiles, so layers that respond only in extreme scenarios
receive a proportionally higher risk charge. The rate on line
decreases more slowly than the loss on line, reflecting the
market reality that excess layers are expensive per unit of
expected loss.

### Visualising the Price Curve

<!--pytest.mark.skip-->

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

excesses = [layer.excess / 1e6 for layer in tower.layers]
els, prices, loadings = [], [], []
for layer in tower.layers:
    rec = layer.apply(losses).recoveries.aggregate()
    rm = proportional_hazards_transform(rec, alpha=0.5)
    els.append(rec.mean() / layer.limit * 100)
    prices.append(rm.value / layer.limit * 100)
    loadings.append((rm.value / rec.mean() - 1) * 100)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(
    x=excesses, y=els, name="Loss on Line %", marker_color="steelblue",
))
fig.add_trace(go.Bar(
    x=excesses, y=prices, name="Rate on Line %", marker_color="coral",
))
fig.add_trace(go.Scatter(
    x=excesses, y=loadings, name="Loading %",
    mode="lines+markers", line=dict(color="black", width=2),
), secondary_y=True)

fig.update_layout(
    title="XoL Pricing by Attachment Point (PH α=0.5)",
    xaxis_title="Attachment Point (£m)",
    barmode="group",
)
fig.update_yaxes(title_text="% of Limit", secondary_y=False)
fig.update_yaxes(title_text="Loading %", secondary_y=True)
fig.show()
```

The chart shows loss on line and rate on line as bars (both
declining with attachment), and the loading as a line (rising
steeply). This is the signature pattern of distortion pricing:
the market charges more per unit of risk for remote layers.

## Summary

| Function | Type | Key Parameter |
|----------|------|---------------|
| `tvar` | Spectral | α — confidence level |
| `proportional_hazards_transform` | Spectral | α — survival distortion power |
| `wang_transform` | Spectral | α — normal CDF shift |
| `dual_power_transform` | Spectral | β — CDF distortion power |
| `exponential_transform` | Spectral | γ — exponential risk aversion |
| `svar` | Spectral | lower, upper — percentile window |
| `standard_deviation_principle` | Euler | k — std deviation loading |
| `percentile_layer` | Layer-based | capital — amount to allocate |

All functions return a `RiskMeasureResult`. Use `.value` for the
risk measure and `.allocate()` for capital allocation.
