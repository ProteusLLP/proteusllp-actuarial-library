# Property Reinsurance Exposure Rating with MBBEFD

This tutorial starts with a classical analytical property exposure-rating calculation and then shows what simulation adds. For ordinary occurrence-only layers, the analytical MBBEFD calculation gives the expected loss directly. Simulation becomes useful when the contract contains annual aggregate features, or when the same loss model is also needed as an input to a wider capital model.

The complete runnable example is `examples/example_property_exposure_rating.py`, with sample data in `examples/data/property_exposures.csv`.

## 1. Exposure data

The sample schedule contains one row per insured risk:

| Field | Meaning |
|---|---|
| `maximum_loss` | Maximum ground-up loss used to scale the MBBEFD damage ratio |
| `policy_limit` | Maximum insurer payment above the policy deductible |
| `policy_deductible` | Ground-up deductible |
| `subject_premium` | Premium used as the exposure-rating base |
| `expected_loss_ratio` | Expected policy loss divided by subject premium |
| `mbbefd_c` | One-parameter Swiss Re curve parameter |

Claim frequency is deliberately **not** an input. It is implied by premium, loss ratio and the policy severity model. `maximum_loss` is separate from the policy limit, so it can represent the full property exposure, including business interruption or other additional cover, even where the contractual limit is lower.

```python
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pal import set_random_seed
from pal.contracts import XoLTower
from pal.distributions import MBBEFD, Poisson
from pal.empirical import Empirical
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel
from pal.variables import ProteusVariable, StochasticScalar

N_SIMS = 100_000
DATA_PATH = Path("examples/data/property_exposures.csv")

exposure_df = pd.read_csv(DATA_PATH)
exposure_df.head()
```

The first few rows are:

| | maximum_loss | policy_limit | policy_deductible | subject_premium | expected_loss_ratio | mbbefd_c |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8,000,000 | 5,000,000 | 100,000 | 60,000 | 0.55 | 2.0 |
| 1 | 12,000,000 | 10,000,000 | 250,000 | 80,000 | 0.60 | 2.5 |
| 2 | 20,000,000 | 15,000,000 | 500,000 | 120,000 | 0.60 | 3.0 |
| 3 | 30,000,000 | 20,000,000 | 1,000,000 | 180,000 | 0.62 | 3.5 |
| 4 | 45,000,000 | 30,000,000 | 1,000,000 | 260,000 | 0.65 | 4.0 |

The dataframe is converted into PAL objects straight away. Each field becomes a `StochasticScalar`, with one value for each exposure row:

<!--pytest-codeblocks:cont-->

```python
exposure = ProteusVariable(
    dim_name="field",
    values={
        column: StochasticScalar(exposure_df[column])
        for column in exposure_df.columns
    },
)
```

## 2. Calibrate the policy loss model

Let {math}`X_i\in[0,1]` be the MBBEFD damage ratio for risk {math}`i`, conditional on a ground-up claim, and let {math}`M_i` be its maximum loss. Ground-up loss is

```{math}
L_i=M_iX_i.
```

For policy deductible {math}`D_i` and limit {math}`P_i`, policy loss is

```{math}
Y_i=\min\{(L_i-D_i)_+,P_i\}.
```

The MBBEFD exposure curve is

```{math}
G_i(x)=\frac{E[\min(X_i,x)]}{E[X_i]}.
```

Therefore

```{math}
E[Y_i\mid\text{claim}]
=M_iE[X_i]
\left[
G_i\!\left(\frac{D_i+P_i}{M_i}\right)
-G_i\!\left(\frac{D_i}{M_i}\right)
\right],
```

with the curve arguments capped at one.

PAL evaluates the MBBEFD mean and exposure curve directly across the exposure vectors:

<!--pytest-codeblocks:cont-->

```python
maximum_loss = exposure["maximum_loss"]
policy_limit = exposure["policy_limit"]
deductible = exposure["policy_deductible"]

mbbefd = MBBEFD.from_c(exposure["mbbefd_c"])

lower = np.minimum(deductible / maximum_loss, 1.0)
upper = np.minimum((deductible + policy_limit) / maximum_loss, 1.0)
policy_share = mbbefd.exposure_curve(upper) - mbbefd.exposure_curve(lower)

expected_policy_severity = maximum_loss * mbbefd.mean() * policy_share
expected_policy_loss = exposure["subject_premium"] * exposure["expected_loss_ratio"]
frequencies = expected_policy_loss / expected_policy_severity
```

The expected annual policy loss is

```{math}
\mu_i=\text{premium}_i\times\text{expected loss ratio}_i,
```

so the annual ground-up claim frequency is

```{math}
\lambda_i=\frac{\mu_i}{E[Y_i\mid\text{claim}]}.
```

For this schedule the total annual ground-up claim frequency is approximately 1.883 claims.

## 3. Analytical exposure rating

Suppose the reinsurance programme contains three occurrence layers:

<!--pytest-codeblocks:cont-->

```python
tower = XoLTower(
    name=["5m xs 5m", "10m xs 10m", "20m xs 20m"],
    limit=[5_000_000, 10_000_000, 20_000_000],
    excess=[5_000_000, 10_000_000, 20_000_000],
    premium=[0.0, 0.0, 0.0],
)
```

For a layer with excess {math}`A` and limit {math}`U`, the corresponding ground-up thresholds for a policy with deductible {math}`D` and limit {math}`P` are

```{math}
D+\min(A,P)
\quad\text{and}\quad
D+\min(A+U,P).
```

The expected layer loss is the expected policy loss multiplied by the ratio of the layer exposure-curve increment to the policy exposure-curve increment:

<!--pytest-codeblocks:cont-->

```python
total_subject_premium = exposure["subject_premium"].sum()

analytical_rates = []
layer_names = []

for layer in tower.layers:
    layer_lower = np.minimum(
        (deductible + np.minimum(layer.excess, policy_limit)) / maximum_loss,
        1.0,
    )
    layer_upper = np.minimum(
        (deductible + np.minimum(layer.excess + layer.limit, policy_limit))
        / maximum_loss,
        1.0,
    )
    layer_share = (
        mbbefd.exposure_curve(layer_upper)
        - mbbefd.exposure_curve(layer_lower)
    )

    analytical_expected_loss = (
        expected_policy_loss * layer_share / policy_share
    ).sum()

    layer_names.append(layer.name)
    analytical_rates.append(
        analytical_expected_loss / total_subject_premium
    )
```

This gives:

| Layer | Analytical exposure rate |
|---|---:|
| 5m xs 5m | 11.635% |
| 10m xs 10m | 12.292% |
| 20m xs 20m | 10.155% |

For these occurrence-only layers, the analytical calculation is sufficient for the expected loss. It is fast and has no Monte Carlo error.

## 4. What simulation adds

Simulation is useful when the pricing problem depends on more than the expected loss from each individual occurrence. In particular, an **annual aggregate deductible** depends on the total layer loss over the year, and an **annual aggregate limit** depends on how multiple recoveries accumulate through the year. A simulation also gives a full annual loss distribution that can be reused in an internal capital model rather than being collapsed immediately to one expected-loss number.

The analytical occurrence calculation therefore remains the benchmark. The simulation extends it to annual contract features and downstream risk calculations for which the annual pattern of losses matters.

## 5. Simulate the portfolio losses

The portfolio claim count is

```{math}
N\sim\operatorname{Poisson}\left(\sum_i\lambda_i\right).
```

Conditional on a portfolio claim occurring, risk {math}`i` is selected with probability

```{math}
\Pr(I=i)=\frac{\lambda_i}{\sum_j\lambda_j}.
```

A weighted empirical distribution over exposure-row numbers represents this selection directly:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)

row_distribution = Empirical(
    samples=np.arange(len(exposure_df)),
    weights=frequencies,
)

claim_rows = FrequencySeverityModel(
    Poisson(frequencies.sum()),
    row_distribution,
).generate(N_SIMS)
```

`claim_rows` is a `FreqSevSims`. Its `sim_index` identifies the simulation year containing each ground-up claim, while its values identify the exposure row that produced the claim.

The sampled row numbers then select the corresponding policy terms and MBBEFD parameter:

<!--pytest-codeblocks:cont-->

```python
row_index = StochasticScalar(claim_rows.values)

selected_maximum_loss = exposure["maximum_loss"][row_index]
selected_policy_limit = exposure["policy_limit"][row_index]
selected_deductible = exposure["policy_deductible"][row_index]
selected_c = exposure["mbbefd_c"][row_index]

damage_ratio = MBBEFD.from_c(selected_c).generate(len(row_index))

policy_loss = np.minimum(
    np.maximum(
        damage_ratio * selected_maximum_loss - selected_deductible,
        0.0,
    ),
    selected_policy_limit,
)

policy_losses = FreqSevSims(
    claim_rows.sim_index,
    policy_loss.values,
    claim_rows.n_sims,
)
```

The resulting `FreqSevSims` is an ordinary occurrence-level policy-loss simulation and can be passed to PAL's reinsurance contracts directly:

<!--pytest-codeblocks:cont-->

```python
tower_result = tower.apply(policy_losses)
```

## 6. Validate the simulation

The simulation should reproduce the expected policy loss used to calibrate it and the analytical occurrence-layer rates.

<!--pytest-codeblocks:cont-->

```python
target_expected_loss = expected_policy_loss.sum()
simulated_expected_loss = policy_losses.aggregate().mean()

calibration = pd.DataFrame(
    {
        "Target": [
            frequencies.sum(),
            target_expected_loss,
            target_expected_loss / total_subject_premium,
        ],
        "Simulation": [
            claim_rows.count().mean(),
            simulated_expected_loss,
            simulated_expected_loss / total_subject_premium,
        ],
    },
    index=[
        "Annual ground-up claim count",
        "Annual policy loss",
        "Portfolio loss ratio",
    ],
)

calibration
```

For 100,000 CPU simulations with seed 42:

| Metric | Target | Simulation |
|---|---:|---:|
| Annual ground-up claim count | 1.882649 | 1.883590 |
| Annual policy loss | £1,422,100 | £1,430,459 |
| Portfolio loss ratio | 66.144% | 66.533% |

The small differences are Monte Carlo error.

The paid-claim severity distribution is available directly from PAL:

<!--pytest-codeblocks:cont-->

```python
severity = StochasticScalar(policy_losses.values)
paid_severity = severity[severity > 0]
severity_figure = paid_severity.show_histogram(
    title="Portfolio Paid-Claim Severity"
)
```

<div id="property-claim-severity" style="width: 100%; height: 440px;"></div>
<script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>
<script src="../_static/js/property_exposure_plots.js"></script>

## 7. Add an annual aggregate deductible

Now suppose each reinsurance layer has a £1m annual aggregate deductible. The occurrence terms are unchanged, but the reinsurer does not pay the first £1m of aggregate recoveries from each layer during the year:

<!--pytest-codeblocks:cont-->

```python
aggregate_tower = XoLTower(
    name=layer_names,
    limit=[5_000_000, 10_000_000, 20_000_000],
    excess=[5_000_000, 10_000_000, 20_000_000],
    premium=[0.0, 0.0, 0.0],
    aggregate_deductible=[1_000_000, 1_000_000, 1_000_000],
)

aggregate_tower_result = aggregate_tower.apply(policy_losses)
```

An aggregate limit can be introduced in exactly the same way with the `aggregate_limit` argument. Because the annual deductible depends on the sum of all recoveries in a year, it is not represented by the simple occurrence exposure-curve increment above. The simulation already contains the required annual pattern of occurrences.

The effect on expected burn is available from the layer summaries:

<!--pytest-codeblocks:cont-->

```python
occurrence_only_rates = [
    layer.summary["mean"] / total_subject_premium
    for layer in tower.layers
]
aggregate_rates = [
    layer.summary["mean"] / total_subject_premium
    for layer in aggregate_tower.layers
]

comparison = pd.DataFrame(
    {
        "Analytical exposure rate": analytical_rates,
        "Simulated occurrence-only burn": occurrence_only_rates,
        "Simulated with £1m aggregate deductible": aggregate_rates,
    },
    index=layer_names,
)

comparison
```

For the same 100,000 simulations:

| Layer | Analytical exposure rate | Simulated occurrence-only burn | With £1m aggregate deductible |
|---|---:|---:|---:|
| 5m xs 5m | 11.635% | 11.742% | 8.962% |
| 10m xs 10m | 12.292% | 12.355% | 10.715% |
| 20m xs 20m | 10.155% | 10.305% | 9.551% |

Across the whole tower, the simulated mean reinsurance burn rate falls from 34.402% to 29.227% of subject premium. The analytical exposure rates remain the right benchmark for the occurrence-only structure; the lower burn for the aggregate structure is the effect of the annual contract term.

A grouped bar chart makes the impact clear:

<!--pytest-codeblocks:cont-->

```python
comparison_figure = go.Figure(
    [
        go.Bar(
            x=layer_names,
            y=analytical_rates,
            name="Analytical exposure rate",
        ),
        go.Bar(
            x=layer_names,
            y=occurrence_only_rates,
            name="Simulated occurrence-only burn",
        ),
        go.Bar(
            x=layer_names,
            y=aggregate_rates,
            name="With £1m aggregate deductible",
        ),
    ]
)
comparison_figure.update_layout(
    title="Impact of Aggregate Terms on Layer Burn Rate",
    xaxis_title="Layer",
    yaxis_title="Rate on subject premium",
    yaxis_tickformat=".1%",
    barmode="group",
)
comparison_figure.show()
```

<div id="property-layer-rate" style="width: 100%; height: 440px;"></div>

## 8. Reuse the simulations in a capital model

The simulation is also useful beyond pricing. Nothing needs to be resimulated to obtain annual gross, reinsurance and net property losses:

<!--pytest-codeblocks:cont-->

```python
annual_gross_loss = policy_losses.aggregate()
annual_reinsurance_recovery = aggregate_tower_result.recoveries.aggregate()
annual_net_loss = annual_gross_loss - annual_reinsurance_recovery
```

These are `StochasticScalar` objects, so `annual_net_loss` can be used directly as the property underwriting-risk component of an internal capital model. It can be combined with casualty, market, credit or other simulated risks, dependence can be imposed with PAL copulas, and portfolio VaR, TVaR and capital allocations can then be calculated with the normal PAL risk-measure API.

The important distinction is that the pricing model has retained the **full annual distribution**, not just the expected burn rate. The same model can therefore support both exposure pricing and downstream capital analysis.

## 9. Extensions

The same pattern works with richer exposure schedules. MBBEFD parameters can vary by occupancy or construction class, and the policy transformation can include coinsurance or additional terms. More advanced simulations can replace the independent row-selection model with catastrophe footprints or other location dependence, while the analytical occurrence exposure rate remains a useful expected-loss benchmark.

## See also

- [Distributions Guide](distributions_guide.md) — distribution construction and simulation
- [Frequency-Severity Modelling](frequency_severity_modelling.md) — compound loss models and `FreqSevSims`
- [Pricing an Excess-of-Loss Reinsurance Program](xol_reinsurance.md) — XoL contracts, towers and aggregate terms
- [Risk Measures and Capital Allocation](risk_measures_and_allocation.md) — using annual simulations in a capital model
