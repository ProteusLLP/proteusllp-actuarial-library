# Property Reinsurance Exposure Rating with MBBEFD

This tutorial combines classical property exposure rating with a portfolio
frequency-severity simulation. We read a schedule of property risks from CSV,
use an MBBEFD damage distribution for each risk, infer claim frequency from
premium and expected loss ratio, combine the resulting policy severities in a
weighted empirical distribution, and pass simulated policy losses through PAL's
`XoLTower`.

The complete runnable example is
`examples/example_property_exposure_rating.py`, with sample data in
`examples/data/property_exposures.csv`.

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

Claim frequency is deliberately **not** an input. It is implied by premium,
loss ratio and the policy severity model.

`maximum_loss` is separate from the policy limit. It can therefore represent the
full property exposure, including business interruption or other additional
cover, even where the contractual limit is lower.

```python
from pathlib import Path

import pandas as pd

exposures = pd.read_csv(Path("examples/data/property_exposures.csv"))
```

## 2. From MBBEFD damage ratio to policy loss

Let \(X_i\in[0,1]\) be the MBBEFD damage ratio for risk \(i\), conditional on a
ground-up claim, and let \(M_i\) be its maximum loss. Ground-up loss is

\[
L_i=M_iX_i.
\]

For policy deductible \(D_i\) and limit \(P_i\), the policy loss is

\[
Y_i=\min\{(L_i-D_i)_+,P_i\}.
\]

For simulation we discretise this conditional policy-loss distribution using a
midpoint quantile grid. With \(K\) points,

\[
u_k=\frac{k-1/2}{K},\qquad k=1,\ldots,K,
\]

and

\[
y_{ik}=\min\left\{\left(M_iF_i^{-1}(u_k)-D_i\right)_+,P_i\right\},
\]

where \(F_i\) is the MBBEFD CDF for the risk.

Using midpoint quantiles rather than a preliminary random sample makes the
empirical severity deterministic and gives an accurate numerical representation
with relatively few points.

## 3. Infer each risk's claim frequency

The exposure data give the expected annual policy loss directly:

\[
\mu_i=\text{premium}_i\times\text{expected loss ratio}_i.
\]

The empirical conditional mean policy severity is

\[
\hat m_i=\frac{1}{K}\sum_{k=1}^K y_{ik}.
\]

We therefore infer annual ground-up claim frequency as

\[
\hat\lambda_i=\frac{\mu_i}{\hat m_i}.
\]

This is the important calibration step: **frequency is a consequence of premium,
loss ratio and severity, not another assumption in the exposure file.**

The complete example does this while it builds the empirical severity support:

<!--pytest.mark.skip-->

```python
for _, row in exposures.iterrows():
    policy_losses = policy_loss_samples(row, n_points)
    expected_annual_loss = row["subject_premium"] * row["expected_loss_ratio"]
    frequency = expected_annual_loss / policy_losses.mean()
```

## 4. Combine the risks with `Empirical`

The portfolio claim count has mean

\[
\lambda=\sum_i\hat\lambda_i.
\]

Conditional on a claim occurring, risk \(i\) should be selected with probability

\[
\frac{\hat\lambda_i}{\sum_j\hat\lambda_j}.
\]

We can represent this without a custom severity class. Put all of the
\(y_{ik}\) values into one `Empirical` distribution and give each point from risk
\(i\) weight

\[
\frac{\hat\lambda_i}{K}.
\]

`Empirical` normalises the weights internally, so the required risk-selection
probabilities follow automatically.

<!--pytest.mark.skip-->

```python
from pal import Empirical

severity = Empirical(
    samples=xp.concatenate(samples),
    weights=xp.concatenate(weights),
)
```

The runnable example uses PAL's active array backend for the concatenation so the
same code works in CPU and GPU mode without moving the severity support between
devices.

## 5. Build the frequency-severity model

The full portfolio model is now just a Poisson frequency and the weighted
empirical policy severity:

<!--pytest.mark.skip-->

```python
from pal import distributions
from pal.frequency_severity import FrequencySeverityModel

model = FrequencySeverityModel(
    distributions.Poisson(float(frequencies.sum())),
    severity,
)
policy_losses = model.generate(100_000)
```

`policy_losses` is a `FreqSevSims`: each value is an individual policy loss and
each value retains the simulation in which it occurred.

## 6. Apply the reinsurance tower

The policy losses can be passed directly to the normal PAL `XoLTower`:

<!--pytest.mark.skip-->

```python
from pal import XoLTower

tower = XoLTower(
    name=["5m xs 5m", "10m xs 10m", "20m xs 20m"],
    limit=[5_000_000, 10_000_000, 20_000_000],
    excess=[5_000_000, 10_000_000, 20_000_000],
    premium=[0.0, 0.0, 0.0],
)

tower.apply(policy_losses)
```

Because the tower receives policy losses, layer attachment and limit are
expressed relative to policy loss rather than underlying maximum loss.

## 7. Compare with the analytical exposure rate

The MBBEFD exposure curve is

\[
G(x)=\frac{E[\min(X,x)]}{E[X]}.
\]

For a reinsurance layer with excess \(A\) and limit \(U\), the corresponding
ground-up thresholds for a policy with deductible \(D\) and limit \(P\) are

\[
D+\min(A,P)
\quad\text{and}\quad
D+\min(A+U,P).
\]

The expected layer loss for a risk is its expected annual policy loss multiplied
by the ratio of the layer exposure-curve increment to the policy exposure-curve
increment. Summing over risks gives the classical exposure-rated expected layer
loss.

The pure exposure rate is

\[
\text{exposure rate}
=\frac{\text{expected reinsurance loss}}
{\text{total subject premium}}.
\]

The example reports this alongside the simulated mean recovery from `XoLTower`.
With a sufficiently fine empirical grid and enough annual simulations, the two
should agree closely.

Run the complete example from the repository root:

```bash
python examples/example_property_exposure_rating.py
```

It reports the inferred frequency for each risk, the total portfolio Poisson
mean, and analytical versus simulated expected losses and rates for each layer.

## 8. Extensions

The same pattern works with richer exposure files. MBBEFD parameters can vary by
occupancy or construction class, and the policy transformation can include
coinsurance or other terms before the losses are put into `Empirical`. More
advanced simulations can add catastrophe footprints or dependence, while the
classical exposure-rate calculation remains a useful expected-loss benchmark.

## See also

- [Distributions Guide](distributions_guide.md) — distribution construction and simulation
- [Frequency-Severity Modelling](frequency_severity_modelling.md) — compound loss models and `FreqSevSims`
- [Pricing an Excess-of-Loss Reinsurance Program](xol_reinsurance.md) — XoL contracts, towers and reinstatements
- [Risk Measures and Capital Allocation](risk_measures_and_allocation.md) — downstream risk analysis of simulated losses
