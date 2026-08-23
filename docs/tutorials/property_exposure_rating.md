# Property Reinsurance Exposure Rating with MBBEFD

This tutorial combines classical property exposure rating with occurrence-level
simulation. We read a schedule of property risks from CSV, infer each risk's
claim frequency from premium and expected loss ratio, use a weighted empirical
distribution to decide which exposure row produces each claim, draw the claim
severity from that row's MBBEFD distribution, and pass the resulting policy
losses through PAL's existing `XoLTower`.

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
cover, even where the contractual policy limit is lower.

```python
from pathlib import Path

import pandas as pd

exposures = pd.read_csv(Path("examples/data/property_exposures.csv"))
```

## 2. Infer claim frequency from premium and loss ratio

Let \(X_i\in[0,1]\) be the MBBEFD damage ratio for risk \(i\), conditional on a
ground-up claim, and let \(M_i\) be its maximum loss. Ground-up loss is

\[
L_i=M_iX_i.
\]

For policy deductible \(D_i\) and limit \(P_i\), policy loss is

\[
Y_i=\min\{(L_i-D_i)_+,P_i\}.
\]

The MBBEFD exposure curve is

\[
G_i(x)=\frac{E[\min(X_i,x)]}{E[X_i]}.
\]

Therefore

\[
E[Y_i\mid\text{claim}]
=M_iE[X_i]
\left[
G_i\!\left(\frac{D_i+P_i}{M_i}\right)
-G_i\!\left(\frac{D_i}{M_i}\right)
\right],
\]

with the curve arguments capped at one.

PAL exposes the MBBEFD mean directly on the distribution, so this conditional
severity is calculated analytically rather than by simulation.

The expected annual policy loss is

\[
\mu_i=\text{premium}_i\times\text{expected loss ratio}_i,
\]

which gives annual ground-up claim frequency

\[
\lambda_i=\frac{\mu_i}{E[Y_i\mid\text{claim}]}.
\]

The portfolio claim count is then

\[
N\sim\operatorname{Poisson}\left(\sum_i\lambda_i\right).
\]

## 3. Use `Empirical` to select the exposure rows with claims

Conditional on a portfolio claim occurring, risk \(i\) should be selected with
probability

\[
\Pr(I=i)=\frac{\lambda_i}{\sum_j\lambda_j}.
\]

This is exactly a weighted empirical distribution over exposure-row numbers.
There is no need for a custom severity class:

<!--pytest.mark.skip-->

```python
row_distribution = Empirical(
    samples=xp.arange(len(exposures)),
    weights=frequencies,
)

claim_rows = FrequencySeverityModel(
    distributions.Poisson(frequencies.sum()),
    row_distribution,
).generate(100_000)
```

`claim_rows` is a `FreqSevSims`. Its `sim_index` says which annual simulation
each claim belongs to, while its values are the exposure-row numbers which have
claims.

This is the only use of the empirical distribution. The claim severity itself is
**not** approximated by an empirical distribution.

## 4. Draw each severity from the selected row's MBBEFD distribution

The row numbers select the MBBEFD parameter and policy terms for every simulated
claim:

<!--pytest.mark.skip-->

```python
row_index = claim_rows.values.astype(int)

maximum_loss = xp.asarray(exposures["maximum_loss"].to_numpy())[row_index]
policy_limit = xp.asarray(exposures["policy_limit"].to_numpy())[row_index]
deductible = xp.asarray(exposures["policy_deductible"].to_numpy())[row_index]
c = xp.asarray(exposures["mbbefd_c"].to_numpy())[row_index]
```

`MBBEFD` is vectorised, so the selected \(c\) values can be passed directly to
`MBBEFD.from_c`. Each event therefore gets the correct row-specific distribution
without any custom class or intermediate stochastic-parameter wrapper:

<!--pytest.mark.skip-->

```python
damage_ratio = MBBEFD.from_c(c).generate(len(row_index))

policy_loss = np.minimum(
    np.maximum(damage_ratio * maximum_loss - deductible, 0.0),
    policy_limit,
)
```

We retain the original claim simulation indices and replace the row-number values
with policy losses:

<!--pytest.mark.skip-->

```python
policy_losses = FreqSevSims(
    claim_rows.sim_index,
    policy_loss.values,
    claim_rows.n_sims,
)
```

The resulting `FreqSevSims` is now an ordinary occurrence-level policy-loss
simulation and can be used by the rest of PAL without any property-specific
simulation class.

## 5. Apply the existing reinsurance tower

The policy losses can be passed directly to the normal PAL `XoLTower`:

<!--pytest.mark.skip-->

```python
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

## 6. Compare with the analytical exposure rate

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
Because the simulation draws directly from the MBBEFD distribution, any
difference is Monte Carlo error rather than a severity-discretisation error.

Run the complete example from the repository root:

```bash
python examples/example_property_exposure_rating.py
```

It reports the inferred frequency for each risk, the total portfolio Poisson
mean, and analytical versus simulated expected losses and rates for each layer.

## 7. Extensions

The same pattern works with richer exposure schedules. MBBEFD parameters can
vary by occupancy or construction class, and the policy transformation can
include coinsurance or additional terms. More advanced simulations can replace
the independent row-selection model with catastrophe footprints or other
location dependence, while the analytical exposure rate remains a useful
expected-loss benchmark.

## See also

- [Distributions Guide](distributions_guide.md) — distribution construction and simulation
- [Frequency-Severity Modelling](frequency_severity_modelling.md) — compound loss models and `FreqSevSims`
- [Pricing an Excess-of-Loss Reinsurance Program](xol_reinsurance.md) — XoL contracts, towers and reinstatements
- [Risk Measures and Capital Allocation](risk_measures_and_allocation.md) — downstream risk analysis of simulated losses
