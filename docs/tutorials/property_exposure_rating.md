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

Equivalently, this first grosses the expected policy loss back up through the
exposure curve to obtain the expected ground-up loss and then divides by the
ground-up mean severity \(M_iE[X_i]\). This is the usual exposure-rating
assumption that the expected loss ratio applies consistently before and after
policy terms.

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
    samples=np.arange(len(exposures)),
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

The sampled row numbers are themselves claim-level stochastic values. PAL's
`StochasticScalar` supports indexing one stochastic vector with another, so the
row selection can be used directly to pick the policy terms and MBBEFD parameter:

<!--pytest.mark.skip-->

```python
row_index = StochasticScalar(claim_rows.values)

maximum_loss = StochasticScalar(exposures["maximum_loss"])[row_index]
policy_limit = StochasticScalar(exposures["policy_limit"])[row_index]
deductible = StochasticScalar(exposures["policy_deductible"])[row_index]
c = StochasticScalar(exposures["mbbefd_c"])[row_index]
```

`StochasticScalar` accepts one-dimensional array-like inputs such as pandas
Series directly. It takes care of the active CPU or GPU backend and preserves the
coupling created by the sampled row index. The selected \(c\) values can therefore
be passed directly to `MBBEFD.from_c`, giving every event its row-specific damage
ratio distribution:

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

## 6. Check the portfolio calibration

Before looking at the reinsurance, it is useful to verify that the simulation
reproduces the assumptions used to calibrate it. The target annual policy loss is

\[
\sum_i \text{premium}_i\times\text{ELR}_i,
\]

and the simulated annual policy loss is simply the mean of
`policy_losses.aggregate()`.

The example produces a compact calibration table:

| Metric | Target | Simulation |
|---|---:|---:|
| Total subject premium | £2,150,000 | £2,150,000 |
| Annual ground-up claim count | 1.882649 | 1.883590 |
| Annual policy loss | £1,422,100 | £1,430,459 |
| Portfolio loss ratio | 66.144% | 66.533% |

These simulation figures use 100,000 simulations and seed 42 on the CPU. The
small difference in loss ratio is Monte Carlo error; a GPU run uses a different
random-number generator and will not reproduce the same sample exactly.

The runnable example constructs the table with:

<!--pytest.mark.skip-->

```python
calibration = make_calibration_table(
    exposures,
    frequencies,
    claim_rows,
    policy_losses,
)
```

This is an important check because it validates the whole chain at once:
frequency inference, weighted risk selection, MBBEFD severity simulation and
application of policy terms.

## 7. Inspect the portfolio severity distribution

The policy severity distribution is a mixture across all the risks in the
schedule. Each risk contributes according to its ground-up claim frequency, and
each selected risk has its own maximum loss, deductible, limit and MBBEFD curve.

For pricing purposes it is usually most informative to look at **paid claims**,
so the example removes the zero policy losses generated by ground-up claims that
fall below the deductible. It then shows a Plotly histogram alongside the
exceedance curve:

<!--pytest.mark.skip-->

```python
severity_figure = make_claim_severity_figure(policy_losses)
severity_figure.show()
```

<div id="property-claim-severity" style="width: 100%; height: 480px;"></div>
<script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>
<script src="../_static/js/property_exposure_plots.js"></script>

The histogram is displayed through the 99.5th percentile so that a handful of
very large claims do not compress the body of the distribution. The exceedance
curve retains the simulated tail through the 99.9th percentile.

The resulting distribution is continuous between the contractual features, with
visible concentrations created by the different policy limits. This is a useful
diagnostic that the simulation is genuinely drawing MBBEFD severities rather
than resampling a discretised severity table.

## 8. Look at annual gross burn rate

For exposure pricing, absolute aggregate loss is often less informative than
the annual burn rate on the subject premium. Define

\[
B_{\mathrm{gross}}
=
\frac{\text{annual policy loss}}
{\text{total subject premium}}.
\]

Because `FreqSevSims.aggregate()` returns one value per simulation year, this is
a one-line PAL calculation:

<!--pytest.mark.skip-->

```python
gross_burn_rate = policy_losses.aggregate() / total_subject_premium
```

The example plots both the histogram and exceedance curve:

<!--pytest.mark.skip-->

```python
gross_burn_figure = make_gross_burn_rate_figure(
    policy_losses,
    total_subject_premium,
)
gross_burn_figure.show()
```

<div id="property-gross-burn" style="width: 100%; height: 480px;"></div>

The mean of this distribution is the simulated portfolio loss ratio. The full
distribution additionally shows the year-to-year volatility around that pricing
assumption. As with the severity plot, the histogram is limited to the 99.5th
percentile for readability while the exceedance curve retains the tail.

## 9. Layer burn-rate distributions

The same idea is particularly useful for the reinsurance layers. For layer
\(k\), define its annual burn rate on subject premium as

\[
B_k
=
\frac{\text{annual ceded loss to layer }k}
{\text{total subject premium}}.
\]

The example obtains the annual recoveries from the existing `XoL` objects and
plots their exceedance curves together:

<!--pytest.mark.skip-->

```python
layer_burn_rates = make_layer_burn_rates(
    tower,
    policy_losses,
    total_subject_premium,
)

layer_burn_figure = make_layer_burn_rate_figure(layer_burn_rates)
layer_burn_figure.show()
```

<div id="property-layer-burn" style="width: 100%; height: 480px;"></div>

This gives more information than the mean exposure rate alone. A high-attaching
layer can have a modest mean burn rate while still showing a highly skewed
annual distribution with many zero years and occasional large burns.

## 10. Compare with the analytical exposure rate

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

For the sample schedule, the analytical and simulated results are:

| Layer | Analytical expected loss | Simulated expected loss | Analytical rate | Simulated mean burn rate |
|---|---:|---:|---:|---:|
| 5m xs 5m | £250,159 | £252,463 | 11.635% | 11.742% |
| 10m xs 10m | £264,272 | £265,628 | 12.292% | 12.355% |
| 20m xs 20m | £218,333 | £221,561 | 10.155% | 10.305% |

The example also creates a grouped Plotly bar chart so that the analytical
exposure rates and simulated mean burn rates can be compared visually:

<!--pytest.mark.skip-->

```python
comparison = make_layer_comparison(
    exposures,
    tower,
    total_subject_premium,
)

comparison_figure = make_layer_rate_comparison_figure(comparison)
comparison_figure.show()
```

<div id="property-layer-rate" style="width: 100%; height: 440px;"></div>

Because the simulation draws directly from the same MBBEFD distributions used
for the analytical calculation, the two approaches are estimating the same
expected loss. Their remaining differences are simulation error rather than
severity discretisation error.

Run the complete example from the repository root:

<!--pytest.mark.skip-->

```bash
python examples/example_property_exposure_rating.py
```

It prints the inferred frequencies, the calibration and layer-comparison tables,
then opens four Proteus-styled Plotly diagnostics:

1. paid-claim portfolio severity;
2. annual gross portfolio burn rate;
3. annual layer burn-rate exceedance curves; and
4. analytical exposure rate versus simulated mean burn rate.

## 11. Extensions

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
