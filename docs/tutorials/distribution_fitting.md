# Fitting Probability Distributions

PAL estimates distribution parameters by **maximum likelihood estimation (MLE)**.
The result is a fitted PAL distribution, together with its estimated parameters
and likelihood diagnostics. You can use the distribution for probability
calculations and simulation just like one constructed with known parameters.

The examples use synthetic data. Importantly, **calculations using fitted
parameters treat those estimates as fixed**: they do not account for
parameter uncertainty. This matters particularly when estimating rare,
large losses.

## 1. Fit a Gamma distribution

Suppose we have claim severity data for a portfolio with positively skewed
losses. We can fit a Gamma distribution and compare its cumulative
distribution function (CDF) with the empirical CDF.

```python
import numpy as np
import plotly.graph_objects as go

from pal.distributions import Gamma

rng = np.random.default_rng(42)
losses = rng.gamma(shape=2.8, scale=35_000, size=500)

gamma_fit = Gamma.fit(losses)
print("Maximum-likelihood estimates:", gamma_fit.parameters)
print("Log-likelihood:", gamma_fit.log_likelihood)
print("AIC:", gamma_fit.aic)

assert gamma_fit.converged
assert gamma_fit.parameters["alpha"] > 0
assert gamma_fit.parameters["theta"] > 0
```

The Gamma parameters are \(\alpha\) (shape) and \(\theta\) (scale). The
fitting result also provides AICc, the observation count and the convergence
status. Here MLE uses a numerical optimisation; other distributions such as
an unconstrained Normal or a Poisson have analytical MLEs.

<!--pytest-codeblocks:cont-->

```python
sorted_losses = np.sort(losses)
cdf_grid = np.linspace(0, sorted_losses[-1], 250)

cdf_figure = go.Figure()
cdf_figure.add_scatter(
    x=sorted_losses,
    y=np.arange(1, len(losses) + 1) / len(losses),
    mode="lines",
    name="Empirical CDF",
    line_shape="hv",
)
cdf_figure.add_scatter(
    x=cdf_grid,
    y=gamma_fit.distribution.cdf(cdf_grid),
    mode="lines",
    name="Fitted Gamma CDF",
)
cdf_figure.update_layout(
    title="Claim sizes: empirical and fitted CDF",
    xaxis_title="Claim size (£)",
    yaxis_title="Cumulative probability",
)
cdf_figure
```

We can now calculate a tail probability or a percentile using the fitted
distribution:

<!--pytest-codeblocks:cont-->

```python
probability_above_200k = 1 - gamma_fit.distribution.cdf(200_000)
ninety_ninth_percentile = gamma_fit.distribution.invcdf(0.99)

assert 0 < probability_above_200k < 1
assert ninety_ninth_percentile > 0
```

## 2. Fit large losses with reporting thresholds and policy limits

A large-loss listing often differs from a sample of complete ground-up
claim sizes in two ways:

- **Left truncation:** claims below the reporting threshold are not included.
  The likelihood is conditional on a loss exceeding that threshold.
- **Right censoring:** a loss hits its policy limit, so we observe the payment
  up to that limit, but not the size of the underlying loss beyond it. The
  likelihood uses the probability of exceeding the limit.

The purpose of fitting censored losses is to estimate the **underlying
severity distribution before applying policy limits**. Capped payments
should not be treated as if they were complete ground-up losses.

We will generate a synthetic, heavy-tailed claim sample, select large
reported claims and apply a policy limit.

<!--pytest-codeblocks:cont-->

```python
from pal.distributions import GPD, LogNormal

rng = np.random.default_rng(141)
ultimate_losses = rng.lognormal(mean=np.log(90_000), sigma=1.0, size=2_500)

reporting_threshold = 75_000.0
policy_limit = 300_000.0

reported_losses = ultimate_losses[ultimate_losses > reporting_threshold]
censored = reported_losses >= policy_limit
observed_losses = np.minimum(reported_losses, policy_limit)

lognormal_fit = LogNormal.fit(
    observed_losses, threshold=reporting_threshold, censored=censored
)
gpd_fit = GPD.fit(
    observed_losses, threshold=reporting_threshold, censored=censored
)

assert lognormal_fit.n_censored == int(censored.sum())
assert gpd_fit.n_observations == len(reported_losses)
assert np.all(observed_losses <= policy_limit)
```

A censored observation records the policy limit, not an estimated ultimate
loss. PAL fits a conditional likelihood for the reporting threshold and uses
the right-censoring indicators for losses reaching the policy limit.

The fitted curves describe **pre-limit claim severity above the reporting
threshold**. They do not identify the distribution below that threshold.
A policy-limited payment distribution must be obtained by applying the
limit to the relevant ground-up severity model.

<!--pytest-codeblocks:cont-->

```python
candidate_fits = {
    LogNormal: lognormal_fit,
    GPD: gpd_fit,
}
preferred_class = min(candidate_fits, key=lambda cls: candidate_fits[cls].aic)
preferred_fit = candidate_fits[preferred_class]

for distribution_class, fit in candidate_fits.items():
    print(distribution_class.__name__, "AIC:", round(fit.aic, 1))
print("Lowest AIC:", preferred_class.__name__)
```

The AIC values are comparable here because each model uses the same
observations, reporting threshold and censoring information. AIC is useful
for comparing candidates but cannot establish whether the tail extrapolation
is reliable.

## 3. Examine the tail with a mean-excess plot

For a pre-limit claim severity \(X\), the theoretical mean excess above
attachment \(d\) is

```{math}
e(d) = \mathbb{E}[X-d \mid X>d].
```

An empirical mean-excess plot compares the average amount by which observed
claims exceed a range of thresholds. A roughly increasing linear trend is
consistent with a heavy-tailed Generalised Pareto model with positive shape,
although sampling variability becomes substantial at high thresholds.

In this *synthetic example*, we have the unobserved ultimate losses, so we
can include their empirical mean excess as a reference. **In real capped
claims data, this reference curve is not available**: simply calculating
mean excess from capped payments or deleting censored claims gives a biased
picture of the underlying tail.

<!--pytest-codeblocks:cont-->

```python
attachments = np.linspace(reporting_threshold, 380_000, 65)

# Synthetic-data diagnostic only: actual ultimate losses are not normally observed.
empirical_mean_excess = np.array([
    np.mean(reported_losses[reported_losses > d] - d)
    for d in attachments
])

mean_excess_figure = go.Figure()
mean_excess_figure.add_scatter(
    x=attachments,
    y=empirical_mean_excess,
    name="Empirical (uncapped synthetic losses)",
    mode="lines+markers",
)
for distribution_class, fit in candidate_fits.items():
    mean_excess_figure.add_scatter(
        x=attachments,
        y=fit.distribution.mean_excess(attachments),
        name=f"Fitted {distribution_class.__name__}",
        mode="lines",
    )

mean_excess_figure.add_vline(
    x=policy_limit, line_dash="dash",
    annotation_text="Policy limit",
)
mean_excess_figure.update_layout(
    title="Mean excess by attachment",
    xaxis_title="Attachment (£)",
    yaxis_title="Mean excess (£)",
)
mean_excess_figure
```

![Mean-excess plot with empirical uncapped synthetic losses and fitted LogNormal
and Generalised Pareto curves](images/mean_excess_large_losses.svg)

*Mean excess in the synthetic example. The vertical dashed line marks the
policy limit; fitted mean excess above it is an extrapolation.*

In practice, use uncensored data where available and techniques that
properly account for censoring. Above the policy limit the fitted mean
excess is model extrapolation, so its sensitivity to the assumed family
and sample size is important.

For a pre-limit loss \(X\), attachment \(d\), and fitted survival probability,
the expected unlimited excess loss **per claim** is

```{math}
\mathbb{E}[(X-d)_+] = \Pr(X>d)\,e(d).
```

<!--pytest-codeblocks:cont-->

```python
attachment = 200_000.0
severity = preferred_fit.distribution

attachment_probability = 1.0 - severity.cdf(attachment)
conditional_mean_excess = severity.mean_excess(attachment)
expected_excess_per_claim = attachment_probability * conditional_mean_excess

assert 0 < attachment_probability < 1
assert conditional_mean_excess > 0
assert expected_excess_per_claim > 0
```

This is an unlimited excess calculation. For a finite reinsurance layer,
the expected payment also depends on the layer limit. For a distribution
with an infinite first moment, unlimited mean excess is infinite.

These are **plug-in estimates based on the MLE parameters**. They do not
allow for parameter uncertainty, which can be material for sparse
large-loss data and high attachments.

## 4. Fit claim frequencies

Count distributions are fitted using probability *masses*, not continuous
densities. A Poisson mean has a simple analytical maximum-likelihood estimate.
For a binomial model, the number of trials must be specified.

<!--pytest-codeblocks:cont-->

```python
from pal.distributions import Binomial, Poisson

annual_claim_counts = np.array([1, 0, 3, 2, 1, 4, 2, 0, 2, 3])
poisson_fit = Poisson.fit(annual_claim_counts)

# Number of successes out of eight trials, observed across periods.
successes = np.array([1, 2, 0, 3, 4, 2, 1])
binomial_fit = Binomial.fit(successes, n=8)

assert np.isclose(poisson_fit.parameters["mean"], annual_claim_counts.mean())
assert np.isclose(binomial_fit.parameters["p"], successes.mean() / 8)
```

Negative binomial fitting estimates its count dispersion using a profile
likelihood. A finite dispersion estimate is not identifiable for data
at the Poisson boundary. Hypergeometric fitting requires a known population
size and number of draws, then estimates the integer number of successes
in the population.

The fitted count and severity distributions can be used in a compound model.
For illustration we use the fitted lognormal severity model from the large-loss
example. The frequency and severity assumptions should each be justified
against the exposure and observation basis of a real portfolio.

<!--pytest-codeblocks:cont-->

```python
from pal import FrequencySeverityModel, config, set_random_seed

config.n_sims = 256
set_random_seed(42)

model = FrequencySeverityModel(
    freq_dist=poisson_fit.distribution,
    sev_dist=lognormal_fit.distribution,
)
annual_loss = model.generate().aggregate()

assert annual_loss.mean() >= 0
```

## 5. Fit bounded or labelled distributions

**Beta** models proportions on a fixed support interval. Its two shape
parameters are fitted conditional on the chosen `loc` and `scale` (by default
the interval from 0 to 1). The interval must reflect the actual quantity
being modelled.

**Categorical** estimates the probability of each label. **Uniform** uses
the observed interval endpoints, and can fit calendar dates as well as
ordinary numeric values.

<!--pytest-codeblocks:cont-->

```python
import datetime as dt

from pal.distributions import Beta, Categorical, Uniform

loss_ratios = [0.35, 0.42, 0.53, 0.64, 0.48, 0.37, 0.59]
beta_fit = Beta.fit(loss_ratios, loc=0.0, scale=1.0)

claim_types = ["property", "motor", "property", "liability", "property"]
category_fit = Categorical.fit(claim_types)

event_dates = [dt.date(2026, 1, 3), dt.date(2026, 1, 8), dt.date(2026, 1, 12)]
date_fit = Uniform.fit(event_dates)

assert beta_fit.parameters["alpha"] > 0
assert np.isclose(sum(category_fit.parameters["probabilities"]), 1.0)
assert date_fit.distribution.invcdf(0.0) == np.datetime64("2026-01-03")
```

A categorical distribution has no numeric mean excess. Calendar dates
likewise do not have a monetary-loss interpretation.

## 6. Special cases and fitting decisions

Some distributions require extra care:

| Distribution | Fitting consideration |
|---|---|
| Generalised Pareto (GPD) | A freely estimated location cannot be identified from a left-truncated sample; use the fixed-threshold fit for exceedances |
| Pareto | The scale/support boundary affects which observed claims are admissible |
| MBBEFD | The likelihood has both a continuous density and a discrete point mass at a total loss; two-parameter fitting needs both types of observation |
| Beta | Fix the support interval before fitting its shape parameters |
| Binomial | Supply the number of trials (`n`) |
| Hypergeometric | Supply `population_size` and `n_draws` |
| Categorical | Optionally supply the full `values` list to preserve categories with no observed cases |

All fitting uses MLE on observed data on the CPU. The resulting distribution can
subsequently be simulated with PAL. Non-negative observation weights are
supported; integer frequency weights represent repeated data points.
For non-integer weights, PAL does not report the usual frequency-sample
interpretation of AICc.

## Further reading

The [Distributions Guide](distributions_guide.md) lists all supported PAL
families and their parameters. [Frequency-Severity Modelling](frequency_severity_modelling.md)
explains how a fitted frequency and severity model can be combined, simulated
and subjected to reinsurance contracts.
