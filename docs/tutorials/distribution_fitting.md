# Fitting Probability Distributions

PAL estimates distribution parameters by **maximum likelihood estimation (MLE)**.
The result is a fitted PAL distribution, together with its estimated parameters
and likelihood diagnostics. You can use the distribution for probability
calculations and simulation just like one constructed with known parameters.

The examples use synthetic data. By default, **a fitted distribution uses
fixed maximum-likelihood parameter estimates**. PAL also offers an optional
joint asymptotic parameter-uncertainty approximation for common severity
families. This matters particularly when estimating rare, large losses.

## 1. Fit a Gamma distribution

Suppose we have claim severity data for a portfolio with positively skewed
losses. We can fit a Gamma distribution and compare its cumulative
distribution function (CDF) with the empirical CDF.

```python
import numpy as np
import plotly.graph_objects as go

from pal.distributions import Gamma
from pal.maths import asnumpy

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

### Optional starting values

Numerical fitting uses automatic data-dependent starting estimates. To
investigate a difficult fit, you can supply some or all of the estimated
parameters in their usual distribution units:

<!--pytest-codeblocks:cont-->

```python
custom_fit = Gamma.fit(losses, initial_parameters={"alpha": 2.0, "theta": 40_000.0})
assert custom_fit.converged
```

These values are starting points, not fixed parameters. Unspecified
parameters use the automatic estimates. Invalid or infeasible starting
values raise an error. The automatic start remains available for recovery
if the supplied start fails. Analytical estimators do not need initial
values. For model averaging, supply different initial parameters through
`per_family_options`.

### Including parameter uncertainty

The fitted Gamma shape and scale are estimated from the data, so they are
uncertain. PAL can estimate their joint uncertainty using a large-sample
approximation to the likelihood.

<!--pytest-codeblocks:cont-->

```python
fit_with_uncertainty = Gamma.fit(losses, uncertainty_method="asymptotic")

uncertainty = fit_with_uncertainty.uncertainty
assert uncertainty is not None
print("Parameter standard errors:", uncertainty.standard_errors)

# This returns an ordinary Gamma distribution with stochastic parameters.
uncertain_gamma = uncertainty.sample_distribution()
assert isinstance(uncertain_gamma, Gamma)
simulated_losses = uncertain_gamma.generate()
```

`uncertain_gamma` is an ordinary PAL `Gamma` instance whose
`alpha` and `theta` parameters are `StochasticScalar` objects.
PAL draws the two parameters **together**, retaining their estimated
correlation. They are **coupled** with the simulated losses, so every
claim within a given scenario can use the same parameter pair in a
frequency-severity model. PAL also keeps parameter draws within their
admissible bounds.

The point-estimate distribution remains available as
`fit_with_uncertainty.distribution`. Calling `Gamma.fit(losses)`
without an uncertainty method continues to use fixed MLE parameters.

The covariance-based approach uses a large-sample approximation.
Alternatives include parameter bootstrapping, Bayesian posterior
prediction, and predictive calibration. For examples, see Jewson and
Sweeting (2026),
[Including parameter uncertainty in loss distributions using calibrating priors](https://doi.org/10.1017/S1748499525100237),
and Norman,
[Exact Frequentist Prediction Intervals for a Gamma Distribution with Unknown Shape and Scale](https://github.com/ProteusLLP/proteus-research/blob/main/gamma_prediction/paper/gamma_prediction_interval_final.tex),
which derives finite-sample prediction intervals using an augmented
sufficient statistic. The latter constructs prediction limits directly
rather than simulating uncertain Gamma parameters.

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
    y=asnumpy(gamma_fit.distribution.cdf(cdf_grid)),
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

`asnumpy` converts GPU-backed results to ordinary NumPy arrays for Plotly;
on CPU it leaves NumPy arrays in a compatible form.

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
from pal.fitting import empirical_mean_excess

empirical_excess = empirical_mean_excess(reported_losses, attachments)

mean_excess_figure = go.Figure()
mean_excess_figure.add_scatter(
    x=attachments,
    y=empirical_excess,
    name="Empirical (uncapped synthetic losses)",
    mode="lines+markers",
)
for distribution_class, fit in candidate_fits.items():
    mean_excess_figure.add_scatter(
        x=attachments,
        y=asnumpy(fit.distribution.mean_excess(attachments)),
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

**Beta** can be used to model *recovery rates on defaulted corporate loans*,
measured as the fraction of outstanding principal recovered. These rates
lie between zero and one, making them a standard application of a bounded
continuous distribution. Its two shape parameters are fitted conditional on
the specified `loc` and `scale` (by default 0 and 1).

**Categorical** estimates the probability of each label. **Uniform** uses
the observed interval endpoints, and can fit calendar dates as well as
ordinary numeric values.

<!--pytest-codeblocks:cont-->

```python
import datetime as dt

from pal.distributions import Beta, Categorical, Uniform

# Synthetic realised recovery rates on defaulted corporate loans.
loan_recovery_rates = [0.12, 0.24, 0.31, 0.45, 0.19, 0.37, 0.56, 0.28]
beta_fit = Beta.fit(loan_recovery_rates, loc=0.0, scale=1.0)

claim_types = ["property", "motor", "property", "liability", "property"]
category_fit = Categorical.fit(claim_types)

event_dates = [dt.date(2026, 1, 3), dt.date(2026, 1, 8), dt.date(2026, 1, 12)]
date_fit = Uniform.fit(event_dates)

assert beta_fit.parameters["alpha"] > 0
assert np.isclose(sum(category_fit.parameters["probabilities"]), 1.0)
assert date_fit.distribution.invcdf(0.0) == np.datetime64("2026-01-03")
```

The ordinary Beta model has no probability mass at exactly zero or one.
If the portfolio includes loans with zero or full recovery, a mixture model
with point masses at those endpoints may be more appropriate.

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

## 7. Distribution model uncertainty with AICc weights

A fitted severity model can be uncertain in two different ways: its *parameters*
are estimated from a finite sample, and the *family* itself may be uncertain.
PAL represents the second uncertainty as a weighted mixture of candidate
distributions:

\[
F(x) = \sum_{j=1}^m w_j F_j(x), \qquad
w_j = \frac{\exp[-(\mathrm{AICc}_j-\min_k\mathrm{AICc}_k)/2]}
           {\sum_\ell \exp[-(\mathrm{AICc}_\ell-\min_k\mathrm{AICc}_k)/2]}.
\]

The following example estimates both sources of uncertainty in one operation.
It uses two families whose likelihoods are comparable on the same severity
observations.

```python
import numpy as np

from pal.distributions import Gamma, LogNormal, WeightedDistribution
from pal.fitting import UncertaintyMethod

sample = np.random.default_rng(73).gamma(2.4, 25000.0, 600)
model_fit = WeightedDistribution.fit(
    sample,
    [Gamma, LogNormal],
    uncertainty_method=UncertaintyMethod.ASYMPTOTIC,
)
print(model_fit.model_weights)          # Model family -> AICc weight
print(model_fit.aicc)                   # Model family -> AICc
print(model_fit.fits[0].parameters)     # Individual MLE diagnostics

# An ordinary model-averaged (marginal) CDF and quantile.
model_cdf = model_fit.distribution.cdf(100_000)
model_p99 = model_fit.distribution.invcdf(0.99)

# One model is chosen per simulation scenario; for that model its parameters
# are drawn jointly using the asymptotic covariance from the original fit.
scenario_model = model_fit.sample_distribution(n_sims=10_000)
first_loss = scenario_model.generate(n_sims=10_000)
second_loss = scenario_model.generate(n_sims=10_000)

assert first_loss.n_sims == second_loss.n_sims == 10_000
assert first_loss.coupled_variable_group is second_loss.coupled_variable_group
```

The scenario-specific model identity is **retained** across these two draws.
The parameter vector for that scenario is retained as well. This is important
when the modelling question is about uncertainty in the loss-generating model
for an entire portfolio, rather than a different choice of family for each
claim. For one-off predictive draws, using
`model_fit.distribution.generate()` instead draws a model independently
for each observation.

For the same fitted candidate models, you can override AICc weights while
preserving the individual MLE estimates and diagnostics:

```python
import numpy as np

from pal.distributions import Gamma, LogNormal, WeightedDistribution

sample = np.random.default_rng(73).gamma(2.4, 25_000.0, 600)
manual_fit = WeightedDistribution.fit(
    sample,
    [Gamma, LogNormal],
    model_weights={Gamma: 0.7, LogNormal: 0.3},
)
```

Here, `model_weights` refers to distribution-family weights;
`weights` in a family's fitting function continues to mean observation
frequency weights.

Manual weights are also possible when constructing a distribution directly:

```python
from pal.distributions import Gamma, LogNormal, WeightedDistribution

weighted = WeightedDistribution(
    [Gamma(alpha=2.0, theta=5000), LogNormal(mu=8.5, sigma=0.9)],
    weights=[0.7, 0.3],
)
```

PAL requires finite AICc values and matching likelihood conventions for all
candidates. AICc is undefined for insufficient effective sample size or
noninteger frequency weights. Families with incompatible probability measures
must not be compared by AICc. The candidate models are all fitted to the
*same* observations, censoring and truncation specification. Family-specific
options, such as a known Binomial trial count, can be supplied using
`per_family_options`.

AICc weights quantify relative support **within the selected candidate
family set**; they are not posterior probabilities or statistically calibrated
confidence levels for the true data-generating model. PAL currently treats
these weights as fixed. Joint asymptotic parameter uncertainty is conditional
on the chosen family, so this does not include uncertainty in estimated
model weights, structural misspecification shared by all candidates, or
predictive coverage corrections. Parameter fitting remains on CPU; subsequent
scenario sampling follows PAL's NumPy/CuPy and MC/RQMC configuration.

## Further reading

The [Distributions Guide](distributions_guide.md) lists all supported PAL
families and their parameters. [Frequency-Severity Modelling](frequency_severity_modelling.md)
explains how a fitted frequency and severity model can be combined, simulated
and subjected to reinsurance contracts.
