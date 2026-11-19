# Fitting Probability Distributions

PAL can estimate the parameters of its probability distributions from observed
data. The fitted distribution is a normal PAL distribution, so it can be used
for probabilities, simulation and frequency-severity modelling.

This tutorial starts with a simple example, then covers censored claim data,
model comparison, frequency fitting and distributions with restricted support.
All examples use synthetic data.

## 1. Fit a normal distribution

Suppose we have observed a series of annual changes in a risk factor. The
normal maximum-likelihood estimates are the sample mean and the population
standard deviation (with divisor \(n\), rather than \(n-1\)).

```python
import numpy as np

from pal.distributions import Normal

changes = np.array([-1.4, 0.3, 1.8, -0.7, 0.5, 2.1, 0.8])
fitted = Normal.fit(changes)

print(fitted.parameters)
print(fitted.log_likelihood, fitted.aic, fitted.converged)

assert np.isclose(fitted.parameters["mu"], changes.mean())
assert np.isclose(fitted.parameters["sigma"], changes.std(ddof=0))
```

The result contains the estimated parameters, the PAL distribution,
log-likelihood, AIC, corrected AIC (AICc), sample size and convergence status.
An analytical fit, such as this normal example, does not require numerical
optimisation.

The fitted object supports PAL's usual probability and simulation methods:

<!--pytest-codeblocks:cont-->

```python
probability_below_zero = fitted.distribution.cdf(0.0)
ninety_ninth_percentile = fitted.distribution.invcdf(0.99)

assert 0 < probability_below_zero < 1
assert ninety_ninth_percentile > fitted.parameters["mu"]
```

## 2. Fit severity distributions with incomplete observations

Claim listings frequently contain **left truncation** and **right censoring**.

- With left truncation, only claims exceeding a reporting threshold are
  included. Fitting uses the likelihood conditional on exceeding that threshold.
- With right censoring, we know a claim exceeds a recorded amount, but not its
  ultimate value. A censored observation contributes a survival probability
  instead of a density.

In this synthetic example, claims below £60,000 are unreported and those above
£350,000 are recorded only as exceeding £350,000.

<!--pytest-codeblocks:cont-->

```python
from pal.distributions import Gamma, LogNormal

rng = np.random.default_rng(141)
ultimate_losses = rng.lognormal(mean=np.log(120_000), sigma=0.9, size=160)

reporting_threshold = 60_000.0
censoring_limit = 350_000.0

reported_losses = ultimate_losses[ultimate_losses > reporting_threshold]
censored = reported_losses > censoring_limit
observed_losses = np.minimum(reported_losses, censoring_limit)

lognormal_fit = LogNormal.fit(
    observed_losses,
    threshold=reporting_threshold,
    censored=censored,
)
gamma_fit = Gamma.fit(
    observed_losses,
    threshold=reporting_threshold,
    censored=censored,
)

assert np.isfinite(lognormal_fit.log_likelihood)
assert np.isfinite(gamma_fit.log_likelihood)
assert lognormal_fit.n_censored == int(censored.sum())
```

For censored records, pass the amount at which observation stops, **not** an
imputed ultimate claim. An attachment or reporting threshold is different from
a censoring limit: one selects which claims enter the dataset, the other limits
what we know about claims already selected.

Both fits use the same observations and censoring conventions, so we can
compare AIC. Lower values indicate a better trade-off between goodness of fit
and the number of fitted parameters. AIC is not an absolute test of fit.

<!--pytest-codeblocks:cont-->

```python
candidate_fits = {
    "LogNormal": lognormal_fit,
    "Gamma": gamma_fit,
}
preferred_name = min(candidate_fits, key=lambda name: candidate_fits[name].aic)
preferred_fit = candidate_fits[preferred_name]

print("Preferred by AIC:", preferred_name)
print("AIC:", preferred_fit.aic)
print("Estimated parameters:", preferred_fit.parameters)
```

Inspect tail probabilities, observed-versus-fitted quantiles, parameter
stability and sensitivity to the reporting threshold before accepting a
severity model. AIC alone is not a substitute for these checks.

## 3. Use the fitted severity for a reinsurance attachment

For a ground-up severity \(X\) and attachment \(d\), the theoretical
mean excess is

```{math}
e(d) = \mathbb{E}[X-d \mid X>d].
```

Multiplying this conditional mean by the probability of exceeding the
attachment gives the expected excess loss **per ground-up claim**:

```{math}
\mathbb{E}[(X-d)_+] = \Pr(X>d)\,e(d).
```

<!--pytest-codeblocks:cont-->

```python
attachment = 200_000.0
severity = preferred_fit.distribution

attachment_probability = 1.0 - severity.cdf(attachment)
mean_excess = severity.mean_excess(attachment)
expected_excess_per_claim = attachment_probability * mean_excess

assert 0 < attachment_probability < 1
assert mean_excess > 0
assert expected_excess_per_claim > 0
```

This is an unlimited excess calculation. For a finite reinsurance layer,
the expected layer payment also depends on its limit. If the fitted
distribution has an infinite first moment, its unlimited mean excess is
infinite. At or above a finite upper endpoint, mean excess is undefined.

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
For illustration we use the lognormal severity model from the previous
section; the choice of frequency model and severity model should be justified
separately for a real portfolio.

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

All fitting uses observed data on the CPU. The resulting distribution can
subsequently be simulated with PAL. Non-negative observation weights are
supported; integer frequency weights represent repeated data points.
For non-integer weights, PAL does not report the usual frequency-sample
interpretation of AICc.

## Further reading

The [Distributions Guide](distributions_guide.md) lists all supported PAL
families and their parameters. [Frequency-Severity Modelling](frequency_severity_modelling.md)
explains how a fitted frequency and severity model can be combined, simulated
and subjected to reinsurance contracts.
