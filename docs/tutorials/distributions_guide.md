# Distributions Guide

PAL provides a comprehensive set of statistical distributions for
actuarial modelling. This tutorial covers how to choose, parameterise
and use them.

## Setup

```python
import numpy as np

from pal import config, set_random_seed
from pal.distributions import Gamma, LogNormal, Poisson
from pal.multivariate_distributions import InverseWishart, MultivariateStudentsT

config.n_sims = 10_000
set_random_seed(42)
```

## Generating Samples

Every distribution has a `generate()` method that returns a
`StochasticScalar` — a vector of simulated values:

<!--pytest-codeblocks:cont-->

```python
loss = LogNormal(mu=10, sigma=1.5).generate()

loss.mean()                           # => 68,673
loss.std()                            # => 205,459
loss.percentile(99.5)                 # => 1,111,353
```

The number of samples is controlled by `config.n_sims` (default
100,000).

## Analytical Functions

Distributions also provide `cdf()` and `invcdf()` without needing to
generate samples:

<!--pytest-codeblocks:cont-->

```python
ln = LogNormal(mu=10, sigma=1.5)

ln.cdf(50_000)       # => 0.7076  (P(X ≤ 50,000))
ln.invcdf(0.5)       # => 22,026  (median)
ln.invcdf(0.995)     # => 1,049,416  (99.5th percentile)
```

These are useful for quick calculations, curve-fitting checks and
validating simulation results.

## Fitting claim severity distributions

Severity distributions can be estimated from observed claim sizes using
**maximum likelihood estimation (MLE)**. Predictions from the fitted parameters
treat the estimates as fixed and therefore do **not** account for parameter
uncertainty, which can be substantial for large-loss tails. Supported families include Pareto, GPD, lognormal, gamma,
Weibull, Burr, log-logistic, inverse Gaussian, paralogistic,
inverse Burr, inverse gamma, inverse Weibull (Fréchet), log-gamma,
exponential, inverse exponential, inverse paralogistic, generalized beta
prime (GB2), generalized inverse Gaussian and generalized extreme value (GEV).
Fitting returns a PAL distribution, estimated parameters and likelihood
diagnostics.

<!--pytest-codeblocks:cont-->

```python
import numpy as np

from pal.distributions import Pareto

# The last two losses hit the policy limit; their underlying sizes are unknown.
losses = np.array([12.0, 15.0, 21.0, 50.0, 50.0])
censored = np.array([False, False, False, True, True])

fitted = Pareto.fit(losses, threshold=10.0, censored=censored)
shape = fitted.parameters["shape"]
tail_probability = 1.0 - fitted.distribution.cdf(50.0)
assert shape > 0 and 0 <= tail_probability <= 1
```

The reporting threshold conditions the likelihood on losses above that
amount. Right censoring occurs when a claim **hits the policy limit** and its
underlying size beyond the limit is unknown. A censored observation
contributes the survival probability at that policy limit, rather than
a density. The fitted curve estimates the **ground-up loss severity before
policy limits are applied**, not the distribution of capped payments. Optional
`weights` support repeated observations or other non-negative observation
weights; zero-weight rows are ignored. The GPD location is fixed at the
threshold by default. Pass `free_location=True` to `GPD.fit(...)` to
estimate all three parameters from an untruncated sample. With left
truncation, GPD location and scale are not separately identifiable, so
PAL rejects a free location when a positive threshold is supplied.
Other supported shifted positive severity distributions are fitted with zero location. GEV estimates its location.

`fitted.log_likelihood`, `fitted.aic`, `fitted.aicc` and
`fitted.converged` provide initial model-fit diagnostics. AICc is available
for integer frequency weights with sufficient observations. Numerical fitting
runs on the CPU using SciPy; the resulting PAL distribution supports the usual
simulation and probability methods.

Theoretical mean excess is available on these severity distributions. For a
loss severity X and threshold d, this is E[X - d | X > d]. It is useful for
checking the fitted tail and the average severity of claims reaching a layer.

<!--pytest-codeblocks:cont-->

```python
mean_excess_at_50 = fitted.distribution.mean_excess(50.0)
assert mean_excess_at_50 > 0
```

For shape parameters with an infinite first moment, mean excess is infinite.
It is undefined at and beyond a finite upper endpoint. The calculation uses
analytical expressions where available and numerical survival integration for
the remaining distributions.

For worked Gamma fitting, large-loss model comparison and a
**mean-excess plot**, see [Fitting Probability Distributions](distribution_fitting.md).
An empirical mean-excess curve calculated from capped losses is biased as a
diagnostic for the underlying severity; censored-data methods or the
theoretical mean excess from a censoring-aware fit are needed.

## Available Distributions

### Severity (Continuous) Distributions

| Distribution | Parameters | Typical Use |
|-------------|------------|-------------|
| `LogNormal` | `mu`, `sigma` | Claim sizes |
| `LogGamma` | `alpha`, `beta` | Claims with a log-gamma tail |
| `Gamma` | `alpha`, `theta`, `loc=0` | Aggregate losses, waiting times |
| `Pareto` | `shape`, `scale` | Large/catastrophe losses |
| `GPD` | `shape`, `scale`, `loc` | Excess losses above a threshold |
| `Burr` | `power`, `shape`, `scale`, `loc` | Heavy-tailed loss distributions |
| `Weibull` | `shape`, `scale`, `loc=0` | Time-to-failure, survival analysis |
| `Normal` | `mu`, `sigma` | Symmetric risks, economic variables |
| `Beta` | `alpha`, `beta`, `scale=1`, `loc=0` | Credit recovery rates and other proportions |
| `Exponential` | `scale`, `loc=0` | Inter-arrival times, simple decay |
| `LogLogistic` | `shape`, `scale`, `loc=0` | Income distributions, survival |
| `Logistic` | `mu`, `sigma` | Growth models |
| `Uniform` | `a`, `b` | Equal-likelihood scenarios |
| `Uniform` | `a`, `b` | Numeric values or dates sampled uniformly between bounds |
| `Categorical` | `values`, `probabilities` | Regions, claim types and other categories |
| `InverseGamma` | `alpha`, `theta`, `loc=0` | Bayesian priors |
| `GeneralizedInverseGaussian` | `p`, `chi`, `psi`, `loc=0` | Flexible positive, right-skewed risks |
| `MBBEFD` | `g`, `b` | Property exposure rating and capped loss-to-value ratios |
| `NonCentralChiSquared` | `df`, `nonc` | Quadratic forms and sums of squared non-zero-mean normal variables |
| `Paralogistic` | `shape`, `scale`, `loc=0` | Heavy-tailed alternatives |
| `InverseBurr` | `power`, `shape`, `scale`, `loc` | Flexible heavy tails |
| `InverseParalogistic` | `shape`, `scale`, `loc=0` | Heavy-tailed alternatives |
| `InverseWeibull` | `shape`, `scale`, `loc=0` | Extreme value modelling |
| `InverseExponential` | `scale`, `loc=0` | Extreme value modelling |

### Frequency (Discrete) Distributions

| Distribution | Parameters | Typical Use |
|-------------|------------|-------------|
| `Poisson` | `mean` | Claim counts (fixed exposure) |
| `NegBinomial` | `n`, `p` | Over-dispersed claim counts |
| `Binomial` | `n`, `p` | Events out of fixed trials |
| `HyperGeometric` | `ngood`, `nbad`, `population_size` | Sampling without replacement |

### Multivariate Distributions

| Distribution | Parameters | Typical Use |
|-------------|------------|-------------|
| `MultivariateNormal` | `mean`, `covariance` | Correlated symmetric risk factors |
| `MultivariateStudentsT` | `nu`, `mean`, `scale` | Correlated heavy-tailed risk factors |
| `Multinomial` | `n`, `p` | Allocate a fixed count across categories |
| `Dirichlet` | `alpha` | Random proportions with a fixed total |
| `InvertedDirichlet` | `alpha` | Positively dependent ratios and positive vectors |
| `GeneralizedDirichlet` | `alpha`, `beta` | Proportions with a richer covariance structure |
| `InvertedGeneralizedDirichlet` | `alpha`, `beta` | Positive vectors with a richer covariance structure |
| `Wishart` | `df`, `scale` | Random covariance or precision matrices |
| `InverseWishart` | `df`, `scale` | Uncertain covariance matrices |

Multivariate samples contain one `StochasticScalar` per named component:

<!--pytest-codeblocks:cont-->

```python
economic_factors = MultivariateStudentsT(
    nu=6,
    mean=[0.02, 0.04],
    scale=[[0.0004, 0.0001], [0.0001, 0.0025]],
    component_names=["inflation", "equity_return"],
).generate()

economic_factors["inflation"].mean()
economic_factors["equity_return"].std()
```

The generalized Dirichlet returns the final unallocated remainder as an
explicit component, so all generated components sum to one. The inverted
Dirichlet uses the final `alpha` value as the common denominator parameter and
therefore returns one fewer component than the number of parameters.

Wishart and inverse Wishart samples use nested named dimensions. The outer
variable contains matrix rows and each row contains the corresponding columns:

<!--pytest-codeblocks:cont-->

```python
covariance = InverseWishart(
    df=10,
    scale=[[0.04, 0.01], [0.01, 0.09]],
    component_names=["property", "casualty"],
).generate()

covariance["property"]["casualty"].mean()
```

## Comparing Severity Distributions

The choice of severity distribution significantly affects tail
behaviour. Here are several distributions simulated with similar
central tendency but very different tails:

```
Distribution                               Mean          Std        99.5th
------------------------------------------------------------------------
LogNormal(mu=10, sigma=1.5)              68,673      205,459     1,111,353
Gamma(alpha=5, theta=1000)                5,020        2,247        12,551
Pareto(shape=2, scale=10000)             20,107       32,032       149,509
GPD(shape=0.5, scale=1000, loc=0)         2,021        6,406        27,902
Weibull(shape=1.5, scale=1000)              898          615         3,082
```

**Key observations:**

- **LogNormal** has the heaviest tail — the 99.5th percentile is 16×
  the mean. Suitable for large-loss classes where extreme events
  dominate.
- **Pareto** also has a heavy tail (99.5th is 7.4× the mean) but its
  minimum value is bounded by the scale parameter.
- **GPD** is the natural choice for modelling excesses above a
  threshold (peaks-over-threshold approach).
- **Gamma** is lighter-tailed (99.5th is only 2.5× the mean) and
  suited for aggregate losses or attritional classes.
- **Weibull** is even lighter — useful for modelling time-to-failure
  or operational risks.

## Comparing Frequency Distributions

```
Distribution                             Mean     Std      Max
--------------------------------------------------------------
Poisson(mean=5)                           5.0     2.3       15
Poisson(mean=50)                         50.0     7.0       75
NegBinomial(n=5, p=0.5)                   5.0     3.2       22
Binomial(n=100, p=0.1)                   10.0     3.0       24
```

- **Poisson** — variance equals the mean. Standard choice when claims
  arrive independently at a constant rate.
- **Negative Binomial** — variance exceeds the mean
  (over-dispersed). Use when there is parameter uncertainty or
  heterogeneity in the claim arrival rate.
- **Binomial** — bounded count (0 to n). Use when there is a fixed
  number of exposures and each can generate at most one claim.

## Choosing a Severity Distribution

A practical decision tree:

1. **Do you have data above a threshold?** → `GPD` (peaks over
   threshold)
2. **Is the tail very heavy (power-law)?** → `Pareto` or `Burr`
3. **Is the distribution right-skewed with moderate tail?** →
   `LogNormal` or `Gamma`
4. **Is it symmetric?** → `Normal` or `Logistic`
5. **Is it bounded between 0 and 1?** → `Beta`
6. **Modelling time or duration?** → `Weibull` or `Exponential`

## Stochastic Parameters

Distribution parameters can themselves be stochastic. Pass a
`StochasticScalar` as a parameter to create a **mixed distribution**:

<!--pytest-codeblocks:cont-->

```python
set_random_seed(42)

# Uncertain claim rate: mean is itself random
uncertain_rate = Gamma(alpha=25, theta=2).generate()
claims = Poisson(mean=uncertain_rate).generate()
```

This produces over-dispersed counts because the Poisson mean varies
across simulations, adding an extra layer of variability (this is
equivalent to a Negative Binomial in the Poisson-Gamma case).

## Working with Generated Variables

`StochasticScalar` objects support standard arithmetic and numpy
operations:

<!--pytest.mark.skip-->

```python
loss = LogNormal(mu=14, sigma=0.5).generate()

# Arithmetic
with_expenses = loss * 1.10
capped = np.minimum(loss, 5_000_000)

# Statistics
loss.mean()
loss.std()
loss.percentile([25, 50, 75, 95, 99, 99.5])

# Visualisation of generated simulations
cdf_fig = loss.cdf_plot("Loss Distribution")
cdf_fig.show()

histogram_fig = loss.histogram_plot("Loss Distribution")
histogram_fig.show()
```

```{only} html
![Empirical CDF of the simulated loss](../_static/generated/distributions_guide_cdf.svg)

![Histogram of the simulated loss](../_static/generated/distributions_guide_histogram.svg)
```

The naming deliberately distinguishes analytical distribution methods such as
`ln.cdf(x)`, which calculate probabilities, from simulation plotting methods
such as `loss.cdf_plot()`, which return Plotly figures.

## Fitting the other PAL distributions

Fitting is available for every concrete PAL distribution class. Count and
discrete distributions use probability mass likelihoods, rather than the
continuous-severity likelihood. Poisson, binomial (with a known number of
trials), Bernoulli and unconstrained Normal have analytical estimators.
Negative binomial, noncentral chi-squared, Student's t, logistic and
other families use deterministic likelihood optimisation.

The following examples show the different interfaces:

```python
from pal.distributions import Binomial, Categorical, Normal, Poisson

poisson_fit = Poisson.fit([0, 1, 0, 3, 2, 1])
binomial_fit = Binomial.fit([1, 3, 2, 0, 4], n=8)
normal_fit = Normal.fit([-1.5, 0.2, 1.0, 1.8, 3.1])
category_fit = Categorical.fit(["motor", "property", "motor"])

assert poisson_fit.parameters["mean"] > 0
assert 0 < binomial_fit.parameters["p"] < 1
assert normal_fit.parameters["sigma"] > 0
assert category_fit.parameters["probabilities"] == [2 / 3, 1 / 3]
```

For binomial fitting, the trial count `n` must be known. Hypergeometric
fitting requires `population_size` and `n_draws`, and estimates the integer
number of successes in the population. An unconstrained negative binomial
has no finite size estimate at the Poisson boundary, so PAL rejects
samples that do not exhibit overdispersion.

`Beta.fit(..., loc=..., scale=...)` estimates both shape parameters
conditional on a specified support interval, which defaults to (0, 1).
Normal, logistic, Student's t, noncentral chi-squared and Beta support
right censoring and left truncation. The other non-severity families
currently fit complete observations.

Uniform distribution fitting uses the observed numeric minimum and maximum
as the maximum-likelihood interval endpoints. Date-valued Uniform also
supports an inclusive calendar-date range. Categorical fitting preserves
observed label order; optionally supply `values=[...]` to fix the category
set and retain unobserved categories with zero estimated probability.
The fitted result holds category labels and probabilities, or date endpoints,
alongside its PAL distribution.

MBBEFD fitting accounts for the discrete probability of a total policy
loss at exactly one as well as its continuous severity density. Estimating
both parameters requires interior observations and at least one total loss.

Numeric distributions provide a theoretical `mean_excess(attachment)`
method, including the count distributions. This is undefined for categorical
labels or temporal Uniform distributions, where excess has no established
numeric loss interpretation.

## See Also

- [Getting Started](getting_started.md) — first steps with PAL
- [Frequency-Severity Modelling](frequency_severity_modelling.md) —
  combining frequency and severity distributions
- [Coupling Groups, Copulas and Variable Reordering](coupling_groups_and_copulas.md) —
  adding dependencies between variables
