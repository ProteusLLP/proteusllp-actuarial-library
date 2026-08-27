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

## Available Distributions

### Severity (Continuous) Distributions

| Distribution | Parameters | Typical Use |
|-------------|------------|-------------|
| `LogNormal` | `mu`, `sigma` | Claim sizes |
| `Gamma` | `alpha`, `theta`, `loc=0` | Aggregate losses, waiting times |
| `Pareto` | `shape`, `scale` | Large/catastrophe losses |
| `GPD` | `shape`, `scale`, `loc` | Excess losses above a threshold |
| `Burr` | `power`, `shape`, `scale`, `loc` | Heavy-tailed loss distributions |
| `Weibull` | `shape`, `scale`, `loc=0` | Time-to-failure, survival analysis |
| `Normal` | `mu`, `sigma` | Symmetric risks, economic variables |
| `Beta` | `alpha`, `beta`, `scale=1`, `loc=0` | Loss ratios, probabilities |
| `Exponential` | `scale`, `loc=0` | Inter-arrival times, simple decay |
| `LogLogistic` | `shape`, `scale`, `loc=0` | Income distributions, survival |
| `Logistic` | `mu`, `sigma` | Growth models |
| `Uniform` | `a`, `b` | Equal-likelihood scenarios |
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

## See Also

- [Getting Started](getting_started.md) — first steps with PAL
- [Frequency-Severity Modelling](frequency_severity_modelling.md) —
  combining frequency and severity distributions
- [Coupling Groups, Copulas and Variable Reordering](coupling_groups_and_copulas.md) —
  adding dependencies between variables
