# PAL quick reference for coding assistants

This page is a compact map from common modelling intentions to PAL's public Python API. It is useful for coding assistants and for users who want a terse reminder of the main concepts.

## Install and import

```bash
pip install proteusllp-actuarial-library
```

```python
from pal import config, copulas, distributions, set_random_seed
```

The PyPI distribution is `proteusllp-actuarial-library`; the installed Python package is `pal`.

## Configure simulations

```python
config.n_sims = 10_000
set_random_seed(42)
```

`config.n_sims` is the default number of Monte Carlo simulations. Use `set_random_seed` when reproducibility matters.

## Generate a stochastic variable

```python
loss = distributions.LogNormal(mu=14, sigma=0.5).generate()
```

`generate()` returns a `StochasticScalar` for an ordinary univariate distribution. A `StochasticScalar` represents one simulated value per simulation.

Use PAL's methods for common simulation statistics:

```python
mean_loss = loss.mean()
standard_deviation = loss.std()
var_995 = loss.percentile(99.5)
```

Prefer `loss.percentile(99.5)` to reaching into `loss.values` and calling `numpy.percentile` unless the raw array is specifically required for another operation.

## Arithmetic and derived variables

Arithmetic is element-wise across simulations:

```python
expenses = 0.10 * loss
gross = loss + expenses
```

Derived variables remain coupled to their inputs. This matters because PAL can later reorder simulations to impose dependence while keeping related quantities aligned.

## Add dependence with a copula

Generate marginal variables first, then apply a copula:

```python
motor = distributions.LogNormal(mu=14, sigma=0.5).generate()
property_loss = distributions.LogNormal(mu=15, sigma=0.8).generate()

copula = copulas.GaussianCopula([[1.0, 0.5], [0.5, 1.0]])
copula.apply([motor, property_loss])

portfolio = motor + property_loss
```

A copula changes dependence by reordering simulations. It should not change either marginal distribution.

Do not manually reorder `.values` when PAL coupling groups should propagate the reordering to related variables.

## Frequency-severity models

For a random number of random-sized claims:

```python
from pal.frequency_severity import FrequencySeverityModel

model = FrequencySeverityModel(
    freq_dist=distributions.Poisson(mean=100),
    sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
)

events = model.generate()
aggregate_loss = events.aggregate()
```

Use the event-level result when claim-level information matters. Use `aggregate()` for one total loss per simulation.

## Plot a stochastic result

PAL plotting methods return Plotly figures rather than displaying them automatically:

```python
fig = aggregate_loss.cdf_plot("Aggregate Loss")
```

Call `fig.show()` only in an interactive environment.

## GPU execution

Install PAL with the GPU extra in a compatible CUDA environment:

```bash
pip install "proteusllp-actuarial-library[gpu]"
```

Set `PAL_USE_GPU=1` before importing/running PAL when GPU execution is required. Code using the public PAL API should normally be backend-independent.

Avoid converting PAL/CuPy data to NumPy merely to perform an operation that PAL already supports, because that can introduce expensive device-to-host transfers.

## How to discover an unfamiliar API

Prefer the public API and introspection before guessing a class or parameter name:

```python
import inspect

from pal import distributions

print(inspect.signature(distributions.Gamma))
help(distributions.Gamma)
```

The generated [API reference](api/modules.html) is the authoritative browsable catalogue of documented classes and methods.

Useful conceptual guides are:

- [Getting started](tutorials/getting_started.html)
- [Distributions guide](tutorials/distributions_guide.html)
- [Frequency-severity modelling](tutorials/frequency_severity_modelling.html)
- [Coupling groups and copulas](tutorials/coupling_groups_and_copulas.html)
- [XoL reinsurance](tutorials/xol_reinsurance.html)
- [Risk measures and capital allocation](tutorials/risk_measures_and_allocation.html)

## Common mistakes to avoid

- Do not assume similarly named distributions use the same parameterisation as SciPy or another library; inspect PAL's signature and docstring.
- Do not discard coupling relationships by extracting and rebuilding raw arrays without a reason.
- Do not assume two generated risks are dependent until a dependence structure has been applied; derived variables, however, remain aligned with their inputs.
- Do not write CPU-only NumPy conversions into code intended to work on PAL's GPU backend.
- Do not infer undocumented public APIs from internal helper names. Prefer documented imports and classes.
- For percentiles and other operations already exposed by `StochasticScalar`, prefer the PAL method because it expresses the modelling intention directly.
