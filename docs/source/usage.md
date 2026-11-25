# Usage Guide

This guide provides comprehensive examples of using the Proteus Actuarial Library (PAL).

## Creating Stochastic Variables

### Basic Stochastic Variables

A `StochasticScalar` is a vector of simulated values:

```python
from pal import StochasticScalar

# Create from array
svariable = StochasticScalar([1, 2, 3, 4])
```

### Statistical Distributions

Statistical distributions are imported directly from `pal.distributions`:

```python
from pal.distributions import Gamma, LogNormal

# Create gamma distribution
gamma_var = Gamma(alpha=2.5, theta=2).generate()

# Create log-normal distribution
lognormal_var = LogNormal(mu=1, sigma=0.5).generate()
```

For heavy-tailed claim severities, the generalized beta prime distribution
(also known as GB2) allows the shape of the main body and upper tail to vary:

```python
from pal.distributions import GeneralizedBetaPrime

claim_severities = GeneralizedBetaPrime(a=2.0, b=100000.0, p=1.5, q=3.0).generate(10000)
```

## Variable Containers

Variables can be grouped into containers with the `ProteusVariable` class:

```python
from pal import ProteusVariable
from pal.distributions import Gamma, LogNormal

# Create individual variables
motor_losses = Gamma(alpha=2.5, theta=2).generate()
property_losses = LogNormal(mu=1, sigma=0.5).generate()

# Group into container
portfolio = ProteusVariable(
    dim_name="line",
    values={"Motor": motor_losses, "Property": property_losses}
)
```

Variable containers support numpy operations and can be added, multiplied together etc. Operations involving multiple variable containers will attempt to match on dictionary labels.

### Data interchange with CSV, Parquet and pandas

For exchanging results with spreadsheets, analytics tools and other systems,
`ProteusVariable.to_dataframe()`, `to_csv()` and `to_parquet()`
produce ordinary tables without PAL-specific metadata. A one-dimensional
portfolio of simulation arrays uses a **wide** table by default, with one
row per simulation and one column per portfolio component:

```python
from pal import ProteusVariable, StochasticScalar

losses = ProteusVariable("class", {
    "Motor": StochasticScalar([100, 200, 150]),
    "Property": StochasticScalar([300, 250, 400]),
})

data = losses.to_dataframe()
assert list(data.columns) == ["Simulation", "Motor", "Property"]
recreated_from_data = ProteusVariable.from_dataframe(data, dim_name="class")
```

Use `layout="long"` for a conventional dimension / simulation / value
table, or for per-claim `FreqSevSims` data. Other programs can read these
files directly with pandas, Arrow, Excel or their own CSV readers.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from pal import ProteusVariable, StochasticScalar

losses = ProteusVariable("class", {
    "Motor": StochasticScalar([100, 200]),
    "Property": StochasticScalar([300, 250]),
})
with TemporaryDirectory() as directory:
    path = Path(directory) / "losses.csv"
    losses.to_csv(path, layout="long")
    imported = ProteusVariable.from_csv(path, "class", "value")
    assert imported["Motor"].n_sims == 2
```

Parquet is preferred for large numerical tables; install the optional
`parquet` extra to enable pandas Parquet import/export. For an ordinary
table, use `losses.to_parquet("losses.parquet")` and
`ProteusVariable.from_parquet("losses.parquet", dim_name="class")`.

**Data interchange is not PAL object persistence.** Coupling relationships,
some Python/NumPy type distinctions, and unobserved trailing zero-claim
simulations cannot in general be inferred from an ordinary table. For an
event listing, specify `n_sims` when importing if the final simulations
have no claims.

### Saving and restoring PAL objects

Use `pal.serialization.save()` and `load()` when the file is a snapshot
that PAL will read later. This preserves the nested variable hierarchy,
deterministic and stochastic leaf types, NumPy dtypes, complete simulation
counts, event ordering, and coupling relationships *among saved leaves*.
Coupling to variables outside the saved object is not retained.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from pal import ProteusVariable, StochasticScalar
from pal.serialization import load, save

portfolio = ProteusVariable("class", {
    "Motor": StochasticScalar([100, 200]),
    "Property": StochasticScalar([300, 250]),
})
with TemporaryDirectory() as directory:
    path = Path(directory) / "portfolio.csv"
    save(portfolio, path)
    restored = load(path)
    assert restored["Motor"].n_sims == 2
```

For large object snapshots, use a `.parquet` path instead. The versioned
snapshot format uses compact typed Arrow columns and PAL schema metadata;
it is intended for PAL-to-PAL storage rather than third-party data exchange.
CSV snapshots are available but much slower. These are distinct APIs and
their file formats should not be used interchangeably.

An existing ordinary scenario CSV remains importable with the familiar
`ProteusVariable.from_csv(path, dim_name, values_column)` call.

### Date and categorical simulations

Simulated values can represent claim dates, regions, policy types and other
labels. For a calendar date, `Uniform` samples each date from the first through
the last day with equal probability; `Categorical` samples named categories
with the specified probabilities.

```python
from datetime import date
from pal import ProteusVariable, StochasticScalar
from pal.distributions import Categorical, Uniform

claim_dates = Uniform(date(2026, 1, 1), date(2026, 12, 31)).generate(4)
regions = Categorical(["UK", "US"], [0.7, 0.3]).generate(4)

by_region = ProteusVariable("region", {
    "UK": StochasticScalar([100, 200, 300, 400]),
    "US": StochasticScalar([10, 20, 30, 40]),
})
selected = by_region[regions]  # One value for each simulated region
```

A `FreqSevSims` can also contain dates or category labels for individual
events. When such an object indexes a `ProteusVariable`, the result retains
its event indices. Numeric summaries such as `aggregate()` are intended for
numeric event values; `count()` works for any event type. Dates and strings
remain NumPy values when PAL runs with GPU acceleration.

## Copulas and Dependencies

Statistical dependencies between PAL variables can be modeled using copulas:

```python
from pal.copulas import GumbelCopula
from pal.distributions import Gamma, LogNormal

# Create independent variables
var1 = Gamma(alpha=2.5, theta=2).generate()
var2 = LogNormal(mu=1, sigma=0.5).generate()

# Apply copula to create dependency
GumbelCopula(theta=1.2).apply([var1, var2])
```

### Variable Coupling

PAL automatically tracks variables that have been used in formulas together (coupled variables):

```python
from pal.distributions import Gamma, LogNormal

# These variables become coupled
var1 = Gamma(alpha=2.5, theta=2).generate()
var2 = LogNormal(mu=1, sigma=0.5).generate()
var3 = var1 + var2  # var1, var2, and var3 are now coupled

# If a copula reorders var3, var1 and var2 are automatically reordered too
```

## Configuration

### Simulation Settings

Configure the global number of simulations:

```python
from pal import config, set_random_seed

# Change simulation count (default is 100,000)
config.n_sims = 1000000

# Set random seed for reproducibility
set_random_seed(123456)
```

PAL uses the `default_rng` class from `numpy.random`, which can also be configured via `config.rng`.

### GPU Acceleration

For CUDA-compatible GPUs, install PAL from PyPI with the GPU extra:

<!--pytest.mark.skip-->

```bash
pip install "proteusllp-actuarial-library[gpu]"
```

Enable GPU mode by setting the environment variable:

<!--pytest.mark.skip-->

```bash
# Linux
export PAL_USE_GPU=1

# Windows
set PAL_USE_GPU=1
```
