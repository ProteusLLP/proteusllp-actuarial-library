# Usage Guide

This guide provides comprehensive examples of using the Proteus Actuarial Library (PAL).

## Creating Stochastic Variables

### Basic Stochastic Variables

A `StochasticScalar` represents one simulated scalar value in each Monte Carlo simulation. It is the basic simulation-level variable in PAL: arithmetic operates scenario by scenario, and methods such as `mean()`, `std()` and `percentile()` summarise the simulated distribution.

```python
from pal import StochasticScalar

# Create from array
svariable = StochasticScalar([1, 2, 3, 4])
```

In normal modelling code you will usually obtain a `StochasticScalar` by generating simulations from a distribution rather than constructing one directly.

### Statistical Distributions

Statistical distributions are imported directly from `pal.distributions`:

```python
from pal.distributions import Gamma, LogNormal

# Create gamma distribution
gamma_var = Gamma(alpha=2.5, theta=2).generate()

# Create log-normal distribution
lognormal_var = LogNormal(mu=1, sigma=0.5).generate()
```

## Variable Containers

A `ProteusVariable` groups related values under a named dimension. It is useful for structures such as lines of business, regions or risk types, and PAL aligns labelled entries when containers are combined in calculations.

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
