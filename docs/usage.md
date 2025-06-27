# Usage Guide

This guide provides comprehensive examples of using the Proteus Actuarial Library (PAL).

## Creating Stochastic Variables

### Basic Stochastic Variables

Stochastic variables can be created with the `StochasticScalar` class:

```python
from pal import StochasticScalar

# Create from array
svariable = StochasticScalar([1, 2, 3, 4])
```

### Statistical Distributions

Statistical distributions are available in the distributions module:

```python
from pal import distributions

# Create gamma distribution
gamma_var = distributions.Gamma(alpha=2.5, beta=2).generate()

# Create log-normal distribution
lognormal_var = distributions.LogNormal(mu=1, sigma=0.5).generate()
```

## Variable Containers

Variables can be grouped into containers with the `ProteusVariable` class:

```python
from pal import ProteusVariable, distributions

# Create individual variables
motor_losses = distributions.Gamma(alpha=2.5, beta=2).generate()
property_losses = distributions.LogNormal(mu=1, sigma=0.5).generate()

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
from pal import copulas, distributions

# Create independent variables
var1 = distributions.Gamma(alpha=2.5, beta=2).generate()
var2 = distributions.LogNormal(mu=1, sigma=0.5).generate()

# Apply copula to create dependency
copulas.GumbelCopula(alpha=1.2, n=2).apply([var1, var2])
```

### Variable Coupling

PAL automatically tracks variables that have been used in formulas together (coupled variables):

```python
# These variables become coupled
var1 = distributions.Gamma(alpha=2.5, beta=2).generate()
var2 = distributions.LogNormal(mu=1, sigma=0.5).generate()
var3 = var1 + var2  # var1, var2, and var3 are now coupled

# If a copula reorders var3, var1 and var2 are automatically reordered too
```

## Configuration

### Simulation Settings

Configure the global number of simulations:

```python
from pal import config

# Change simulation count (default is 100,000)
config.n_sims = 1000000

# Set random seed for reproducibility
config.set_random_seed(123456)
```

PAL uses the `default_rng` class from `numpy.random`, which can also be configured via `config.rng`.

### GPU Acceleration

For CUDA-compatible GPUs, install GPU dependencies:

```bash
pdm install -G gpu
```

Enable GPU mode by setting the environment variable:

```bash
# Linux
export PAL_USE_GPU=1

# Windows  
set PAL_USE_GPU=1
```

Set to any other value to revert to CPU mode.

## Advanced Examples

For more complex examples including reinsurance modeling and catastrophe simulations, see the [examples directory](../examples/) in this repository.

## See Also

- [Development Guide](development.md) - Setting up the development environment
- [Main README](../README.md) - Project overview and quick start

## Performance Tips

1. **Use appropriate simulation counts** - Start with smaller counts for development
2. **Leverage GPU acceleration** for large simulations if available
3. **Consider memory usage** when working with very large portfolios
4. **Use vectorized operations** where possible for better performance