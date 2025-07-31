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

### Environment Variables

PAL supports several environment variables for configuration:

#### GPU Acceleration

For CUDA-compatible GPUs, install GPU dependencies:

```bash
pdm install -G gpu
```

Enable GPU mode:

```bash
# Linux/macOS
export PAL_USE_GPU=1

# Windows  
set PAL_USE_GPU=1
```

Set to any other value to revert to CPU mode.

#### Plotting Control

When running in headless/CLI environments (without display), suppress plot output:

```bash
# Linux/macOS
export PAL_SUPPRESS_PLOTS=true

# Windows
set PAL_SUPPRESS_PLOTS=true
```

This prevents `show_cdf()` and other plotting methods from attempting to display plots in browsers or GUI windows.

## Interactive Examples (Jupyter Notebooks)

PAL includes interactive Jupyter notebooks that demonstrate key features with live plots and step-by-step explanations.

### Using Notebooks in VS Code

All development work is done inside the devcontainer. VS Code provides native Jupyter support through the `ms-toolsai.jupyter` extension (pre-configured).

#### Step-by-Step Instructions

1. **Open the project in VS Code** with the Dev Containers extension installed
2. **Reopen in Container** when prompted, or use Command Palette (Ctrl+Shift+P): "Dev Containers: Reopen in Container"
3. **Wait for container setup** - first time takes a few minutes to build and install dependencies
4. **Open a notebook** from the VS Code Explorer: `examples/example_catastrophes.ipynb`
5. **Select Kernel** when prompted:
   - Click "Select Kernel" in the top-right of the notebook
   - Choose "Python Environments"
   - Select the PDM environment (should show `/workspace/.venv/bin/python`)
6. **Run cells** using:
   - **Ctrl+Enter** - Run current cell
   - **Shift+Enter** - Run current cell and move to next
   - **▶ Play button** in each cell
   - **Run All** button in the toolbar

#### Troubleshooting

- **"Kernel not found"**: Ensure you've selected the correct Python interpreter (`/workspace/.venv/bin/python`)
- **Import errors**: Make sure the container finished building and installed all dependencies
- **Plots not showing**: They should display inline automatically - no additional setup needed

### Available Notebooks

- **`example_catastrophes.ipynb`** - Catastrophe modeling with reinsurance recoveries
- More notebooks coming soon...

### Features

- **Native VS Code integration** - No separate Jupyter server needed
- **Live plots** displayed inline within VS Code
- **Interactive debugging** - Full VS Code debugging support in notebooks
- **Integrated development** - IntelliSense, linting, and formatting work seamlessly
- **GitHub rendering** - notebooks display with plots when viewed on GitHub

## Advanced Examples

For additional Python scripts, see the [examples directory](../examples/) in this repository.

## See Also

- [Development Guide](development.md) - Setting up the development environment
- [Main README](../README.md) - Project overview and quick start

## Performance Tips

1. **Use appropriate simulation counts** - Start with smaller counts for development
2. **Leverage GPU acceleration** for large simulations if available
3. **Consider memory usage** when working with very large portfolios
4. **Use vectorized operations** where possible for better performance