<div style="display: flex; align-items: center; gap: 15px;">
  <img src="https://raw.githubusercontent.com/ProteusLLP/proteusllp-actuarial-library/main/PAL.svg" alt="PAL Logo" width="80"/>
  <div>
    <h1 style="margin: 0;">Proteus Actuarial Library</h1>
    <p style="margin: 5px 0 0 0;">
      <a href="https://proteusllp-actuarial-library.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/proteusllp-actuarial-library/badge/?version=latest" alt="Documentation Status"/>
      </a>
    </p>
  </div>
</div>

<br/>

An actuarial stochastic modeling library in python.

**Note**
This library is still in beta!

📚 **[Development Guide](docs/development.md)** - Get started with development setup and testing

## Introduction

The Proteus Actuarial Library (PAL) is a fast, lightweight framework for building simulation-based actuarial and financial models. It handles complex statistical dependencies using copulas while providing simple, intuitive syntax.

**Key Features:**
- Built on NumPy/SciPy for performance
- Optional GPU acceleration with CuPy
- Automatic dependency tracking between variables
- Comprehensive statistical distributions
- Clean, Pythonic API

## Quick Start

```python
from pal import distributions, copulas

# Create stochastic variables
losses = distributions.Gamma(alpha=2.5, theta=2).generate()
expenses = distributions.LogNormal(mu=1, sigma=0.5).generate()

# Apply statistical dependencies
copulas.GumbelCopula(theta=1.2, n=2).apply([losses, expenses])

# Variables are now correlated
total = losses + expenses
```

## Installation

<!--pytest.mark.skip-->

```bash
# Basic installation
pip install proteus-actuarial-library

# With GPU support
pip install proteus-actuarial-library[gpu]
```

## Documentation

**[Read the full documentation on Read the Docs](https://proteusllp-actuarial-library.readthedocs.io/)**


- [Usage Guide](docs/usage.md) - Comprehensive examples and API documentation
- [Development Guide](docs/development.md) - Setting up the development environment and running tests
- [Examples](examples/) - Example scripts showing how to use the library

## Project Status

PAL is currently in early release preview (beta). There are a limited number of supported distributions and reinsurance contracts. We are working on:

* Adding more distributions and loss generation types
* Making it easier to work with multi-dimensional variables
* Adding support for Catastrophe loss generation
* Adding support for more reinsurance contract types (Surplus, Stop Loss etc)
* Stratified sampling and Quasi-Monte Carlo methods
* Reporting dashboards

## Issues

Please log issues on our github [page](https://github.com/ProteusLLP/proteusllp-actuarial-library/issues).

## Contributing

You are welcome to contribute pull requests. Please see the [Contributer License Agreement](./CLA.md)
