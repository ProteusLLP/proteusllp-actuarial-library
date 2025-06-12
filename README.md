# Proteus Actuarial Library

An actuarial stochastic modeling library in python.

**Note**
This library is still in development!

## Introduction

The Proteus Actuarial Library (PAL) is a simple, fast and lightweight framework for building simulation-based actuarial and financial models. It is originated from the ![rippy](https://github.com/pythactuary/rippy) package for reinsurance modeling.

PAL is designed to look after the complicated stuff, such as copulas and simulation re-ordering, providing easy to use objects and clear syntax. 

PAL is based on the scientific python stack of numpy and scipy for fast performance. It can optionally run on a GPU using the cupy package for extremely fast performance. It is designed for interoperability with numpy and ndarrays.


### Creating stochastic variables and variable containers

Stochastic variables can be created with the ```StochasticScalar``` class:

```python
svariable = StochasticScalar([1,2,3,4])
```

Statistical distributions are available in the distributions module

```python
svariable = distributions.Gamma(alpha=2.5,beta=2).generate()
```

Variables can be grouped into containers with the ```ProteusVariable``` class. ```ProteusVariables``` have a dimension and values that can either be a list or dictionary of other variables.

```python
svariable1 = distributions.Gamma(alpha=2.5,beta=2).generate()
svariable2 = distributions.LogNormal(mu=1,sigma=0.5).generate()
variable_container = ProteusVariable(dim_name="line",values={"Motor":svariable1,"Property":svariable2})
```

Variable containers can be operated on with numpy functions, and can be added, multiplied together etc. If the ```values``` are a dictionary then operations involving multiple variable containers will attempt to match on the labels of the dictionary.

### Copulas and Couplings

Statistical dependencies between PAL variables can be modelled using copulas. The idea is that the marginal distributions can be sampled independently, and then re-ordered relative to one another using the relative ordering from a sample from a copula.

```python
svariable1 = distributions.Gamma(alpha=2.5,beta=2).generate()
svariable2 = distributions.LogNormal(mu=1,sigma=0.5).generate()
copulas.GumbelCopula(alpha=1.2,n=2).apply([svariable1,svariable2])
```

The PAL library ensures variables that have been used in formula with other variables (i.e. variables that are *coupled*) are re-ordered consistently. For example

```python
svariable1 = distributions.Gamma(alpha=2.5,beta=2).generate()
svariable2 = distributions.LogNormal(mu=1,sigma=0.5).generate()
svariable3 = svariable1+svariable2
```
Because svariable1 and svariable2 have been used in the formula for svariable3, svariable1,svariable2 and svariable3 are *coupled* together.

If applying a copula between svariable3 and another variable svariable4 results in svariable3 being reordered, svariable1 and svariable2 will be reordered automatically.


### Configuring the simulation settings

The global number of simulations can be changed from the ```config``` class (the default is 100,000 simulations)

```python
from rippy import config
config.n_sims = 1000000
```

The global random seed can also be configured from the ```config``` class

```python
config.set_random_seed(123456)
```

PAL uses the ```default_rng``` class of the ```numpy.random``` module. This can also be configured using the ```config.rng``` property.

### Using a GPU

GPU support requires a CUDA compatible GPU. Internally PAL uses the cupy library. Install the dependencies by running

```
pip install pal[gpu]
```

To enable GPU mode, set the RIPPY_USE_GPU environment variable to 1.
```linux
export RIPPY_USE_GPU=1
```
on Linux or
```
set RIPPY_USE_GPU=1
```
on Windows. Set it to anything else to revert to using a CPU


## Project Status

PAL is currently a proof of concept. There are a limited number of supported distributions and reinsurance contracts. We are working on:

* Adding more distributions and loss generation types
* Making it easier to work with multi-dimensional variables
* Adding support for Catastrophe loss generation
* Adding support for more reinsurance contract types (Surplus, Stop Loss etc)
* Stratified sampling and Quasi-Monte Carlo methods
* Reporting dashboards

## Issues

Please log issues in github

## Contributing

You are welcome to contribute pull requests

