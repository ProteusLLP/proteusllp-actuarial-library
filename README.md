# Python Capital Model

An insurance capital modeling library in python.

## Introduction

The Python Capital Model (PCM) is a simple, fast and lightweight simulation-based capital modeling package, designed for non-life insurance companies. It is originated from the ![rippy](https://github.com/pythactuary/rippy) package for reinsurance modeling.

PCM is designed to look after the complicated stuff, such as copulas and simulation re-ordering, providing easy to use objects and clear syntax. 

PCM is based on the scientific python stack of numpy and scipy for fast performance. It can optionally run on a GPU for extremely fast performance. It is designed for interoperability with numpy and ndarrays.


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

PCM variables can

The PCM library ensures variables that have been used in formula with other variables (i.e. variables that are *coupled*) are re-ordered consistently. For example

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

PCM uses the ```default_rng``` class of the ```numpy.random``` module. This can also be configured using the ```config.rng``` property.

### Using a GPU

GPU support requires a CUDA compatible GPU. Internally PCM uses the cupy library. Install the dependencies by running

```
pip install rippy[gpu]
```

To enable GPU mode, set the RIPPY_USE_GPU environment variable to 1.
```linux
export RIPPY_USE_GPU=1
```
on Linux or
```
set RIPPY_USE_GPU=1
```
on Windows. Set it to anythin else to revert to using a CPU


## Project Status

PCM is currently a proof of concept. There are a limited number of supported distributions and reinsurance contracts. We are working on:

* Adding more distributions and loss generation types
* Adding support for Catastrophe loss generation and reinsurance contracts
* Adding support for more reinsurance contract types (Surplus, Stop Loss etc)
* Grouping reinsurance contracts into programs and structures
* Stratified sampling and Quasi-Monte Carlo methods
* Reporting dashboards

## Issues

Please log issues in github

## Contributing

You are welcome to contribute pull requests

