Copulas
=======

The copulas module provides copula functions for modeling dependencies between stochastic variables.

.. automodule:: pal.copulas
   :members:
   :undoc-members:
   :show-inheritance:

Available Copulas
-----------------

The following copulas are available for modeling dependencies:

- **GaussianCopula**: Gaussian (Normal) copula for modeling linear correlations
- **GumbelCopula**: Gumbel copula for modeling upper tail dependence
- **ClaytonCopula**: Clayton copula for modeling lower tail dependence
- **FrankCopula**: Frank copula for symmetric dependence

Usage Example
-------------

.. code-block:: python

   from pal import distributions, copulas

   # Create independent variables
   var1 = distributions.Gamma(alpha=2.5, beta=2).generate()
   var2 = distributions.LogNormal(mu=1, sigma=0.5).generate()

   # Apply Gumbel copula to create dependency
   copulas.GumbelCopula(alpha=1.2, n=2).apply([var1, var2])
