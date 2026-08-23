Copulas
=======

The copulas module provides copula models for dependence between stochastic variables. Each concrete copula has its own reference page.

Elliptical copulas
-----------------

.. autosummary::
   :toctree: copulas
   :nosignatures:

   pal.copulas.GaussianCopula
   pal.copulas.StudentsTCopula

Archimedean copulas
-------------------

.. autosummary::
   :toctree: copulas
   :nosignatures:

   pal.copulas.ClaytonCopula
   pal.copulas.GumbelCopula
   pal.copulas.FrankCopula
   pal.copulas.JoeCopula

Extreme-value and other copulas
-------------------------------

.. autosummary::
   :toctree: copulas
   :nosignatures:

   pal.copulas.GalambosCopula
   pal.copulas.HuslerReissCopula
   pal.copulas.ExtremalTCopula
   pal.copulas.MM1Copula
   pal.copulas.PlackettCopula

Supporting API
--------------

.. autosummary::
   :nosignatures:

   pal.copulas.Copula
   pal.copulas.EllipticalCopula
   pal.copulas.ArchimedeanCopula

Usage example
-------------

.. code-block:: python

   from pal import copulas, distributions

   var1 = distributions.Gamma(alpha=2.5, theta=2).generate()
   var2 = distributions.LogNormal(mu=1, sigma=0.5).generate()

   copulas.GumbelCopula(theta=1.2).apply([var1, var2])
