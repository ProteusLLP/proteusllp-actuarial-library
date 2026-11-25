Copulas
=======

The copulas module provides copula models for dependence between stochastic variables. Each concrete copula has its own reference page.

Elliptical copulas
-----------------

Gaussian and Student's t copulas require positive-definite correlation
matrices. By default, PAL repairs invalid matrices with Higham's nearest
correlation algorithm and emits a ``UserWarning`` when an adjustment is
necessary. Valid matrices are used unchanged, without a warning. Inspect
``adjusted_correlation_matrix`` to see the coefficients used in simulation;
``was_repaired`` records whether repair was necessary.

Pass ``repair="none"`` to reject invalid matrices, or ``weights`` (a positive
vector of row/column weights) and ``pattern`` (fixed index pairs) to control
the Higham repair. The ``matrix_type="chol"`` constructor also exposes the
implied correlation matrix.

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
   pal.copulas.FRA1Copula

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
