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

Vine copulas
------------

A vine copula represents a multivariate dependence structure as a sequence of
bivariate copulas. Each edge can use a different family, dependence parameter
and rotation. Higher trees describe dependence conditional on other risks.
This allows different upper- and lower-tail dependence structures across a
portfolio without requiring all risks to follow the same copula family.

PAL supports canonical (C-vine) and drawable (D-vine) structures under the
simplifying assumption that conditional copula parameters do not vary with
the conditioning variables. The first tree contains d-1 edges, the second
contains d-2, and so on. An independent pair can be used to truncate the vine.
For C-vines, the first margin is the root of the first tree. For D-vines,
the first tree is the chain of adjacent margins.

For example, in a three-risk C-vine, the first risk is a common driver;
the second tree describes the remaining dependence of the other two risks
conditional on that common driver. The following model uses upper-tail
dependence for the first pair, Gaussian dependence for the second pair,
and conditional lower-tail dependence between the other two risks.

.. code-block:: python

   from pal.copulas import PairCopula, VineCopula

   vine = VineCopula(
       [
           [PairCopula("gumbel", 2.0), PairCopula("gaussian", 0.4)],
           [PairCopula("clayton", 1.5)],
       ],
       structure="c",
   )
   uniforms = vine.generate()

The result is a coupled ProteusVariable of uniform margins, in variable
order. As with other PAL copulas, use ``vine.apply(variables)`` to reorder
existing independent marginal simulations while preserving their values.
Both Monte Carlo and randomised quasi-Monte Carlo sampling are supported.
The supported pair families are independence, Gaussian, Clayton, Gumbel
and Frank. The pair copula rotation can be 0, 90, 180 or 270 degrees.

.. autosummary::
   :toctree: copulas
   :nosignatures:

   pal.copulas.PairCopula
   pal.copulas.VineCopula

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
