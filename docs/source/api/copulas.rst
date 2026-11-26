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
           [PairCopula.Gumbel(2.0), PairCopula.Gaussian(0.4)],
           [PairCopula.Clayton(1.5)],
       ],
       structure="c",
   )
   uniforms = vine.generate()

The result is a coupled ProteusVariable of uniform margins, in variable
order. As with other PAL copulas, use ``vine.apply(variables)`` to reorder
existing independent marginal simulations while preserving their values.
Both Monte Carlo and randomised quasi-Monte Carlo sampling are supported.
All PAL copula families have a bivariate representation for use in vines.
The named constructors take the dependence parameters of the corresponding
copula family:

.. code-block:: python

   from pal.copulas import PairCopula

   independent = PairCopula.Independence()
   gaussian = PairCopula.Gaussian(0.4)
   student_t = PairCopula.StudentsT(0.4, dof=5)
   clayton = PairCopula.Clayton(1.5)
   gumbel = PairCopula.Gumbel(1.2)
   frank = PairCopula.Frank(3)
   joe = PairCopula.Joe(2)
   galambos = PairCopula.Galambos(1.5)
   plackett = PairCopula.Plackett(2)
   husler_reiss = PairCopula.HuslerReiss(0.75)
   extremal_t = PairCopula.ExtremalT(0.4, nu=3)
   mm1 = PairCopula.MM1(theta=1.2, delta=1.5)

The bivariate MM1 copula is equivalent to a Gumbel copula with the product
of its two parameters. Student's t and extremal-t pair copulas require
both a correlation and a positive degrees-of-freedom parameter.

For each non-independent family, the rotation may be 0, 90, 180 or 270
degrees. For programmatic model construction, the string-based constructor
is also available. Its optional second parameter is named
``parameter2``; it is used for the degrees of freedom or the MM1 delta.

.. code-block:: python

   from pal.copulas import PairCopula

   assert PairCopula.Gumbel(1.2, rotation=180) == PairCopula("gumbel", 1.2, rotation=180)
   assert PairCopula.StudentsT(0.4, 5) == PairCopula("students_t", 0.4, parameter2=5)

Gaussian, Student's t, Clayton and Frank use analytical conditional
quantile transformations. Other families use vectorised numerical
inversion on the active CPU or GPU array backend.

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
