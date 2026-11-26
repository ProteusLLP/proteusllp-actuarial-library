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

PAL supports regular vines (R-vines), including canonical (C-vine) and
drawable (D-vine) structures, under the simplifying assumption that conditional copula parameters do not vary with
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

Regular vines
~~~~~~~~~~~~~

A regular vine allows branching between groups of risks instead of requiring
a single root or a chain. Specify each edge by the two variable indices,
the pair copula and, for higher trees, the conditioning variables.
Indices are zero-based and results are returned in index order.

For example, the first tree below connects two groups of three risks around
risks 0 and 3. The second tree describes their remaining pairwise dependence
conditional on the shared risk. The higher trees are omitted.

.. code-block:: python

   from pal.copulas import PairCopula, VineCopula, VineEdge

   vine = VineCopula.from_edges(
       [
           [
               VineEdge(0, 1, PairCopula.Gumbel(1.5)),
               VineEdge(0, 2, PairCopula.Galambos(2)),
               VineEdge(0, 3, PairCopula.Gaussian(0.4)),
               VineEdge(3, 4, PairCopula.StudentsT(0.5, dof=4)),
               VineEdge(3, 5, PairCopula.Clayton(1.5, rotation=180)),
           ],
           [
               VineEdge(1, 2, PairCopula.Gaussian(0.2), (0,)),
               VineEdge(2, 3, PairCopula.Gaussian(0.1), (0,)),
               VineEdge(0, 4, PairCopula.Independence(), (3,)),
               VineEdge(4, 5, PairCopula.Gaussian(0.3), (3,)),
           ],
       ]
   )
   uniforms = vine.generate()
   assert vine.dimension == 6
   assert vine.truncation_level == 2

Each tree must be connected and acyclic. Two nodes in a higher tree may
only be connected if the corresponding edges of the preceding tree share
a node (the proximity condition). The constructor validates these
requirements and determines a valid simulation order automatically.
The order of edges within a tree is immaterial, but the order of the
two variable indices in an individual edge matters for asymmetric rotations.

The compact C-vine and D-vine syntax also supports truncation by omitting
higher rows. Both forms use the same regular-vine simulation algorithm.

.. autosummary::
   :toctree: copulas
   :nosignatures:

   pal.copulas.PairCopula
   pal.copulas.VineEdge
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
