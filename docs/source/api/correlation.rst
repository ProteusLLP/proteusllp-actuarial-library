Correlation matrix repair
=========================

A correlation matrix must be symmetric, positive semidefinite and have ones
on its diagonal. Estimated or manually specified correlation matrices can
violate these conditions, making them unsuitable for copulas and multivariate
simulation.

PAL provides two repair algorithms through :func:`pal.correlation.nearest_correlation`:

* **Higham** (the default) minimises the Frobenius distance to the input
  subject to the correlation-matrix constraints. It uses alternating
  projections with Dykstra's correction.
* **Spectral** clips negative eigenvalues and rescales to unit diagonal.
  It is a convenient one-step approximation, but does not generally yield
  the closest correlation matrix.

The :func:`pal.correlation.near_corr` function preserves the Proteus Correlation
Toolkit interface, including the ability to preserve selected entries with a
list of fixed index pairs (the pattern argument). PAL also implements
Higham's row-and-column weighted Frobenius norm. Given a strictly
positive weight vector ``w`` of length ``n``, it minimises
``sum(w[i] * w[j] * (C[i,j] - A[i,j])**2)``.
The weights are **only a vector**, not an elementwise weight matrix. Define
``D = diag(sqrt(w))``. The congruence transform ``D @ C @ D`` reduces
the weighted problem to ordinary Frobenius projections with a prescribed
diagonal. Multiplying every weight by the same constant does not change
the solution.

Both repair algorithms work with NumPy and, when PAL is configured for GPU use, CuPy.
A positive minimum eigenvalue can be requested when positive definiteness
is required for Cholesky factorisation. Gaussian and Student's t copulas and
multivariate normal and Student's t distributions apply Higham repair by
default when their input correlation matrices require adjustment, emitting
a warning and providing an ``adjusted_correlation_matrix`` property for
inspection. Set ``repair="none"`` on those classes to disable automatic repair.

.. autofunction:: pal.correlation.near_corr

.. autofunction:: pal.correlation.nearest_correlation

.. autofunction:: pal.correlation.is_psd

Correlation matrix completion
-----------------------------

When some pairwise correlations are unknown, put NaN in the corresponding
lower-triangular entries and use the maximum-determinant completion. The
known correlations are held fixed. The `auto` setting chooses between
primal (dense) and dual (sparse) formulations.

.. autofunction:: pal.correlation.complete_corr

.. autofunction:: pal.correlation.complete_corr_sparse

.. autofunction:: pal.correlation.complete_corr_dense

Clustering and graph diagnostics
--------------------------------

Cluster correlated risks by hierarchical linkage, obtaining a reordered
matrix, linkage matrix, and original-variable ordering.

.. autofunction:: pal.correlation.cluster_corr

.. autoclass:: pal.correlation.ClusteringResult
   :members:

.. autofunction:: pal.correlation.analyze_threshold_ranges

.. autofunction:: pal.correlation.extract_clusters

.. autofunction:: pal.correlation.compute_cluster_statistics

The clique functions identify maximally connected sets of specified
pairwise correlations. They can detect infeasible groups of coefficients
before attempting matrix completion.

.. autofunction:: pal.correlation.get_all_cliques

.. autofunction:: pal.correlation.get_big_cliques_from_edges

.. autofunction:: pal.correlation.get_edges_from_incomplete_matrix

.. autofunction:: pal.correlation.get_edges_from_adjacency_matrix

.. autofunction:: pal.correlation.get_adjacency_matrx

.. autofunction:: pal.correlation.check_all_cliques_in_correlation_matrix

.. autofunction:: pal.correlation.check_incomplete_matrix_is_chordal

.. autofunction:: pal.correlation.find_non_pd_submatrix

Backend behaviour
-----------------

Unweighted and weighted nearest-correlation repair run on PAL's active
NumPy or CuPy backend using eigenvalue decompositions and array operations.

Matrix completion uses SciPy root-finding on the CPU. Hierarchical linkage
also uses SciPy on the CPU. Both accept CuPy inputs but transfer the matrix
to the CPU for those calculations, and return the resulting correlation
matrix on PAL's active backend. Clustering linkage metadata and graph
clique enumeration are CPU-side. This transfer is not a GPU acceleration
of the underlying optimisation or graph algorithms.

Usage example
-------------

.. code-block:: python

   import numpy as np
   from pal.correlation import nearest_correlation

   estimated = np.array([
       [1.0, 0.9, 0.9],
       [0.9, 1.0, -0.9],
       [0.9, -0.9, 1.0],
   ])
   repaired = nearest_correlation(estimated, minimum_eigenvalue=1e-8)

   assert np.allclose(np.diag(repaired), 1.0)
   assert np.linalg.eigvalsh(repaired).min() >= -1e-8

   # Prioritise correlations involving the first variable.
   from pal.correlation import near_corr
   weighted = near_corr(estimated, weights=np.array([5.0, 1.0, 1.0]))
   assert weighted.shape == estimated.shape

Reference
---------

Higham, N. J. (2002), *Computing the nearest correlation matrix:
a problem from finance*, IMA Journal of Numerical Analysis, 22, 329-343.
