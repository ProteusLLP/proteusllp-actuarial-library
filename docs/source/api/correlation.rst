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

The :func:`pal.correlation.near_corr` function preserves the Proteus Correlation\nToolkit interface, including the ability to preserve selected entries with a\nlist of fixed index pairs (the pattern argument). The original toolkit also\naccepted a weights argument but never applied it; PAL raises an error for\nnon-None weights rather than silently ignoring them.\n\nBoth repair algorithms work with NumPy and, when PAL is configured for GPU use, CuPy.
A positive minimum eigenvalue can be requested when positive definiteness
is required for Cholesky factorisation.

.. autofunction:: pal.correlation.near_corr\n\n.. autofunction:: pal.correlation.nearest_correlation\n\n.. autofunction:: pal.correlation.is_psd

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

Reference
---------

Higham, N. J. (2002), *Computing the nearest correlation matrix:
a problem from finance*, IMA Journal of Numerical Analysis, 22, 329-343.
