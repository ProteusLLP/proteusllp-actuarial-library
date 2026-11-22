"""Weighted nearest correlation matrices using convex ADMM projections.

The objective is one half of the sum of w_ij * (C_ij - A_ij)^2.
This is distinct from the diagonal congruence weighting in Higham (2002).
"""

from __future__ import annotations

import typing as t

from ._maths import xp


def weighted_nearest_correlation(
    matrix: t.Any,
    weights: t.Any,
    *,
    pattern: t.Sequence[tuple[int, int]] | None = None,
    minimum_eigenvalue: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 10000,
) -> t.Any:
    """Minimise elementwise weighted squared distance with correlation constraints.

    All weights must be finite and strictly positive. Equality constraints
    from pattern are imposed in both triangles, including unit diagonal.
    Updates run entirely on the active NumPy/CuPy backend, except scalar
    convergence checks.
    """
    source = xp.asarray(matrix, dtype=xp.float64)
    weight = xp.asarray(weights, dtype=xp.float64)
    if weight.shape != source.shape or weight.ndim != 2:
        raise ValueError("weights must have the same square shape as matrix.")
    if not bool(xp.all(xp.isfinite(weight))) or not bool(xp.all(weight > 0)):
        raise ValueError("weights must contain strictly positive finite values.")
    if not bool(xp.allclose(weight, weight.T)):
        raise ValueError("weights must be symmetric.")
    weight = (weight + weight.T) / 2
    source = (source + source.T) / 2
    n = source.shape[0]
    diagonal = xp.arange(n)
    source[diagonal, diagonal] = 1
    if n == 1:
        return xp.ones((1, 1), dtype=xp.float64)

    fixed_rows = xp.asarray([i for i, j in pattern] if pattern else [], dtype=int)
    fixed_cols = xp.asarray([j for i, j in pattern] if pattern else [], dtype=int)
    if pattern:
        for i, j in pattern:
            if not 0 <= i < n or not 0 <= j < n:
                raise ValueError("pattern indices must be within the matrix.")

    # The weight ratio, not its absolute scale, determines the minimiser.
    weight = weight / float(xp.mean(weight).item())
    rho = 1.0
    solution = source.copy()
    dual = xp.zeros_like(source)
    for _ in range(max_iterations):
        eigenvalues, eigenvectors = xp.linalg.eigh((solution - dual + solution.T - dual.T) / 2)
        positive = (eigenvectors * xp.maximum(eigenvalues, minimum_eigenvalue)) @ eigenvectors.T
        old_solution = solution
        solution = (weight * source + rho * (positive + dual)) / (weight + rho)
        solution = (solution + solution.T) / 2
        solution[diagonal, diagonal] = 1
        if pattern:
            solution[fixed_rows, fixed_cols] = source[fixed_rows, fixed_cols]
            solution[fixed_cols, fixed_rows] = source[fixed_rows, fixed_cols]
        dual = dual + positive - solution

        primal = float(xp.linalg.norm(positive - solution).item())
        movement = float(xp.linalg.norm(solution - old_solution).item())
        if max(primal, movement) <= tolerance * max(1, n):
            eigmin = float(xp.linalg.eigvalsh(solution)[0].item())
            if eigmin >= minimum_eigenvalue - 10 * tolerance * max(1, n):
                return (solution + solution.T) / 2
    raise RuntimeError("Weighted nearest correlation matrix did not converge; check feasibility or increase max_iterations.")
