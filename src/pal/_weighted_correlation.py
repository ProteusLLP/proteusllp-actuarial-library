"""Higham's diagonal weighted Frobenius nearest-correlation projection.

For positive row/column weights w, solve

    min ||diag(sqrt(w)) @ (C - A) @ diag(sqrt(w))||_F**2

over symmetric correlation matrices C. A congruence transformation converts
the weighted problem to ordinary Frobenius projection with a prescribed,
non-unit diagonal. This runs on PAL's configured NumPy or CuPy backend.
"""

from __future__ import annotations

import typing as t

from ._maths import xp


def _row_weights(weights: t.Any, size: int) -> t.Any:
    """Validate and normalise a positive vector of row/column weights.

    Args:
        size: Number of rows and columns in the correlation matrix.
        weights: Weights applied to the correlation repair objective.
    """
    vector = xp.asarray(weights, dtype=xp.float64)
    if vector.ndim != 1 or vector.shape != (size,):
        raise ValueError("weights must be a one-dimensional vector of length n.")
    if not bool(xp.all(xp.isfinite(vector))) or not bool(xp.all(vector > 0)):
        raise ValueError("weights must be finite and strictly positive.")
    # Multiplying all weights by a constant does not change the minimiser.
    # Normalise before constructing the congruence matrix.
    return vector / xp.mean(vector)


def weighted_nearest_correlation(
    matrix: t.Any,
    weights: t.Any,
    *,
    pattern: t.Sequence[tuple[int, int]] | None = None,
    minimum_eigenvalue: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 10000,
) -> t.Any:
    """Compute Higham's nearest correlation matrix in a diagonal weighted norm.

    A positive weight vector w defines the quadratic objective
    sum_ij w_i*w_j*(C_ij - A_ij)**2. The weights argument must be a
    strictly positive vector of length n. Each weight rescales one row
    and its corresponding column through D = diag(sqrt(w)); an
    elementwise n-by-n matrix is not accepted.

    Fixed correlations in pattern are preserved symmetrically, and
    minimum_eigenvalue imposes C >= minimum_eigenvalue * I. Dykstra's
    correction yields convergence towards the weighted Frobenius minimiser.

    Args:
        matrix: Matrix to validate, complete or use in the correlation calculation.
        max_iterations: Maximum number of iterations of the repair algorithm.
        minimum_eigenvalue: Minimum permissible eigenvalue of the repaired correlation matrix.
        pattern: Optional pattern of fixed, missing or weighted matrix entries.
        tolerance: Numerical tolerance used to assess convergence.
        weights: Weights applied to the correlation repair objective.
    """
    source = xp.asarray(matrix, dtype=xp.float64)
    n = source.shape[0]
    weight = _row_weights(weights, n)
    if n == 1:
        return xp.ones((1, 1), dtype=xp.float64)
    root_weight = xp.sqrt(weight)
    outer_root = root_weight[:, None] * root_weight[None, :]
    source = (source + source.T) * 0.5
    index = xp.arange(n)
    source[index, index] = 1.0
    if float(xp.linalg.eigvalsh(source)[0].item()) >= minimum_eigenvalue:
        return source.copy()

    # Let Y = D (C - mI) D. Then Y must be PSD with diagonal (1-m)*w,
    # while the inverse transform C = mI + D**-1 Y D**-1 preserves the floor.
    result = source * outer_root
    result[index, index] = (1.0 - minimum_eigenvalue) * weight
    correction = xp.zeros_like(result)

    fixed_rows = xp.asarray([i for i, _ in pattern] if pattern else [], dtype=int)
    fixed_cols = xp.asarray([j for _, j in pattern] if pattern else [], dtype=int)
    for _ in range(max_iterations):
        residual = result - correction
        eigenvalues, eigenvectors = xp.linalg.eigh((residual + residual.T) * 0.5)
        psd = (eigenvectors * xp.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        correction = psd - residual
        previous = result
        result = psd.copy()
        result[index, index] = (1.0 - minimum_eigenvalue) * weight
        if pattern:
            result[fixed_rows, fixed_cols] = source[fixed_rows, fixed_cols] * outer_root[fixed_rows, fixed_cols]
            result[fixed_cols, fixed_rows] = source[fixed_rows, fixed_cols] * outer_root[fixed_rows, fixed_cols]

        change = float((xp.linalg.norm(result - previous) / xp.maximum(1.0, xp.linalg.norm(result))).item())
        if change <= tolerance:
            candidate = (result + result.T) * (0.5 / outer_root)
            # Inverting Y = D(C - mI)D restores mI to the diagonal.
            candidate[index, index] = 1.0
            # Check feasibility on the original correlation scale.
            if float(xp.linalg.eigvalsh(candidate)[0].item()) >= minimum_eigenvalue - 10.0 * tolerance:
                return candidate

    raise RuntimeError("Weighted nearest correlation matrix did not converge; increase max_iterations.")
