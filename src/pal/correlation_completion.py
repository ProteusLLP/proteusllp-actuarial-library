"""Maximum-determinant completion of partially specified correlation matrices.

The observed entries are taken from the lower triangle. Completion uses
SciPy's CPU root solver; CuPy inputs are transferred to the host once, and
completed matrices are returned to PAL's active array backend.
"""

from __future__ import annotations

import typing as t

import numpy as np
from scipy.optimize import root

from ._maths import asnumpy, xp


def _prepare(matrix: t.Any) -> np.ndarray:
    """Copy lower-triangular observations into a symmetric matrix."""
    raw = asnumpy(matrix, dtype=float).copy()
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.shape[0] == 0:
        raise ValueError("incomplete_matrix must be a nonempty square matrix.")
    rows, cols = np.triu_indices(raw.shape[0], 1)
    raw[rows, cols] = raw[cols, rows]
    if not np.all(np.isfinite(np.diag(raw))) or np.any(np.diag(raw) <= 0):
        raise ValueError("The diagonal must contain finite positive variances.")
    if np.any(np.isinf(raw)):
        raise ValueError("Specified correlations must be finite.")
    return raw


def _complete_sparse_cpu(source: np.ndarray) -> np.ndarray:
    """Solve the maximum-entropy dual over observed precision entries."""
    n = len(source)
    rows, cols = np.where(np.tril(np.isfinite(source)))
    target = source[rows, cols]
    if len(target) == n * (n + 1) // 2:
        if np.linalg.eigvalsh(source)[0] <= 0:
            raise ValueError("A fully specified matrix must be positive definite.")
        return source.copy()

    start = np.zeros(len(target))
    diagonal = rows == cols
    start[diagonal] = 1 / target[diagonal]

    def residuals(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        precision = np.zeros((n, n))
        precision[rows, cols] = x
        precision[cols, rows] = x
        try:
            completed = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            return np.full(len(target), 1e10), np.eye(len(target)) * 1e10
        error = completed[rows, cols] - target
        jacobian = -(
            completed[rows[:, None], rows[None, :]] * completed[cols[:, None], cols[None, :]]
            + completed[rows[:, None], cols[None, :]]
            * completed[cols[:, None], rows[None, :]]
            * (rows[None, :] != cols[None, :])
        )
        return 1e5 * error, 1e5 * jacobian

    fitted = root(residuals, start, method="lm", jac=True)
    precision = np.zeros((n, n))
    precision[rows, cols] = fitted.x
    precision[cols, rows] = fitted.x
    try:
        result = np.linalg.inv(precision)
    except np.linalg.LinAlgError as error:
        raise ValueError("No positive-definite completion found.") from error
    return (result + result.T) / 2


def _complete_dense_cpu(source: np.ndarray) -> np.ndarray:
    """Solve for missing covariance entries with inverse zeros at those entries."""
    from .correlation import near_corr

    n = len(source)
    rows, cols = np.where(np.tril(np.isnan(source), k=-1))
    if len(rows) == 0:
        if np.linalg.eigvalsh(source)[0] <= 0:
            raise ValueError("A fully specified matrix must be positive definite.")
        return source.copy()
    fixed_rows, fixed_cols = np.where(np.tril(np.isfinite(source)))
    observed_offdiag = source[np.tril_indices(n, -1)]
    observed_offdiag = observed_offdiag[np.isfinite(observed_offdiag)]
    fill = float(np.mean(observed_offdiag)) if observed_offdiag.size else 0.0
    initial = np.nan_to_num(source, nan=fill)
    std = np.sqrt(np.diag(initial))
    correlation = initial / std[:, None] / std[None, :]
    # The projection seeks a strictly positive definite interior point.
    if np.linalg.eigvalsh(initial)[0] < 1e-8:
        fixed_pairs = [(int(i), int(j)) for i, j in zip(fixed_rows, fixed_cols)]
        initial = asnumpy(near_corr(correlation, pattern=fixed_pairs))
        initial *= std[:, None] * std[None, :]
    base = initial.copy()

    def residuals(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = base.copy()
        matrix[rows, cols] = x
        matrix[cols, rows] = x
        try:
            inv = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.full(len(x), 1e10), np.eye(len(x)) * 1e10
        error = inv[rows, cols]
        jacobian = -(
            inv[rows[:, None], rows[None, :]] * inv[cols[:, None], cols[None, :]]
            + inv[rows[:, None], cols[None, :]] * inv[cols[:, None], rows[None, :]]
        )
        return error, jacobian

    fitted = root(residuals, initial[rows, cols], jac=True, method="lm", tol=1e-12)
    result = source.copy()
    result[rows, cols] = fitted.x
    result[cols, rows] = fitted.x
    return result


def _complete(matrix: t.Any, method: str) -> t.Any:
    """Run the toolkit-compatible completion and validate the result."""
    source = _prepare(matrix)
    if method not in ("auto", "sparse", "dense"):
        raise ValueError("method must be 'auto', 'sparse', or 'dense'.")
    if method == "auto":
        observed = int(np.count_nonzero(np.tril(np.isfinite(source))))
        method = "sparse" if observed <= len(source) * (len(source) + 1) / 4 else "dense"
    result = _complete_sparse_cpu(source) if method == "sparse" else _complete_dense_cpu(source)
    fixed = np.isfinite(source)
    error = float(np.max(np.abs(result[fixed] - source[fixed])))
    if error > 1e-6 or np.linalg.eigvalsh(result)[0] < -1e-7:
        raise ValueError("Could not find a positive-definite completion preserving specified entries.")
    return xp.asarray(result)


def complete_corr(incomplete_matrix: t.Any, method: str = "auto") -> t.Any:
    """Maximise determinant while filling missing correlations.

    The lower triangle defines the observed coefficients; missing entries
    are NaN. Methods 'sparse', 'dense', and 'auto' preserve the toolkit API.
    The interior-point equations are solved by SciPy on CPU, with the
    output transferred back to PAL's configured NumPy or CuPy backend.
    """
    return _complete(incomplete_matrix, method.lower())


def complete_corr_sparse(incomplete_matrix: t.Any) -> t.Any:
    """Complete a sparse matrix through a precision-matrix formulation."""
    return _complete(incomplete_matrix, "sparse")


def complete_corr_dense(incomplete_matrix: t.Any) -> t.Any:
    """Complete a dense matrix through covariance-matrix optimisation."""
    return _complete(incomplete_matrix, "dense")


__all__ = ["complete_corr", "complete_corr_sparse", "complete_corr_dense"]
