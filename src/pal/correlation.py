"""Nearest correlation matrix repair on PAL's NumPy or CuPy backend."""

from __future__ import annotations

import math
import typing as t

import numpy.typing as npt

from ._maths import xp

__all__ = ["nearest_correlation"]

Method = t.Literal["higham", "spectral"]


def _positive_semidefinite_projection(matrix: t.Any, eigenvalue_floor: float) -> t.Any:
    """Project a symmetric matrix onto matrices with bounded eigenvalues."""
    eigenvalues, eigenvectors = xp.linalg.eigh((matrix + matrix.T) * 0.5)
    return (eigenvectors * xp.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T


def _spectral_repair(matrix: t.Any, minimum_eigenvalue: float) -> t.Any:
    """Clip eigenvalues, rescale the diagonal, and optionally shrink to identity."""
    # A positive floor avoids division by zero for negative-definite input.
    positive = _positive_semidefinite_projection(matrix, float(xp.finfo(xp.float64).eps))
    scaling = xp.sqrt(xp.diag(positive))
    result = positive / scaling[:, None] / scaling[None, :]
    indices = xp.arange(result.shape[0])
    result[indices, indices] = 1.0

    if minimum_eigenvalue > 0.0:
        smallest = float(xp.linalg.eigvalsh(result)[0].item())
        if smallest < minimum_eigenvalue:
            # Shrinking to identity preserves the diagonal and enforces the floor.
            fraction = (minimum_eigenvalue - smallest) / (1.0 - smallest)
            result = (1.0 - fraction) * result + fraction * xp.eye(result.shape[0])
    return (result + result.T) * 0.5


def nearest_correlation(
    matrix: npt.ArrayLike,
    *,
    method: Method = "higham",
    minimum_eigenvalue: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> t.Any:
    """Repair a matrix into a symmetric, positive semidefinite correlation matrix.

    The Higham method minimises unweighted Frobenius distance to the input
    subject to a unit diagonal and the eigenvalue lower bound, up to numerical
    tolerance. It uses alternating projections with Dykstra's correction.
    The spectral method clips eigenvalues and normalises the diagonal; it is
    faster but is generally not the nearest correlation matrix.

    The symmetric part of the input is used; the input is not modified.
    Outputs remain on PAL's active NumPy or CuPy backend.

    Args:
        matrix: Finite non-empty square matrix.
        method: 'higham' (nearest) or 'spectral' (one-step approximation).
        minimum_eigenvalue: Lower bound on eigenvalues, less than one. A small
            positive value can make the result suitable for Cholesky methods.
        tolerance: Relative convergence tolerance for the Higham method.
        max_iterations: Maximum number of alternating projections.

    Returns:
        A unit-diagonal correlation matrix on the configured array backend.

    Raises:
        ValueError: Invalid matrix or parameters.
        RuntimeError: Higham iterations fail to converge.

    References:
        Higham, N. J. (2002), Computing the nearest correlation matrix:
        a problem from finance, IMA Journal of Numerical Analysis, 22, 329-343.
    """
    values = xp.asarray(matrix, dtype=xp.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square matrix.")
    if not bool(xp.all(xp.isfinite(values))):
        raise ValueError("matrix must contain only finite values.")
    if method not in ("higham", "spectral"):
        raise ValueError("method must be 'higham' or 'spectral'.")
    if not math.isfinite(minimum_eigenvalue) or not 0.0 <= minimum_eigenvalue < 1.0:
        raise ValueError("minimum_eigenvalue must be finite and in [0, 1).")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")

    source = (values + values.T) * 0.5
    if source.shape[0] == 1:
        return xp.ones((1, 1), dtype=xp.float64)
    indices = xp.arange(source.shape[0])
    if bool(xp.all(xp.diag(source) == 1.0)):
        smallest = float(xp.linalg.eigvalsh(source)[0].item())
        if smallest >= minimum_eigenvalue:
            return source.copy()

    if method == "spectral":
        return _spectral_repair(source, minimum_eigenvalue)

    # Dykstra's correction yields the Frobenius-nearest point in the intersection.
    result = source.copy()
    correction = xp.zeros_like(source)
    for _ in range(max_iterations):
        residual = result - correction
        projected = _positive_semidefinite_projection(residual, minimum_eigenvalue)
        correction = projected - residual
        previous = result
        result = projected.copy()
        result[indices, indices] = 1.0
        relative_change = float(
            (xp.linalg.norm(result - previous) / xp.maximum(1.0, xp.linalg.norm(result))).item()
        )
        if relative_change <= tolerance:
            smallest = float(xp.linalg.eigvalsh(result)[0].item())
            if smallest >= minimum_eigenvalue - 10.0 * tolerance:
                return (result + result.T) * 0.5

    raise RuntimeError("Nearest correlation matrix iteration did not converge; increase max_iterations.")
