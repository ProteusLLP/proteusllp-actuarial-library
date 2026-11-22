"""Nearest correlation matrix repair on PAL's NumPy or CuPy backend."""

from __future__ import annotations

import math
import numbers
import typing as t

import numpy.typing as npt

from ._maths import xp
from ._weighted_correlation import weighted_nearest_correlation

__all__ = [
    "near_corr",
    "nearest_correlation",
    "is_psd",
    "complete_corr",
    "complete_corr_sparse",
    "complete_corr_dense",
    "cluster_corr",
    "ClusteringResult",
    "analyze_threshold_ranges",
    "extract_clusters",
    "compute_cluster_statistics",
    "get_adjacency_matrx",
    "get_adjacency_matrix",
    "get_edges_from_adjacency_matrix",
    "get_edges_from_incomplete_matrix",
    "get_all_cliques",
    "get_big_cliques_from_edges",
    "check_all_cliques_in_correlation_matrix",
    "check_incomplete_matrix_is_chordal",
    "find_non_pd_submatrix",
]

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
    pattern: t.Sequence[tuple[int, int]] | None = None,
    weights: npt.ArrayLike | None = None,
) -> t.Any:
    """Repair a matrix into a symmetric, positive semidefinite correlation matrix.

    The Higham method minimises Frobenius distance to the input, or its
    diagonally weighted version when weights are supplied, subject to a
    unit diagonal and the eigenvalue lower bound, up to numerical tolerance.
    It uses alternating projections with Dykstra's correction.
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
        pattern: Optional positions whose symmetric values must remain fixed.
            Only supported by the Higham method.
        weights: Positive row/column weights for Higham's weighted Frobenius
            norm, supplied as a length-n vector, a diagonal matrix, or a
            rank-one outer product. Cannot be combined with spectral repair.

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
    # The original diagonal has no effect on distance when it is fixed to 1.
    indices = xp.arange(source.shape[0])
    source[indices, indices] = 1.0

    fixed_rows: list[int] = []
    fixed_cols: list[int] = []
    rows = xp.asarray(fixed_rows, dtype=int)
    cols = xp.asarray(fixed_cols, dtype=int)
    if pattern is not None:
        if method == "spectral":
            raise ValueError("Fixed pattern entries require method='higham'.")
        for entry in pattern:
            if len(entry) != 2 or not all(isinstance(index, numbers.Integral) for index in entry):
                raise ValueError("Each pattern element must be a pair of integer indices.")
            i, j = (int(index) for index in entry)
            if not (0 <= i < source.shape[0] and 0 <= j < source.shape[0]):
                raise ValueError("pattern indices must be within the matrix.")
            fixed_rows.append(i)
            fixed_cols.append(j)
        if fixed_rows:
            rows = xp.asarray(fixed_rows, dtype=int)
            cols = xp.asarray(fixed_cols, dtype=int)

    if weights is not None:
        if method == "spectral":
            raise ValueError("Weighted repair requires method='higham'.")
        return weighted_nearest_correlation(
            source,
            weights,
            pattern=pattern,
            minimum_eigenvalue=minimum_eigenvalue,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    if source.shape[0] == 1:
        return xp.ones((1, 1), dtype=xp.float64)
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
        if fixed_rows:
            result[rows, cols] = source[rows, cols]
            result[cols, rows] = source[rows, cols]
        relative_change = float((xp.linalg.norm(result - previous) / xp.maximum(1.0, xp.linalg.norm(result))).item())
        if relative_change <= tolerance:
            smallest = float(xp.linalg.eigvalsh(result)[0].item())
            if smallest >= minimum_eigenvalue - 10.0 * tolerance:
                return (result + result.T) * 0.5

    raise RuntimeError("Nearest correlation matrix iteration did not converge; increase max_iterations.")


def near_corr(
    matrix: npt.ArrayLike,
    pattern: list[tuple[int, int]] | None = None,
    weights: npt.ArrayLike | None = None,
) -> t.Any:
    """Return the nearest correlation matrix, optionally fixing selected entries.

    This preserves the Proteus Correlation Toolkit's public signature.
    The input is symmetrised and its diagonal constrained to one. Entries
    identified by pattern remain fixed, including their transposes.

    Args:
        matrix: Square matrix of correlation coefficients.
        pattern: Zero-based index pairs of coefficients to hold fixed.
        weights: Positive row/column weights for Higham's weighted norm.
            Accepts an n-vector, a positive diagonal matrix, or an outer
            product of row/column weights, not arbitrary entrywise weights.

    Returns:
        Nearest correlation matrix on the active NumPy or CuPy backend.

    Raises:
        ValueError: Invalid matrix or pattern.
        RuntimeError: The algorithm does not converge.
    """
    return nearest_correlation(
        matrix,
        method="higham",
        pattern=pattern,
        weights=weights,
        minimum_eigenvalue=1e-8,
        tolerance=1e-10,
        max_iterations=10000,
    )


def is_psd(matrix: npt.ArrayLike) -> bool:
    """Test whether a finite symmetric matrix is positive semidefinite."""
    values = xp.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        return False
    if not bool(xp.all(xp.isfinite(values))) or not bool(xp.allclose(values, values.T)):
        return False
    return float(xp.linalg.eigvalsh(values)[0].item()) >= -1e-12


# Import dependent modules after the primary repair functions are defined.
from .correlation_cliques import (  # noqa: E402
    check_all_cliques_in_correlation_matrix,
    check_incomplete_matrix_is_chordal,
    find_non_pd_submatrix,
    get_adjacency_matrix,
    get_adjacency_matrx,
    get_all_cliques,
    get_big_cliques_from_edges,
    get_edges_from_adjacency_matrix,
    get_edges_from_incomplete_matrix,
)
from .correlation_clustering import (  # noqa: E402
    ClusteringResult,
    analyze_threshold_ranges,
    cluster_corr,
    compute_cluster_statistics,
    extract_clusters,
)
from .correlation_completion import complete_corr, complete_corr_dense, complete_corr_sparse  # noqa: E402
