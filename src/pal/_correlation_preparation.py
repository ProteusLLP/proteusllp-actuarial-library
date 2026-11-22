"""Shared correlation-matrix preparation for elliptical stochastic models."""

from __future__ import annotations

import typing as t
import warnings

import numpy.typing as npt

from ._maths import xp
from .correlation import nearest_correlation

RepairMode = t.Literal["higham", "none"]

# A positive floor allows the repaired matrix to be factorised by Cholesky.
_CHOLESKY_EIGENVALUE_FLOOR = 1e-8


def prepare_correlation_matrix(
    matrix: npt.ArrayLike,
    *,
    repair: RepairMode = "higham",
    weights: npt.ArrayLike | None = None,
    pattern: t.Sequence[tuple[int, int]] | None = None,
    name: str = "Correlation matrix",
) -> tuple[t.Any, bool]:
    """Return a usable correlation matrix and whether repair was necessary.

    Higham's nearest-correlation algorithm is the default. An adjustment
    emits a UserWarning; valid correlation matrices are left unchanged
    and do not emit a warning. Set repair="none" to reject invalid inputs.
    """
    if repair not in ("higham", "none"):
        raise ValueError("repair must be 'higham' or 'none'.")
    values = xp.asarray(matrix, dtype=xp.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not values.shape[0]:
        raise ValueError(f"{name} must be a non-empty square matrix.")
    if not bool(xp.all(xp.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values.")

    symmetric = bool(xp.allclose(values, values.T, rtol=1e-10, atol=1e-12))
    unit_diagonal = bool(xp.allclose(xp.diag(values), 1.0, rtol=1e-10, atol=1e-12))
    symmetrised = (values + values.T) * 0.5
    smallest = float(xp.linalg.eigvalsh(symmetrised)[0].item())
    needs_repair = not (symmetric and unit_diagonal and smallest > _CHOLESKY_EIGENVALUE_FLOOR)

    if repair == "none":
        if weights is not None or pattern is not None:
            raise ValueError("weights and pattern require repair='higham'.")
        if needs_repair:
            raise ValueError(f"{name} must be symmetric, unit-diagonal and positive definite.")
        return values.copy(), False

    if not needs_repair and weights is None and pattern is None:
        return values.copy(), False

    repaired = nearest_correlation(
        values,
        method="higham",
        minimum_eigenvalue=_CHOLESKY_EIGENVALUE_FLOOR,
        tolerance=1e-10,
        max_iterations=10000,
        weights=weights,
        pattern=pattern,
    )
    if needs_repair:
        warnings.warn(
            f"{name} required Higham correlation-matrix repair; "
            "inspect adjusted_correlation_matrix for the coefficients used.",
            UserWarning,
            stacklevel=3,
        )
    return repaired, needs_repair
