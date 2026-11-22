"""Regression tests for automatic Higham repair in elliptical models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pal._maths import xp
from pal.copulas import GaussianCopula, StudentsTCopula
from pal.maths import asnumpy
from pal.multivariate_distributions import MultivariateNormal, MultivariateStudentsT


INDEFINITE = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.8], [0.9, -0.8, 1.0]])


@pytest.mark.parametrize("student_t", [False, True])
def test_elliptical_copulas_repair_by_default_and_warn(student_t: bool) -> None:
    """Both families should simulate using the announced, repaired matrix."""
    with pytest.warns(UserWarning, match="Higham"):
        copula = StudentsTCopula(INDEFINITE, dof=5) if student_t else GaussianCopula(INDEFINITE)
    corrected = asnumpy(copula.adjusted_correlation_matrix)
    assert copula.was_repaired
    np.testing.assert_allclose(np.diag(corrected), 1.0, atol=1e-12)
    np.testing.assert_allclose(corrected, corrected.T, atol=1e-12)
    assert np.linalg.eigvalsh(corrected)[0] >= 1e-8 - 1e-9
    np.testing.assert_allclose(asnumpy(copula._chol @ copula._chol.T), corrected, atol=1e-9)
    # A mutable copy must not allow accidental modifications to simulation settings.
    exposed = copula.adjusted_correlation_matrix
    exposed[0, 1] = 0.0
    assert float(copula.adjusted_correlation_matrix[0, 1]) != 0.0


@pytest.mark.parametrize("student_t", [False, True])
def test_valid_copula_is_not_repaired_or_warned(student_t: bool) -> None:
    correlation = np.array([[1.0, -0.4], [-0.4, 1.0]])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        copula = StudentsTCopula(correlation, dof=6) if student_t else GaussianCopula(correlation)
    assert not captured
    assert not copula.was_repaired
    np.testing.assert_array_equal(asnumpy(copula.adjusted_correlation_matrix), correlation)


@pytest.mark.parametrize("student_t", [False, True])
def test_copula_strict_mode_rejects_indefinite_matrix(student_t: bool) -> None:
    with pytest.raises(ValueError, match="positive definite"):
        if student_t:
            StudentsTCopula(INDEFINITE, dof=5, repair="none")
        else:
            GaussianCopula(INDEFINITE, repair="none")


@pytest.mark.parametrize("student_t", [False, True])
def test_cholesky_input_exposes_resulting_correlation_matrix(student_t: bool) -> None:
    correlation = np.array([[1.0, 0.4], [0.4, 1.0]])
    factor = np.linalg.cholesky(correlation)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        copula = (
            StudentsTCopula(factor, dof=8, matrix_type="chol")
            if student_t
            else GaussianCopula(factor, matrix_type="chol")
        )
    assert not captured
    assert not copula.was_repaired
    np.testing.assert_allclose(asnumpy(copula.correlation_matrix), correlation, atol=1e-14)
    np.testing.assert_allclose(asnumpy(copula.adjusted_correlation_matrix), correlation, atol=1e-14)


@pytest.mark.parametrize("student_t", [False, True])
def test_multivariate_repair_preserves_diagonal_scales(student_t: bool) -> None:
    standard_deviations = np.array([2.0, 3.0, 4.0])
    scale_matrix = INDEFINITE * np.outer(standard_deviations, standard_deviations)
    with pytest.warns(UserWarning, match="Higham"):
        model = (
            MultivariateStudentsT(5, [0.0] * 3, scale_matrix)
            if student_t
            else MultivariateNormal([0.0] * 3, scale_matrix)
        )
    assert model.was_repaired
    corrected = asnumpy(model.adjusted_correlation_matrix)
    adjusted_scale = asnumpy(model.adjusted_scale_matrix if student_t else model.adjusted_covariance_matrix)
    np.testing.assert_allclose(np.diag(adjusted_scale), np.diag(scale_matrix), atol=1e-12)
    np.testing.assert_allclose(np.diag(corrected), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(corrected)[0] >= 1e-8 - 1e-9
    np.testing.assert_allclose(asnumpy(model._chol @ model._chol.T), adjusted_scale, atol=1e-8)


@pytest.mark.parametrize("student_t", [False, True])
def test_valid_multivariate_does_not_warn(student_t: bool) -> None:
    scale = np.array([[4.0, 1.2], [1.2, 9.0]])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = MultivariateStudentsT(5, [0, 0], scale) if student_t else MultivariateNormal([0, 0], scale)
    assert not captured
    assert not model.was_repaired
    expected_correlation = scale / np.outer(np.sqrt(np.diag(scale)), np.sqrt(np.diag(scale)))
    np.testing.assert_allclose(asnumpy(model.adjusted_correlation_matrix), expected_correlation)


def test_higham_vector_weights_and_fixed_entries_are_available_to_copulas() -> None:
    with pytest.warns(UserWarning, match="Higham"):
        copula = GaussianCopula(INDEFINITE, weights=[10.0, 1.0, 1.0], pattern=[(0, 1)])
    corrected = asnumpy(copula.adjusted_correlation_matrix)
    assert corrected[0, 1] == pytest.approx(INDEFINITE[0, 1], abs=1e-10)
    assert np.linalg.eigvalsh(corrected)[0] > -1e-8


def test_higham_vector_weights_are_available_to_multivariate_distributions() -> None:
    covariance = INDEFINITE * np.outer([2.0, 3.0, 4.0], [2.0, 3.0, 4.0])
    with pytest.warns(UserWarning, match="Higham"):
        model = MultivariateNormal([0.0] * 3, covariance, weights=[10.0, 1.0, 1.0])
    assert model.was_repaired
    np.testing.assert_allclose(asnumpy(model.adjusted_covariance_matrix).diagonal(), [4.0, 9.0, 16.0])


@pytest.mark.parametrize("factory", [
    lambda: GaussianCopula(INDEFINITE, repair="none"),
    lambda: StudentsTCopula(INDEFINITE, 5, repair="none"),
    lambda: MultivariateNormal([0.0] * 3, INDEFINITE, repair="none"),
    lambda: MultivariateStudentsT(5, [0.0] * 3, INDEFINITE, repair="none"),
])
def test_repair_none_requires_valid_correlations(factory: object) -> None:
    with pytest.raises(ValueError, match="positive definite"):
        factory()  # type: ignore[operator]


def test_gpu_repair_returns_device_arrays_without_host_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    if xp.__name__ != "cupy":
        pytest.skip("GPU-specific regression")

    def reject_host_copy(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected GPU-to-CPU conversion")

    monkeypatch.setattr(xp, "asnumpy", reject_host_copy)
    correlations = xp.asarray(INDEFINITE)
    with pytest.warns(UserWarning, match="Higham"):
        copula = GaussianCopula(correlations)
    assert isinstance(copula.adjusted_correlation_matrix, xp.ndarray)

    scales = xp.asarray([2.0, 3.0, 4.0])
    covariance = correlations * scales[:, None] * scales[None, :]
    with pytest.warns(UserWarning, match="Higham"):
        model = MultivariateNormal([0.0] * 3, covariance)
    assert isinstance(model.adjusted_correlation_matrix, xp.ndarray)
    assert isinstance(model.adjusted_covariance_matrix, xp.ndarray)
