"""Numerical and input-validation tests for correlation matrix repair."""

import numpy as np
import pytest

from pal.correlation import nearest_correlation
from pal.maths import asnumpy


@pytest.mark.parametrize("method", ["higham", "spectral"])
def test_equicorrelation_projection_has_known_answer(method: str) -> None:
    """An exchangeable 3x3 matrix has a nearest off-diagonal value of -1/2."""
    invalid = np.full((3, 3), -0.9)
    np.fill_diagonal(invalid, 1.0)
    unchanged = invalid.copy()

    result = nearest_correlation(invalid, method=method)

    expected = np.full((3, 3), -0.5)
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(asnumpy(result), expected, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(invalid, unchanged)


@pytest.mark.parametrize("method", ["higham", "spectral"])
@pytest.mark.parametrize("eigenvalue_floor", [0.0, 0.02])
def test_repaired_matrix_is_a_correlation_matrix(method: str, eigenvalue_floor: float) -> None:
    invalid = np.array([[1.0, 0.8, 0.8, 0.6], [0.8, 1.0, -0.4, 0.7], [0.8, -0.4, 1.0, -0.9], [0.6, 0.7, -0.9, 1.0]])
    result = nearest_correlation(invalid, method=method, minimum_eigenvalue=eigenvalue_floor)
    np.testing.assert_allclose(asnumpy(result), asnumpy(result.T), atol=1e-12)
    np.testing.assert_allclose(np.diag(asnumpy(result)), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(asnumpy(result))[0] >= eigenvalue_floor - 1e-8


def test_higham_is_closer_than_spectral_approximation() -> None:
    """Compare two algorithms on an asymmetric-sign indefinite example."""
    invalid = np.array([[1.0, 0.8, 0.8, 0.6], [0.8, 1.0, -0.4, 0.7], [0.8, -0.4, 1.0, -0.9], [0.6, 0.7, -0.9, 1.0]])
    nearest = nearest_correlation(invalid)
    approximate = nearest_correlation(invalid, method="spectral")
    assert np.linalg.norm(asnumpy(nearest) - invalid) < np.linalg.norm(asnumpy(approximate) - invalid) - 1e-4


def test_valid_matrix_is_preserved_and_not_aliased() -> None:
    valid = np.array([[1.0, 0.35], [0.35, 1.0]])
    result = nearest_correlation(valid)
    np.testing.assert_array_equal(asnumpy(result), valid)
    assert result is not valid


def test_nonsymmetric_input_uses_its_symmetric_part() -> None:
    input_matrix = np.array([[1.0, 0.9], [0.1, 1.0]])
    np.testing.assert_allclose(asnumpy(nearest_correlation(input_matrix)), [[1.0, 0.5], [0.5, 1.0]])


def test_one_dimensional_matrix_is_unit_correlation() -> None:
    np.testing.assert_array_equal(asnumpy(nearest_correlation([[-3.0]])), [[1.0]])


@pytest.mark.parametrize(
    "input_matrix",
    [[], [1, 2], [[1, 2, 3], [4, 5, 6]], [[np.nan]], [[np.inf]]],
)
def test_invalid_matrix_rejected(input_matrix: object) -> None:
    with pytest.raises(ValueError):
        nearest_correlation(input_matrix)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"method": "other"}, "method"),
        ({"minimum_eigenvalue": -0.1}, "minimum_eigenvalue"),
        ({"minimum_eigenvalue": 1.0}, "minimum_eigenvalue"),
        ({"minimum_eigenvalue": float("nan")}, "minimum_eigenvalue"),
        ({"tolerance": 0.0}, "tolerance"),
        ({"max_iterations": 0}, "max_iterations"),
    ],
)
def test_invalid_options_rejected(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        nearest_correlation([[1.0, 0.5], [0.5, 1.0]], **options)


def test_convergence_failure_is_reported() -> None:
    invalid = np.full((3, 3), -0.9)
    np.fill_diagonal(invalid, 1.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        nearest_correlation(invalid, max_iterations=1)


def test_public_api_is_discoverable() -> None:
    from pal import api, correlation

    assert correlation.nearest_correlation is nearest_correlation
    assert "pal.correlation" in api.modules()
    assert api.describe("nearest_correlation")["qualified_name"] == "pal.correlation.nearest_correlation"
