"""Regression tests for weighted repair, completion, clustering and graph diagnostics."""

import numpy as np
import pytest

from pal._maths import xp
from pal.correlation import (
    analyze_threshold_ranges,
    check_all_cliques_in_correlation_matrix,
    check_incomplete_matrix_is_chordal,
    cluster_corr,
    complete_corr,
    compute_cluster_statistics,
    extract_clusters,
    get_all_cliques,
    get_edges_from_incomplete_matrix,
    near_corr,
)
from pal.maths import asnumpy


def test_weighted_projection_respects_importance_and_constraints() -> None:
    """Heavier entries should move less in an indefinite matrix."""
    matrix = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.8], [0.9, -0.8, 1.0]])
    weights = np.array([10.0, 1.0, 1.0])
    repaired = asnumpy(near_corr(matrix, weights=weights))
    unweighted = asnumpy(near_corr(matrix))
    assert abs(repaired[0, 1] - 0.9) < abs(unweighted[0, 1] - 0.9)
    assert np.linalg.eigvalsh(repaired)[0] > -1e-7
    np.testing.assert_allclose(np.diag(repaired), 1.0)
    weighted_with_pattern = asnumpy(near_corr(matrix, pattern=[(0, 1)], weights=weights))
    assert abs(weighted_with_pattern[0, 1] - 0.9) < 1e-10


def test_weighted_projection_matches_independent_slsqp_reference() -> None:
    """Independent constrained optimisation checks the weighted objective."""
    from scipy.optimize import minimize

    matrix = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.8], [0.9, -0.8, 1.0]])
    row_weights = np.array([1.0, 20.0, 4.0])
    weight_matrix = np.outer(row_weights, row_weights)

    def build(x: np.ndarray) -> np.ndarray:
        return np.array([[1, x[0], x[1]], [x[0], 1, x[2]], [x[1], x[2], 1]])

    def objective(x: np.ndarray) -> float:
        delta = build(x) - matrix
        return float(0.5 * np.sum(weight_matrix * delta * delta))

    reference = minimize(
        objective,
        x0=[0.2, 0.2, 0.2],
        method="SLSQP",
        bounds=[(-1, 1)] * 3,
        constraints={"type": "ineq", "fun": lambda x: np.linalg.eigvalsh(build(x))[0]},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert reference.success
    result = asnumpy(near_corr(matrix, weights=row_weights))
    assert objective(np.array([result[0, 1], result[0, 2], result[1, 2]])) <= reference.fun + 1e-7


@pytest.mark.parametrize(
    "bad_weights",
    [
        np.diag([1.0, 0.0, 1.0]),
        np.ones((2, 2)),
        np.full((3, 3), np.nan),
        np.array([-1.0, 1.0, 1.0]),
        np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    ],
)
def test_weight_validation(bad_weights: np.ndarray) -> None:
    with pytest.raises(ValueError, match="weights"):
        near_corr(np.eye(3), weights=bad_weights)


def test_weighted_vector_and_matrix_forms_are_equivalent() -> None:
    """Diagonal and rank-one weight matrices encode the same row weights."""
    matrix = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.8], [0.9, -0.8, 1.0]])
    weights = np.array([2.0, 5.0, 1.0])
    vector = asnumpy(near_corr(matrix, weights=weights))
    diagonal = asnumpy(near_corr(matrix, weights=np.diag(weights)))
    outer = asnumpy(near_corr(matrix, weights=np.outer(weights, weights)))
    scaled = asnumpy(near_corr(matrix, weights=3.0 * weights))
    np.testing.assert_allclose(vector, diagonal, atol=1e-8)
    np.testing.assert_allclose(vector, outer, atol=1e-8)
    np.testing.assert_allclose(vector, scaled, atol=1e-8)


def test_weighted_minimum_eigenvalue_floor() -> None:
    """Positive definiteness constraints survive congruence scaling."""
    from pal.correlation import nearest_correlation

    matrix = np.array([[1.0, 0.95, 0.9], [0.95, 1.0, -0.8], [0.9, -0.8, 1.0]])
    result = asnumpy(nearest_correlation(matrix, weights=[1.0, 10.0, 4.0], minimum_eigenvalue=1e-4))
    np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-11)
    assert np.linalg.eigvalsh(result)[0] >= 1e-4 - 1e-8


def test_non_chordal_four_cycle_and_cliques() -> None:
    missing = np.array([[1, 0.4, np.nan, 0.1], [0.4, 1, 0.2, np.nan], [np.nan, 0.2, 1, 0.3], [0.1, np.nan, 0.3, 1]])
    assert not check_incomplete_matrix_is_chordal(missing)
    edges = get_edges_from_incomplete_matrix(missing)
    assert len(edges) == 4
    assert {tuple(c) for c in get_all_cliques(missing)} == {(0, 1), (0, 3), (1, 2), (2, 3)}
    ok, sub, indices = check_all_cliques_in_correlation_matrix(missing)
    assert ok and sub is None and indices is None


def test_detects_infeasible_three_variable_clique() -> None:
    matrix = [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]
    ok, sub, indices = check_all_cliques_in_correlation_matrix(matrix)
    assert not ok
    assert not np.all(np.linalg.eigvalsh(asnumpy(sub)) >= 0)
    assert set(asnumpy(indices).tolist()) == {0, 1, 2}


def test_cluster_result_and_utilities() -> None:
    matrix = np.array([[1, 0.8, 0.2], [0.8, 1, 0.2], [0.2, 0.2, 1]])
    result = cluster_corr(matrix)
    assert set(result.reorder_indices) == {0, 1, 2}
    expected = matrix[np.ix_(result.reorder_indices, result.reorder_indices)]
    np.testing.assert_allclose(asnumpy(result.reordered_matrix), expected)
    labels, members = extract_clusters(result.linkage_matrix, 2)
    assert sum(map(len, members.values())) == 3
    assert len(labels) == 3
    summary = analyze_threshold_ranges(matrix)
    assert summary["min_distance"] <= summary["max_distance"]
    stats = compute_cluster_statistics(matrix, labels)
    assert sum(info["size"] for info in stats.values()) == 3


def test_gpu_backend_residency_for_weighted_repair_and_returned_results() -> None:
    if xp.__name__ != "cupy":
        pytest.skip("GPU-specific test")
    matrix = xp.asarray([[1, 0.9, 0.9], [0.9, 1, -0.8], [0.9, -0.8, 1]], dtype=float)
    weights = xp.asarray([1.0, 3.0, 2.0])
    fixed = near_corr(matrix, weights=weights)
    assert isinstance(fixed, xp.ndarray)
    assert float(xp.linalg.eigvalsh(fixed)[0].item()) >= -1e-7
    incomplete = xp.asarray([[1, xp.nan, xp.nan], [0.5, 1, xp.nan], [0.2, xp.nan, 1]])
    filled = complete_corr(incomplete)
    assert isinstance(filled, xp.ndarray)
    clustered = cluster_corr(filled)
    assert isinstance(clustered.reordered_matrix, xp.ndarray)
