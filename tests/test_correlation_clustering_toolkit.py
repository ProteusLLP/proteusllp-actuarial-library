# ruff: noqa: D, I, N, E501
# fmt: off
"""
Comprehensive clustering tests - systematic coverage of all clustering functionality.

This test suite provides complete coverage of the clustering module, including:
- All 10 valid method+metric combinations
- Threshold features (distance and correlation)
- ClusteringResult structure validation
- Mathematical accuracy and edge cases
- Error handling and parameter validation
"""

import numpy as np
import pytest
from typing import Any

from pal.correlation import ClusteringResult, cluster_corr
from pal.correlation import is_psd
from pal._maths import xp

pytestmark = pytest.mark.skipif(xp.__name__ == "cupy", reason="toolkit reference expects NumPy arrays")

# Type alias for numpy arrays in tests
NDArray = np.ndarray[Any, np.dtype[Any]]


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def simple_3x3_matrix() -> NDArray:
    """Simple 3x3 correlation matrix for basic testing."""
    return np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])


@pytest.fixture
def block_correlation_matrix() -> NDArray:
    """4x4 matrix with clear block structure for clustering validation."""
    return np.array([
        [1.0, 0.9, 0.2, 0.1],  # Block 1: variables 0,1
        [0.9, 1.0, 0.1, 0.2],
        [0.2, 0.1, 1.0, 0.8],  # Block 2: variables 2,3
        [0.1, 0.2, 0.8, 1.0]
    ])


@pytest.fixture
def identity_matrix() -> NDArray:
    """Identity matrix (perfect clustering case)."""
    return np.eye(3)


@pytest.fixture
def perfect_correlation_matrix() -> NDArray:
    """Matrix with perfect correlations (edge case)."""
    return np.array([
        [1.0, 1.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ])


# =============================================================================
# 1. BASIC FUNCTIONALITY TESTS (10 tests)
# =============================================================================

@pytest.mark.parametrize("method,metric", [
    ("ward", "euclidean"),
    ("complete", "euclidean"), ("complete", "correlation"), ("complete", "cosine"),
    ("average", "euclidean"), ("average", "correlation"), ("average", "cosine"),
    ("single", "euclidean"), ("single", "correlation"), ("single", "cosine"),
    ("ward", "correlation"), ("ward", "cosine")
])
def test_all_method_metric_combinations(
    simple_3x3_matrix: NDArray, method: str, metric: str
) -> None:
    """Test all valid method+metric combinations work correctly."""
    result = cluster_corr(simple_3x3_matrix, method=method, metric=metric)

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3
    assert is_psd(result.reordered_matrix)


def test_ward_euclidean_basic(simple_3x3_matrix: NDArray) -> None:
    """Test ward+euclidean combination (required constraint)."""
    result = cluster_corr(simple_3x3_matrix, method='ward', metric='euclidean')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)  # n-1 x 4 for linkage
    assert len(result.reorder_indices) == 3
    assert is_psd(result.reordered_matrix)


def test_complete_euclidean_basic(simple_3x3_matrix: NDArray) -> None:
    """Test complete+euclidean combination."""
    result = cluster_corr(simple_3x3_matrix, method='complete', metric='euclidean')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_complete_correlation_basic(simple_3x3_matrix: NDArray) -> None:
    """Test complete+correlation combination."""
    result = cluster_corr(simple_3x3_matrix, method='complete', metric='correlation')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_complete_cosine_basic(simple_3x3_matrix: NDArray) -> None:
    """Test complete+cosine combination."""
    result = cluster_corr(simple_3x3_matrix, method='complete', metric='cosine')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_average_euclidean_basic(simple_3x3_matrix: NDArray) -> None:
    """Test average+euclidean combination."""
    result = cluster_corr(simple_3x3_matrix, method='average', metric='euclidean')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_single_euclidean_basic(simple_3x3_matrix: NDArray) -> None:
    """Test single+euclidean combination."""
    result = cluster_corr(simple_3x3_matrix, method='single', metric='euclidean')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_average_correlation_basic(simple_3x3_matrix: NDArray) -> None:
    """Test average+correlation combination."""
    result = cluster_corr(simple_3x3_matrix, method='average', metric='correlation')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_average_cosine_basic(simple_3x3_matrix: NDArray) -> None:
    """Test average+cosine combination."""
    result = cluster_corr(simple_3x3_matrix, method='average', metric='cosine')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_single_correlation_basic(simple_3x3_matrix: NDArray) -> None:
    """Test single+correlation combination."""
    result = cluster_corr(simple_3x3_matrix, method='single', metric='correlation')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_single_cosine_basic(simple_3x3_matrix: NDArray) -> None:
    """Test single+cosine combination."""
    result = cluster_corr(simple_3x3_matrix, method='single', metric='cosine')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


# =============================================================================
# 2. PARAMETER VALIDATION TESTS
# =============================================================================

def test_ward_correlation_basic(simple_3x3_matrix: NDArray) -> None:
    """Test ward method with correlation metric works (not a constraint violation)."""
    result = cluster_corr(simple_3x3_matrix, method='ward', metric='correlation')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_ward_cosine_basic(simple_3x3_matrix: NDArray) -> None:
    """Test ward method with cosine metric works (not a constraint violation)."""
    result = cluster_corr(simple_3x3_matrix, method='ward', metric='cosine')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert result.linkage_matrix.shape == (2, 4)
    assert len(result.reorder_indices) == 3


def test_invalid_method(simple_3x3_matrix: NDArray) -> None:
    """Test invalid clustering method."""
    with pytest.raises((ValueError, RuntimeError)):
        cluster_corr(simple_3x3_matrix, method='invalid_method')


def test_invalid_metric(simple_3x3_matrix: NDArray) -> None:
    """Test invalid distance metric."""
    with pytest.raises((ValueError, RuntimeError)):
        cluster_corr(simple_3x3_matrix, method='complete', metric='invalid_metric')


def test_non_square_matrix() -> None:
    """Test non-square matrix validation."""
    non_square = np.array([[1.0, 0.8], [0.8, 1.0], [0.6, 0.4]])

    with pytest.raises(ValueError):
        cluster_corr(non_square)


def test_empty_matrix() -> None:
    """Test empty matrix validation."""
    empty_matrix = np.array([])

    with pytest.raises(ValueError):
        cluster_corr(empty_matrix)


def test_non_symmetric_matrix() -> None:
    """Test that non-symmetric matrix still works (function is permissive)."""
    non_symmetric = np.array([
        [1.0, 0.8, 0.3],
        [0.7, 1.0, 0.5],  # Different from symmetric position
        [0.3, 0.5, 1.0]
    ])

    # Function works with non-symmetric matrices too
    result = cluster_corr(non_symmetric)
    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert len(result.reorder_indices) == 3


# =============================================================================
# 3. THRESHOLD TESTS
# =============================================================================

def test_distance_threshold_only(block_correlation_matrix: NDArray) -> None:
    """Test clustering with distance threshold only."""
    result = cluster_corr(
        block_correlation_matrix,
        method='complete',
        metric='euclidean',
        distance_threshold=0.5
    )

    assert isinstance(result, ClusteringResult)
    assert len(result.reorder_indices) == 4


def test_correlation_threshold_only(block_correlation_matrix: NDArray) -> None:
    """Test clustering with correlation threshold only."""
    result = cluster_corr(
        block_correlation_matrix,
        method='complete',
        metric='correlation',
        correlation_threshold=0.7
    )

    assert isinstance(result, ClusteringResult)
    assert len(result.reorder_indices) == 4


def test_both_thresholds(block_correlation_matrix: NDArray) -> None:
    """Test clustering with both distance and correlation thresholds."""
    result = cluster_corr(
        block_correlation_matrix,
        method='complete',
        metric='correlation',
        distance_threshold=0.5,
        correlation_threshold=0.7
    )

    assert isinstance(result, ClusteringResult)
    assert len(result.reorder_indices) == 4


# =============================================================================
# 4. EDGE CASE TESTS
# =============================================================================

def test_identity_matrix_clustering(identity_matrix: NDArray) -> None:
    """Test clustering behavior with identity matrix."""
    result = cluster_corr(identity_matrix, method='complete', metric='euclidean')

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert len(result.reorder_indices) == 3
    # Identity matrix should remain positive semidefinite
    assert is_psd(result.reordered_matrix)


def test_perfect_correlation_clustering(perfect_correlation_matrix: NDArray) -> None:
    """Test clustering with perfect correlations (edge case)."""
    result = cluster_corr(
        perfect_correlation_matrix, method='complete', metric='correlation'
    )

    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert len(result.reorder_indices) == 3


# =============================================================================
# 5. CLUSTERING RESULT VALIDATION TESTS
# =============================================================================

def test_clustering_result_field_types(simple_3x3_matrix: NDArray) -> None:
    """Test that ClusteringResult fields have correct types."""
    result = cluster_corr(simple_3x3_matrix, method='ward', metric='euclidean')

    # Test field types
    assert isinstance(result.reordered_matrix, np.ndarray)
    assert isinstance(result.linkage_matrix, np.ndarray)
    assert isinstance(result.reorder_indices, list)
    assert all(isinstance(idx, (int, np.integer)) for idx in result.reorder_indices)


def test_clustering_result_matrix_consistency(simple_3x3_matrix: NDArray) -> None:
    """Test that reordered matrix matches manual reordering."""
    result = cluster_corr(simple_3x3_matrix, method='ward', metric='euclidean')

    # Manual reordering should match result
    manual_reorder: NDArray = simple_3x3_matrix[
        np.ix_(result.reorder_indices, result.reorder_indices)
    ]

    np.testing.assert_array_almost_equal(result.reordered_matrix, manual_reorder)


def test_clustering_result_indices_validity(simple_3x3_matrix: NDArray) -> None:
    """Test that reorder indices are valid permutation."""
    result = cluster_corr(simple_3x3_matrix, method='ward', metric='euclidean')

    n: int = simple_3x3_matrix.shape[0]

    # Should be permutation of [0, 1, ..., n-1]
    assert len(result.reorder_indices) == n
    assert set(result.reorder_indices) == set(range(n))
    assert len(set(result.reorder_indices)) == n  # No duplicates


def test_clustering_result_reproducibility(simple_3x3_matrix: NDArray) -> None:
    """Test that clustering results are reproducible."""
    result1 = cluster_corr(simple_3x3_matrix, method='ward', metric='euclidean')
    result2 = cluster_corr(simple_3x3_matrix, method='ward', metric='euclidean')

    # Results should be identical
    np.testing.assert_array_equal(result1.reordered_matrix, result2.reordered_matrix)
    np.testing.assert_array_equal(result1.linkage_matrix, result2.linkage_matrix)
    assert result1.reorder_indices == result2.reorder_indices


# =============================================================================
# 6. MATHEMATICAL PROPERTY TESTS
# =============================================================================

def test_correlation_metric_mathematical_properties(simple_3x3_matrix: NDArray) -> None:
    """Test mathematical properties of correlation distance metric."""
    result = cluster_corr(simple_3x3_matrix, method='complete', metric='correlation')

    # Reordered matrix should preserve correlation properties
    assert np.allclose(np.diag(result.reordered_matrix), 1.0)  # Diagonal should be 1
    # Should be symmetric
    assert np.allclose(result.reordered_matrix, result.reordered_matrix.T)
    assert is_psd(result.reordered_matrix)  # Should remain PSD


def test_euclidean_metric_mathematical_properties(simple_3x3_matrix: NDArray) -> None:
    """Test mathematical properties of euclidean distance metric."""
    result = cluster_corr(simple_3x3_matrix, method='complete', metric='euclidean')

    # Correlation matrix properties should be preserved
    original_values: NDArray = np.sort(simple_3x3_matrix.flatten())
    reordered_values: NDArray = np.sort(result.reordered_matrix.flatten())

    # Same values, just reordered
    np.testing.assert_array_almost_equal(original_values, reordered_values)
    assert is_psd(result.reordered_matrix)


def test_clustering_preserves_matrix_size(simple_3x3_matrix: NDArray) -> None:
    """Test that clustering preserves matrix dimensions."""
    result = cluster_corr(simple_3x3_matrix, method='complete', metric='euclidean')

    assert result.reordered_matrix.shape == simple_3x3_matrix.shape
    assert result.linkage_matrix.shape[0] == simple_3x3_matrix.shape[0] - 1
    assert result.linkage_matrix.shape[1] == 4  # Standard linkage matrix format


def test_known_matrix_clustering_result() -> None:
    """Test clustering with known matrix and verify specific expected outcome."""
    # Use a well-defined correlation matrix with known clustering structure
    known_matrix = np.array([
        [1.0, 0.9, 0.1],  # Variables 0,1 should cluster together
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0]   # Variable 2 should be separate
    ])

    result = cluster_corr(known_matrix, method='complete', metric='correlation')

    # Verify basic structure
    assert isinstance(result, ClusteringResult)
    assert result.reordered_matrix.shape == (3, 3)
    assert len(result.reorder_indices) == 3

    # Variables 0 and 1 should be adjacent in the reordered matrix (first cluster)
    reorder_pos_0 = result.reorder_indices.index(0)
    reorder_pos_1 = result.reorder_indices.index(1)

    # They should be adjacent (difference of 1) due to high correlation
    assert abs(reorder_pos_0 - reorder_pos_1) == 1

    # Matrix should remain PSD and correlation-like
    assert is_psd(result.reordered_matrix)
    assert np.allclose(np.diag(result.reordered_matrix), 1.0)  # Diagonal = 1
