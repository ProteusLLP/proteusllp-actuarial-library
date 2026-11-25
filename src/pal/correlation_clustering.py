"""Hierarchical clustering of correlation matrices.

SciPy hierarchy calculation is CPU-based. Reordering uses the PAL array
backend, preserving CuPy residency for the reordered correlation matrix.
"""

from __future__ import annotations

import typing as t
from typing import NamedTuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from ._maths import asnumpy, xp


class ClusteringResult(NamedTuple):
    """Reordered correlation matrix, SciPy linkage, and variable permutation.

    Attributes:
        linkage_matrix: Hierarchical clustering linkage matrix describing the successive merges.
        reorder_indices: Permutation of the original variable indices into dendrogram leaf order.
        reordered_matrix: Correlation matrix with its rows and columns reordered together.
    """

    reordered_matrix: t.Any
    linkage_matrix: np.ndarray
    reorder_indices: list[int]


def _check(matrix: t.Any) -> t.Any:
    """Symmetrise a complete correlation matrix and reset its diagonal.

    Args:
        matrix: Matrix to validate, complete or use in the correlation calculation.
    """
    values = xp.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("Correlation matrix must be square and at least 2x2.")
    values = (values + values.T) / 2
    if not bool(xp.all(xp.isfinite(values))):
        raise ValueError("Correlation matrix must contain finite values.")
    values = values.copy()
    idx = xp.arange(values.shape[0])
    values[idx, idx] = 1.0
    if bool(xp.any((values < -1) | (values > 1))):
        raise ValueError("Correlation values must be between -1 and 1.")
    return values


def _distance(matrix: np.ndarray, metric: str) -> np.ndarray:
    """Return condensed symmetric dissimilarity matrix.

    Args:
        matrix: Matrix to validate, complete or use in the correlation calculation.
        metric: Distance metric used to compare correlation profiles.
    """
    if metric == "correlation":
        values = 1.0 - np.abs(matrix)
    elif metric == "euclidean":
        squared = np.sum(matrix**2, axis=1)
        values = np.sqrt(np.maximum(0, squared[:, None] + squared[None, :] - 2 * matrix @ matrix.T))
    elif metric == "cosine":
        normalised = matrix / np.linalg.norm(matrix, axis=1)[:, None]
        values = 1.0 - normalised @ normalised.T
    else:
        raise ValueError("metric must be 'correlation', 'euclidean', or 'cosine'.")
    values = (values + values.T) / 2
    np.fill_diagonal(values, 0.0)
    return np.maximum(values, 0.0)


def _tree(matrix: t.Any, method: str, metric: str, optimal_ordering: bool) -> np.ndarray:
    """Calculate linkage on host using SciPy.

    Args:
        matrix: Matrix to validate, complete or use in the correlation calculation.
        method: Algorithm selected for the correlation or clustering calculation.
        metric: Distance metric used to compare correlation profiles.
        optimal_ordering: Whether to minimise adjacent leaf distances in the dendrogram.
    """
    if method not in ("ward", "complete", "average", "single"):
        raise ValueError("method must be 'ward', 'complete', 'average', or 'single'.")
    distance = _distance(asnumpy(matrix), metric)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method=method)
    if optimal_ordering and method in ("ward", "complete", "average"):
        tree = optimal_leaf_ordering(tree, condensed)
    return tree


def cluster_corr(
    correlation_matrix: t.Any,
    method: str = "ward",
    metric: str = "euclidean",
    optimal_ordering: bool = True,
    distance_threshold: float | None = None,
    correlation_threshold: float | None = None,
) -> ClusteringResult:
    """Cluster correlated variables and return their reordered matrix.

    Preserves the Proteus Correlation Toolkit call signature. The linkage
    calculation and metadata use CPU SciPy; the reordered matrix remains on
    PAL's active NumPy/CuPy backend.

    Args:
        correlation_matrix: Correlation matrix with one row and column per variable.
        correlation_threshold: Correlation threshold used when cutting the clustering tree.
        distance_threshold: Maximum linkage distance at which clusters are joined.
        method: Algorithm selected for the correlation or clustering calculation.
        metric: Distance metric used to compare correlation profiles.
        optimal_ordering: Whether to minimise adjacent leaf distances in the dendrogram.
    """
    matrix = _check(correlation_matrix)
    if distance_threshold is not None and not np.isfinite(distance_threshold):
        raise ValueError("distance_threshold must be finite.")
    if correlation_threshold is not None and not np.isfinite(correlation_threshold):
        raise ValueError("correlation_threshold must be finite.")
    hierarchy = _tree(matrix, method, metric, optimal_ordering)
    threshold = (1 - correlation_threshold) if correlation_threshold is not None else distance_threshold
    if threshold is None:
        indices = leaves_list(hierarchy).tolist()
    else:
        groups = fcluster(hierarchy, threshold, criterion="distance")
        indices = [i for group in np.unique(groups) for i in np.flatnonzero(groups == group)]
    indices = [int(i) for i in indices]
    return ClusteringResult(
        reordered_matrix=matrix[xp.ix_(xp.asarray(indices), xp.asarray(indices))],
        linkage_matrix=hierarchy,
        reorder_indices=indices,
    )


def analyze_threshold_ranges(
    correlation_matrix: t.Any,
    metric: str = "euclidean",
    method: str = "ward",
    num_samples: int = 20,
) -> dict[str, t.Any]:
    """Summarise the range of linkage distances and suggested cluster cuts.

    Args:
        correlation_matrix: Correlation matrix with one row and column per variable.
        method: Algorithm selected for the correlation or clustering calculation.
        metric: Distance metric used to compare correlation profiles.
        num_samples: Number of threshold levels to examine.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    hierarchy = _tree(_check(correlation_matrix), method, metric, False)
    distances = hierarchy[:, 2]
    low = float(np.percentile(distances, 25))
    high = float(np.percentile(distances, 75))
    a = fcluster(hierarchy, low, criterion="distance")
    b = fcluster(hierarchy, high, criterion="distance")
    n_low = len(np.unique(a))
    n_high = len(np.unique(b))
    return {
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "suggested_min": low,
        "suggested_max": high,
        "optimal_range": f"{low:.3f} - {high:.3f}",
        "num_clusters_at_min": n_low,
        "num_clusters_at_max": n_high,
        "guidance": f"Try values between {low:.3f} and {high:.3f} for {n_high}-{n_low} clusters",
    }


def extract_clusters(linkage_matrix: t.Any, n_clusters: int) -> tuple[list[int], dict[int, list[int]]]:
    """Extract labels and variable indices for a requested number of clusters.

    Args:
        linkage_matrix: Hierarchical clustering linkage matrix.
        n_clusters: Requested number of clusters.
    """
    labels = [int(x) for x in fcluster(asnumpy(linkage_matrix), n_clusters, criterion="maxclust")]
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return labels, groups


def compute_cluster_statistics(
    correlation_matrix: t.Any,
    cluster_assignments: list[int],
) -> dict[int, dict[str, t.Any]]:
    """Summarise within-cluster and between-cluster absolute correlations.

    Args:
        cluster_assignments: Cluster label associated with each variable.
        correlation_matrix: Correlation matrix with one row and column per variable.
    """
    matrix = asnumpy(correlation_matrix)
    if len(cluster_assignments) != len(matrix):
        raise ValueError("cluster_assignments must match matrix dimension.")
    result: dict[int, dict[str, t.Any]] = {}
    for label in sorted(set(cluster_assignments)):
        inside = [i for i, value in enumerate(cluster_assignments) if value == label]
        outside = [i for i, value in enumerate(cluster_assignments) if value != label]
        within = np.abs(matrix[np.ix_(inside, inside)])
        between = np.abs(matrix[np.ix_(inside, outside)]) if outside else np.array([])
        a = float((np.sum(within) - len(inside)) / (len(inside) * (len(inside) - 1))) if len(inside) > 1 else 1.0
        b = float(np.mean(between)) if between.size else 0.0
        result[label] = {
            "size": len(inside),
            "avg_intra_correlation": a,
            "avg_inter_correlation": b,
            "cohesion": a - b if b > 0 else a,
            "variable_indices": inside,
        }
    return result


__all__ = [
    "ClusteringResult",
    "cluster_corr",
    "analyze_threshold_ranges",
    "extract_clusters",
    "compute_cluster_statistics",
]
