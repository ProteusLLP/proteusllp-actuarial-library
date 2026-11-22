"""Graph and clique diagnostics for partially specified correlation matrices.

These functions preserve the Proteus Correlation Toolkit utility names,
including its original get_adjacency_matrx spelling. Graph operations use
Python sets; CuPy inputs transfer only a boolean adjacency matrix to CPU.
"""

from __future__ import annotations

import typing as t

import numpy as np
import scipy.linalg

from ._maths import asnumpy, xp
from .correlation import is_psd


def get_adjacency_matrx(incomplete_matrix: t.Any) -> t.Any:
    """Return the symmetric mask of observed entries (lower triangle wins)."""
    matrix = xp.asarray(incomplete_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    result = xp.isfinite(matrix)
    rows, cols = xp.triu_indices(matrix.shape[0], 1)
    result[rows, cols] = result[cols, rows]
    return result


def get_adjacency_matrix(incomplete_matrix: t.Any) -> t.Any:
    """Correctly spelled alias for get_adjacency_matrx."""
    return get_adjacency_matrx(incomplete_matrix)


def get_edges_from_adjacency_matrix(adj_matrix: t.Any) -> list[list[int]]:
    """Return undirected edges between distinct observed variables."""
    mask = asnumpy(adj_matrix).astype(bool, copy=False)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    return [[int(i), int(j)] for i, j in np.argwhere(np.triu(mask, 1))]


def get_edges_from_incomplete_matrix(incomplete_matrix: t.Any) -> list[list[int]]:
    """Return edges connecting pairs with specified correlations."""
    return get_edges_from_adjacency_matrix(get_adjacency_matrx(incomplete_matrix))


def _maximal_cliques(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Bron-Kerbosch enumeration with pivoting and deterministic ordering."""
    result: list[list[int]] = []

    def visit(clique: set[int], candidates: set[int], excluded: set[int]) -> None:
        if not candidates and not excluded:
            result.append(sorted(clique))
            return
        pivot_options = candidates | excluded
        pivot = max(sorted(pivot_options), key=lambda u: len(candidates & adjacency[u])) if pivot_options else None
        available = candidates - adjacency[pivot] if pivot is not None else candidates.copy()
        for vertex in sorted(available):
            neighbours = adjacency[vertex]
            visit(clique | {vertex}, candidates & neighbours, excluded & neighbours)
            candidates.remove(vertex)
            excluded.add(vertex)

    visit(set(), set(adjacency), set())
    return sorted(result)


def _graph_from_edges(edges: t.Iterable[t.Sequence[int]], nodes: t.Iterable[int] = ()) -> dict[int, set[int]]:
    result = {int(node): set() for node in nodes}
    for i, j in edges:
        a, b = int(i), int(j)
        result.setdefault(a, set())
        result.setdefault(b, set())
        if a != b:
            result[a].add(b)
            result[b].add(a)
    return result


def get_all_cliques(incomplete_matrix: t.Any) -> t.Iterator[list[int]]:
    """Iterate maximal cliques of the observed-correlation graph."""
    n = int(xp.asarray(incomplete_matrix).shape[0])
    graph = _graph_from_edges(get_edges_from_incomplete_matrix(incomplete_matrix), range(n))
    return iter(_maximal_cliques(graph))


def get_big_cliques_from_edges(edges: list[list[int]]) -> list[list[int]]:
    """Return maximal cliques of at least three nodes."""
    return [clique for clique in _maximal_cliques(_graph_from_edges(edges)) if len(clique) > 2]


def check_incomplete_matrix_is_chordal(incomplete_matrix: t.Any) -> bool:
    """Check graph chordality by maximum-cardinality search."""
    n = int(xp.asarray(incomplete_matrix).shape[0])
    graph = _graph_from_edges(get_edges_from_incomplete_matrix(incomplete_matrix), range(n))
    unvisited = set(graph)
    weights = dict.fromkeys(graph, 0)
    ordering: list[int] = []
    while unvisited:
        vertex = max(sorted(unvisited), key=lambda u: weights[u])
        unvisited.remove(vertex)
        ordering.append(vertex)
        for neighbour in graph[vertex] & unvisited:
            weights[neighbour] += 1
    visited: set[int] = set()
    for vertex in ordering:
        earlier = graph[vertex] & visited
        for neighbour in earlier:
            if not (earlier - {neighbour}) <= graph[neighbour]:
                return False
        visited.add(vertex)
    return True


def find_non_pd_submatrix(matrix: t.Any, max_iter: int = 1000) -> tuple[t.Any, t.Any]:
    """Find a small non-positive-definite principal submatrix and its indices.

    Uses the first failing leading Cholesky principal minor then tries to
    remove vertices without restoring positive definiteness. The reduction
    is deterministic and bounded by max_iter. Returns (None, None) if the
    input is positive definite.
    """
    raw = asnumpy(matrix, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not np.all(np.isfinite(raw)):
        raise ValueError("matrix must be a finite square matrix.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    raw = (raw + raw.T) / 2
    _, info = scipy.linalg.lapack.dpotrf(raw, lower=1)
    if info == 0:
        return None, None
    indices = list(range(int(info))) if info > 0 else list(range(len(raw)))
    trials = 0
    changed = True
    while changed and len(indices) > 2 and trials < max_iter:
        changed = False
        for index in list(indices):
            if trials >= max_iter:
                break
            trials += 1
            trial = [i for i in indices if i != index]
            if np.linalg.eigvalsh(raw[np.ix_(trial, trial)])[0] <= 0:
                indices = trial
                changed = True
                break
    return xp.asarray(raw[np.ix_(indices, indices)]), xp.asarray(indices)


def check_all_cliques_in_correlation_matrix(
    incomplete_matrix: t.Any,
) -> tuple[bool, t.Any, t.Any]:
    """Find a non-PD observed clique or confirm clique-level feasibility."""
    raw = xp.asarray(incomplete_matrix)
    for clique in get_all_cliques(incomplete_matrix):
        if len(clique) <= 2:
            continue
        sub = raw[xp.ix_(xp.asarray(clique), xp.asarray(clique))]
        if not is_psd(sub):
            minor, indices = find_non_pd_submatrix(sub)
            if indices is None:
                continue
            return False, minor, xp.asarray(clique)[indices]
    return True, None, None


__all__ = [
    "find_non_pd_submatrix",
    "get_adjacency_matrx",
    "get_adjacency_matrix",
    "get_edges_from_adjacency_matrix",
    "get_edges_from_incomplete_matrix",
    "get_all_cliques",
    "get_big_cliques_from_edges",
    "check_all_cliques_in_correlation_matrix",
    "check_incomplete_matrix_is_chordal",
]
