"""Submatrix diagnostic tests ported from Proteus Correlation Toolkit."""

import numpy as np
from pal.correlation import find_non_pd_submatrix
from pal.maths import asnumpy


def test_find_non_pd_submatrix_reference():

    matrix = np.array(
        [[1.0, 0, 0.2, 0.8], [0, 1, 0, 0], [0.2, 0, 1, 0.8], [0.8, 0, 0.8, 1]]
    )
    submatrix, indices = find_non_pd_submatrix(matrix)
    submatrix, indices = asnumpy(submatrix), asnumpy(indices)

    assert np.allclose(
        submatrix, np.array([[1, 0.2, 0.8], [0.2, 1, 0.8], [0.8, 0.8, 1]])
    )
    assert np.allclose(indices, np.array([0, 2, 3]))


def test_find_non_pd_submatrix2():
    matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1.0, -0.15738345, -0.1180793, -0.21020922, -0.04846454],
            [0, -0.15738345, 1.0, -0.00312585, -0.00707493, -0.9],
            [0, -0.1180793, -0.00312585, 1.0, -0.01917419, -0.05],
            [0, -0.21020922, -0.00707493, -0.01917419, 1.0, 0.42679104],
            [0, -0.04846454, -0.9, -0.05, 0.42679104, 1.0],
        ]
    )

    expected_submatrix = np.array(
        [
            [1.0, -0.15738345, -0.1180793, -0.21020922, -0.04846454],
            [-0.15738345, 1.0, -0.00312585, -0.00707493, -0.9],
            [-0.1180793, -0.00312585, 1.0, -0.01917419, -0.05],
            [-0.21020922, -0.00707493, -0.01917419, 1.0, 0.42679104],
            [-0.04846454, -0.9, -0.05, 0.42679104, 1.0],
        ]
    )

    submatrix, indices = find_non_pd_submatrix(matrix)
    submatrix, indices = asnumpy(submatrix), asnumpy(indices)

    assert np.allclose(submatrix, expected_submatrix)
    assert np.allclose(indices, np.array([1, 2, 3, 4, 5]))


def test_random_matrix():
    size = 200
    matrix = np.random.rand(size, size) * 2 - 1
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1)
    submatrix, indices = find_non_pd_submatrix(matrix)
    submatrix, indices = asnumpy(submatrix), asnumpy(indices)
