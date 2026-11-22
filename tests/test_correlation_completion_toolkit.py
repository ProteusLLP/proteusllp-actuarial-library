# ruff: noqa: D, I, N, E501
# fmt: off
"""Numerical regression cases ported from Proteus Correlation Toolkit."""

import numpy as np
from numpy import nan
from pal.correlation import complete_corr
from pal.maths import asnumpy
import pytest


def make_symmetric(matrix):
    return np.tril(matrix) + np.triu(matrix.T, 1)


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr(method):
    """this test case is from Example 4.1 of Completion problem with partial correlation
    vines, D. Kurowicka, R.M. Cooke, Linear Algebra and its Applications 405 (2005)
    255-272
    """

    incomplete_matrix = np.array(
        [
            [1, nan, nan, nan, nan, nan],
            [0.7, 1, nan, nan, nan, nan],
            [0.5, 0.3, 1, nan, nan, nan],
            [nan, 0.4, nan, 1, nan, nan],
            [nan, -0.1, nan, -0.2, 1, nan],
            [0.4, -0.2, nan, nan, nan, 1],
        ]
    )

    # note only the lower triangle is required to be filled in

    expected_complete_matrix = np.array(
        [
            [1, nan, nan, nan, nan, nan],
            [0.7, 1, nan, nan, nan, nan],
            [0.5, 0.3, 1, nan, nan, nan],
            [0.28, 0.4, 0.12, 1, nan, nan],
            [-0.07, -0.1, -0.03, -0.2, 1, nan],
            [0.4, -0.2, 0.247, -0.08, 0.02, 1],
        ]
    )

    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))

    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), 1e-3
    )


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr2(method):
    """This example is from Completing Correlation Matrices, Dan Georgescu and
    Nicholas J. Higham,
    https://bankunderground.co.uk/2018/02/01/completing-correlation-matrices/"""
    incomplete_matrix = np.array(
        [
            [1, nan, nan, nan, nan],
            [0.7, 1, nan, nan, nan],
            [0.85, 0.6, 1, nan, nan],
            [0.85, 0.5, nan, 1, nan],
            [0.75, 0.85, nan, 0.75, 1],
        ]
    )
    expected_complete_matrix = np.array(
        [
            [1, nan, nan, nan, nan],
            [0.7, 1, nan, nan, nan],
            [0.85, 0.6, 1, nan, nan],
            [0.85, 0.5, 0.72, 1, nan],
            [0.75, 0.85, 0.64, 0.75, 1],
        ]
    )

    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))

    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), 1e-2
    )


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr3(method):
    """This example is from Explicit solutions to correlation matrix completion
    problems, with an application to risk management and insurance,
    Dan I. Georgescu, Nicholas J. Higham and Gareth W. Peters,
    https://royalsocietypublishing.org/doi/pdf/10.1098/rsos.172348
    """
    incomplete_matrix = np.array(
        [
            [1, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [0.25, 1, nan, nan, nan, nan, nan, nan, nan, nan],
            [0.6, 0, 1, nan, nan, nan, nan, nan, nan, nan],
            [0.55, 0, 0.75, 1, nan, nan, nan, nan, nan, nan],
            [0.65, 0, 0.75, 0.5, 1, nan, nan, nan, nan, nan],
            [0, 0, 0, 0, 0, 1, nan, nan, nan, nan],
            [0.4, nan, nan, nan, nan, nan, 1, nan, nan, nan],
            [0.6, nan, nan, nan, nan, nan, 0.25, 1, nan, nan],
            [0.2, nan, nan, nan, nan, nan, 0.25, 0.25, 1, nan],
            [0.3, nan, nan, nan, nan, nan, 0.5, 0, 0, 1],
        ]
    )
    expected_complete_matrix = np.array(
        [
            [1, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [0.25, 1, nan, nan, nan, nan, nan, nan, nan, nan],
            [0.6, 0, 1, nan, nan, nan, nan, nan, nan, nan],
            [0.55, 0, 0.75, 1, nan, nan, nan, nan, nan, nan],
            [0.65, 0, 0.75, 0.5, 1, nan, nan, nan, nan, nan],
            [0, 0, 0, 0, 0, 1, nan, nan, nan, nan],
            [0.4, 0.1, 0.24, 0.22, 0.26, 0, 1, nan, nan, nan],
            [0.6, 0.15, 0.36, 0.33, 0.39, 0, 0.25, 1, nan, nan],
            [0.2, 0.05, 0.12, 0.11, 0.13, 0, 0.25, 0.25, 1, nan],
            [0.3, 0.075, 0.18, 0.165, 0.195, 0, 0.5, 0, 0, 1],
        ]
    )
    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))

    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), 1e-12
    )


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr4(method):
    """This example if from Complete the Correlation matrix, Kahl, C and Gunther, M,
    2005"""
    incomplete_matrix = np.array(
        [
            [1, nan, nan, nan, nan, nan, nan, nan],
            [0.2, 1, nan, nan, nan, nan, nan, nan],
            [0, 0.4, 1, nan, nan, nan, nan, nan],
            [0.5, 0, 0.6, 1, nan, nan, nan, nan],
            [-0.7, nan, nan, nan, 1, nan, nan, nan],
            [nan, -0.8, nan, nan, nan, 1, nan, nan],
            [
                nan,
                nan,
                -0.9,
                nan,
                nan,
                nan,
                1,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                -0.8,
                nan,
                nan,
                nan,
                1,
            ],
        ]
    )
    expected_complete_matrix = np.array(
        [
            [1, nan, nan, nan, nan, nan, nan, nan],
            [0.2, 1, nan, nan, nan, nan, nan, nan],
            [0, 0.4, 1, nan, nan, nan, nan, nan],
            [0.5, 0, 0.6, 1, nan, nan, nan, nan],
            [-0.7, -0.14, 0, -0.35, 1, nan, nan, nan],
            [-0.16, -0.8, -0.32, 0, 0.112, 1, nan, nan],
            [
                0,
                -0.36,
                -0.9,
                -0.54,
                0,
                0.288,
                1,
                nan,
            ],
            [
                -0.4,
                0,
                -0.48,
                -0.8,
                0.28,
                0,
                0.432,
                1,
            ],
        ]
    )
    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))

    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), 1e-8
    )


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr5(method):
    """Example from https://www.cse.iitb.ac.in/~cs709/notes/code/cvx/examples/cvxbook/Ch04_cvx_opt_probs/html/max_det_psd_completion.html"""  # noqa
    incomplete_matrix = np.array(
        [
            [3, nan, nan, nan],
            [0.5, 2, nan, nan],
            [nan, 0.75, 1, nan],
            [0.25, nan, nan, 5],
        ]
    )
    expected_complete_matrix = np.array(
        [
            [3, nan, nan, nan],
            [0.5, 2, nan, nan],
            [0.1875, 0.75, 1, nan],
            [0.25, 0.0417, 0.0156, 5],
        ]
    )

    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))
    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), atol=1e-4
    )


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr6(method):
    """Example 1 from Section V of Matrix Completion à la Dempster
    by the Principle of Parsimony, Augusto Ferrante and Michele Pavon,IEEE TRANSACTIONS
    ON INFORMATION THEORY, VOL. 57, NO. 6, JUNE 2011 3925,
    https://www.math.unipd.it/~pavon/Site/Publications_files/Matrix_Completion.pdf
    """
    incomplete_matrix = np.array(
        [
            [120 / 929, nan, nan, nan],
            [4 / 929, 124 / 929, nan, nan],
            [-15 / 929, -1 / 1858, 118 / 929, nan],
            [nan, -63 / 1858, 2 / 929, 126 / 929],
        ]
    )
    expected_complete_matrix = np.array(
        [
            [120 / 929, nan, nan, nan],
            [4 / 929, 124 / 929, nan, nan],
            [-15 / 929, -1 / 1858, 118 / 929, nan],
            [-79 / 58527, -63 / 1858, 2 / 929, 126 / 929],
        ]
    )

    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))
    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), atol=1e-12
    )


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr7(method):
    """"""
    incomplete_matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.5, 1, nan, nan],
            [0.5, nan, 1, nan],
            [nan, nan, 0.5, 1],
        ]
    )
    expected_complete_matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.5, 1, nan, nan],
            [0.5, 0.25, 1, nan],
            [0.25, 0.125, 0.5, 1],
        ]
    )

    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))
    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), atol=1e-12
    )


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr8(method):
    incomplete_matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.8, 1, nan, nan],
            [0.8, 0.3, 1, nan],
            [0.8, nan, nan, 1],
        ]
    )
    expected_complete_matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.8, 1, nan, nan],
            [0.8, 0.3, 1, nan],
            [0.8, 0.64, 0.64, 1],
        ]
    )
    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))
    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), atol=1e-12
    )


@pytest.mark.parametrize("method", ["sparse", "dense"])
def test_complete_corr9(method):
    incomplete_matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.5, 1, nan, nan],
            [0.25, nan, 1, nan],
            [nan, nan, nan, 1],
        ]
    )
    expected_complete_matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.5, 1, nan, nan],
            [0.25, 0.125, 1, nan],
            [0.0, 0.0, 0.0, 1],
        ]
    )
    complete_matrix = asnumpy(complete_corr(incomplete_matrix, method=method))
    assert np.allclose(
        np.tril(complete_matrix), np.tril(expected_complete_matrix), atol=1e-12
    )


def test_large():
    n = 100
    incomplete_matrix = np.ones((n, n)) * 0.5 + np.eye(n) * 0.5
    incomplete_matrix[0, 1] = nan
    incomplete_matrix[1, 7] = nan
    incomplete_matrix[2, 9] = nan
    complete_matrix = asnumpy(complete_corr(incomplete_matrix))
    expected_complete_matrix = np.ones((n, n)) * 0.5 + np.eye(n) * 0.5
    assert np.allclose(complete_matrix, expected_complete_matrix, atol=1e-12)


"""
def test_difficult():

    incomplete_matrix = np.array(
        [
            [1, nan, nan, nan, nan, nan],
            [nan, 1, nan, nan, nan, nan],
            [0.8, 0.8, 1, nan, nan, nan],
            [0.2, 0.9, nan, 1, nan, nan],
            [0.2, 0.8, nan, 0.5, 1, nan],
            [0.2, 0.2, 0.5, nan, 0.5, 1],
        ]
    )
    complete_matrix = complete_corr(incomplete_matrix, method="sparse")


test_difficult()"""
