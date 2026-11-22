"""Proteus Correlation Toolkit numerical regression tests, adapted to PAL."""

import numpy as np
from numpy import nan

from pal.correlation import is_psd, near_corr
from pal.maths import asnumpy


def make_symmetric(matrix):
    matrix[np.triu_indices(len(matrix))] = matrix.T[np.triu_indices(len(matrix))]


def test_near_corr():
    matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.8, 1, nan, nan],
            [0.8, 0.3, 1, nan],
            [0.8, 0.5, 0.5, 1],
        ]
    )
    make_symmetric(matrix)
    pattern = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 3)]
    near_corr_matrix = near_corr(matrix, pattern=pattern)
    expected_near_corr_matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.8, 1, nan, nan],
            [0.8, 0.3, 1, nan],
            [0.8, 0.58, 0.58, 1],
        ]
    )

    make_symmetric(expected_near_corr_matrix)
    assert np.allclose(asnumpy(near_corr_matrix), expected_near_corr_matrix, atol=1e-12)


def test_example1():
    """From section 4 of Higham."""
    matrix = np.array([[2.0, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
    expected_result = np.array(
        [
            [1.0000, -0.8084, 0.1916, 0.1068],
            [-0.8084, 1.0000, -0.6562, 0.1916],
            [0.1916, -0.6562, 1.0000, -0.8084],
            [0.1068, 0.1916, -0.8084, 1.000],
        ]
    )
    assert np.allclose(asnumpy(near_corr(matrix)), expected_result, atol=1e-4)


def test_example2():
    """From section 4 of Higham."""
    matrix = np.array([[1.0, 1, 0], [1, 1, 1], [0, 1, 1]])

    expected_result = np.array(
        [
            [1.0, 0.76068985, 0.15729811],
            [0.76068985, 1.0, 0.76068985],
            [0.15729811, 0.76068985, 1.0],
        ]
    )
    assert np.allclose(asnumpy(near_corr(matrix)), expected_result, atol=1e-8)


def test_example3():
    """Test reference example from Lucas (2001), Section 4.3."""
    matrix = np.array(
        [
            [1.0000, -0.3250, 0.1881, 0.5760, 0.0064, -0.6111, -0.0724, -0.1589],
            [-0.3250, 1.0000, 0.2048, 0.2436, 0.4058, 0.2730, 0.2869, 0.4241],
            [0.1881, 0.2048, 1.0000, -0.1325, 0.7658, 0.2765, -0.6172, 0.9006],
            [0.5760, 0.2436, -0.1325, 1.0000, 0.3041, 0.0126, 0.6452, -0.3210],
            [0.0064, 0.4058, 0.7658, 0.3041, 1.0000, 0.6652, -0.3293, 0.9939],
            [-0.6111, 0.2730, 0.2765, 0.0126, 0.6652, 1.0000, 0.0492, 0.5964],
            [-0.0724, 0.2869, -0.6172, 0.6452, -0.3293, 0.0492, 1.0000, -0.3983],
            [-0.1589, 0.4241, 0.9006, -0.3210, 0.9939, 0.5964, -0.3982, 1.00000],
        ]
    )

    expected_result = np.array(
        [
            [1.0000, -0.3112, 0.1889, 0.5396, 0.0268, -0.5925, -0.0621, -0.1921],
            [-0.3112, 1.0000, 0.2050, 0.2265, 0.4148, 0.2822, 0.2915, 0.4088],
            [0.1889, 0.2050, 1.0000, -0.1468, 0.7880, 0.2727, -0.6085, 0.8802],
            [0.5396, 0.2265, -0.1468, 1.0000, 0.2137, 0.0015, 0.6069, -0.2208],
            [0.0268, 0.4148, 0.7880, 0.2137, 1.0000, 0.6580, -0.2812, 0.8762],
            [-0.5925, 0.2822, 0.2727, 0.0015, 0.6580, 1.0000, 0.0479, 0.5932],
            [-0.0621, 0.2915, -0.6085, 0.6069, -0.2812, 0.0479, 1.0000, -0.4470],
            [-0.1921, 0.4088, 0.8802, -0.2208, 0.8762, 0.5932, -0.4470, 1.0000],
        ]
    )
    assert np.allclose(asnumpy(near_corr(matrix)), expected_result, atol=1e-4)


def test_example4():
    """Test reference example from Lucas (2001), Section 4.3."""
    matrix = np.array(
        [
            [1.0000, -0.3250, 0.1881, 0.5760, 0.0064, -0.6111, -0.0724, -0.1589],
            [-0.3250, 1.0000, 0.2048, 0.2436, 0.4058, 0.2730, 0.2869, 0.4241],
            [0.1881, 0.2048, 1.0000, -0.1325, 0.7658, 0.2765, -0.6172, 0.9006],
            [0.5760, 0.2436, -0.1325, 1.0000, 0.3041, 0.0126, 0.6452, -0.3210],
            [0.0064, 0.4058, 0.7658, 0.3041, 1.0000, 0.6652, -0.3293, 0.9939],
            [-0.6111, 0.2730, 0.2765, 0.0126, 0.6652, 1.0000, 0.0492, 0.5964],
            [-0.0724, 0.2869, -0.6172, 0.6452, -0.3293, 0.0492, 1.0000, -0.3983],
            [-0.1589, 0.4241, 0.9006, -0.3210, 0.9939, 0.5964, -0.3982, 1.00000],
        ]
    )

    pattern = [(i, i) for i in range(0, len(matrix))] + [(0, 1), (0, 2), (1, 2)]

    expected_result = np.array(
        [
            [1.0000, -0.3250, 0.1881, 0.5375, 0.0258, -0.5899, -0.0625, -0.1927],
            [-0.3250, 1.0000, 0.2048, 0.2251, 0.4145, 0.2838, 0.2914, 0.4081],
            [0.1881, 0.2048, 1.0000, -0.1462, 0.7882, 0.2720, -0.6084, 0.8805],
            [0.5375, 0.2251, -0.1462, 1.0000, 0.2141, 0.0001, 0.6071, -0.2203],
            [0.0258, 0.4145, 0.7882, 0.2141, 1.0000, 0.6570, -0.2810, 0.8762],
            [-0.5899, 0.2838, 0.2720, 0.0001, 0.6570, 1.0000, 0.0475, 0.5929],
            [-0.0625, 0.2914, -0.6084, 0.6071, -0.2810, 0.0475, 1.0000, -0.4469],
            [-0.1927, 0.4081, 0.8805, -0.2203, 0.8762, 0.5929, -0.4469, 1.0000],
        ]
    )
    assert np.allclose(asnumpy(near_corr(matrix, pattern)), expected_result, atol=1e-4)


def test_psd():
    matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
    expected_result = matrix
    assert np.allclose(asnumpy(near_corr(matrix)), expected_result, atol=1e-16)


def test_ispsd():
    matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
    assert is_psd(matrix)


def test_isnotpsd():
    matrix = np.array([[1.0, -0.5, 0.9], [-0.5, 1, 0.7], [0.9, 0.7, 1]])
    assert not is_psd(matrix)


def test_near_corr_with_fixed_entries():
    matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.8, 1, nan, nan],
            [0.8, 0.3, 1, nan],
            [0.8, 0.5, 0.5, 1],
        ]
    )
    make_symmetric(matrix)
    pattern = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 3)]
    near_corr_matrix = near_corr(matrix, pattern=pattern)
    expected_near_corr_matrix = np.array(
        [
            [1, nan, nan, nan],
            [0.8, 1, nan, nan],
            [0.8, 0.3, 1, nan],
            [0.8, 0.5, 0.6632144, 1],
        ]
    )

    make_symmetric(expected_near_corr_matrix)
    assert np.allclose(asnumpy(near_corr_matrix), expected_near_corr_matrix, atol=1e-12)
