from pal.variables import StochasticScalar
from pal.copulas import apply_copula, GumbelCopula
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel
from pal.distributions import GPD, Poisson, Normal
import numpy as np
import scipy


def test_fs_reordering():
    x = FreqSevSims([0, 0, 1, 2, 4], [10, 21, 30, 40, 50], 6)
    y = FreqSevSims([0, 1, 1, 3, 5], [12, 22, 32, 42, 52], 6)
    a = x.aggregate()  # a is [31,30,40,0,50,0]
    b = y.aggregate()  # b is [12,54,0,42,0,52]
    copula_samples = [
        StochasticScalar([0, 2, 1, 5, 4, 3]),
        StochasticScalar([1, 0, 3, 2, 5, 4]),
    ]
    apply_copula([a, b], copula_samples)
    assert (a.values == [31, 30, 40, 0, 50, 0]).all()
    assert (b.values == [52, 0, 54, 0, 12, 42]).all()
    # expected_x
    expected_x = FreqSevSims([0, 0, 1, 2, 4], [10, 21, 30, 40, 50], 6)
    # expected_y
    expected_y = FreqSevSims([4, 2, 2, 5, 0], [12, 22, 32, 42, 52], 6)

    assert (expected_x.sim_index == x.sim_index).all()
    assert (expected_x.values == x.values).all()
    assert (expected_y.sim_index == y.sim_index).all()
    assert (expected_y.values == y.values).all()


def test_fs_reordering2():
    """Tests that other variables attached to the FreqSevSims are correctly reordered."""
    x = FreqSevSims([0, 0, 1, 2, 4], [10, 21, 30, 40, 50], 6)
    y = FreqSevSims([0, 1, 1, 3, 5], [12, 22, 32, 42, 52], 6)
    x1 = x * 2
    y1 = y * 3
    a = x1.aggregate()  # a is [31,30,40,0,50,0]*2
    b = y1.aggregate()  # b is [12,54,0,42,0,52]*3
    copula_samples = [
        StochasticScalar([0, 2, 1, 5, 4, 3]),
        StochasticScalar([1, 0, 3, 2, 5, 4]),
    ]
    apply_copula([a, b], copula_samples)
    assert (a.values == [62, 60, 80, 0, 100, 0]).all()
    assert (b.values == [156, 0, 162, 0, 36, 126]).all()
    # expected_x
    expected_x = FreqSevSims([0, 0, 1, 2, 4], [10, 21, 30, 40, 50], 6)
    # expected_y
    expected_y = FreqSevSims([4, 2, 2, 5, 0], [12, 22, 32, 42, 52], 6)

    assert (expected_x.sim_index == x.sim_index).all()
    assert (expected_x.values == x.values).all()
    assert (expected_y.sim_index == y.sim_index).all()
    assert (expected_y.values == y.values).all()

    # aggregate x
    x_agg = x.aggregate()
    y_agg = y.aggregate()
    expected_x_agg = StochasticScalar([31, 30, 40, 0, 50, 0])
    expected_y_agg = StochasticScalar([52, 0, 54, 0, 12, 42])
    assert (x_agg == expected_x_agg).values.all()
    assert (y_agg == expected_y_agg).values.all()


def test_fs_reordering3():
    """Tests that other variables attached to the FreqSevSims are correctly reordered."""
    x = FrequencySeverityModel(
        Poisson(mean=2), GPD(shape=0.33, scale=100000, loc=0)
    ).generate()
    y = FrequencySeverityModel(
        Poisson(mean=2), GPD(shape=0.33, scale=100000, loc=0)
    ).generate()
    x1 = x * 2
    y1 = y * 3
    a = x1.aggregate()
    b = y1.aggregate()
    GumbelCopula(1.5, 2).apply([a, b])
    # check the copula has been applied correctly
    calculated_tau = scipy.stats.kendalltau(a.values, b.values).statistic
    assert np.isclose(calculated_tau, 1 - 1 / 1.5, atol=1e-2)
    # check that when x and y are re_calculated and reaggregated, they give the same result
    re_calculated_a = (x * 2).aggregate()
    re_calculated_b = (y * 3).aggregate()
    assert np.allclose(re_calculated_a.values, a.values, atol=1e-10)
    assert np.allclose(re_calculated_b.values, b.values, atol=1e-10)
