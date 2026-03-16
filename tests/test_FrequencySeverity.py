"""Tests for frequency-severity simulation functionality.

Comprehensive tests for FreqSevSims class covering aggregation operations,
arithmetic operations, and integration with other stochastic types.
"""

import numpy as np
import pal.maths as pnp
from pal.frequency_severity import FreqSevSims
from pal.variables import StochasticScalar


def test_aggregate():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    assert np.array_equal(fs.aggregate(), np.array([3.0, 12.0, 30.0]))


def test_occurrence():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    assert np.array_equal(fs.occurrence(), np.array([2.0, 5.0, 9.0]))


def test_copy():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    fs_copy = fs.copy()
    assert np.array_equal(fs_copy.sim_index, fs.sim_index)
    assert np.array_equal(fs_copy, fs)
    assert fs_copy.n_sims == fs.n_sims


def test_apply():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index, values, n_sims)
    fs_squared = fs.apply(lambda x: x**2)
    assert np.array_equal(fs_squared, np.array([1, 4, 9, 16, 25, 36, 49, 64, 81]))


def test_math_operations():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs1 = FreqSevSims(sim_index, values, n_sims)
    fs2 = FreqSevSims(sim_index, values, n_sims)
    fs_sum = fs1 + fs2
    assert np.array_equal(fs_sum, np.array([2, 4, 6, 8, 10, 12, 14, 16, 18]))
    fs_plus_one = fs1 + 1
    assert np.array_equal(fs_plus_one, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
    fs_times_five = fs1 * 5
    assert np.array_equal(fs_times_five, np.array([5, 10, 15, 20, 25, 30, 35, 40, 45]))
    fs1_less1 = -1 + fs1
    assert np.array_equal(fs1_less1, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
    fs_divide2 = fs1 / 2
    assert np.array_equal(fs_divide2, np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]))
    fs_squared = fs1**2
    assert np.array_equal(fs_squared, np.array([1, 4, 9, 16, 25, 36, 49, 64, 81]))
    fs1_times_fs2 = fs1 * fs2
    assert np.array_equal(fs1_times_fs2, np.array([1, 4, 9, 16, 25, 36, 49, 64, 81]))
    fs1_divide_fs2 = fs1 / fs2
    assert np.array_equal(fs1_divide_fs2, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    two_to_power_fs1 = 2**fs1
    assert np.array_equal(two_to_power_fs1, np.array([2, 4, 8, 16, 32, 64, 128, 256, 512]))
    # test with numpy array
    arr = np.array([1, 2, 3])
    fs1_times_arr = fs1 * arr
    assert np.array_equal(fs1_times_arr, np.array([1, 2, 6, 8, 10, 18, 21, 24, 27]))
    fs1_divide_arr = fs1 / arr
    assert np.array_equal(fs1_divide_arr, np.array([1, 2, 1.5, 2, 2.5, 2, 7 / 3, 8 / 3, 3]))
    fs1_plus_arr = fs1 + arr
    assert np.array_equal(fs1_plus_arr, np.array([2, 3, 5, 6, 7, 9, 10, 11, 12]))
    fs1_minus_arr = fs1 - arr
    assert np.array_equal(fs1_minus_arr, np.array([0, 1, 1, 2, 3, 3, 4, 5, 6]))
    # test reverse operations
    fs1_times_arr = arr * fs1
    assert np.array_equal(fs1_times_arr, np.array([1, 2, 6, 8, 10, 18, 21, 24, 27]))
    fs1_divide_arr = arr / fs1
    assert np.array_equal(
        fs1_divide_arr,
        np.array([1, 0.5, 2 / 3, 0.5, 0.4, 0.5, 3 / 7, 0.375, 1 / 3]),
    )
    fs1_plus_arr = arr + fs1
    assert np.array_equal(fs1_plus_arr, np.array([2, 3, 5, 6, 7, 9, 10, 11, 12]))
    fs1_minus_arr = arr - fs1
    assert np.array_equal(fs1_minus_arr, np.array([0, -1, -1, -2, -3, -3, -4, -5, -6]))
    fs1_to_power_arr = fs1**arr
    assert np.array_equal(fs1_to_power_arr, np.array([1, 2, 9, 16, 25, 216, 343, 512, 729]))
    arr_to_power_fs1 = arr**fs1
    assert np.array_equal(arr_to_power_fs1, np.array([1, 1, 8, 16, 32, 729, 2187, 6561, 19683]))


def test_get_sims():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    selected_values = fs[1]
    assert np.array_equal(selected_values, [3, 4, 5])


def test_stochastic_scalar():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    scalar = StochasticScalar([1, 2, 3])
    result = fs + scalar
    assert np.array_equal(result, np.array([2, 3, 5, 6, 7, 9, 10, 11, 12]))
    assert result.coupled_variable_group == fs.coupled_variable_group == scalar.coupled_variable_group
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    scalar = StochasticScalar([1, 2, 3])
    result = fs - scalar
    assert result.coupled_variable_group == fs.coupled_variable_group == scalar.coupled_variable_group
    assert np.array_equal(result, np.array([0, 1, 1, 2, 3, 3, 4, 5, 6]))
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    scalar = StochasticScalar([1, 2, 3])
    result = fs * scalar
    assert result.coupled_variable_group == fs.coupled_variable_group == scalar.coupled_variable_group
    assert np.array_equal(result, np.array([1, 2, 6, 8, 10, 18, 21, 24, 27]))
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    scalar = StochasticScalar([1, 2, 3])
    result = fs / scalar
    assert np.array_equal(result, np.array([1, 2, 1.5, 2, 2.5, 2, 2 + 1 / 3, 2 + 2 / 3, 3]))
    assert result.coupled_variable_group == fs.coupled_variable_group == scalar.coupled_variable_group
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    scalar = StochasticScalar([1, 2, 3])
    result = fs**scalar
    assert np.array_equal(result, np.array([1, 2, 9, 16, 25, 216, 343, 512, 729]))
    assert result.coupled_variable_group == fs.coupled_variable_group == scalar.coupled_variable_group
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    result = fs + 1
    assert np.array_equal(result, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    result = fs - 1
    assert np.array_equal(result, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))

    # reverse operations
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    scalar = StochasticScalar([1, 2, 3])
    result = scalar + fs
    assert np.array_equal(result, np.array([2, 3, 5, 6, 7, 9, 10, 11, 12]))
    assert result.coupled_variable_group == fs.coupled_variable_group == scalar.coupled_variable_group

    # reverse operations
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=3)
    scalar = 1
    result = scalar + fs
    assert np.array_equal(result, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert result.coupled_variable_group == fs.coupled_variable_group


def test_where():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=n_sims)
    condition = fs > 5
    result = np.where(condition, fs, 0)
    assert np.array_equal(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7, 8, 9]))
    assert condition.coupled_variable_group == fs.coupled_variable_group


def test_minimum():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=n_sims)
    result = np.minimum(fs, 5)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 5, 5, 5, 5]))


def test_maximum():
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_sims = 3
    fs = FreqSevSims(sim_index=sim_index, values=values, n_sims=n_sims)
    result = np.maximum(fs, 5)
    assert np.array_equal(result, np.array([5, 5, 5, 5, 5, 6, 7, 8, 9]))


def test_safe_divide_freqsevsims():
    """Test safe_divide with FreqSevSims arguments."""
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values_num = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    values_denom = np.array([2, 0, 5, 0, 10, 6, 0, 8, 9])
    n_sims = 3
    numerator = FreqSevSims(sim_index=sim_index, values=values_num, n_sims=n_sims)
    denominator = FreqSevSims(sim_index=sim_index, values=values_denom, n_sims=n_sims)
    default = 99
    result = pnp.safe_divide(numerator, denominator, default)
    assert np.array_equal(result, np.array([5, 99, 6, 99, 5, 10, 99, 10, 10]))
    assert numerator.coupled_variable_group == denominator.coupled_variable_group == result.coupled_variable_group


def test_safe_divide_freqsevsims_scalar_denominator():
    """Test safe_divide with FreqSevSims and scalar denominator."""
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    n_sims = 3
    numerator = FreqSevSims(sim_index=sim_index, values=values, n_sims=n_sims)
    denominator = 5
    default = 99
    result = pnp.safe_divide(numerator, denominator, default)
    assert np.array_equal(result, np.array([2, 4, 6, 8, 10, 12, 14, 16, 18]))
    assert numerator.coupled_variable_group == result.coupled_variable_group


def test_safe_divide_freqsevsims_scalar_denominator_zero():
    """Test safe_divide with FreqSevSims and zero scalar denominator."""
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    n_sims = 3
    numerator = FreqSevSims(sim_index=sim_index, values=values, n_sims=n_sims)
    denominator = 0
    default = -1
    result = pnp.safe_divide(numerator, denominator, default)
    assert np.array_equal(result, np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1]))
    assert numerator.coupled_variable_group == result.coupled_variable_group


def test_safe_divide_freqsevsims_scalar_numerator():
    """Test safe_divide with scalar numerator and FreqSevSims denominator."""
    sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    values = np.array([2, 0, 5, 0, 10, 6, 0, 8, 9])
    n_sims = 3
    numerator = 100
    denominator = FreqSevSims(sim_index=sim_index, values=values, n_sims=n_sims)
    default = -999
    result = pnp.safe_divide(numerator, denominator, default)
    expected = np.array([50, -999, 20, -999, 10, 100 / 6, -999, 12.5, 100 / 9])
    assert np.allclose(result.values, expected)
    assert denominator.coupled_variable_group == result.coupled_variable_group
