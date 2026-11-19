"""Regression coverage for uncommon NumPy paths and scenario coupling."""

from __future__ import annotations

import numpy as np
import pytest

from pal import StochasticScalar
from pal._maths import asnumpy


def test_masked_ufunc_out_updates_existing_variable_and_joins_all_sources() -> None:
    left = StochasticScalar([1.0, 2.0, 3.0])
    right = StochasticScalar([10.0, 20.0, 30.0])
    mask = StochasticScalar([True, False, True])
    destination = StochasticScalar([-1.0, -1.0, -1.0])

    returned = np.add(left, right, out=destination, where=mask)

    assert returned is destination
    np.testing.assert_array_equal(asnumpy(destination.values), [11.0, -1.0, 33.0])
    assert all(
        item.coupled_variable_group is left.coupled_variable_group
        for item in (right, mask, destination)
    )


def test_invalid_ufunc_out_axis_and_mask_are_atomic() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    destination = StochasticScalar([-1.0, -1.0, -1.0])
    mismatched = StochasticScalar([True])
    group_source = source.coupled_variable_group
    group_destination = destination.coupled_variable_group
    initial = asnumpy(destination.values).copy()

    with pytest.raises(ValueError, match="matching n_sims"):
        np.add(source, 1, out=destination, where=mismatched)
    with pytest.raises(TypeError, match="boolean"):
        np.add(source, 1, out=destination, where=[1, 0, 1])
    with pytest.raises(ValueError, match="out="):
        np.add(source, np.ones((2, 3)), out=destination)

    assert source.coupled_variable_group is group_source
    assert destination.coupled_variable_group is group_destination
    np.testing.assert_array_equal(asnumpy(destination.values), initial)


def test_multivariate_output_ufunc_writes_both_outputs_atomically() -> None:
    source = StochasticScalar([1.25, -2.5, 3.75])
    mask = StochasticScalar([True, False, True])
    fractions = StochasticScalar([-9.0, -9.0, -9.0])
    integers = StochasticScalar([-8.0, -8.0, -8.0])

    actual_fractions, actual_integers = np.modf(source, out=(fractions, integers), where=mask)

    assert actual_fractions is fractions
    assert actual_integers is integers
    np.testing.assert_allclose(asnumpy(fractions.values), [0.25, -9.0, 0.75])
    np.testing.assert_allclose(asnumpy(integers.values), [1.0, -8.0, 3.0])
    assert all(
        item.coupled_variable_group is source.coupled_variable_group
        for item in (mask, fractions, integers)
    )


def test_invalid_second_ufunc_output_does_not_write_first() -> None:
    source = StochasticScalar([1.25, 2.5, 3.75])
    fractions = StochasticScalar([-1.0, -1.0, -1.0])
    bad_integers = StochasticScalar([0.0])
    original_group = fractions.coupled_variable_group

    with pytest.raises(ValueError, match="matching n_sims"):
        np.modf(source, out=(fractions, bad_integers))

    np.testing.assert_array_equal(asnumpy(fractions.values), [-1.0] * 3)
    assert fractions.coupled_variable_group is original_group
    assert fractions.coupled_variable_group is not source.coupled_variable_group


def test_partial_multioutput_out_links_new_and_existing_leaves() -> None:
    source = StochasticScalar([1.25, 2.5, 3.75])
    destination = StochasticScalar([0.0, 0.0, 0.0])
    fraction, integer = np.modf(source, out=(None, destination))
    assert isinstance(fraction, StochasticScalar)
    assert integer is destination
    assert fraction.coupled_variable_group is source.coupled_variable_group
    assert destination.coupled_variable_group is source.coupled_variable_group


def test_masked_ufunc_without_out_is_rejected_to_avoid_undefined_values() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="out="):
        np.add(source, 1, where=[True, False, True])


def test_numpy_out_with_plain_array_preserves_return_identity() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    raw = np.zeros(3, dtype=float)
    result = np.add(source, 2.0, out=raw)
    assert result is raw
    np.testing.assert_allclose(raw, [3.0, 4.0, 5.0])


def test_numpy_where_and_clip_preserve_all_scenario_dependencies() -> None:
    condition = StochasticScalar([True, False, True])
    a = StochasticScalar([-4.0, 7.0, 9.0])
    b = StochasticScalar([20.0, -2.0, 1.0])
    selected = np.where(condition, a, b)
    destination = StochasticScalar([0.0, 0.0, 0.0])

    clipped = np.clip(selected, 0.0, 5.0, out=destination)

    assert clipped is destination
    np.testing.assert_allclose(asnumpy(destination.values), [0.0, 0.0, 5.0])
    assert all(item.coupled_variable_group is condition.coupled_variable_group for item in (a, b, selected, clipped))


@pytest.mark.parametrize("function", [
    lambda x: np.sort(x),
    lambda x: np.flip(x),
    lambda x: np.roll(x, 1),
    lambda x: np.cumsum(x),
    lambda x: np.add.accumulate(x),
    lambda x: np.add.reduceat(x, np.arange(x.n_sims)),
])
def test_reordering_and_cumulative_operations_are_detached(function) -> None:
    source = StochasticScalar([3.0, 1.0, 4.0])
    result = function(source)
    assert isinstance(result, StochasticScalar)
    assert result.coupled_variable_group is not source.coupled_variable_group


def test_one_argument_where_returns_raw_indices_without_coupling() -> None:
    condition = StochasticScalar([True, False, True])
    indices = np.where(condition)
    assert isinstance(indices, tuple)
    np.testing.assert_array_equal(asnumpy(indices[0]), [0, 2])


def test_keepdims_reduction_is_not_coupled_to_scenarios() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    result = np.add.reduce(source, keepdims=True)
    assert not isinstance(result, StochasticScalar)
    np.testing.assert_allclose(asnumpy(result), [6.0])


def test_mutating_numpy_array_function_is_rejected() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    with pytest.raises(NotImplementedError, match="controlled stochastic"):
        np.copyto(source, np.array([9.0, 9.0, 9.0]))
    with pytest.raises(NotImplementedError, match="ufunc.at"):
        np.add.at(source, [0], 1)
    np.testing.assert_allclose(asnumpy(source.values), [1.0, 2.0, 3.0])
