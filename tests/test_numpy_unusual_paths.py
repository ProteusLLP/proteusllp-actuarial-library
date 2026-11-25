"""Regression coverage for uncommon NumPy paths and scenario coupling."""

from __future__ import annotations

import numpy as np
import pytest

from pal import FreqSevSims, StochasticScalar
from pal._maths import asnumpy


def test_masked_ufunc_out_updates_existing_variable_and_joins_all_sources() -> None:
    left = StochasticScalar([1.0, 2.0, 3.0])
    right = StochasticScalar([10.0, 20.0, 30.0])
    mask = StochasticScalar([True, False, True])
    destination = StochasticScalar([-1.0, -1.0, -1.0])

    returned = np.add(left, right, out=destination, where=mask)

    assert returned is destination
    np.testing.assert_array_equal(asnumpy(destination.values), [11.0, -1.0, 33.0])
    assert all(item.coupled_variable_group is left.coupled_variable_group for item in (right, mask, destination))


def test_invalid_ufunc_out_axis_and_mask_are_atomic() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    destination = StochasticScalar([-1.0, -1.0, -1.0])
    mismatched = StochasticScalar([True, False])
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
    assert all(item.coupled_variable_group is source.coupled_variable_group for item in (mask, fractions, integers))


def test_invalid_second_ufunc_output_does_not_write_first() -> None:
    source = StochasticScalar([1.25, 2.5, 3.75])
    fractions = StochasticScalar([-1.0, -1.0, -1.0])
    bad_integers = StochasticScalar([0.0])
    original_group = fractions.coupled_variable_group

    with pytest.raises(ValueError, match="out= destination"):
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


@pytest.mark.parametrize(
    "function",
    [
        lambda x: np.sort(x),
        lambda x: np.flip(x),
        lambda x: np.roll(x, 1),
        lambda x: np.cumsum(x),
        lambda x: np.add.accumulate(x),
        lambda x: np.add.reduceat(x, np.arange(x.n_sims)),
    ],
)
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


def test_event_out_preserves_original_object_and_claim_layout() -> None:
    events = FreqSevSims([0, 0, 2], [3.0, 4.0, 5.0], 3)
    destination = FreqSevSims([0, 0, 2], [-1.0, -1.0, -1.0], 3)
    mask = StochasticScalar([True, False, True])

    returned = np.add(events, 2.0, out=destination, where=mask)

    assert returned is destination
    np.testing.assert_allclose(asnumpy(destination.values), [5.0, 6.0, 7.0])
    assert destination.coupled_variable_group is events.coupled_variable_group
    assert mask.coupled_variable_group is events.coupled_variable_group


def test_event_multioutput_ufunc_links_both_results() -> None:
    events = FreqSevSims([0, 1, 1], [1.25, 2.5, 3.75], 2)
    fraction, integer = np.modf(events)

    assert isinstance(fraction, FreqSevSims)
    assert isinstance(integer, FreqSevSims)
    assert fraction.coupled_variable_group is events.coupled_variable_group
    assert integer.coupled_variable_group is events.coupled_variable_group
    np.testing.assert_allclose(asnumpy(fraction.values), [0.25, 0.5, 0.75])
    np.testing.assert_allclose(asnumpy(integer.values), [1.0, 2.0, 3.0])


def test_event_multioutput_invalid_second_destination_is_atomic() -> None:
    events = FreqSevSims([0, 1, 1], [1.25, 2.5, 3.75], 2)
    first = FreqSevSims([0, 1, 1], [-1.0, -1.0, -1.0], 2)
    second = FreqSevSims([0, 0, 1], [-1.0, -1.0, -1.0], 2)
    original = first.coupled_variable_group

    with pytest.raises(ValueError, match="Event simulation indices"):
        np.modf(events, out=(first, second))

    assert first.coupled_variable_group is original
    assert events.coupled_variable_group is not original
    np.testing.assert_array_equal(asnumpy(first.values), [-1.0] * 3)


def test_event_where_broadcasts_scenario_branches_by_claim_index() -> None:
    condition = FreqSevSims([0, 1, 1], [True, True, False], 2)
    scenario_values = StochasticScalar([10.0, 20.0])
    fallback = FreqSevSims([0, 1, 1], [0.0, 0.0, 3.0], 2)

    result = np.where(condition, scenario_values, fallback)

    assert isinstance(result, FreqSevSims)
    np.testing.assert_allclose(asnumpy(result.values), [10.0, 20.0, 3.0])
    assert scenario_values.coupled_variable_group is condition.coupled_variable_group
    assert fallback.coupled_variable_group is condition.coupled_variable_group


def test_event_masked_call_without_out_rejects_undefined_values() -> None:
    events = FreqSevSims([0, 1], [2.0, 3.0], 2)
    with pytest.raises(ValueError, match="out="):
        np.add(events, 1.0, where=[True, False])


def test_event_round_out_preserves_identity_and_coupling() -> None:
    events = FreqSevSims([0, 1], [2.35, 3.45], 2)
    destination = FreqSevSims([0, 1], [0.0, 0.0], 2)
    returned = np.round(events, 1, out=destination)
    assert returned is destination
    np.testing.assert_allclose(asnumpy(destination.values), [2.4, 3.4])
    assert destination.coupled_variable_group is events.coupled_variable_group
