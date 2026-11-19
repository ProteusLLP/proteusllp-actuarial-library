"""Less common NumPy dispatch paths must not bypass scenario coupling."""

from __future__ import annotations

import numpy as np
import pytest

from pal import FreqSevSims, StochasticScalar
from pal._maths import asnumpy


def test_masked_scalar_reduce_writes_the_scalar_out_not_a_scenario_mask() -> None:
    values = StochasticScalar([1.0, 2.0, 3.0])
    mask = StochasticScalar([True, False, True])
    destination = np.array(-1.0)
    original_group = values.coupled_variable_group

    returned = np.add.reduce(values, where=mask, out=destination)

    assert returned is destination
    assert destination.item() == 4.0
    # Reductions do not represent scenarios and must not couple their inputs.
    assert values.coupled_variable_group is original_group
    assert mask.coupled_variable_group is not original_group


def test_masked_minimum_reduce_without_identity_uses_initial() -> None:
    values = StochasticScalar([3.0, 1.0, 4.0])
    mask = StochasticScalar([True, False, True])
    destination = np.array(-1.0)

    returned = np.minimum.reduce(values, where=mask, initial=100.0, out=destination)

    assert returned is destination
    assert destination.item() == 3.0
    assert mask.coupled_variable_group is not values.coupled_variable_group


def test_masked_multiply_reduce_uses_neutral_element() -> None:
    values = StochasticScalar([2.0, 5.0, 7.0])
    mask = StochasticScalar([True, False, True])

    result = np.multiply.reduce(values, where=mask)

    assert result == 14.0
    assert mask.coupled_variable_group is not values.coupled_variable_group


def test_masked_event_reduce_uses_event_mask_and_scalar_out() -> None:
    events = FreqSevSims([0, 0, 2], [2.0, 3.0, 4.0], 3)
    mask = StochasticScalar([True, False, False])
    destination = np.array(-1.0)
    original_group = events.coupled_variable_group

    returned = np.add.reduce(events, where=mask, out=destination)

    assert returned is destination
    assert destination.item() == 5.0
    assert events.coupled_variable_group is original_group
    assert mask.coupled_variable_group is not original_group


def test_masked_multioutput_scalar_requires_every_destination() -> None:
    source = StochasticScalar([1.25, 2.5, 3.75])
    existing = StochasticScalar([-5.0, -5.0, -5.0])
    original_group = existing.coupled_variable_group

    with pytest.raises(ValueError, match="out="):
        np.modf(source, out=(None, existing), where=[True, False, True])

    np.testing.assert_array_equal(asnumpy(existing.values), [-5.0, -5.0, -5.0])
    assert existing.coupled_variable_group is original_group
    assert source.coupled_variable_group is not original_group


def test_masked_multioutput_events_requires_every_destination() -> None:
    events = FreqSevSims([0, 1, 1], [1.25, 2.5, 3.75], 2)
    existing = FreqSevSims([0, 1, 1], [-5.0, -5.0, -5.0], 2)
    original_group = existing.coupled_variable_group

    with pytest.raises(ValueError, match="out="):
        np.modf(events, out=(None, existing), where=[True, False, True])

    np.testing.assert_array_equal(asnumpy(existing.values), [-5.0, -5.0, -5.0])
    assert existing.coupled_variable_group is original_group
    assert events.coupled_variable_group is not original_group


def test_positional_clip_out_preserves_scalar_destination_and_coupling() -> None:
    source = StochasticScalar([-3.0, 2.0, 9.0])
    destination = StochasticScalar([-1.0, -1.0, -1.0])

    returned = np.clip(source, 0.0, 5.0, destination)

    assert returned is destination
    np.testing.assert_array_equal(asnumpy(destination.values), [0.0, 2.0, 5.0])
    assert destination.coupled_variable_group is source.coupled_variable_group


def test_positional_round_out_preserves_scalar_destination_and_coupling() -> None:
    source = StochasticScalar([1.24, 2.35, 3.46])
    destination = StochasticScalar([0.0, 0.0, 0.0])

    returned = np.round(source, 1, destination)

    assert returned is destination
    np.testing.assert_allclose(asnumpy(destination.values), [1.2, 2.4, 3.5])
    assert destination.coupled_variable_group is source.coupled_variable_group


def test_positional_round_out_preserves_event_destination_and_coupling() -> None:
    events = FreqSevSims([0, 1, 1], [1.24, 2.35, 3.46], 2)
    destination = FreqSevSims([0, 1, 1], [0.0, 0.0, 0.0], 2)

    returned = np.round(events, 1, destination)

    assert returned is destination
    np.testing.assert_allclose(asnumpy(destination.values), [1.2, 2.4, 3.5])
    assert destination.coupled_variable_group is events.coupled_variable_group


@pytest.mark.parametrize("positional", [False, True])
def test_event_sum_scalar_out_is_written_and_returned(positional: bool) -> None:
    events = FreqSevSims([0, 0, 2], [2.0, 3.0, 4.0], 3)
    destination = np.array(-1.0)
    original_group = events.coupled_variable_group

    returned = np.sum(events, None, None, destination) if positional else np.sum(events, out=destination)

    assert returned is destination
    assert destination.item() == 9.0
    assert events.coupled_variable_group is original_group


@pytest.mark.parametrize("positional", [False, True])
def test_event_mean_scalar_out_uses_per_simulation_aggregate(positional: bool) -> None:
    events = FreqSevSims([0, 0, 2, 2], [2.0, 3.0, 4.0, 5.0], 3)
    destination = np.array(-1.0)

    returned = np.mean(events, None, None, destination) if positional else np.mean(events, out=destination)

    assert returned is destination
    assert destination.item() == pytest.approx(14.0 / 3.0)


def test_non_scenario_positional_out_is_rejected_before_mutation() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    destination = StochasticScalar([-1.0, -1.0, -1.0])
    original_group = destination.coupled_variable_group

    with pytest.raises(ValueError, match="scenario-preserving"):
        np.sum(source, None, None, destination)

    np.testing.assert_array_equal(asnumpy(destination.values), [-1.0, -1.0, -1.0])
    assert destination.coupled_variable_group is original_group


@pytest.mark.parametrize("positional", [False, True])
def test_nan_to_num_copy_false_is_rejected_without_mutation(positional: bool) -> None:
    source = StochasticScalar([1.0, np.nan, np.inf])
    original_group = source.coupled_variable_group

    with pytest.raises(ValueError, match="copy=False"):
        if positional:
            np.nan_to_num(source, False)
        else:
            np.nan_to_num(source, copy=False)

    np.testing.assert_array_equal(asnumpy(source.values), [1.0, np.nan, np.inf])
    assert source.coupled_variable_group is original_group


def test_nan_to_num_copy_true_retains_coupling() -> None:
    source = StochasticScalar([1.0, np.nan, np.inf])
    cleaned = np.nan_to_num(source)
    assert isinstance(cleaned, StochasticScalar)
    assert cleaned.coupled_variable_group is source.coupled_variable_group
    np.testing.assert_array_equal(asnumpy(cleaned.values), [1.0, 0.0, np.finfo(float).max])


def test_invalid_second_raw_ufunc_output_does_not_write_first() -> None:
    source = StochasticScalar([1.25, 2.5, 3.75])
    first = np.full(3, -1.0)
    second = np.zeros(3, dtype=np.int32)
    original_group = source.coupled_variable_group

    with pytest.raises(TypeError, match="casting"):
        np.modf(source, out=(first, second))

    np.testing.assert_array_equal(first, [-1.0, -1.0, -1.0])
    assert source.coupled_variable_group is original_group


def test_outer_ufunc_raw_destination_is_not_scenario_coupled() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    raw = np.zeros((3, 2))
    original_group = source.coupled_variable_group

    returned = np.add.outer(source, np.array([10.0, 20.0]), out=raw)

    assert returned is raw
    np.testing.assert_array_equal(raw, [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]])
    assert source.coupled_variable_group is original_group
