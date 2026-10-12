"""Tests for recursive, coupling-aware ProteusVariable upsampling."""

import numpy as np
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar


def test_upsample_recurses_through_nested_proteus_variables() -> None:
    first = StochasticScalar([1.0, 2.0, 3.0])
    second = StochasticScalar([10.0, 20.0, 30.0])
    first.coupled_variable_group.merge(second.coupled_variable_group)

    left = ProteusVariable("left", {"value": first})
    right = ProteusVariable("right", {"value": second})
    variable = ProteusVariable("outer", {"left": left, "right": right})

    result = variable.upsample(9)

    resampled_first = result["left"]["value"]
    resampled_second = result["right"]["value"]
    assert isinstance(resampled_first, StochasticScalar)
    assert isinstance(resampled_second, StochasticScalar)
    assert resampled_first.n_sims == 9
    assert resampled_second.n_sims == 9
    assert resampled_first.coupled_variable_group is resampled_second.coupled_variable_group
    np.testing.assert_allclose(np.asarray(resampled_second), 10.0 * np.asarray(resampled_first))


def test_upsample_keeps_independent_groups_independent() -> None:
    first = StochasticScalar([1.0, 2.0, 3.0])
    second = StochasticScalar([10.0, 20.0, 30.0])
    variable = ProteusVariable("risk", {"first": first, "second": second})

    result = variable.upsample(9)

    assert result["first"].coupled_variable_group is not result["second"].coupled_variable_group


def test_upsample_freqsev_preserves_shared_simulation_selection() -> None:
    sim_index = [0, 0, 2, 3]
    first = FreqSevSims(sim_index, [1.0, 2.0, 4.0, 5.0], n_sims=4)
    second = FreqSevSims(sim_index, [10.0, 20.0, 40.0, 50.0], n_sims=4)
    first.coupled_variable_group.merge(second.coupled_variable_group)
    variable = ProteusVariable("risk", {"first": first, "second": second})

    result = variable.upsample(10)

    resampled_first = result["first"]
    resampled_second = result["second"]
    assert resampled_first.n_sims == 10
    assert resampled_second.n_sims == 10
    assert resampled_first.coupled_variable_group is resampled_second.coupled_variable_group
    np.testing.assert_allclose(
        np.asarray(resampled_second.aggregate()),
        10.0 * np.asarray(resampled_first.aggregate()),
    )
    np.testing.assert_array_equal(np.asarray(resampled_first.sim_index), [0, 0, 2, 3, 4, 4, 6, 7, 8, 8])
    np.testing.assert_array_equal(np.asarray(resampled_first.values), [1, 2, 4, 5, 1, 2, 4, 5, 1, 2])
    assert type(resampled_first.sim_index) is type(first.sim_index)
    assert type(resampled_first.values) is type(first.values)


def test_nested_variable_simulation_count_and_no_op_upsample() -> None:
    leaf = StochasticScalar([1.0, 2.0, 3.0])
    child = ProteusVariable("child", {"value": leaf})
    variable = ProteusVariable("outer", {"child": child})

    assert variable.n_sims == 3
    assert variable.upsample(3) is variable

    result = variable.upsample(8)
    assert result.n_sims == 8
    assert result["child"].n_sims == 8
    assert result.upsample(8) is result
    np.testing.assert_array_equal(np.asarray(result["child"]["value"]), [1, 2, 3, 1, 2, 3, 1, 2])


def test_nested_variable_rejects_mismatched_simulation_counts() -> None:
    first = ProteusVariable("child", {"value": StochasticScalar([1.0, 2.0, 3.0])})
    second = ProteusVariable("child", {"value": StochasticScalar([1.0, 2.0])})

    with pytest.raises(ValueError, match="Number of simulations do not match"):
        ProteusVariable("outer", {"first": first, "second": second})


def test_upsample_freqsev_with_no_events() -> None:
    empty = FreqSevSims(np.array([], dtype=int), np.array([], dtype=float), n_sims=3)
    result = ProteusVariable("risk", {"empty": empty}).upsample(8)

    assert result.n_sims == 8
    assert result["empty"].n_sims == 8
    assert result["empty"].sim_index.size == 0
    assert result["empty"].values.size == 0
