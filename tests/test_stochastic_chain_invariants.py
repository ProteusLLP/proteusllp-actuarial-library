"""Cross-module regression tests for stochastic dependency chains."""

from __future__ import annotations

import numpy as np
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal.copulas import apply_copula
from tests._assertions import assert_allclose, assert_array_equal, host_values


def _rank_matched_indices(margin: np.ndarray, scores: list[int]) -> np.ndarray:
    """Match each score rank to a marginal observation, with stable ties.

    The first copula margin and first model margin are in ascending order in
    these fixtures, so their ordering is unchanged.
    """
    margin_order = sorted(range(len(margin)), key=lambda i: (margin[i], i))
    score_order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    score_ranks = [0] * len(scores)
    for rank, position in enumerate(score_order):
        score_ranks[position] = rank
    return np.asarray([margin_order[rank] for rank in score_ranks], dtype=int)


def _events_by_simulation(events: FreqSevSims) -> list[list[float]]:
    """Retain the original event order within each simulation."""
    indices = host_values(events.sim_index)
    values = host_values(events)
    return [[float(value) for index, value in zip(indices, values) if index == sim] for sim in range(events.n_sims)]


def test_three_margin_copula_reorders_all_attached_frequency_severity_chains() -> None:
    """Reorder complete event groups and every dependent scenario-level value."""
    first = StochasticScalar([10, 20, 30, 40, 50, 60])
    events = FreqSevSims(
        [4, 0, 2, 5, 2, 0, 3],
        [400.0, 10.0, 200.0, 500.0, 220.0, 20.0, 300.0],
        6,
    )
    factor = StochasticScalar([1, 2, 3, 4, 5, 6])
    adjusted = events * factor
    aggregate = events.aggregate()
    adjusted_aggregate = adjusted.aggregate()
    frequency = events.count()

    third = StochasticScalar([6, 1, 4, 3, 5, 2])
    third_derived = (third * 10) + 3
    second_scores = [3, 0, 4, 2, 5, 1]
    third_scores = [5, 2, 0, 4, 1, 3]

    old_indices = host_values(events.sim_index).copy()
    old_values = host_values(events).copy()
    old_adjusted = host_values(adjusted).copy()
    old_aggregate = host_values(aggregate).copy()
    old_adjusted_aggregate = host_values(adjusted_aggregate).copy()
    old_factor = host_values(factor).copy()
    old_frequency = host_values(frequency).copy()
    old_third = host_values(third).copy()

    second_selection = _rank_matched_indices(old_aggregate, second_scores)
    third_selection = _rank_matched_indices(old_third, third_scores)
    apply_copula(
        [first, aggregate, third],
        [
            StochasticScalar([0, 1, 2, 3, 4, 5]),
            StochasticScalar(second_scores),
            StochasticScalar(third_scores),
        ],
    )

    # An event keeps its value and position within its original simulation;
    # only the simulation label changes under the copula permutation.
    new_indices = np.argsort(second_selection)[old_indices]
    assert_array_equal(events.sim_index, new_indices)
    assert_array_equal(events.values, old_values)
    assert_array_equal(adjusted.sim_index, new_indices)
    assert_array_equal(adjusted.values, old_adjusted)

    assert_array_equal(aggregate, old_aggregate[second_selection])
    assert_array_equal(adjusted_aggregate, old_adjusted_aggregate[second_selection])
    assert_array_equal(factor, old_factor[second_selection])
    assert_array_equal(frequency, old_frequency[second_selection])
    assert_array_equal(third, old_third[third_selection])
    assert_allclose(third_derived, 10 * host_values(third) + 3)
    assert_allclose(events.aggregate(), aggregate)
    assert_allclose(adjusted.aggregate(), adjusted_aggregate)
    assert_array_equal(events.count(), frequency)
    assert_allclose(
        adjusted_aggregate,
        host_values(factor) * host_values(aggregate),
    )
    assert all(
        variable.coupled_variable_group is first.coupled_variable_group
        for variable in (
            events,
            factor,
            adjusted,
            aggregate,
            adjusted_aggregate,
            frequency,
            third,
            third_derived,
        )
    )


def test_second_copula_reorders_every_variable_from_first_copula_chain() -> None:
    """An already coupled group can be the second margin of another copula."""
    first = StochasticScalar([10, 20, 30, 40, 50, 60])
    second = StochasticScalar([3, 6, 1, 5, 2, 4])
    derived = (second * 2) + 5

    apply_copula(
        [first, second],
        [StochasticScalar([0, 1, 2, 3, 4, 5]), StochasticScalar([5, 1, 2, 0, 4, 3])],
    )
    bridge = first + derived
    original_first = np.sort(host_values(first))
    original_second = np.sort(host_values(second))

    independent = StochasticScalar([60, 50, 40, 30, 20, 10])
    apply_copula(
        [independent, bridge],
        [StochasticScalar([0, 1, 2, 3, 4, 5]), StochasticScalar([1, 5, 3, 0, 4, 2])],
    )

    assert_array_equal(np.sort(host_values(first)), original_first)
    assert_array_equal(np.sort(host_values(second)), original_second)
    assert_allclose(derived, 2 * host_values(second) + 5)
    assert_allclose(bridge, host_values(first) + host_values(derived))
    assert_array_equal(independent, [60, 50, 40, 30, 20, 10])
    assert all(
        variable.coupled_variable_group is independent.coupled_variable_group
        for variable in (first, second, derived, bridge)
    )


def test_copula_rank_matching_uses_stable_ties_across_derived_variables() -> None:
    """Repeated marginal observations must retain their stable relative order."""
    anchor = StochasticScalar([1, 2, 3, 4, 5, 6])
    tied = StochasticScalar([20, 20, 10, 10, 30, 30])
    derived = (tied + 1) * 5

    apply_copula(
        [anchor, tied],
        [
            StochasticScalar([0, 1, 2, 3, 4, 5]),
            StochasticScalar([1, 3, 0, 5, 2, 4]),
        ],
    )

    assert_array_equal(tied, [10, 20, 10, 30, 20, 30])
    assert_array_equal(derived, [55, 105, 55, 155, 105, 155])
    assert tied.coupled_variable_group is derived.coupled_variable_group
    assert tied.coupled_variable_group is anchor.coupled_variable_group


def test_nested_upsample_after_copula_preserves_mixed_stochastic_chains() -> None:
    """Cycling scenarios must keep event lists, factors and aggregates aligned."""
    events = FreqSevSims([2, 0, 2, 3, 0], [100.0, 10.0, 200.0, 300.0, 20.0], 4)
    factor = StochasticScalar([2, 3, 5, 7])
    adjusted = events * factor
    aggregate = events.aggregate()
    adjusted_aggregate = adjusted.aggregate()
    frequency = events.count()
    driver = StochasticScalar([10, 20, 30, 40])

    apply_copula(
        [driver, aggregate],
        [StochasticScalar([0, 1, 2, 3]), StochasticScalar([2, 0, 3, 1])],
    )
    independent = StochasticScalar([9, 8, 7, 6])
    source = ProteusVariable(
        "portfolio",
        {
            "risk": ProteusVariable(
                "risk",
                {
                    "events": events,
                    "adjusted": adjusted,
                    "aggregate": aggregate,
                    "adjusted_aggregate": adjusted_aggregate,
                    "frequency": frequency,
                    "factor": factor,
                },
            ),
            "driver": driver,
            "independent": independent,
        },
    )

    expected_events = _events_by_simulation(events)
    expected_adjusted = _events_by_simulation(adjusted)
    original_indices = host_values(events.sim_index).copy()
    original_values = host_values(events).copy()
    selection = np.arange(11) % 4

    result = source.upsample(11)
    risk = result["risk"]
    resampled_events = risk["events"]
    resampled_adjusted = risk["adjusted"]
    assert resampled_events.n_sims == 11
    assert _events_by_simulation(resampled_events) == [expected_events[i] for i in selection]
    assert _events_by_simulation(resampled_adjusted) == [expected_adjusted[i] for i in selection]

    for key, original in (
        ("aggregate", aggregate),
        ("adjusted_aggregate", adjusted_aggregate),
        ("frequency", frequency),
        ("factor", factor),
    ):
        assert_array_equal(risk[key], host_values(original)[selection])
    assert_array_equal(result["driver"], host_values(driver)[selection])
    assert_array_equal(result["independent"], host_values(independent)[selection])
    assert_array_equal(resampled_events.sim_index, resampled_adjusted.sim_index)
    assert_allclose(resampled_events.aggregate(), risk["aggregate"])
    assert_allclose(resampled_adjusted.aggregate(), risk["adjusted_aggregate"])
    assert_array_equal(resampled_events.count(), risk["frequency"])
    assert_allclose(
        resampled_adjusted.values,
        host_values(resampled_events) * host_values(risk["factor"])[host_values(resampled_events.sim_index)],
    )
    assert all(
        risk[key].coupled_variable_group is result["driver"].coupled_variable_group
        for key in ("events", "adjusted", "aggregate", "adjusted_aggregate", "frequency", "factor")
    )
    assert result["independent"].coupled_variable_group is not result["driver"].coupled_variable_group
    assert events.coupled_variable_group is not resampled_events.coupled_variable_group
    assert_array_equal(events.sim_index, original_indices)
    assert_array_equal(events.values, original_values)


@pytest.mark.parametrize("invalid", ["coupled_margins", "sample_overlap", "sample_length"])
def test_invalid_copula_inputs_leave_dependency_chains_unchanged(invalid: str) -> None:
    """All validation must finish before any variable or event is reordered."""
    first = StochasticScalar([1, 2, 3, 4])
    second = StochasticScalar([4, 3, 2, 1])
    derived = second * 10
    before = [host_values(variable).copy() for variable in (first, second, derived)]
    original_groups = (first.coupled_variable_group, second.coupled_variable_group)

    margins = [first, second]
    samples = [StochasticScalar([0, 1, 2, 3]), StochasticScalar([3, 2, 1, 0])]
    if invalid == "coupled_margins":
        margins = [second, derived]
        match = "not independent"
    elif invalid == "sample_overlap":
        samples[1] = second
        match = "coupled to marginal"
    else:
        samples[1] = StochasticScalar([3, 2, 1])
        match = "same number of simulations"

    with pytest.raises(ValueError, match=match):
        apply_copula(margins, samples)

    for variable, expected in zip((first, second, derived), before):
        assert_array_equal(variable, expected)
    assert first.coupled_variable_group is original_groups[0]
    assert second.coupled_variable_group is original_groups[1]
    assert derived.coupled_variable_group is original_groups[1]
    assert first.coupled_variable_group is not second.coupled_variable_group
