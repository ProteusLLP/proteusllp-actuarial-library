"""Regression tests for frequency-severity random-access indexing."""

import numpy as np
import pytest

from benchmarks.benchmark_core import build_workload
from pal._maths import asnumpy, xp
from pal.frequency_severity import FreqSevSims
from tests._assertions import assert_array_equal


@pytest.mark.parametrize("reordered", [False, True])
def test_random_access_preserves_event_order_and_empty_scenarios(reordered: bool) -> None:
    indices = xp.asarray([0, 0, 2, 3, 3, 3, 5], dtype=int)
    values = xp.asarray([10.0, -2.0, 8.0, 2.0, 3.0, 7.0, 11.0])
    events = FreqSevSims(indices, values, n_sims=6)
    if reordered:
        events._reorder_sims(xp.asarray([5, 0, 4, 3, 1, 2]))
    expected_indices = asnumpy(events.sim_index)
    expected_values = asnumpy(events.values)
    for sim in (0, 5, 2, 1, 3, 4, 2, 0):
        assert_array_equal(events[sim].values, expected_values[expected_indices == sim])
    assert events._index_is_sorted is (not reordered)


def test_sorted_random_access_does_not_alias_original_event_values() -> None:
    events = FreqSevSims(xp.asarray([0, 0, 2]), xp.asarray([1.0, 2.0, 3.0]), n_sims=4)
    selected = events[0]
    selected.values[0] = 999.0
    assert_array_equal(events.values, [1.0, 2.0, 3.0])


def test_random_access_handles_empty_and_out_of_range_scenarios() -> None:
    events = FreqSevSims(xp.asarray([], dtype=int), xp.asarray([], dtype=float), n_sims=4)
    for index in (-1, 0, 1, 3, 4):
        assert len(events[index].values) == 0


def test_random_access_reordering_invalidates_cached_sorted_status() -> None:
    events = FreqSevSims(xp.asarray([0, 1, 2, 3]), xp.asarray([10, 20, 30, 40]), n_sims=4)
    assert_array_equal(events[1].values, [20])
    assert events._index_is_sorted is True
    events._reorder_sims(xp.asarray([3, 0, 2, 1]))
    assert events._index_is_sorted is None
    for sim in range(4):
        indices = asnumpy(events.sim_index)
        assert_array_equal(events[sim].values, asnumpy(events.values)[indices == sim])
    assert events._index_is_sorted is False


def test_random_access_supports_nonnumeric_values_after_copula_reordering() -> None:
    events = FreqSevSims(xp.asarray([0, 1, 1, 3]), np.array(["a", "b", "c", "d"]), 4)
    assert_array_equal(events[1].values, ["b", "c"])
    events._reorder_sims(xp.asarray([3, 1, 0, 2]))
    indices = asnumpy(events.sim_index)
    values = asnumpy(events.values)
    for sim in range(4):
        assert_array_equal(events[sim].values, values[indices == sim])


def test_random_access_benchmark_matches_full_scan_baseline() -> None:
    fast = build_workload("freqsev_random_access", 40)()
    baseline = build_workload("freqsev_random_access_mask_baseline", 40)()
    assert len(fast) == len(baseline)
    for actual, expected in zip(fast, baseline):
        assert_array_equal(actual.values, expected.values)
