"""Regression tests for efficient iteration over event-level simulations."""

import numpy as np
import pytest

from pal._maths import asnumpy, xp
from pal.frequency_severity import FreqSevSims


@pytest.mark.parametrize("reordered", [False, True])
def test_iteration_matches_per_scenario_indexing(reordered: bool):
    """Retain event values and order even after a copula relabels simulations."""
    indices = xp.asarray([0, 0, 2, 3, 3, 3, 5], dtype=int)
    values = xp.asarray([10.0, -2.0, 8.0, 2.0, 3.0, 7.0, 11.0])
    fs = FreqSevSims(indices, values, n_sims=6)
    if reordered:
        fs._reorder_sims(xp.asarray([5, 0, 4, 3, 1, 2]))
    expected_indices = asnumpy(fs.sim_index)
    expected_values = asnumpy(fs.values)
    observed = list(fs)
    assert len(observed) == 6
    for i, events in enumerate(observed):
        np.testing.assert_array_equal(asnumpy(events.values), expected_values[expected_indices == i])


def test_iteration_includes_all_empty_scenarios():
    fs = FreqSevSims(xp.asarray([], dtype=int), xp.asarray([], dtype=float), n_sims=4)
    assert [len(event.values) for event in fs] == [0, 0, 0, 0]


def test_iteration_handles_host_string_values_on_cuda_and_cpu():
    fs = FreqSevSims(xp.asarray([2, 0, 2, 1]), np.array(["a", "b", "c", "d"]), 4)
    expected = [["b"], ["d"], ["a", "c"], []]
    assert [asnumpy(events.values).tolist() for events in fs] == expected


def test_iteration_does_not_run_a_full_scan_for_each_simulation(monkeypatch: pytest.MonkeyPatch):
    fs = FreqSevSims(xp.asarray([0, 99, 199]), xp.asarray([2.0, 3.0, 4.0]), 200)

    def fail_getitem(self: FreqSevSims, index: int):
        raise AssertionError("iteration called the quadratic per-simulation lookup")

    monkeypatch.setattr(FreqSevSims, "__getitem__", fail_getitem)
    observations = [asnumpy(x.values).tolist() for x in fs]
    assert observations[0] == [2.0]
    assert observations[99] == [3.0]
    assert observations[199] == [4.0]
    assert sum(map(len, observations)) == 3
