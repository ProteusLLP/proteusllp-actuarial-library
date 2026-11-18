"""Behavioural tests for per-simulation frequency-severity event lookup.

These tests are retained from the unmerged optimisation in PR #153. They
exercise the public indexing behaviour, not its internal implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pal._maths import asnumpy, xp
from pal.frequency_severity import FreqSevSims
from tests._assertions import assert_array_equal


@pytest.mark.parametrize("reordered", [False, True])
def test_lookup_preserves_event_order_with_missing_scenarios(reordered: bool) -> None:
    """Random access retains original within-simulation order after relabelling."""
    indices = xp.asarray([0, 0, 2, 3, 3, 3, 5], dtype=int)
    values = xp.asarray([10.0, -2.0, 8.0, 2.0, 3.0, 7.0, 11.0])
    events = FreqSevSims(indices, values, n_sims=6)
    if reordered:
        events._reorder_sims(xp.asarray([5, 0, 4, 3, 1, 2]))

    expected_indices = asnumpy(events.sim_index)
    expected_values = asnumpy(events.values)
    for sim in (0, 5, 2, 1, 3, 4, 2, 0):
        selection = events[sim]
        assert_array_equal(selection.values, expected_values[expected_indices == sim])
        assert selection not in events.coupled_variable_group


def test_lookup_results_do_not_alias_original_event_values() -> None:
    """Changing a returned event slice must not change the source event data."""
    events = FreqSevSims(xp.asarray([0, 0, 2]), xp.asarray([1.0, 2.0, 3.0]), n_sims=4)
    selection = events[0]
    selection.values[0] = 999.0
    assert_array_equal(events.values, [1.0, 2.0, 3.0])


def test_lookup_all_empty_scenarios() -> None:
    events = FreqSevSims(xp.asarray([], dtype=int), xp.asarray([], dtype=float), n_sims=4)
    assert [len(events[index].values) for index in range(4)] == [0, 0, 0, 0]


def test_lookup_nonnumeric_values_after_reordering() -> None:
    """Nonnumeric event values retain their type and order after relabelling."""
    events = FreqSevSims(xp.asarray([0, 1, 1, 3]), np.array(["a", "b", "c", "d"]), 4)
    assert_array_equal(events[1].values, ["b", "c"])
    events._reorder_sims(xp.asarray([3, 1, 0, 2]))

    indices = asnumpy(events.sim_index)
    values = asnumpy(events.values)
    for sim in range(4):
        assert_array_equal(events[sim].values, values[indices == sim])
