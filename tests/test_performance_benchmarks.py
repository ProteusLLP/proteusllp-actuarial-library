"""Regression checks for event-index generation and the benchmark harness."""

import json

import numpy as np
import pytest

from benchmarks.benchmark_core import comparison, markdown_report, run_cases
from pal._maths import asnumpy, xp
from pal.frequency_severity import _get_sims_of_events


@pytest.mark.parametrize(
    "counts",
    [
        [1],
        [0, 0, 0],
        [2, 0, 3, 1, 0, 4],
        [0, 100, 0, 1, 0],
    ],
)
def test_event_to_simulation_indices_match_reference(counts):
    values = xp.asarray(counts, dtype=xp.int64)
    actual = _get_sims_of_events(values)
    expected = np.repeat(np.arange(len(counts)), counts)
    np.testing.assert_array_equal(asnumpy(actual), expected)
    assert actual.dtype.kind in "iu"


def test_event_to_simulation_indices_random_portfolio():
    """Include sparse and dense portfolios without changing event order."""
    rng = np.random.default_rng(2031)
    for mean in (0.1, 2, 12):
        counts = rng.poisson(mean, size=1000).astype(np.int64)
        actual = asnumpy(_get_sims_of_events(xp.asarray(counts)))
        np.testing.assert_array_equal(actual, np.repeat(np.arange(len(counts)), counts))


def test_empty_cpu_counts():
    if xp is not np:
        pytest.skip("The existing GPU path requires at least one simulation")
    assert _get_sims_of_events(np.empty(0, dtype=np.int64)).shape == (0,)


def test_benchmark_results_are_machine_readable():
    report = run_cases(["event_indexing", "aggregate", "stochastic_arithmetic"], [8], warmup=0, repeat=1)
    copy = json.loads(json.dumps(report))
    assert copy["schema_version"] == 1
    assert len(copy["results"]) == 3
    for row in copy["results"]:
        assert row["status"] == "ok", row.get("error")
        assert row["median_ms"] >= 0
        assert row["n_sims"] == 8
    assert "PAL performance benchmarks" in markdown_report(report, {})


def test_baseline_comparison_is_stack_matched():
    present = {
        "environment": {"backend": "cpu", "python": "3.13", "cpu": "A"},
        "results": [{"name": "aggregate", "n_sims": 10, "median_ms": 12.0, "status": "ok"}],
    }
    past = {
        "environment": dict(present["environment"]),
        "results": [{"name": "aggregate", "n_sims": 10, "median_ms": 10.0, "status": "ok"}],
    }
    assert comparison(present, past)[("aggregate", 10)] == pytest.approx(20.0)
    past["environment"]["cpu"] = "B"
    with pytest.raises(ValueError, match="environment"):
        comparison(present, past)
