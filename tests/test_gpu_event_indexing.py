"""Correctness and regression checks for candidate GPU event-index algorithms."""

import numpy as np
import pytest
from benchmarks.benchmark_gpu_event_indexing import METHODS, _method, make_counts

from pal._maths import asnumpy, xp
from pal.frequency_severity import FreqSevSims


@pytest.mark.parametrize("scenario", ["sparse", "dense", "skewed"])
def test_event_index_fixture_is_reproducible(scenario):
    first = make_counts(scenario, 1050)
    np.testing.assert_array_equal(first, make_counts(scenario, 1050))
    assert first.dtype == np.int64
    assert np.all(first >= 0)
    if scenario == "skewed":
        assert np.count_nonzero(first) == 2


@pytest.mark.parametrize("scenario", ["sparse", "dense", "skewed"])
@pytest.mark.parametrize("n_sims", [0, 1, 1031])
@pytest.mark.parametrize("method", METHODS)
def test_gpu_event_index_candidates_match_numpy(scenario, n_sims, method):
    if xp is np:
        pytest.skip("CUDA-only indexing candidates")
    counts = make_counts(scenario, n_sims)
    try:
        actual = _method(method)(xp.asarray(counts))
    except (TypeError, NotImplementedError) as exc:
        if method != "repeat":
            raise
        pytest.skip(f"This version of CuPy cannot repeat device counts: {exc}")
    assert isinstance(actual, xp.ndarray)
    assert actual.dtype == xp.int64
    np.testing.assert_array_equal(asnumpy(actual), np.repeat(np.arange(n_sims), counts))


@pytest.mark.parametrize("method", METHODS)
def test_gpu_event_indices_preserve_aggregate_and_frequency(method):
    if xp is np:
        pytest.skip("CUDA-only event-indexing regression")
    counts = np.array([0, 3, 0, 2, 0, 1], dtype=np.int64)
    try:
        index = _method(method)(xp.asarray(counts))
    except (TypeError, NotImplementedError) as exc:
        if method != "repeat":
            raise
        pytest.skip(f"This version of CuPy cannot repeat device counts: {exc}")

    values = xp.asarray([2.0, 4.0, 6.0, 1.5, 2.5, 7.0])
    events = FreqSevSims(index, values, len(counts))
    np.testing.assert_array_equal(asnumpy(events.count().values), counts)
    np.testing.assert_array_equal(asnumpy(events.aggregate().values), [0, 12, 0, 4, 0, 7])
