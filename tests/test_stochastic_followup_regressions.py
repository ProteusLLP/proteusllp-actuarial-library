"""Regression tests for the follow-up review of nonnumeric stochastic values."""

import datetime as dt

import numpy as np
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal.distributions import Categorical, Uniform
from tests._assertions import assert_array_equal


def test_numeric_object_arrays_remain_supported() -> None:
    scalar = StochasticScalar(np.array([1, 2], dtype=object))
    assert_array_equal(scalar.values, [1, 2])
    assert_array_equal((scalar + 1).values, [2, 3])

    events = FreqSevSims([0, 1], np.array([2, 3], dtype=object), 2)
    assert_array_equal(events.aggregate().values, [2, 3])


def test_mixed_numpy_and_python_dates_preserve_precision() -> None:
    dates = StochasticScalar([dt.date(2026, 1, 1), np.datetime64("2026-01-02")])
    assert dates.values.dtype.kind == "M"
    assert dates.tolist() == [dt.date(2026, 1, 1), dt.date(2026, 1, 2)]

    precise = StochasticScalar([dt.date(2026, 1, 1), np.datetime64("2026-01-02T12:00:00.123456789")])
    assert precise.values.dtype == np.dtype("datetime64[ns]")
    assert precise.values[1] == np.datetime64("2026-01-02T12:00:00.123456789")

    categories = Categorical([dt.date(2026, 1, 1), np.datetime64("2026-01-02")])
    assert np.isclose(categories.cdf(dt.date(2026, 1, 1)), 0.5)


def test_uniform_date_cast_and_categorical_nat_rejection() -> None:
    dist = Uniform(dt.date(2026, 1, 1), dt.date(2026, 1, 3))
    assert dist.invcdf(1.0) == np.datetime64("2026-01-03")
    with pytest.raises(ValueError, match="NaT"):
        Categorical([np.datetime64("NaT")])


def test_categorical_tail_zero_mass_and_date_query() -> None:
    labels = [str(i) for i in range(11)]
    dist = Categorical(labels, [0.1] * 10 + [0.0])
    assert dist.invcdf(1.0) == "9"
    assert dist.invcdf(StochasticScalar([1.0])).tolist() == ["9"]

    dates = Categorical([dt.date(2026, 1, 1), dt.datetime(2026, 1, 2, 12)])
    first_cdf = dates.cdf(dt.date(2026, 1, 1))
    second_cdf = dates.cdf(dt.datetime(2026, 1, 2, 12))
    assert np.ndim(first_cdf) == 0
    assert np.ndim(second_cdf) == 0
    assert np.isclose(first_cdf, 0.5)
    assert np.isclose(second_cdf, 1.0)
    with pytest.raises(ValueError, match="Unknown"):
        dates.cdf(dt.date(2026, 1, 3))


def test_categorical_large_vector_lookup_without_cross_product() -> None:
    labels = [f"category_{i:04}" for i in range(1000)]
    distribution = Categorical(labels)
    queries = np.resize(np.asarray(labels), 25_000)
    probabilities = distribution.cdf(queries)
    assert probabilities.shape == (25_000,)
    assert np.isclose(probabilities[123], 0.124)
    assert np.isclose(probabilities[999], 1.0)


def test_unsigned_stochastic_indices_validate_before_cast() -> None:
    p = ProteusVariable("item", {"a": StochasticScalar([10]), "b": StochasticScalar([20])})
    with pytest.raises(IndexError, match="out of range"):
        p[StochasticScalar(np.array([np.iinfo(np.uint64).max], dtype=np.uint64))]
    assert_array_equal(p[StochasticScalar(np.array([0, 1], dtype=np.uint64))], [10, 20])
    assert_array_equal(p[StochasticScalar(np.array([-1, 0], dtype=np.int64))], [20, 10])


def test_numeric_selection_with_stochastic_scalar_and_event_selector() -> None:
    p = ProteusVariable("item", {"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([4, 5, 6])})
    assert_array_equal(p[StochasticScalar([0, 1, 0])], [1, 5, 3])
    selected = p[FreqSevSims([0, 1, 1], [0, 1, 0], 3)]
    assert_array_equal(selected.values, [1, 5, 2])


def test_where_with_string_branches_and_datetime_scalars() -> None:
    condition = StochasticScalar([True, False])
    assert np.where(condition, ["UK", "US"], ("US", "UK")).tolist() == ["UK", "UK"]
    dates = np.where(condition, np.datetime64("2026-01-01"), np.datetime64("2026-01-02"))
    assert dates.values.dtype.kind == "M"
    assert dates.tolist() == [dt.date(2026, 1, 1), dt.date(2026, 1, 2)]

    events = FreqSevSims([0, 1], [True, False], 2)
    assert np.where(events, ["UK", "US"], ("US", "UK")).values.tolist() == ["UK", "UK"]


def test_date_ufunc_with_raw_host_operand() -> None:
    offsets = StochasticScalar([1, 2])
    dates = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")
    added = np.add(offsets, dates)
    assert added.values.dtype.kind == "M"
    assert added.tolist() == [dt.date(2026, 1, 2), dt.date(2026, 1, 4)]

    events = FreqSevSims([0, 1], [1, 2], 2)
    shifted = np.add(events, dates)
    assert shifted.values.dtype.kind == "M"
    assert shifted.values.tolist() == [dt.date(2026, 1, 2), dt.date(2026, 1, 4)]


def test_host_ufunc_out_buffers_write_back_to_gpu() -> None:
    labels = StochasticScalar(["UK", "US"])
    output = StochasticScalar([False, False])
    np.equal(labels, "UK", out=output)
    assert_array_equal(output, [True, False])

    fs = FreqSevSims([0, 1], ["UK", "US"], 2)
    fs_out = FreqSevSims([0, 1], [False, False], 2)
    np.equal(fs, "UK", out=fs_out)
    assert_array_equal(fs_out.values, [True, False])


def test_event_lookup_chooses_dtype_from_selected_entries() -> None:
    numeric = FreqSevSims([0, 1], [5, 6], 2)
    dates = FreqSevSims([0, 1], [dt.date(2026, 1, 1), dt.date(2026, 1, 2)], 2)
    p = ProteusVariable("value", {"numeric": numeric, "date": dates})
    selected = p[StochasticScalar(["date", "date"])]
    assert selected.values.dtype.kind == "M"
    assert selected.values.tolist() == [dt.date(2026, 1, 1), dt.date(2026, 1, 2)]


def test_host_where_mask_with_date_string_ufunc() -> None:
    labels = StochasticScalar(["UK", "US"])
    mask = StochasticScalar([True, False])
    out = StochasticScalar([False, False])
    np.equal(labels, "UK", out=out, where=mask.values)
    assert_array_equal(out, [True, False])
