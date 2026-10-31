"""Date, categorical and keyed stochastic simulation regression tests."""

import datetime as dt

import numpy as np
import pytest

from pal import FreqSevSims, FrequencySeverityModel, ProteusVariable, StochasticScalar
from pal.distributions import Categorical, Poisson, Uniform


def test_stochastic_scalar_strings_and_dates() -> None:
    labels = StochasticScalar(["UK", "US", "UK"])
    assert labels.values.dtype.kind == "U"
    np.testing.assert_array_equal(np.asarray(labels == "UK"), [True, False, True])
    np.testing.assert_array_equal(np.asarray(labels), ["UK", "US", "UK"])
    np.testing.assert_array_equal(np.where(labels == "UK", "GBP", "USD").values, ["GBP", "USD", "GBP"])
    assert labels.upsample(5).tolist() == ["UK", "US", "UK", "UK", "US"]

    dates = StochasticScalar([dt.date(2026, 1, 1), dt.date(2026, 1, 3)])
    assert dates.values.dtype == np.dtype("datetime64[D]")
    np.testing.assert_array_equal(np.asarray(dates >= dt.date(2026, 1, 2)), [False, True])
    assert dates.tolist() == [dt.date(2026, 1, 1), dt.date(2026, 1, 3)]

    datetimes = StochasticScalar([dt.datetime(2026, 1, 1, 10), dt.datetime(2026, 1, 1, 12)])
    assert datetimes.values.dtype == np.dtype("datetime64[us]")
    assert len(datetimes[StochasticScalar([True, False])]) == 1
    with pytest.raises(TypeError, match="consistent"):
        StochasticScalar([dt.date(2026, 1, 1), "different"])


def test_frequency_severity_categorical_event_values() -> None:
    locations = FreqSevSims([0, 0, 2], ["London", "Paris", "London"], 3)
    assert locations[0].tolist() == ["London", "Paris"]
    assert locations[1].tolist() == []
    assert locations.count().tolist() == [2.0, 0.0, 1.0]
    assert locations.copy().values.dtype.kind == "U"
    np.testing.assert_array_equal(np.asarray(locations == "London"), [True, False, True])
    with pytest.raises(TypeError, match="numeric"):
        locations.aggregate()

    dates = FreqSevSims(
        [0, 2],
        [dt.date(2026, 1, 1), dt.date(2026, 3, 2)],
        3,
    )
    assert dates.values.dtype.kind == "M"
    np.testing.assert_array_equal(np.asarray(dates < dt.date(2026, 2, 1)), [True, False])


def test_uniform_dates_are_inclusive_and_have_exact_cdf() -> None:
    dist = Uniform(dt.date(2026, 1, 1), dt.date(2026, 1, 3))
    assert dist.invcdf(0) == np.datetime64("2026-01-01")
    assert dist.invcdf(1) == np.datetime64("2026-01-03")
    np.testing.assert_allclose(
        [
            dist.cdf(dt.date(2025, 12, 31)),
            dist.cdf(dt.date(2026, 1, 1)),
            dist.cdf(dt.date(2026, 1, 2)),
            dist.cdf(dt.date(2026, 1, 3)),
        ],
        [0.0, 1 / 3, 2 / 3, 1.0],
    )
    generated = dist.generate(5000, np.random.default_rng(5))
    assert generated.values.dtype == np.dtype("datetime64[D]")
    counts = np.array([(generated.values == np.datetime64(f"2026-01-0{i}")).sum() for i in range(1, 4)])
    assert np.all(np.abs(counts / counts.sum() - 1 / 3) < 0.03)

    times = Uniform(dt.datetime(2026, 1, 1, 12), dt.datetime(2026, 1, 1, 14))
    assert times.invcdf(0.5) == np.datetime64("2026-01-01T13:00:00")
    assert times.cdf(dt.datetime(2026, 1, 1, 13)) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="upper date"):
        Uniform(dt.date(2026, 1, 3), dt.date(2026, 1, 1))


def test_categorical_distribution_and_frequency_severity_model() -> None:
    dist = Categorical(["Motor", "Property", "Liability"], [0.2, 0.3, 0.5])
    assert dist.invcdf(0.1) == "Motor"
    assert dist.invcdf(0.5) == "Liability"
    assert dist.cdf("Property") == pytest.approx(0.5)
    labels = dist.generate(8000, np.random.default_rng(4))
    assert labels.values.dtype.kind == "U"
    assert np.mean(labels.values == "Liability") == pytest.approx(0.5, abs=0.025)
    with pytest.raises(ValueError, match="sum to one"):
        Categorical(["A", "B"], [0.2, 0.4])

    model = FrequencySeverityModel(
        Poisson(mean=2),
        Uniform(dt.date(2026, 1, 1), dt.date(2026, 1, 31)),
    )
    events = model.generate(100, np.random.default_rng(42))
    assert events.values.dtype.kind == "M"
    assert len(events.values) == int(events.count().values.sum())


def test_stochastic_dereferencing_scenarios() -> None:
    values = ProteusVariable(
        "region",
        {
            "UK": StochasticScalar([1, 2, 3, 4]),
            "US": StochasticScalar([10, 20, 30, 40]),
        },
    )
    region = StochasticScalar(["UK", "US", "UK", "US"])
    result = values[region]
    assert isinstance(result, StochasticScalar)
    assert result.tolist() == [1, 20, 3, 40]
    assert region in result.coupled_variable_group
    assert values["UK"] in result.coupled_variable_group
    assert values["US"] in result.coupled_variable_group

    assert values[StochasticScalar([0, 1, 1, 0])].tolist() == [1, 20, 30, 4]
    with pytest.raises(KeyError, match="Unknown region"):
        values[StochasticScalar(["UK", "FR", "UK", "US"])]
    with pytest.raises(IndexError, match="out of range"):
        values[StochasticScalar([0, 1, 2, 0])]


def test_stochastic_dereferencing_events_and_event_outputs() -> None:
    by_region = ProteusVariable(
        "region",
        {
            "UK": StochasticScalar([100, 200, 300]),
            "US": StochasticScalar([10, 20, 30]),
        },
    )
    event_region = FreqSevSims([0, 0, 2], ["UK", "US", "US"], 3)
    selected = by_region[event_region]
    assert isinstance(selected, FreqSevSims)
    assert selected.values.tolist() == [100, 10, 30]
    assert selected.sim_index.tolist() == [0, 0, 2]
    assert event_region in selected.coupled_variable_group

    event_values = ProteusVariable(
        "region",
        {
            "UK": FreqSevSims([0, 1, 1, 2], [1, 2, 3, 4], 3),
            "US": FreqSevSims([0, 0, 1, 2], [10, 20, 30, 40], 3),
        },
    )
    region = StochasticScalar(["US", "UK", "US"])
    selected_events = event_values[region]
    assert isinstance(selected_events, FreqSevSims)
    assert selected_events.sim_index.tolist() == [0, 0, 1, 1, 2]
    assert selected_events.values.tolist() == [10, 20, 2, 3, 40]
