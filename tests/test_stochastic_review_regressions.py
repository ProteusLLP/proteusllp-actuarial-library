"""Regressions for coupling and dtype safety in stochastic lookups."""

import datetime as dt

import numpy as np

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal.distributions import Categorical


def test_event_slice_has_independent_simulation_axis() -> None:
    parent = FreqSevSims([0, 0, 1], [10.0, 20.0, 30.0], 3)
    aggregate = parent.aggregate()
    subset = parent[0]
    assert subset.n_sims == 2
    assert subset not in parent.coupled_variable_group

    # A copula permutes the source's three simulations, not the two events.
    for member in list(aggregate.coupled_variable_group):
        member._reorder_sims(np.array([2, 0, 1]))
    assert subset.tolist() == [10.0, 20.0]


def test_scalar_indexing_does_not_couple_incompatible_axes() -> None:
    source = StochasticScalar([10, 20, 30])
    mask = StochasticScalar([True, False, True])
    selected = source[mask]
    assert selected.tolist() == [10, 30]
    assert selected not in source.coupled_variable_group
    assert selected not in mask.coupled_variable_group

    # Reordering the original simulation axis must not reorder the subset.
    for member in list(source.coupled_variable_group):
        member._reorder_sims(np.array([2, 1, 0]))
    assert selected.tolist() == [10, 30]

    # Positional indexing can also change the number of simulations.
    positions = StochasticScalar([0, 1])
    sampled = source[positions]
    assert sampled.n_sims == 2
    assert sampled not in source.coupled_variable_group
    assert sampled in positions.coupled_variable_group


def test_resample_events_after_copula_relabels_indices() -> None:
    source = FreqSevSims([1, 0], [10, 20], 2)
    variable = ProteusVariable("line", {"A": source})
    resampled = variable.get_value_at_sim(StochasticScalar([0, 1]))["A"]
    assert isinstance(resampled, FreqSevSims)
    assert np.asarray(resampled.sim_index).tolist() == [0, 1]
    assert resampled.values.tolist() == [20, 10]

    # This also applies to upsampling, which preserves complete events.
    upsampled = variable.upsample(4)["A"]
    assert isinstance(upsampled, FreqSevSims)
    assert np.asarray(upsampled.sim_index).tolist() == [0, 1, 2, 3]
    assert upsampled.values.tolist() == [20, 10, 20, 10]
    assert source not in upsampled.coupled_variable_group


def test_resampled_event_leaves_share_target_coupling_not_source() -> None:
    first = FreqSevSims([1, 0], [10, 20], 2)
    second = first * 2
    pv = ProteusVariable("line", {"one": first, "two": second})
    updated = pv.upsample(4)
    left, right = updated["one"], updated["two"]
    assert left.coupled_variable_group is right.coupled_variable_group
    assert first not in left.coupled_variable_group
    assert second not in left.coupled_variable_group

    for member in list(left.coupled_variable_group):
        member._reorder_sims(np.array([3, 2, 1, 0]))
    assert first.n_sims == 2
    assert np.asarray(first.sim_index).tolist() == [1, 0]


def test_stochastic_lookup_promotes_numeric_and_text_dtypes() -> None:
    numeric = ProteusVariable(
        "type", {"integer": StochasticScalar([1]), "fraction": StochasticScalar([0.75])}
    )
    selector = StochasticScalar(["integer", "fraction", "fraction"])
    result = numeric[selector]
    np.testing.assert_allclose(np.asarray(result), [1.0, 0.75, 0.75])
    assert numeric["integer"] not in result.coupled_variable_group
    assert numeric["fraction"] not in result.coupled_variable_group
    assert selector in result.coupled_variable_group

    names = ProteusVariable(
        "region", {"short": StochasticScalar(["UK"]), "long": StochasticScalar(["Australia"])}
    )
    assert names[StochasticScalar(["long", "short"])].tolist() == ["Australia", "UK"]


def test_stochastic_lookup_preserves_datetime_precision() -> None:
    dates = ProteusVariable(
        "period",
        {
            "date": StochasticScalar([dt.date(2026, 1, 1)]),
            "datetime": StochasticScalar([dt.datetime(2026, 1, 2, 12, 30)]),
        },
    )
    selected = dates[StochasticScalar(["date", "datetime"])]
    assert np.asarray(selected).dtype == np.dtype("datetime64[us]")
    assert selected.tolist() == [dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2, 12, 30)]


def test_mixed_date_datetime_values_do_not_lose_time() -> None:
    mixed = StochasticScalar([dt.date(2026, 1, 1), dt.datetime(2026, 1, 2, 12)])
    assert mixed.values.dtype == np.dtype("datetime64[us]")
    assert mixed.tolist() == [dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2, 12)]
    events = FreqSevSims([0, 1], [dt.date(2026, 1, 1), dt.datetime(2026, 1, 2, 12)], 2)
    assert events.values.dtype == np.dtype("datetime64[us]")
    assert events.values.tolist()[1] == dt.datetime(2026, 1, 2, 12)


def test_categorical_quantile_skips_zero_mass_endpoint() -> None:
    first_only = Categorical(["A", "B"], [1.0, 0.0])
    assert first_only.invcdf(0.0) == "A"
    assert first_only.invcdf(1.0) == "A"
    assert first_only.invcdf(StochasticScalar([0.0, 0.5, 1.0])).tolist() == ["A", "A", "A"]

    middle_only = Categorical(["A", "B", "C"], [0.0, 1.0, 0.0])
    assert middle_only.invcdf(0.0) == "B"
    assert middle_only.invcdf(1.0) == "B"
