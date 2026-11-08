"""Independent invariants and regression tests for frequency-severity simulations."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from pal._maths import xp
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel
from pal.variables import ProteusVariable, StochasticScalar
from tests._assertions import assert_allclose, assert_array_equal, host_values


def _per_simulation_values(indices: np.ndarray, values: np.ndarray, n_sims: int) -> list[list[float]]:
    """Construct an independent, order-preserving event list for each simulation."""
    return [[float(value) for idx, value in zip(indices, values) if int(idx) == sim] for sim in range(n_sims)]


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
@pytest.mark.parametrize("n_sims", [1, 3, 17, 64])
def test_event_reductions_against_independent_python_reference(seed: int, n_sims: int) -> None:
    """Count, aggregate and occurrence must agree with per-scenario event lists."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 6, size=n_sims)
    indices = np.repeat(np.arange(n_sims), counts)
    values = rng.uniform(0.0, 200.0, size=indices.size)
    events = FreqSevSims(indices, values, n_sims)
    reference = _per_simulation_values(indices, values, n_sims)

    assert_array_equal(events.count(), [len(group) for group in reference])
    assert_allclose(events.aggregate(), [sum(group) for group in reference], rtol=1e-12, atol=1e-10)
    assert_allclose(events.occurrence(), [max(group, default=0.0) for group in reference])
    assert np.isclose(host_values(events.aggregate()).sum(), values.sum(), rtol=1e-12)
    assert events.aggregate().coupled_variable_group is events.coupled_variable_group


@pytest.mark.parametrize(
    "indices,values,n_sims",
    [
        ([], [], 4),
        ([2, 2], [-4.0, 1.0], 4),
        ([1, 0, 1, 3], [2.5, -7.0, -1.5, 9.0], 5),
    ],
)
def test_aggregate_signed_losses_and_empty_scenarios(indices: list[int], values: list[float], n_sims: int) -> None:
    """Aggregation must preserve signed values and zero-event simulations."""
    events = FreqSevSims(indices, values, n_sims)
    reference = _per_simulation_values(np.asarray(indices), np.asarray(values), n_sims)
    assert_allclose(events.aggregate(), [sum(group) for group in reference])
    assert_array_equal(events.count(), [len(group) for group in reference])


@pytest.mark.parametrize(
    "counts,target",
    [
        ([2, 0, 1], 7),
        ([2, 0, 1], 3),
        ([2, 0, 1], 2),
        ([0, 0, 0], 8),
        ([1], 5),
        ([0, 3, 2, 0], 9),
    ],
)
def test_direct_upsample_preserves_complete_event_lists(counts: list[int], target: int) -> None:
    """Each target simulation must contain precisely its selected source's events."""
    original_indices = np.repeat(np.arange(len(counts)), counts)
    original_values = np.arange(1, original_indices.size + 1, dtype=float) * 1.25
    original = FreqSevSims(original_indices, original_values, len(counts))
    reference = _per_simulation_values(original_indices, original_values, len(counts))

    result = original.upsample(target)
    expected_groups = [reference[i % len(counts)] for i in range(target)]
    expected_indices = [i for i, group in enumerate(expected_groups) for _ in group]
    expected_values = [value for group in expected_groups for value in group]

    assert result.n_sims == target
    assert_array_equal(result.sim_index, expected_indices)
    assert_allclose(result.values, expected_values)
    assert_allclose(result.aggregate(), [sum(group) for group in expected_groups])
    assert_array_equal(result.count(), [len(group) for group in expected_groups])
    if target == original.n_sims:
        assert original in result.coupled_variable_group  # same-size resampling is a coupled copy
    else:
        assert original not in result.coupled_variable_group


def test_direct_upsample_preserves_event_order_with_unsorted_input() -> None:
    """The original within-scenario order matters even when rows are unsorted."""
    original = FreqSevSims([2, 0, 2, 0, 1], [1.0, 10.0, 2.0, 20.0, 5.0], 3)
    result = original.upsample(5)
    assert_array_equal(result.sim_index, [0, 0, 1, 2, 2, 3, 3, 4])
    assert_allclose(result.values, [10, 20, 5, 1, 2, 10, 20, 5])


@pytest.mark.parametrize("target", [0, -2])
def test_direct_upsample_rejects_invalid_simulation_count(target: int) -> None:
    """Invalid targets must fail before constructing invalid event indices."""
    with pytest.raises(ValueError, match="n_sims"):
        FreqSevSims([0], [2.0], 1).upsample(target)


@pytest.mark.parametrize("seed", [1, 19, 41])
def test_arithmetic_aligns_scenario_values_and_retains_coupling(seed: int) -> None:
    """Arithmetic with scenario-level values must broadcast by event simulation."""
    rng = np.random.default_rng(seed)
    indices = np.repeat(np.arange(9), rng.integers(0, 5, size=9))
    values = rng.uniform(0.5, 10.0, size=indices.size)
    multipliers = rng.uniform(0.5, 3.0, size=9)
    events = FreqSevSims(indices, values, 9)
    factor = StochasticScalar(multipliers)

    result = (events * factor) + factor
    assert_allclose(result.values, values * multipliers[indices] + multipliers[indices])
    assert_array_equal(result.sim_index, indices)
    assert result.coupled_variable_group is events.coupled_variable_group
    assert result.coupled_variable_group is factor.coupled_variable_group
    assert_allclose(result.aggregate(), np.bincount(indices, weights=host_values(result), minlength=9))


@pytest.mark.parametrize("operation", ["add", "multiply", "maximum"])
def test_aligned_event_arrays_combine_elementwise(operation: str) -> None:
    """Event-wise binary operations are meaningful for identical event layouts."""
    left = FreqSevSims([0, 0, 2], [1.0, 3.0, 5.0], 4)
    right = FreqSevSims([0, 0, 2], [2.0, 4.0, 6.0], 4)
    result = getattr(np, operation)(left, right)
    expected = getattr(np, operation)([1.0, 3.0, 5.0], [2.0, 4.0, 6.0])
    assert_allclose(result.values, expected)
    assert_array_equal(result.sim_index, [0, 0, 2])
    assert left.coupled_variable_group is right.coupled_variable_group
    assert result.coupled_variable_group is left.coupled_variable_group


@pytest.mark.parametrize("operation", ["add", "multiply", "maximum"])
def test_binary_operations_reject_differently_assigned_events(operation: str) -> None:
    """Equal array lengths do not imply that individual event rows are aligned."""
    left = FreqSevSims([0, 0, 1], [1.0, 3.0, 5.0], 2)
    right = FreqSevSims([0, 1, 1], [2.0, 4.0, 6.0], 2)
    with pytest.raises(ValueError, match="[Ee]vent.*(index|align|simulat)"):
        getattr(np, operation)(left, right)
    assert left.coupled_variable_group is not right.coupled_variable_group


def test_binary_operations_reject_differing_simulation_counts() -> None:
    """Identical event indices cannot represent the same layout with different n_sims."""
    left = FreqSevSims([0, 1], [10.0, 20.0], 2)
    right = FreqSevSims([0, 1], [30.0, 40.0], 3)
    with pytest.raises(ValueError, match="[Ee]vent.*(index|align|simulat)"):
        _ = left + right


def test_where_rejects_differently_assigned_event_branches() -> None:
    """Conditional event-wise selection must validate all frequency-severity inputs."""
    condition = FreqSevSims([0, 0, 1], [True, False, True], 2)
    left = FreqSevSims([0, 0, 1], [10, 20, 30], 2)
    right = FreqSevSims([0, 1, 1], [40, 50, 60], 2)
    with pytest.raises(ValueError, match="[Ee]vent.*(index|align|simulat)"):
        np.where(condition, left, right)


def test_where_aligned_branches_merge_coupling() -> None:
    """A selected event result depends on both the condition and its branches."""
    condition = FreqSevSims([0, 0, 1], [True, False, True], 2)
    left = FreqSevSims([0, 0, 1], [10, 20, 30], 2)
    right = FreqSevSims([0, 0, 1], [40, 50, 60], 2)
    result = np.where(condition, left, right)
    assert_array_equal(result.values, [10, 50, 30])
    assert_array_equal(result.sim_index, [0, 0, 1])
    assert result.coupled_variable_group is condition.coupled_variable_group
    assert result.coupled_variable_group is left.coupled_variable_group
    assert result.coupled_variable_group is right.coupled_variable_group


@pytest.mark.parametrize(
    "mask_index,mask_n_sims",
    [
        ([0, 1, 1], 2),
        ([0, 0, 1], 3),
    ],
)
def test_ufunc_where_rejects_misaligned_event_mask_before_writing(mask_index: list[int], mask_n_sims: int) -> None:
    """An event-wise where mask must have the same simulation assignment."""
    events = FreqSevSims([0, 0, 1], [10, 20, 30], 2)
    destination = FreqSevSims([0, 0, 1], [-1, -1, -1], 2)
    mask = FreqSevSims(mask_index, [True, False, True], mask_n_sims)

    with pytest.raises(ValueError, match="Event simulation indices must match"):
        np.add(events, 1, out=destination, where=mask)

    assert_array_equal(destination.values, [-1, -1, -1])
    assert destination.coupled_variable_group is not mask.coupled_variable_group
    assert events.coupled_variable_group is not mask.coupled_variable_group


def test_ufunc_where_aligned_event_mask_updates_selected_losses() -> None:
    """The aligned event mask controls only the corresponding event positions."""
    events = FreqSevSims([0, 0, 1], [10, 20, 30], 2)
    destination = FreqSevSims([0, 0, 1], [-1, -1, -1], 2)
    mask = FreqSevSims([0, 0, 1], [True, False, True], 2)

    result = np.add(events, 1, out=destination, where=mask)
    assert_array_equal(destination.values, [11, -1, 31])
    assert_array_equal(result.sim_index, [0, 0, 1])
    assert result.coupled_variable_group is events.coupled_variable_group
    assert result.coupled_variable_group is mask.coupled_variable_group
    assert result.coupled_variable_group is destination.coupled_variable_group


def test_ufunc_where_scenario_level_mask_broadcasts_to_events() -> None:
    """A StochasticScalar mask is applied to every event in its simulation."""
    events = FreqSevSims([0, 0, 1], [10, 20, 30], 2)
    destination = FreqSevSims([0, 0, 1], [-1, -1, -1], 2)
    mask = StochasticScalar([False, True])

    np.add(events, 1, out=destination, where=mask)
    assert_array_equal(destination.values, [-1, -1, 31])
    assert destination.coupled_variable_group is mask.coupled_variable_group


def test_unary_and_scalar_ufuncs_skip_self_layout_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operations involving only this event layout should not scan its indices."""
    events = FreqSevSims([0, 0, 1], [1.0, 2.0, 3.0], 2)
    original_array_equal = xp.array_equal
    calls = 0

    def tracked_array_equal(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_array_equal(*args, **kwargs)

    monkeypatch.setattr(xp, "array_equal", tracked_array_equal)
    transformed = np.exp(events)
    shifted = events + 1.0
    assert calls == 0
    monkeypatch.setattr(xp, "array_equal", original_array_equal)
    assert_allclose(transformed.values, np.exp([1.0, 2.0, 3.0]))
    assert_allclose(shifted.values, [2.0, 3.0, 4.0])


def test_array_equal_allows_differing_event_layouts() -> None:
    """Equality is a predicate, not an aligned binary arithmetic operation."""
    source = FreqSevSims([0, 0, 1], [1, 2, 3], 2)
    differing_layout = FreqSevSims([0, 1, 1], [1, 2, 4], 2)
    differing_sim_count = FreqSevSims([0, 0, 1], [1, 5, 3], 3)

    assert not bool(np.array_equal(source, differing_layout))
    assert not bool(np.array_equal(source, differing_sim_count))
    assert bool(np.array_equal(source, source))
    assert source.coupled_variable_group is not differing_layout.coupled_variable_group


def test_copy_and_deep_copy_have_distinct_coupling_contracts() -> None:
    """Copies have independent buffers; deep_copy is not coupled to its source."""
    original = FreqSevSims([0, 2], [4.0, 8.0], 3)
    copied = original.copy()
    detached = original.deep_copy()
    assert copied.coupled_variable_group is original.coupled_variable_group
    assert detached.coupled_variable_group is not original.coupled_variable_group
    original.values[0] = 99.0
    assert_allclose(copied.values, [4.0, 8.0])
    assert_allclose(detached.values, [4.0, 8.0])


def test_host_date_and_label_event_arrays_preserve_dtype_on_copy_and_resample() -> None:
    """Nonnumeric event values remain valid even with the CuPy backend enabled."""
    labels = FreqSevSims([0, 2], ["UK", "US"], 3)
    dates = FreqSevSims([0, 2], [dt.date(2026, 1, 1), dt.date(2026, 6, 1)], 3)
    for original in (labels, dates):
        copied = original.copy()
        resampled = original.upsample(5)
        assert copied.values.dtype == original.values.dtype
        assert resampled.values.dtype == original.values.dtype
        assert_array_equal(resampled.sim_index, [0, 2, 3])
        assert_array_equal(resampled.values, [original.values[0], original.values[1], original.values[0]])


class _FixedDistribution:
    """Minimal distribution double to verify frequency/severity orchestration."""

    def __init__(self, expected_n: int, values: list[int] | list[float]) -> None:
        self.expected_n = expected_n
        self.values = values

    def generate(self, n_sims: int, rng: object) -> StochasticScalar:
        assert n_sims == self.expected_n
        return StochasticScalar(self.values)


@pytest.mark.parametrize(
    "counts,severities",
    [
        ([2, 0, 1, 0], [10.0, 20.0, 5.0]),
        ([0, 0, 0, 0], [42.0]),
    ],
)
def test_frequency_severity_model_preserves_event_counts_and_empty_output(
    counts: list[int], severities: list[float]
) -> None:
    """The compound model must construct the right event table, including zero events."""
    frequency = _FixedDistribution(len(counts), counts)
    severity = _FixedDistribution(len(severities), severities)
    model = FrequencySeverityModel(frequency, severity)
    result = model.generate(len(counts))
    expected_indices = np.repeat(np.arange(len(counts)), counts)
    assert_array_equal(result.sim_index, expected_indices)
    assert_array_equal(result.count(), counts)
    assert_allclose(result.values, severities if expected_indices.size else [])
    assert result.n_sims == len(counts)
    assert isinstance(result.values, xp.ndarray)


def test_nested_variable_resampling_agrees_with_direct_frequency_severity_upsample() -> None:
    """Direct and nested resampling should use the same scenario selection rule."""
    source = FreqSevSims([0, 0, 2], [10.0, 20.0, 30.0], 3)
    direct = source.upsample(7)
    nested = ProteusVariable("line", {"loss": source}).upsample(7)["loss"]
    assert_array_equal(direct.sim_index, nested.sim_index)
    assert_allclose(direct.values, nested.values)
    assert_allclose(direct.aggregate(), nested.aggregate())
