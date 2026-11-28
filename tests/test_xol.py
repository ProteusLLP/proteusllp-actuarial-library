"""Tests for excess of loss (XoL) reinsurance layer functionality.

Tests covering XoL layer application including limits, excesses, franchise
deductibles, aggregate limits, reinstatement premiums, and complex layering.
"""

import numpy as np
import pytest

from pal.contracts import XoL
from pal.frequency_severity import FreqSevSims
from tests._assertions import allclose


def test_xol_no_agg():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        premium=1000,
    )
    claims = FreqSevSims(np.array([0, 0, 0, 0]), np.array([100000, 800000, 500000, 200000]), 1)
    result = layer.apply(claims)
    assert result.recoveries.values.tolist() == [0, 500000, 250000, 0]
    assert result.reinstatement_premium is None


def test_xol_franchise():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=0,
        premium=1000,
        franchise=250000,
    )
    claims = FreqSevSims(np.array([0, 0, 0, 0]), np.array([100000, 800000, 400000, 200000]), 1)
    result = layer.apply(claims)
    assert result.recoveries.values.tolist() == [0, 500000, 400000, 0]
    assert result.reinstatement_premium is None


def test_xol_reinstatements():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        aggregate_limit=1000000,
        premium=1000,
        reinstatement_cost=[1],
    )
    claims = FreqSevSims(np.array([0, 0, 0, 0]), np.array([100000, 800000, 500000, 200000]), 1)
    result = layer.apply(claims)
    assert result.recoveries.values.tolist() == [0, 500000, 250000, 0]
    assert result.reinstatement_premium is not None
    assert result.reinstatement_premium.tolist() == [1000]


def test_xol_multiple_reinstatements():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        aggregate_limit=2000000,
        premium=1000,
        reinstatement_cost=[1, 0.5, 0, 0],
    )
    claims = FreqSevSims(np.array([0, 0, 0, 0]), np.array([100000, 600000, 500000, 200000]), 1)
    result = layer.apply(claims)
    assert result.recoveries.values.tolist() == [0, 350000, 250000, 0]
    assert result.reinstatement_premium is not None
    # 600,000 recovered: the first 500,000 is reinstated by the first
    # reinstatement at 100% and the next 100,000 (a fifth of the limit) by the
    # second reinstatement at 50%.
    assert allclose(result.reinstatement_premium, np.array([1000 * (1 + 0.5 * 0.2)]))


def _mata_reinstatement_premium(aggregate, limit, premium, costs, reinstatements):
    """Pro rata reinstatement premium of Mata (2000), band by band.

    Losses between k and k+1 limits are reinstated by reinstatement k+1 at
    costs[k]. Only the first `reinstatements` limits of loss are reinstated;
    a fractional final reinstatement restores only that fraction of the limit.
    """
    reinstated = min(aggregate, reinstatements * limit)
    return premium * sum(cost * min(max(reinstated - k * limit, 0), limit) / limit for k, cost in enumerate(costs))


@pytest.mark.parametrize(
    "costs,aggregate_limit",
    [
        ([1, 0.5, 0], 2_000_000),
        ([0, 1, 2], 2_000_000),
        ([1, 1, 1], 2_000_000),
        ([0.25, 0.75], 1_250_000),  # 1.5 reinstatements: the second is partial
        ([1, 0.5, 0.25, 0.1], 2_500_000),  # more costs than reinstatements
    ],
)
def test_reinstatement_premium_charges_each_reinstatement_at_its_own_cost(costs, aggregate_limit):
    """Each band of aggregate recovery is charged at its own reinstatement cost.

    Claims never exceed the limit and the excess is zero, so each simulation's
    aggregate recovery is min(total claims, aggregate limit), independently of
    PAL. Totals include whole multiples of the limit, values just either side
    of them and amounts beyond the aggregate limit.
    """
    limit, premium = 500_000, 1000
    totals = [0, 250_000, 499_999, 500_000, 500_001, 600_000, 999_999, 1_000_000, 1_250_000, 1_500_000, 3_000_000]
    sim_index, losses = [], []
    for sim, total in enumerate(totals):
        full, remainder = divmod(total, limit)
        claims = [limit] * full + ([remainder] if remainder else [])
        sim_index += [sim] * len(claims)
        losses += claims
    layer = XoL(
        "Layer", limit=limit, excess=0, premium=premium, aggregate_limit=aggregate_limit, reinstatement_cost=costs
    )
    result = layer.apply(FreqSevSims(np.array(sim_index), np.array(losses, dtype=float), len(totals)))
    assert result.reinstatement_premium is not None
    reinstatements = aggregate_limit / limit - 1
    expected = [
        _mata_reinstatement_premium(min(total, aggregate_limit), limit, premium, costs, reinstatements)
        for total in totals
    ]
    np.testing.assert_allclose(np.asarray(result.reinstatement_premium.values), expected, rtol=1e-12, atol=1e-9)


def test_reinstatement_premium_never_decreases_as_recoveries_increase():
    """With reinstatement costs that fall, more recovery must still never reduce the premium."""
    limit = 100.0
    totals = np.linspace(0, 450, 901)
    layer = XoL("Layer", limit=limit, excess=0, premium=1, aggregate_limit=400, reinstatement_cost=[1, 0.25, 0])
    claims = FreqSevSims(np.repeat(np.arange(totals.size), 5), np.repeat(totals / 5, 5), totals.size)
    premium = np.asarray(layer.apply(claims).reinstatement_premium.values)
    assert np.all(np.diff(premium) >= -1e-12)
    assert premium[-1] == pytest.approx(1.25)


def test_reinstatement_costs_are_validated():
    """Reinstatement costs need an aggregate limit and a cost for every reinstatement."""
    with pytest.raises(ValueError, match="no aggregate limit"):
        XoL("Layer", limit=100, excess=0, premium=1, reinstatement_cost=[1])
    with pytest.raises(ValueError, match="allows 2 reinstatements but has 1"):
        XoL("Layer", limit=100, excess=0, premium=1, aggregate_limit=300, reinstatement_cost=[1])
    with pytest.raises(ValueError, match="allows 1.5 reinstatements but has 1"):
        XoL("Layer", limit=100, excess=0, premium=1, aggregate_limit=250, reinstatement_cost=[1])
    # 0.3 / 0.1 - 1 rounds above two; two costs still cover both reinstatements.
    XoL("Layer", limit=0.1, excess=0, premium=1, aggregate_limit=0.3, reinstatement_cost=[1, 1])


def test_xol_aggregate_limit():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        aggregate_limit=1000000,
        premium=1000,
        reinstatement_cost=[1],
    )
    claims = FreqSevSims(np.array([0, 0, 0, 0]), np.array([100000, 800000, 500000, 1000000]), 1)
    result = layer.apply(claims)
    assert result.recoveries.aggregate().tolist() == [1000000]
    assert result.recoveries.values.tolist() == [
        0,
        500000 * (1000000 / 1250000),
        250000 * (1000000 / 1250000),
        500000 * (1000000 / 1250000),
    ]
    assert result.reinstatement_premium is not None
    assert allclose(result.reinstatement_premium, np.array([1000]))


def test_xol_aggregate_deductible():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        aggregate_limit=1000000,
        aggregate_deductible=250000,
        premium=1000,
        reinstatement_cost=[1],
    )
    claims = FreqSevSims(np.array([0, 0, 0, 0]), np.array([100000, 600000, 500000, 200000]), 1)
    result = layer.apply(claims)
    assert result.recoveries.aggregate().tolist() == [350000]
    assert result.recoveries.values.tolist() == [
        0,
        350000 * (350000 / 600000),
        250000 * (350000 / 600000),
        0,
    ]
    assert result.reinstatement_premium is not None
    assert allclose(result.reinstatement_premium.values.tolist(), [700])
