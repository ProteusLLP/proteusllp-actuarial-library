from pal import XoL, FreqSevSims
import numpy as np


def test_XoL_no_agg():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        premium=1000,
    )
    claims = FreqSevSims(
        np.array([0, 0, 0, 0]), np.array([100000, 800000, 500000, 200000]), 1
    )
    result = layer.apply(claims)
    assert result.recoveries.values.tolist() == [0, 500000, 250000, 0]
    assert result.reinstatement_premium is None


def test_XoL_franchise():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=0,
        premium=1000,
        franchise=250000,
    )
    claims = FreqSevSims(
        np.array([0, 0, 0, 0]), np.array([100000, 800000, 400000, 200000]), 1
    )
    result = layer.apply(claims)
    assert result.recoveries.values.tolist() == [0, 500000, 400000, 0]
    assert result.reinstatement_premium is None


def test_XoL_reinstatements():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        aggregate_limit=1000000,
        premium=1000,
        reinstatement_cost=[1],
    )
    claims = FreqSevSims(
        np.array([0, 0, 0, 0]), np.array([100000, 800000, 500000, 200000]), 1
    )
    result = layer.apply(claims)
    assert result.recoveries.values.tolist() == [0, 500000, 250000, 0]
    assert result.reinstatement_premium.tolist() == [1000]


def test_XoL_multiple_reinstatements():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        aggregate_limit=2000000,
        premium=1000,
        reinstatement_cost=[1, 0.5, 0, 0],
    )
    claims = FreqSevSims(
        np.array([0, 0, 0, 0]), np.array([100000, 600000, 500000, 200000]), 1
    )
    result = layer.apply(claims)
    assert result.recoveries.values.tolist() == [0, 350000, 250000, 0]
    assert np.allclose(result.reinstatement_premium.tolist(), [1200])


def test_XoL_aggregate_limit():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        aggregate_limit=1000000,
        premium=1000,
        reinstatement_cost=[1],
    )
    claims = FreqSevSims(
        np.array([0, 0, 0, 0]), np.array([100000, 800000, 500000, 1000000]), 1
    )
    result = layer.apply(claims)
    assert result.recoveries.aggregate().tolist() == [1000000]
    assert result.recoveries.values.tolist() == [
        0,
        500000 * (1000000 / 1250000),
        250000 * (1000000 / 1250000),
        500000 * (1000000 / 1250000),
    ]
    assert np.allclose(result.reinstatement_premium.tolist(), [1000])


def test_XoL_aggregate_deductible():
    layer = XoL(
        "Layer 1",
        limit=500000,
        excess=250000,
        aggregate_limit=1000000,
        aggregate_deductible=250000,
        premium=1000,
        reinstatement_cost=[1],
    )
    claims = FreqSevSims(
        np.array([0, 0, 0, 0]), np.array([100000, 600000, 500000, 200000]), 1
    )
    result = layer.apply(claims)
    assert result.recoveries.aggregate().tolist() == [350000]
    assert result.recoveries.values.tolist() == [
        0,
        350000 * (350000 / 600000),
        250000 * (350000 / 600000),
        0,
    ]
    assert np.allclose(result.reinstatement_premium.values.tolist(), [700])
