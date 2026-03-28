"""Tests verifying XoL reinstatement pricing against Mata (2000).

Reproduces Table 2 (pure premiums) and Table 5 (PH transform premiums)
from: Mata, A.J. (2000). "Pricing Excess of Loss Reinsurance with
Reinstatements." ASTIN Bulletin, 30(2), 349-368.

The paper's model:
  - Frequency: Poisson(lambda=10)
  - Severity: Pareto Type II (alpha=3, beta=10), i.e. S(y) = (10/(y+10))^3
  - Layers: "10 xs 10", "10 xs 20", combined "20 xs 10"
  - Free reinstatements (c=0) with K=1 or K=infinity
"""

import pytest

from pal.config import set_default_n_sims, set_random_seed
from pal.contracts import XoL
from pal.distributions import Pareto, Poisson
from pal.frequency_severity import (
    FreqSevSims,
    FrequencySeverityModel,
)
from pal.risk_measures import proportional_hazards_transform


@pytest.fixture(scope="module")
def claims() -> FreqSevSims:
    """Simulate the Poisson-Pareto II compound process from the paper."""
    set_random_seed(42)
    set_default_n_sims(100_000)
    # PAL's Pareto is Type I (support [scale, inf)).
    # Pareto II (Lomax) with alpha=3, beta=10 is obtained by shifting.
    model = FrequencySeverityModel(Poisson(10.0), Pareto(shape=3, scale=10))
    claims = model.generate() - 10.0
    return claims


def _apply_layer(
    claims: FreqSevSims,
    limit: float,
    excess: float,
    aggregate_limit: float | None = None,
):
    """Apply an XoL layer and return aggregate recoveries."""
    kwargs: dict = {
        "name": "test",
        "limit": limit,
        "excess": excess,
        "premium": 1,
    }
    if aggregate_limit is not None:
        kwargs["aggregate_limit"] = aggregate_limit
        kwargs["reinstatement_cost"] = [0.0]
    layer = XoL(**kwargs)
    return layer.apply(claims).recoveries.aggregate()


# ── Table 2: Pure premiums, lambda=10, free reinstatements (c=0) ──


class TestTable2PurePremiums:
    """Pure premiums from Table 2 of Mata (2000), free reinstatements."""

    # K=1 free reinstatement (aggregate_limit = 2 * layer width)

    def test_layer_10xs10_k1(self, claims):
        agg = _apply_layer(claims, 10, 10, aggregate_limit=20)
        assert agg.mean() == pytest.approx(6.6128, rel=0.005)

    def test_layer_10xs20_k1(self, claims):
        agg = _apply_layer(claims, 10, 20, aggregate_limit=20)
        assert agg.mean() == pytest.approx(2.4103, rel=0.005)

    def test_combined_20xs10_k1(self, claims):
        agg = _apply_layer(claims, 20, 10, aggregate_limit=40)
        assert agg.mean() == pytest.approx(9.2173, rel=0.005)

    def test_sum_leq_combined_k1(self, claims):
        """Sum of layer premiums <= combined layer premium (Table 2)."""
        agg1 = _apply_layer(claims, 10, 10, aggregate_limit=20)
        agg2 = _apply_layer(claims, 10, 20, aggregate_limit=20)
        aggc = _apply_layer(claims, 20, 10, aggregate_limit=40)
        assert agg1.mean() + agg2.mean() <= aggc.mean()

    # K=infinity (unlimited reinstatements, no aggregate limit)

    def test_layer_10xs10_unlimited(self, claims):
        agg = _apply_layer(claims, 10, 10)
        assert agg.mean() == pytest.approx(6.9444, rel=0.005)

    def test_layer_10xs20_unlimited(self, claims):
        agg = _apply_layer(claims, 10, 20)
        assert agg.mean() == pytest.approx(2.4305, rel=0.005)

    def test_combined_20xs10_unlimited(self, claims):
        agg = _apply_layer(claims, 20, 10)
        assert agg.mean() == pytest.approx(9.3749, rel=0.005)

    def test_sum_equals_combined_unlimited(self, claims):
        """With unlimited reinstatements, sum of layers = combined."""
        agg1 = _apply_layer(claims, 10, 10)
        agg2 = _apply_layer(claims, 10, 20)
        aggc = _apply_layer(claims, 20, 10)
        assert agg1.mean() + agg2.mean() == pytest.approx(aggc.mean(), rel=0.005)


# ── Table 5: PH transform premiums, lambda=10, K=1 free reinstatement ──

# Columns from Table 5: pi_p(S1*), pi_p(S2*), pi_p(Sc*)
# Paper's p parameter maps to PAL's alpha = 1/p
TABLE_5 = {
    #  p: (pi_p(S1*), pi_p(S2*), pi_p(Sc*))
    1.0: (6.6128, 2.4103, 9.2173),
    1.2: (7.7403, 3.2344, 11.1852),
    1.4: (8.7116, 4.0313, 12.9715),
    1.6: (9.5518, 4.7875, 14.5856),
    1.8: (10.2828, 5.4971, 16.0425),
    2.0: (10.9230, 6.1590, 17.3585),
}


class TestTable5PHTransform:
    """PH transform premiums from Table 5 of Mata (2000)."""

    @pytest.mark.parametrize(
        "p, expected_s1, expected_s2, expected_sc",
        [(p, *vals) for p, vals in TABLE_5.items()],
        ids=[f"p={p}" for p in TABLE_5],
    )
    def test_ph_layer_premiums(self, claims, p, expected_s1, expected_s2, expected_sc):
        alpha = 1.0 / p
        agg1 = _apply_layer(claims, 10, 10, aggregate_limit=20)
        agg2 = _apply_layer(claims, 10, 20, aggregate_limit=20)
        aggc = _apply_layer(claims, 20, 10, aggregate_limit=40)

        ph1 = proportional_hazards_transform(agg1, alpha=alpha)
        ph2 = proportional_hazards_transform(agg2, alpha=alpha)
        phc = proportional_hazards_transform(aggc, alpha=alpha)

        assert ph1.value == pytest.approx(expected_s1, rel=0.005)
        assert ph2.value == pytest.approx(expected_s2, rel=0.005)
        assert phc.value == pytest.approx(expected_sc, rel=0.005)

    @pytest.mark.parametrize("p", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    def test_sub_additivity(self, claims, p):
        """pi_p(S1*+S2*) <= pi_p(S1*) + pi_p(S2*) for all p >= 1."""
        alpha = 1.0 / p
        agg1 = _apply_layer(claims, 10, 10, aggregate_limit=20)
        agg2 = _apply_layer(claims, 10, 20, aggregate_limit=20)

        ph1 = proportional_hazards_transform(agg1, alpha=alpha)
        ph2 = proportional_hazards_transform(agg2, alpha=alpha)
        ph_sum = proportional_hazards_transform(agg1 + agg2, alpha=alpha)

        assert ph_sum.value <= ph1.value + ph2.value + 1e-6

    @pytest.mark.parametrize("p", [1.2, 1.4, 1.6, 1.8, 2.0])
    def test_combined_geq_sum_of_layers(self, claims, p):
        """pi_p(S1*) + pi_p(S2*) <= pi_p(Sc*) for lambda=10."""
        alpha = 1.0 / p
        agg1 = _apply_layer(claims, 10, 10, aggregate_limit=20)
        agg2 = _apply_layer(claims, 10, 20, aggregate_limit=20)
        aggc = _apply_layer(claims, 20, 10, aggregate_limit=40)

        ph1 = proportional_hazards_transform(agg1, alpha=alpha)
        ph2 = proportional_hazards_transform(agg2, alpha=alpha)
        phc = proportional_hazards_transform(aggc, alpha=alpha)

        assert ph1.value + ph2.value <= phc.value + 1e-6
