"""Tests for risk measures and capital allocation."""

import numpy as np
import pytest

from pal.risk_measures import (
    RiskMeasureResult,
    _spectral_risk_measure,
    dual_power_transform,
    exponential_transform,
    percentile_layer,
    proportional_hazards_transform,
    standard_deviation_principle,
    svar,
    tvar,
    wang_transform,
)
from pal.stochastic_scalar import StochasticScalar


@pytest.fixture
def simple_profile() -> StochasticScalar:
    """Small deterministic profile for exact verification."""
    return StochasticScalar([10.0, 30.0, 20.0, 40.0, 50.0])


@pytest.fixture
def large_profile() -> StochasticScalar:
    """Larger profile for statistical properties."""
    rng = np.random.default_rng(42)
    return StochasticScalar(rng.exponential(scale=100, size=10_000))


# --- RiskMeasureResult tests ---


class TestRiskMeasureResult:
    """Tests for RiskMeasureResult."""

    def test_value_property(self, simple_profile: StochasticScalar) -> None:
        """The .value property returns the risk measure scalar."""
        rm = proportional_hazards_transform(simple_profile, 0.5)
        assert isinstance(rm, RiskMeasureResult)
        assert isinstance(rm.value, float)

    def test_weights_property(self, simple_profile: StochasticScalar) -> None:
        """The .weights property returns the StochasticScalar."""
        rm = proportional_hazards_transform(simple_profile, 0.5)
        assert isinstance(rm.weights, StochasticScalar)

    def test_value_equals_allocate_self(self, large_profile: StochasticScalar) -> None:
        """Allocating the risk profile to itself equals .value."""
        rm = proportional_hazards_transform(large_profile, 0.5)
        np.testing.assert_allclose(float(rm.allocate(large_profile)), rm.value, rtol=1e-10)

    def test_allocate_stochastic_scalar(self) -> None:
        """Allocate computes weighted mean of a StochasticScalar."""
        profile = StochasticScalar([10.0, 20.0, 30.0, 40.0])
        rm = proportional_hazards_transform(profile, 0.5)
        result = rm.allocate(profile)
        expected = (rm.weights.values * profile.values).mean()
        np.testing.assert_allclose(float(result), expected)

    def test_repr(self, simple_profile: StochasticScalar) -> None:
        """Repr includes the value."""
        rm = tvar(simple_profile, 0.9)
        assert "RiskMeasureResult" in repr(rm)


# --- Helper function tests ---


class TestSpectralRiskMeasureHelper:
    """Tests for the shared _spectral_risk_measure helper."""

    def test_uniform_weight_fn_gives_equal_weights(self, simple_profile: StochasticScalar) -> None:
        """Uniform weight function produces all-ones weights."""
        rm = _spectral_risk_measure(simple_profile, lambda u: u / u)
        np.testing.assert_allclose(rm.weights.values, 1.0)

    def test_uniform_weight_value_equals_mean(self, simple_profile: StochasticScalar) -> None:
        """Uniform weights give value equal to the mean."""
        rm = _spectral_risk_measure(simple_profile, lambda u: u / u)
        np.testing.assert_allclose(rm.value, simple_profile.mean())

    def test_weights_average_to_one(self, simple_profile: StochasticScalar) -> None:
        """Normalized weights average to 1."""
        rm = _spectral_risk_measure(simple_profile, lambda u: u**2)
        np.testing.assert_allclose(rm.weights.values.mean(), 1.0)

    def test_result_is_risk_measure_result(self, simple_profile: StochasticScalar) -> None:
        """Result type is RiskMeasureResult."""
        rm = _spectral_risk_measure(simple_profile, lambda u: u / u)
        assert isinstance(rm, RiskMeasureResult)

    def test_loss_mode_highest_value_gets_highest_weight(self) -> None:
        """In loss mode, the largest value gets the largest weight."""
        profile = StochasticScalar([1.0, 5.0, 3.0])
        rm = _spectral_risk_measure(profile, lambda u: u, "loss")
        w = rm.weights
        assert w.values[1] == max(w.values)
        assert w.values[0] == min(w.values)

    def test_profit_mode_lowest_value_gets_highest_weight(self) -> None:
        """In profit mode, the smallest value gets the largest weight."""
        profile = StochasticScalar([1.0, 5.0, 3.0])
        rm = _spectral_risk_measure(profile, lambda u: u, "profit")
        w = rm.weights
        assert w.values[0] == max(w.values)
        assert w.values[1] == min(w.values)

    def test_profit_mode_rank_complement(self) -> None:
        """Profit mode correctly complements ranks, not reverses."""
        profile = StochasticScalar([30.0, 10.0, 20.0, 5.0])
        w_loss = _spectral_risk_measure(profile, lambda u: u + 1, "loss").weights
        w_prof = _spectral_risk_measure(profile, lambda u: u + 1, "profit").weights
        assert w_loss.values[0] == max(w_loss.values)
        assert w_prof.values[0] == min(w_prof.values)
        assert w_loss.values[3] == min(w_loss.values)
        assert w_prof.values[3] == max(w_prof.values)


# --- Proportional Hazard Transform ---


class TestProportionalHazard:
    """Tests for proportional_hazards_transform."""

    def test_alpha_one_gives_uniform_weights(self, simple_profile: StochasticScalar) -> None:
        """Alpha=1 means no distortion, all weights equal."""
        rm = proportional_hazards_transform(simple_profile, 1.0)
        np.testing.assert_allclose(rm.weights.values, 1.0, atol=1e-10)

    def test_alpha_one_value_equals_mean(self, simple_profile: StochasticScalar) -> None:
        """Alpha=1 gives value equal to the mean."""
        rm = proportional_hazards_transform(simple_profile, 1.0)
        np.testing.assert_allclose(rm.value, simple_profile.mean(), atol=1e-10)

    def test_weights_average_to_one(self, large_profile: StochasticScalar) -> None:
        """Normalized weights average to 1."""
        rm = proportional_hazards_transform(large_profile, 0.5)
        np.testing.assert_allclose(rm.weights.values.mean(), 1.0, atol=1e-10)

    def test_lower_alpha_increases_value(self, large_profile: StochasticScalar) -> None:
        """Lower alpha produces a higher risk measure value."""
        rm_mild = proportional_hazards_transform(large_profile, 0.8)
        rm_strong = proportional_hazards_transform(large_profile, 0.3)
        assert rm_strong.value > rm_mild.value

    def test_weights_nonnegative(self, large_profile: StochasticScalar) -> None:
        """All weights must be non-negative."""
        rm = proportional_hazards_transform(large_profile, 0.5)
        assert np.all(rm.weights.values >= 0)

    def test_known_weight_values(self) -> None:
        """Verify weights against hand-calculated values."""
        profile = StochasticScalar([100.0, 200.0])
        alpha = 0.5
        rm = proportional_hazards_transform(profile, alpha)
        expected_raw = alpha * np.array([0.75, 0.25]) ** (alpha - 1)
        expected = expected_raw / expected_raw.mean()
        np.testing.assert_allclose(rm.weights.values, expected, rtol=1e-10)


# --- Wang Transform ---


class TestWangTransform:
    """Tests for wang_transform."""

    def test_alpha_zero_gives_uniform_weights(self, simple_profile: StochasticScalar) -> None:
        """Alpha=0 means no distortion."""
        rm = wang_transform(simple_profile, 0.0)
        np.testing.assert_allclose(rm.weights.values, 1.0, atol=1e-10)

    def test_alpha_zero_value_equals_mean(self, simple_profile: StochasticScalar) -> None:
        """Alpha=0 gives value equal to the mean."""
        rm = wang_transform(simple_profile, 0.0)
        np.testing.assert_allclose(rm.value, simple_profile.mean(), atol=1e-10)

    def test_weights_average_to_one(self, large_profile: StochasticScalar) -> None:
        """Normalized weights average to 1."""
        rm = wang_transform(large_profile, 0.5)
        np.testing.assert_allclose(rm.weights.values.mean(), 1.0, atol=1e-10)

    def test_higher_alpha_increases_value(self, large_profile: StochasticScalar) -> None:
        """Higher alpha produces a higher risk measure value."""
        rm_mild = wang_transform(large_profile, 0.3)
        rm_strong = wang_transform(large_profile, 1.0)
        assert rm_strong.value > rm_mild.value

    def test_weights_nonnegative(self, large_profile: StochasticScalar) -> None:
        """All weights must be non-negative."""
        rm = wang_transform(large_profile, 0.5)
        assert np.all(rm.weights.values >= 0)


# --- TVaR ---


class TestTVaR:
    """Tests for tvar."""

    def test_weights_average_to_one(self, large_profile: StochasticScalar) -> None:
        """Normalized weights average to 1."""
        rm = tvar(large_profile, 0.9)
        np.testing.assert_allclose(rm.weights.values.mean(), 1.0, atol=1e-2)

    def test_bottom_sims_get_zero_weight(self) -> None:
        """Simulations below the alpha quantile get zero weight."""
        profile = StochasticScalar(np.arange(1.0, 101.0))
        rm = tvar(profile, 0.9)
        sorted_weights = rm.weights.values[np.argsort(profile.values)]
        assert np.all(sorted_weights[:90] == 0.0)

    def test_top_sims_get_equal_nonzero_weight(self) -> None:
        """Simulations above alpha quantile get equal weight."""
        profile = StochasticScalar(np.arange(1.0, 101.0))
        rm = tvar(profile, 0.9)
        sorted_weights = rm.weights.values[np.argsort(profile.values)]
        top_weights = sorted_weights[90:]
        np.testing.assert_allclose(top_weights, top_weights[0], rtol=1e-10)

    def test_value_equals_tail_mean(self) -> None:
        """Risk measure value equals the mean of the tail."""
        values = np.arange(1.0, 101.0)
        profile = StochasticScalar(values)
        rm = tvar(profile, 0.9)
        tail_mean = values[90:].mean()
        np.testing.assert_allclose(rm.value, tail_mean, rtol=0.05)

    def test_alpha_zero_value_equals_mean(self, large_profile: StochasticScalar) -> None:
        """TVaR at alpha=0 equals the plain mean."""
        rm = tvar(large_profile, 0.0)
        np.testing.assert_allclose(rm.value, large_profile.mean(), rtol=1e-2)


# --- Dual Power ---


class TestDualPower:
    """Tests for dual_power_transform."""

    def test_beta_one_gives_uniform_weights(self, simple_profile: StochasticScalar) -> None:
        """Beta=1 means no distortion, all weights equal."""
        rm = dual_power_transform(simple_profile, 1.0)
        np.testing.assert_allclose(rm.weights.values, 1.0, atol=1e-10)

    def test_weights_average_to_one(self, large_profile: StochasticScalar) -> None:
        """Normalized weights average to 1."""
        rm = dual_power_transform(large_profile, 2.0)
        np.testing.assert_allclose(rm.weights.values.mean(), 1.0, atol=1e-10)

    def test_higher_beta_increases_value(self, large_profile: StochasticScalar) -> None:
        """Higher beta produces a higher risk measure value."""
        rm_mild = dual_power_transform(large_profile, 1.5)
        rm_strong = dual_power_transform(large_profile, 3.0)
        assert rm_strong.value > rm_mild.value

    def test_weights_nonnegative(self, large_profile: StochasticScalar) -> None:
        """All weights must be non-negative."""
        rm = dual_power_transform(large_profile, 2.0)
        assert np.all(rm.weights.values >= 0)


# --- Exponential ---


class TestExponential:
    """Tests for exponential_transform."""

    def test_gamma_zero_gives_uniform_weights(self, simple_profile: StochasticScalar) -> None:
        """Gamma=0 means no distortion, all weights equal."""
        rm = exponential_transform(simple_profile, 0.0)
        np.testing.assert_allclose(rm.weights.values, 1.0, atol=1e-10)

    def test_weights_average_to_one(self, large_profile: StochasticScalar) -> None:
        """Normalized weights average to 1."""
        rm = exponential_transform(large_profile, 2.0)
        np.testing.assert_allclose(rm.weights.values.mean(), 1.0, atol=1e-10)

    def test_higher_gamma_increases_value(self, large_profile: StochasticScalar) -> None:
        """Higher gamma produces a higher risk measure value."""
        rm_mild = exponential_transform(large_profile, 1.0)
        rm_strong = exponential_transform(large_profile, 5.0)
        assert rm_strong.value > rm_mild.value

    def test_weights_nonnegative(self, large_profile: StochasticScalar) -> None:
        """All weights must be non-negative."""
        rm = exponential_transform(large_profile, 3.0)
        assert np.all(rm.weights.values >= 0)


# --- SVaR ---


class TestSVaR:
    """Tests for svar."""

    def test_invalid_bounds_raise(self, simple_profile: StochasticScalar) -> None:
        """Invalid lower/upper bounds raise ValueError."""
        with pytest.raises(ValueError):
            svar(simple_profile, 0.9, 0.5)
        with pytest.raises(ValueError):
            svar(simple_profile, -0.1, 0.5)

    def test_full_range_equals_mean(self, large_profile: StochasticScalar) -> None:
        """SVaR over [0, 1] approximates the mean."""
        rm = svar(large_profile, 0.0, 1.0)
        np.testing.assert_allclose(rm.value, large_profile.mean(), rtol=0.05)

    def test_top_range_approximates_tvar(self, large_profile: StochasticScalar) -> None:
        """SVaR over [alpha, 1] approximates TVaR at alpha."""
        rm_svar = svar(large_profile, 0.9, 1.0)
        rm_tvar = tvar(large_profile, 0.9)
        np.testing.assert_allclose(rm_svar.value, rm_tvar.value, rtol=0.1)


# --- Standard Deviation Principle ---


class TestStandardDeviationPrinciple:
    """Tests for standard_deviation_principle."""

    def test_k_zero_equals_mean(self, large_profile: StochasticScalar) -> None:
        """k=0 gives value equal to the mean."""
        rm = standard_deviation_principle(large_profile, 0.0)
        np.testing.assert_allclose(rm.value, large_profile.mean())

    def test_value_formula(self, large_profile: StochasticScalar) -> None:
        """Value equals mean + k * std."""
        k = 2.0
        rm = standard_deviation_principle(large_profile, k)
        expected = large_profile.mean() + k * large_profile.std()
        np.testing.assert_allclose(rm.value, expected)

    def test_weights_average_to_one(self, large_profile: StochasticScalar) -> None:
        """Weights average to 1."""
        rm = standard_deviation_principle(large_profile, 2.0)
        np.testing.assert_allclose(rm.weights.values.mean(), 1.0, atol=1e-10)

    def test_allocate_self_equals_value(self, large_profile: StochasticScalar) -> None:
        """Allocating the profile to itself equals .value."""
        rm = standard_deviation_principle(large_profile, 2.0)
        np.testing.assert_allclose(float(rm.allocate(large_profile)), rm.value, rtol=1e-10)

    def test_zero_variance_gives_uniform_weights(self) -> None:
        """Constant profile gives uniform weights."""
        profile = StochasticScalar([5.0, 5.0, 5.0, 5.0])
        rm = standard_deviation_principle(profile, 3.0)
        np.testing.assert_allclose(rm.weights.values, 1.0)
        np.testing.assert_allclose(rm.value, 5.0)


# --- Percentile Layer ---


class TestPercentileLayer:
    """Tests for percentile_layer (Capital Allocation by Percentile Layer)."""

    def test_value_equals_capital_when_max_exceeds(self) -> None:
        """Value equals capital when max loss > capital."""
        profile = StochasticScalar([10.0, 20.0, 30.0, 40.0, 50.0])
        rm = percentile_layer(profile, 35.0)
        np.testing.assert_allclose(rm.value, 35.0, rtol=1e-10)

    def test_value_equals_max_when_below_capital(self) -> None:
        """Value equals max loss when all losses < capital."""
        profile = StochasticScalar([10.0, 20.0, 30.0])
        rm = percentile_layer(profile, 100.0)
        np.testing.assert_allclose(rm.value, 30.0, rtol=1e-10)

    def test_allocate_self_equals_value(self) -> None:
        """Allocating the profile to itself equals .value."""
        profile = StochasticScalar([10.0, 20.0, 30.0, 40.0, 50.0])
        rm = percentile_layer(profile, 35.0)
        np.testing.assert_allclose(float(rm.allocate(profile)), rm.value, rtol=1e-10)

    def test_component_allocations_sum_to_value(self) -> None:
        """Component allocations sum to the total."""
        a = np.array([5.0, 15.0, 10.0, 25.0, 30.0])
        b = np.array([5.0, 5.0, 20.0, 15.0, 20.0])
        total = StochasticScalar(a + b)
        rm = percentile_layer(total, 35.0)
        alloc_a = float(rm.allocate(StochasticScalar(a)))
        alloc_b = float(rm.allocate(StochasticScalar(b)))
        np.testing.assert_allclose(alloc_a + alloc_b, rm.value, rtol=1e-10)

    def test_hand_calculated_weights(self) -> None:
        """Verify weights against hand calculation.

        X = [10, 20, 30], C = 25.
        Sorted: X_(0)=10, X_(1)=20, X_(2)=30
        Y = [10, 20, 25], delta = [10, 10, 5]
        Counts = [3, 2, 1]
        Layer alloc = [10/3, 5, 5]
        Cumul alloc = [10/3, 25/3, 40/3]
        w_(i) = 3 * alloc / X_(i) = [1, 1.25, 4/3]
        """
        profile = StochasticScalar([10.0, 20.0, 30.0])
        rm = percentile_layer(profile, 25.0)
        sorted_order = np.argsort(profile.values)
        w_sorted = rm.weights.values[sorted_order]
        expected = np.array([1.0, 1.25, 4.0 / 3.0])
        np.testing.assert_allclose(w_sorted, expected, rtol=1e-10)

    def test_allocations_nondecreasing_in_sorted_order(self) -> None:
        """Per-simulation allocations are non-decreasing when sorted."""
        profile = StochasticScalar([5.0, 15.0, 25.0, 35.0, 45.0])
        rm = percentile_layer(profile, 30.0)
        sorted_order = np.argsort(profile.values)
        # Allocation = w * X / N
        alloc = rm.weights.values[sorted_order] * profile.values[sorted_order]
        assert all(alloc[i] <= alloc[i + 1] + 1e-10 for i in range(len(alloc) - 1))

    def test_zero_losses_get_zero_weight(self) -> None:
        """Simulations with zero loss get zero weight."""
        profile = StochasticScalar([0.0, 0.0, 10.0, 20.0])
        rm = percentile_layer(profile, 15.0)
        assert rm.weights.values[0] == 0.0
        assert rm.weights.values[1] == 0.0

    def test_large_profile(self, large_profile: StochasticScalar) -> None:
        """Percentile layer on a large profile with capital at 99th pctl."""
        capital = float(np.percentile(large_profile.values, 99))
        rm = percentile_layer(large_profile, capital)
        np.testing.assert_allclose(float(rm.allocate(large_profile)), rm.value, rtol=1e-10)
