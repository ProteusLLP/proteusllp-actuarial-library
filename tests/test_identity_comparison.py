"""Tests for identity comparison behavior needed for WeakSet and Pydantic compatibility.

These tests ensure that the fix for the Pydantic model_copy(deep=True) bug
remains functional. The fix adds identity checks to __eq__ and __ne__ methods
to return boolean values when comparing an object with itself.
"""

import weakref

import numpy as np
import pytest
from pal import FreqSevSims, ProteusVariable, StochasticScalar


class TestIdentityComparison:
    """Test identity comparison returns boolean for set/dict compatibility."""

    def test_stochastic_scalar_identity_returns_bool(self):
        """Test that x == x returns True (boolean) for identity comparison."""
        x = StochasticScalar([1.0, 2.0, 3.0])
        result = x == x

        # Identity comparison should return boolean True
        assert result is True
        assert isinstance(result, bool)

    def test_stochastic_scalar_identity_ne_returns_bool(self):
        """Test that x != x returns False (boolean) for identity comparison."""
        x = StochasticScalar([1.0, 2.0, 3.0])
        result = x != x

        # Identity comparison should return boolean False
        assert result is False
        assert isinstance(result, bool)

    def test_stochastic_scalar_value_comparison_returns_array(self):
        """Test that x == y returns StochasticScalar for value comparison."""
        x = StochasticScalar([1.0, 2.0, 3.0])
        y = StochasticScalar([1.0, 2.0, 3.0])
        result = x == y

        # Value comparison should return StochasticScalar
        assert isinstance(result, StochasticScalar)
        assert (result.values == [True, True, True]).all()

    def test_freqsevsims_identity_returns_bool(self):
        """Test that FreqSevSims identity comparison returns boolean."""
        sim_index = np.array([1, 2, 2, 3, 4], dtype=int)
        fs = FreqSevSims(
            sim_index=sim_index,
            values=np.array([500.0, 700.0, 600.0, 800.0, 550.0]),
            n_sims=5,
        )
        result = fs == fs

        # Identity comparison should return boolean True
        assert result is True
        assert isinstance(result, bool)

    def test_freqsevsims_value_comparison_returns_array(self):
        """Test that FreqSevSims value comparison returns FreqSevSims."""
        sim_index = np.array([1, 2, 2, 3, 4], dtype=int)
        fs1 = FreqSevSims(
            sim_index=sim_index,
            values=np.array([500.0, 700.0, 600.0, 800.0, 550.0]),
            n_sims=5,
        )
        fs2 = FreqSevSims(
            sim_index=sim_index,
            values=np.array([500.0, 700.0, 600.0, 800.0, 550.0]),
            n_sims=5,
        )
        result = fs1 == fs2

        # Value comparison should return FreqSevSims
        assert isinstance(result, FreqSevSims)


class TestWeakSetCompatibility:
    """Test that PAL objects work correctly in WeakSets."""

    def test_stochastic_scalar_in_weakset(self):
        """Test that StochasticScalar can be added to WeakSet."""
        x = StochasticScalar([1.0, 2.0, 3.0])
        ws: weakref.WeakSet[StochasticScalar] = weakref.WeakSet()

        # Should not raise
        ws.add(x)
        assert len(ws) == 1

        # Adding same object again should not increase size
        ws.add(x)
        assert len(ws) == 1

    def test_freqsevsims_in_weakset(self):
        """Test that FreqSevSims can be added to WeakSet."""
        sim_index = np.array([1, 2, 2, 3, 4], dtype=int)
        fs = FreqSevSims(
            sim_index=sim_index,
            values=np.array([500.0, 700.0, 600.0, 800.0, 550.0]),
            n_sims=5,
        )
        ws: weakref.WeakSet[FreqSevSims] = weakref.WeakSet()

        # Should not raise
        ws.add(fs)
        assert len(ws) == 1

        # Adding same object again should not increase size
        ws.add(fs)
        assert len(ws) == 1

    def test_multiple_objects_in_weakset(self):
        """Test that different PAL objects are distinct in WeakSet."""
        x = StochasticScalar([1.0, 2.0, 3.0])
        y = StochasticScalar([1.0, 2.0, 3.0])  # Same values, different object

        ws: weakref.WeakSet[StochasticScalar] = weakref.WeakSet()
        ws.add(x)
        ws.add(y)

        # Two different objects should result in size 2
        assert len(ws) == 2


class TestPydanticCompatibility:
    """Test that PAL objects work with Pydantic's model_copy."""

    def test_pydantic_deep_copy_stochastic_scalar(self):
        """Test that StochasticScalar survives Pydantic model_copy(deep=True)."""
        pytest.importorskip("pydantic", reason="Pydantic not installed")
        from pydantic import BaseModel, ConfigDict

        class Model(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            value: StochasticScalar

        original = Model(value=StochasticScalar([1.0, 2.0, 3.0]))
        copied = original.model_copy(deep=True)

        # Should not raise and should work
        assert isinstance(copied.value, StochasticScalar)
        assert (copied.value.values == [1.0, 2.0, 3.0]).all()

    def test_pydantic_deep_copy_freqsevsims(self):
        """Test that FreqSevSims survives Pydantic model_copy(deep=True)."""
        pytest.importorskip("pydantic", reason="Pydantic not installed")
        from pydantic import BaseModel, ConfigDict

        sim_index = np.array([1, 2, 2, 3, 4], dtype=int)
        fs = FreqSevSims(
            sim_index=sim_index,
            values=np.array([500.0, 700.0, 600.0, 800.0, 550.0]),
            n_sims=5,
        )

        class Model(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            value: FreqSevSims

        original = Model(value=fs)
        copied = original.model_copy(deep=True)

        # Should not raise and should work
        assert isinstance(copied.value, FreqSevSims)

    def test_pydantic_deep_copy_with_operations(self):
        """Test the original bug scenario: deep copy then perform operations."""
        pytest.importorskip("pydantic", reason="Pydantic not installed")
        from pydantic import BaseModel, ConfigDict

        # Create coupled FreqSevSims objects (the problematic scenario)
        sim_index = np.array([1, 2, 2, 3, 4], dtype=int)
        losses = ProteusVariable(
            "item",
            {
                "asset1": FreqSevSims(
                    sim_index=sim_index,
                    values=np.array([500.0, 700.0, 600.0, 800.0, 550.0]),
                    n_sims=5,
                ),
                "asset2": FreqSevSims(
                    sim_index=sim_index,
                    values=np.array([1500.0, 2000.0, 1800.0, 2200.0, 1600.0]),
                    n_sims=5,
                ),
            },
        )

        class DataModel(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            losses: ProteusVariable[FreqSevSims]

        model = DataModel(losses=losses)

        # Deep copy should not raise
        copied_model = model.model_copy(deep=True)

        # Operations on copied objects should work (this was the bug)
        copied_items = [copied_model.losses["asset1"], copied_model.losses["asset2"]]
        weights = np.array([0.5, 0.5])

        # This used to raise TypeError but should now work
        combined = sum(item * weights[i] for i, item in enumerate(copied_items))

        # Verify the result makes sense
        assert isinstance(combined, FreqSevSims)
        mean = combined.occurrence().mean()
        assert mean > 0  # Should have positive mean


class TestCouplingGroupMerge:
    """Test that coupling group merge operations work correctly."""

    def test_merge_with_identity_comparison(self):
        """Test that merging coupling groups uses identity comparison correctly."""
        x = StochasticScalar([1.0, 2.0, 3.0])
        y = StochasticScalar([4.0, 5.0, 6.0])

        # Performing an operation should merge coupling groups
        z = x + y

        # All should be in the same coupling group now
        assert x.coupled_variable_group is y.coupled_variable_group
        assert y.coupled_variable_group is z.coupled_variable_group

        # The WeakSet in the coupling group should contain all three
        assert len(x.coupled_variable_group.variables) == 3


class TestBackwardCompatibility:
    """Ensure the fix doesn't break existing functionality."""

    def test_element_wise_equality_still_works(self):
        """Test that element-wise equality comparison still works as before."""
        x = StochasticScalar([1, 2, 3])
        y = StochasticScalar([1, 4, 3])
        result = x == y

        # Should return element-wise comparison
        assert isinstance(result, StochasticScalar)
        assert (result.values == [True, False, True]).all()

    def test_bool_raises_for_ambiguous_cases(self):
        """Test that __bool__ still raises for ambiguous cases."""
        x = StochasticScalar([1, 2, 3])
        y = StochasticScalar([1, 4, 3])
        result = x == y  # [True, False, True] - ambiguous

        # Should still raise TypeError when used in boolean context
        with pytest.raises(TypeError, match="Ambiguous truth value"):
            if result:
                pass

    def test_any_all_methods_still_work(self):
        """Test that .any() and .all() methods still work."""
        x = StochasticScalar([1, 2, 3])
        y = StochasticScalar([1, 4, 3])
        result = x == y  # [True, False, True]

        # Should be able to use .any() and .all()
        assert result.any()  # At least one True
        assert not result.all()  # Not all True
