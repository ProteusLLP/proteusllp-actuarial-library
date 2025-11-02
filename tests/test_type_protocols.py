"""Tests for PAL type system protocols.

This module verifies that our classes correctly implement the expected protocols
for structural typing, ensuring runtime protocol compatibility.
"""

from pal.frequency_severity import FreqSevSims
from pal.stochastic_scalar import StochasticScalar
from pal.types import ProteusLike, VectorLike
from pal.variables import ProteusVariable


def test_stochastic_scalar_implements_vector_like():
    """Test that StochasticScalar implements VectorLike protocol."""
    s = StochasticScalar([1.0, 2.0, 3.0])

    # Runtime protocol check
    assert isinstance(s, VectorLike)

    # Should NOT implement ProteusLike (it's not a container)
    assert not isinstance(s, ProteusLike)


def test_freq_sev_sims_implements_vector_like():
    """Test that FreqSevSims implements VectorLike protocol."""
    fs = FreqSevSims([1, 2, 3], [100.0, 200.0, 300.0], 3)

    # Runtime protocol check
    assert isinstance(fs, VectorLike)

    # Should NOT implement ProteusLike (it's not a container)
    assert not isinstance(fs, ProteusLike)


def test_proteus_variable_implements_proteus_like():
    """Test that ProteusVariable implements ProteusLike protocol."""
    pv = ProteusVariable(
        dim_name="test",
        values={"a": StochasticScalar([1.0, 2.0]), "b": StochasticScalar([3.0, 4.0])},
    )

    # Runtime protocol check
    assert isinstance(pv, ProteusLike)

    # ProteusVariable should also be VectorLike since it can be converted to arrays
    # and supports vector operations
    assert isinstance(pv, VectorLike)


def test_nested_proteus_variable_implements_proteus_like():
    """Test that nested ProteusVariable structures implement ProteusLike."""
    inner = ProteusVariable(
        dim_name="inner", values={"x": StochasticScalar([1.0, 2.0])}
    )

    outer = ProteusVariable(dim_name="outer", values={"nested": inner})

    # Both should implement ProteusLike
    assert isinstance(inner, ProteusLike)
    assert isinstance(outer, ProteusLike)

    # Both should also implement VectorLike
    assert isinstance(inner, VectorLike)
    assert isinstance(outer, VectorLike)


def test_scalar_proteus_variable_implements_proteus_like():
    """Test that ProteusVariable with scalar values implements ProteusLike."""
    pv = ProteusVariable(dim_name="scalars", values={"a": 1.0, "b": 2.0, "c": 3.0})

    # Should implement ProteusLike (it's a container)
    assert isinstance(pv, ProteusLike)

    # Should also implement VectorLike
    assert isinstance(pv, VectorLike)


def test_protocol_attributes_exist():
    """Test that protocol-required attributes exist on implementing classes."""
    # Test StochasticScalar has VectorLike requirements
    s = StochasticScalar([1.0, 2.0, 3.0])
    assert hasattr(s, "__len__")
    assert hasattr(s, "__array__")
    assert hasattr(s, "__array_function__")
    assert hasattr(s, "__array_ufunc__")
    assert hasattr(s, "__add__")
    assert hasattr(s, "__lt__")
    assert len(s) == 3

    # Test ProteusVariable has ProteusLike requirements
    pv = ProteusVariable(dim_name="test", values={"a": 1.0, "b": 2.0})
    assert hasattr(pv, "n_sims")
    assert hasattr(pv, "values")
    assert hasattr(pv, "upsample")
    assert hasattr(pv.values, "keys")  # values should be Mapping-like
    assert len(pv) == 2  # Should support len()


def test_protocol_methods_work():
    """Test that protocol methods actually work as expected."""
    # Test VectorLike operations on StochasticScalar
    s1 = StochasticScalar([1.0, 2.0, 3.0])
    s2 = StochasticScalar([4.0, 5.0, 6.0])

    # Arithmetic operations should work
    result = s1 + s2
    assert isinstance(result, StochasticScalar)

    # Comparison operations should work
    comp_result = s1 < s2
    assert isinstance(comp_result, StochasticScalar)

    # Test ProteusLike operations on ProteusVariable
    pv = ProteusVariable(
        dim_name="test",
        values={"a": StochasticScalar([1.0, 2.0]), "b": StochasticScalar([3.0, 4.0])},
    )

    # Should support iteration (Sequence protocol)
    values_list = list(pv)
    assert len(values_list) == 2
    assert all(isinstance(v, StochasticScalar) for v in values_list)

    # Should support upsample
    upsampled = pv.upsample(10)
    assert isinstance(upsampled, ProteusVariable)
