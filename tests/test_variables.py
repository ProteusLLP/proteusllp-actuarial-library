import numpy as np
import pytest
from pal.variables import FreqSevSims, ProteusVariable, StochasticScalar


def test_empty():
    x = ProteusVariable(dim_name="dim1", values={})
    assert x.values == {}


def test_variable():
    x = ProteusVariable(dim_name="dim1", values={"a": 1, "b": 2, "c": 3})
    y = x + 1
    assert y.values == {"a": 2, "b": 3, "c": 4}


def test_variable2():
    """Test that a variable can be created with a dict of StochasticScalars."""
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = x + 2.2
    assert (
        y
        == ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([3.2, 4.2, 5.2]),
                "b": StochasticScalar([4.2, 5.2, 6.2]),
            },
        )
    ).all()


def test_variable3():
    """Test variable creation with dictionary, label matching, and summing."""
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": 1, "b": 2},
    )
    y = ProteusVariable(
        dim_name="dim1",
        values={"b": 5, "a": 8},
    )
    z = x + y
    assert z.values == {"a": 9, "b": 7}


def test_array_variable_dereferencing():
    x = ProteusVariable(
        dim_name="dim1",
        values={"first": 1, "second": 2},
    )
    assert x[0] == 1  # Gets first value by index
    assert x[1] == 2  # Gets second value by index


def test_sum():
    x = ProteusVariable(dim_name="dim1", values={"a": 1, "b": 2})
    y = sum(x)
    assert y == 3


def test_sum_stochastic():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = sum(x)
    assert (y == StochasticScalar([3, 5, 7])).all()
    assert (
        y.coupled_variable_group
        == x[0].coupled_variable_group
        == x[1].coupled_variable_group
    )


def test_divide():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = x / 2.0
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([0.5, 1, 3 / 2]),
                "b": StochasticScalar([1, 3 / 2, 2]),
            },
        )
        == y
    ).all()


def test_divide_two():
    x = ProteusVariable(dim_name="dim1", values={"a": 1, "b": 2, "c": 3})
    y = x / ProteusVariable(dim_name="dim1", values={"a": 2, "b": 4, "c": 6})
    assert y.values == {"a": 0.5, "b": 0.5, "c": 0.5}


def test_rdivide():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": 1, "b": 2, "c": 3},
    )
    y = 2 / x
    assert y.values == pytest.approx({"a": 2, "b": 1, "c": 2 / 3})


def test_multiply_stochastic():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = StochasticScalar([2, 3, 4])
    z = y * x
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([2, 6, 12]),
                "b": StochasticScalar([4, 9, 16]),
            },
        )
        == z
    ).all()


def test_rmultiply_stochastic():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = StochasticScalar([2, 3, 4])
    z = x * y
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([2, 6, 12]),
                "b": StochasticScalar([4, 9, 16]),
            },
        )
        == z
    ).all()


def test_subtract():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = x - 1
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={"a": StochasticScalar([0, 1, 2]), "b": StochasticScalar([1, 2, 3])},
        )
        == y
    ).all()


def test_rsubtract():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = 1 - x
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([0, -1, -2]),
                "b": StochasticScalar([-1, -2, -3]),
            },
        )
        == y
    ).all()


def test_subtract_two():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": 1, "b": 2, "c": 3},
    )
    y = x - ProteusVariable(dim_name="dim1", values={"a": 2, "b": 4, "c": 6})
    assert y.values == {"a": -1, "b": -2, "c": -3}


def test_sub_stochastic():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = StochasticScalar([2, 3, 4])
    z = y - x
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([1, 1, 1]),
                "b": StochasticScalar([0, 0, 0]),
            },
        )
        == z
    ).all()


def test_rsub_stochastic():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = StochasticScalar([2, 3, 4])
    z = x - y
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([-1, -1, -1]),
                "b": StochasticScalar([0, 0, 0]),
            },
        )
        == z
    ).all()


def test_sub_stochastic_scalar():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = ProteusVariable(
        dim_name="dim1",
        values={"a": 1, "b": 2},
    )
    z = x - y
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([0, 1, 2]),
                "b": StochasticScalar([0, 1, 2]),
            },
        )
        == z
    ).all()


def test_rsub_stochastic_scalar():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = ProteusVariable(
        dim_name="dim1",
        values={"a": 1, "b": 2},
    )
    z = y - x
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": StochasticScalar([0, -1, -2]),
                "b": StochasticScalar([0, -1, -2]),
            },
        )
        == z
    ).all()


def test_sub_2():
    a = StochasticScalar([1, 2, 3])
    b = FreqSevSims([0, 0, 1, 2], [1, 2, 3, 4], 3)
    x = a - b
    assert (x == FreqSevSims([0, 0, 1, 2], [0, -1, -1, -1], 3)).values.all()


def test_sub_3():
    a = StochasticScalar([2, 3, 4])
    b = FreqSevSims([0, 1, 1, 2], [1, 2, 3, 4], 3)
    x = a - b
    assert (x == FreqSevSims([0, 1, 1, 2], [1, 1, 0, 0], 3)).values.all()


def test_sub_stochastic_scalar_frequency_severity():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = ProteusVariable(
        dim_name="dim1",
        values={
            "a": FreqSevSims([0, 0, 1, 2], [1, 2, 3, 4], 3),
            "b": FreqSevSims([0, 1, 1, 2], [1, 2, 3, 4], 3),
        },
    )
    z = x - y
    assert (
        ProteusVariable(
            dim_name="dim1",
            values={
                "a": FreqSevSims([0, 0, 1, 2], [0, -1, -1, -1], 3),
                "b": FreqSevSims([0, 1, 1, 2], [1, 1, 0, 0], 3),
            },
        )
        == z
    ).all()


def test_corr():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 10, 2]), "b": StochasticScalar([2, 3, 4])},
    )
    matrix = x.correlation_matrix()
    assert (np.array(matrix) == np.array([[1, 0.5], [0.5, 1]])).all()


def test_get_value_at_sim():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    assert x.get_value_at_sim(0).values == {"a": 1, "b": 2}
    assert x.get_value_at_sim(1).values == {"a": 2, "b": 3}


def test_get_value_at_sim_stochastic():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    assert (
        x.get_value_at_sim(StochasticScalar([0, 2]))
        == ProteusVariable(
            "dim1", {"a": StochasticScalar([1, 3]), "b": StochasticScalar([2, 4])}
        )
    ).all()


def test_array_ufunc():
    x = ProteusVariable(
        dim_name="dim1",
        values={"foo": StochasticScalar([1, 2, 3])},
    )
    y = np.exp(x)
    assert (
        y.values["foo"] == StochasticScalar([np.exp(1), np.exp(2), np.exp(3)])
    ).all()


def test_array_func2():
    x = ProteusVariable(
        dim_name="dim1",
        values={"foo": StochasticScalar([1, 2, 3]), "bar": StochasticScalar([1, 2, 3])},
    )
    y = np.cumsum(x)
    assert (
        y
        == ProteusVariable(
            "dim1",
            {"foo": StochasticScalar([1, 2, 3]), "bar": StochasticScalar([2, 4, 6])},
        )
    ).all()


def test_from_csv():
    x = ProteusVariable.from_csv("tests/data/variable.csv", "class", "value")
    expected = ProteusVariable(
        dim_name="class",
        values={
            "Motor": StochasticScalar([0.1, 0.4]),
            "Property": StochasticScalar([0.2, 0.5]),
            "Liability": StochasticScalar([0.3, 0.6]),
        },
    )
    assert (x == expected).all()


def test_mean_dict_stochastic():
    """Test mean method with dict values containing StochasticScalar."""
    x = ProteusVariable(
        dim_name="class",
        values={
            "Motor": StochasticScalar([1.0, 2.0, 3.0]),
            "Property": StochasticScalar([4.0, 5.0, 6.0]),
        },
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "class"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"Motor", "Property"}

    # Verify the means
    assert result.values["Motor"] == 2.0  # mean of [1, 2, 3]
    assert result.values["Property"] == 5.0  # mean of [4, 5, 6]


def test_mean_dict_freqsev():
    """Test mean method with dict values containing FreqSevSims."""
    from pal.frequency_severity import FreqSevSims

    # Create some simple FreqSevSims for testing
    freq_sev_1 = FreqSevSims([0, 1], [10.0, 20.0], 2)
    freq_sev_2 = FreqSevSims([0, 1], [30.0, 40.0], 2)

    x = ProteusVariable(
        dim_name="coverage",
        values={
            "CompDamage": freq_sev_1,
            "Collision": freq_sev_2,
        },
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "coverage"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"CompDamage", "Collision"}

    # Verify that FreqSevSims.aggregate().mean() was called
    # The result should be the mean of the aggregated values
    assert result.values["CompDamage"] == 15.0  # mean of [10, 20]
    assert result.values["Collision"] == 35.0  # mean of [30, 40]


def test_mean_dict_scalars():
    """Test mean method with dict values containing scalar values."""
    x = ProteusVariable(
        dim_name="factor",
        values={
            "inflation": 1.03,
            "discount": 0.95,
            "trend": 1.02,
        },
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "factor"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"inflation", "discount", "trend"}

    # Scalar values should be unchanged
    assert result.values["inflation"] == 1.03
    assert result.values["discount"] == 0.95
    assert result.values["trend"] == 1.02


def test_mean_mixed_dict():
    """Test mean method with dict values containing mixed types."""
    x = ProteusVariable(
        dim_name="mixed",
        values={
            "stochastic": StochasticScalar([10.0, 20.0, 30.0]),
            "scalar": 5.0,
        },
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "mixed"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"stochastic", "scalar"}

    # Verify the values
    assert result.values["stochastic"] == 20.0  # mean of [10, 20, 30]
    assert result.values["scalar"] == 5.0  # unchanged scalar


def test_mean_nested_proteus_variable() -> None:
    """Test mean method with nested ProteusVariable objects."""
    inner_var = ProteusVariable(
        dim_name="inner",
        values={"a": StochasticScalar([2.0, 4.0, 6.0]), "b": 10.0},
    )

    x = ProteusVariable(
        dim_name="outer",
        values={
            "nested": inner_var,
            "simple": StochasticScalar([1.0, 3.0, 5.0]),
        },
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "outer"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"nested", "simple"}

    # Nested ProteusVariable should be converted to float via mean
    assert result.values["nested"] == 7.0  # mean of inner_var.mean() = (4.0 + 10.0) / 2
    assert result.values["simple"] == 3.0  # mean of [1, 3, 5]


def test_mean_empty_values():
    """Test mean method with empty values (edge case)."""
    x = ProteusVariable(dim_name="empty", values={})

    result = x.mean()

    # Should return ProteusVariable with empty list
    assert result.dim_name == "empty"
    assert result.values == {}


def test_mean_single_value():
    """Test mean method with single stochastic value."""
    x = ProteusVariable(
        dim_name="single",
        values={"only": StochasticScalar([5.0])},
    )

    result = x.mean()

    # Single value mean should be the value itself
    assert result.dim_name == "single"
    assert result.values["only"] == 5.0


def test_mean_large_dataset():
    """Test mean method with larger dataset to ensure robustness."""
    import numpy as np

    # Create larger arrays for testing
    large_array_1 = np.random.RandomState(42).normal(100, 15, 1000)
    large_array_2 = np.random.RandomState(123).exponential(50, 1000)

    x = ProteusVariable(
        dim_name="large",
        values={
            "normal": StochasticScalar(large_array_1),
            "exponential": StochasticScalar(large_array_2),
        },
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "large"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"normal", "exponential"}

    # Verify means are approximately correct (within tolerance due to randomness)
    assert abs(result.values["normal"] - np.mean(large_array_1)) < 1e-10
    assert abs(result.values["exponential"] - np.mean(large_array_2)) < 1e-10


def test_upsample_same_n_sims():
    """Test upsample method when current n_sims equals target n_sims."""
    x = ProteusVariable(
        dim_name="test",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([4, 5, 6])},
    )
    
    # Should return the same object when n_sims matches
    result = x.upsample(3)
    assert result is x  # Should be the same object, not a copy


def test_upsample_dict_stochastic_scalar():
    """Test upsample method with dict values containing StochasticScalar."""
    x = ProteusVariable(
        dim_name="test",
        values={
            "a": StochasticScalar([1, 2]),
            "b": StochasticScalar([3, 4]),
        },
    )
    
    result = x.upsample(4)
    
    # Verify structure
    assert result.dim_name == "test"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"a", "b"}
    
    # Verify upsampled values (should cycle through original values)
    assert (result.values["a"] == StochasticScalar([1, 2, 1, 2])).all()
    assert (result.values["b"] == StochasticScalar([3, 4, 3, 4])).all()


def test_upsample_dict_scalar_values():
    """Test upsample method with dict values containing scalar values."""
    x = ProteusVariable(
        dim_name="test",
        values={
            "a": 10,
            "b": 20.5,
        },
    )
    
    result = x.upsample(5)
    
    # Verify structure
    assert result.dim_name == "test"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"a", "b"}
    
    # Scalar values should remain unchanged
    assert result.values["a"] == 10
    assert result.values["b"] == 20.5


def test_upsample_dict_mixed_types():
    """Test upsample method with dict values containing mixed types."""
    x = ProteusVariable(
        dim_name="test",
        values={
            "stochastic": StochasticScalar([1, 2, 3]),
            "scalar": 42,
        },
    )
    
    result = x.upsample(6)
    
    # Verify structure
    assert result.dim_name == "test"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"stochastic", "scalar"}
    
    # Verify upsampled stochastic value
    assert (result.values["stochastic"] == StochasticScalar([1, 2, 3, 1, 2, 3])).all()
    # Scalar value should remain unchanged
    assert result.values["scalar"] == 42


def test_upsample_dict_freqsev():
    """Test upsample method with dict values containing FreqSevSims."""
    from pal.frequency_severity import FreqSevSims
    
    freq_sev = FreqSevSims([0, 1], [10.0, 20.0], 2)
    x = ProteusVariable(
        dim_name="test",
        values={
            "coverage": freq_sev,
            "scalar": 5.0,
        },
    )
    
    result = x.upsample(4)
    
    # Verify structure
    assert result.dim_name == "test"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"coverage", "scalar"}
    
    # FreqSevSims should be upsampled
    assert isinstance(result.values["coverage"], FreqSevSims)
    assert result.values["coverage"].n_sims == 4
    
    # Scalar value should remain unchanged
    assert result.values["scalar"] == 5.0


def test_upsample_empty_dict():
    """Test upsample method with empty dict values."""
    x = ProteusVariable(dim_name="test", values={})
    
    result = x.upsample(10)
    
    # Should return a ProteusVariable with empty dict
    assert result.dim_name == "test"
    assert result.values == {}


def test_upsample_single_value():
    """Test upsample method with single stochastic value."""
    x = ProteusVariable(
        dim_name="test",
        values={"single": StochasticScalar([5.0])},
    )
    
    result = x.upsample(3)
    
    # Verify structure
    assert result.dim_name == "test"
    assert isinstance(result.values, dict)
    assert set(result.values.keys()) == {"single"}
    
    # Single value should be repeated
    assert (result.values["single"] == StochasticScalar([5.0, 5.0, 5.0])).all()


def test_upsample_n_sims_property():
    """Test that upsample correctly updates n_sims property."""
    x = ProteusVariable(
        dim_name="test",
        values={"a": StochasticScalar([1, 2, 3])},
    )
    
    # Original n_sims should be 3
    assert x.n_sims == 3
    
    result = x.upsample(6)
    
    # Result n_sims should be 6
    assert result.n_sims == 6


def test_upsample_preserve_dim_name():
    """Test that upsample preserves the dimension name."""
    x = ProteusVariable(
        dim_name="custom_dimension",
        values={"a": StochasticScalar([1, 2])},
    )
    
    result = x.upsample(4)
    
    assert result.dim_name == "custom_dimension"


def test_upsample_large_multiplier():
    """Test upsample with a large multiplier."""
    x = ProteusVariable(
        dim_name="test",
        values={"a": StochasticScalar([1, 2])},
    )
    
    result = x.upsample(100)
    
    # Should cycle through original values 50 times
    expected_values = [1, 2] * 50
    assert (result.values["a"] == StochasticScalar(expected_values)).all()
    assert result.n_sims == 100
