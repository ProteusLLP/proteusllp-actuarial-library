import numpy as np
from pal.variables import FreqSevSims, ProteusVariable, StochasticScalar


def test_empty():
    x = ProteusVariable(dim_name="dim1", values=[])
    assert x.values == []


def test_variable():
    x = ProteusVariable(dim_name="dim1", values=[1, 2, 3])
    y = x + 1
    assert y.values == [2, 3, 4]


def test_variable2():
    """Test that a variable can be created with a list of StochasticScalars."""
    x = ProteusVariable(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = x + 2.2
    assert (
        y
        == ProteusVariable(
            dim_name="dim1",
            values=[
                StochasticScalar([3.2, 4.2, 5.2]),
                StochasticScalar([4.2, 5.2, 6.2]),
            ],
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


def test_dict_variable_dereferencing():
    x = ProteusVariable(
        dim_name="dim1",
        values={"a": 1, "b": 2},
    )
    assert x["a"] == 1
    assert x["b"] == 2
    assert x[0] == 1
    assert x[1] == 2


def test_array_variable_dereferencing():
    x = ProteusVariable(
        dim_name="dim1",
        values=[1, 2],
    )
    assert x[0] == 1
    assert x[1] == 2


def test_sum():
    x = ProteusVariable(dim_name="dim1", values=[1, 2])
    y = sum(x)
    assert y == 3


def test_sum_stochastic():
    x = ProteusVariable(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = sum(x)
    assert (y == StochasticScalar([3, 5, 7])).all()


def test_sum_dict_stochastic():
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
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = x / 2.0
    assert (
        ProteusVariable(
            dim_name="dim1",
            values=[StochasticScalar([0.5, 1, 3 / 2]), StochasticScalar([1, 3 / 2, 2])],
        )
        == y
    ).all()


def test_divide_two():
    x = ProteusVariable(dim_name="dim1", values=[1, 2, 3])
    y = x / ProteusVariable(dim_name="dim1", values=[2, 4, 6])
    assert y.values == [0.5, 0.5, 0.5]


def test_rdivide():
    x = ProteusVariable(
        dim_name="dim1",
        values=[1, 2, 3],
    )
    y = 2 / x
    assert y.values == [2, 1, 2 / 3]


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
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = x - 1
    assert (
        ProteusVariable(
            dim_name="dim1",
            values=[StochasticScalar([0, 1, 2]), StochasticScalar([1, 2, 3])],
        )
        == y
    ).all()


def test_rsubtract():
    x = ProteusVariable(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = 1 - x
    assert (
        ProteusVariable(
            dim_name="dim1",
            values=[StochasticScalar([0, -1, -2]), StochasticScalar([-1, -2, -3])],
        )
        == y
    ).all()


def test_subtract_two():
    x = ProteusVariable(
        dim_name="dim1",
        values=[1, 2, 3],
    )
    y = x - ProteusVariable(dim_name="dim1", values=[2, 4, 6])
    assert y.values == [-1, -2, -3]


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
        values=[StochasticScalar([1, 2, 3])],
    )
    y = np.exp(x)
    assert (y.values[0] == StochasticScalar([np.exp(1), np.exp(2), np.exp(3)])).all()


def test_array_func2():
    x = ProteusVariable(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([1, 2, 3])],
    )
    y = np.cumsum(x)
    assert (
        y
        == ProteusVariable(
            "dim1", [StochasticScalar([1, 2, 3]), StochasticScalar([2, 4, 6])]
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


def test_mean_list_stochastic():
    """Test mean method with list values containing StochasticScalar."""
    x = ProteusVariable(
        dim_name="line",
        values=[
            StochasticScalar([1.0, 3.0, 5.0]),
            StochasticScalar([2.0, 4.0, 6.0]),
            StochasticScalar([10.0, 20.0, 30.0]),  # Same n_sims as others
        ],
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "line"
    assert isinstance(result.values, list)
    assert len(result.values) == 3

    # Verify the means
    assert result.values[0] == 3.0  # mean of [1, 3, 5]
    assert result.values[1] == 4.0  # mean of [2, 4, 6]
    assert result.values[2] == 20.0  # mean of [10, 20, 30]


def test_mean_list_freqsev():
    """Test mean method with list values containing FreqSevSims."""
    from pal.frequency_severity import FreqSevSims

    # Create FreqSevSims for testing
    freq_sev_1 = FreqSevSims([0, 1, 2], [100.0, 200.0, 300.0], 3)
    freq_sev_2 = FreqSevSims([0, 1, 2], [50.0, 150.0, 250.0], 3)  # Same n_sims

    x = ProteusVariable(
        dim_name="peril",
        values=[freq_sev_1, freq_sev_2],
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "peril"
    assert isinstance(result.values, list)
    assert len(result.values) == 2

    # Verify that FreqSevSims.aggregate().mean() was called
    assert result.values[0] == 200.0  # mean of [100, 200, 300]
    assert result.values[1] == 150.0  # mean of [50, 150, 250]


def test_mean_list_scalars():
    """Test mean method with list values containing scalar values."""
    x = ProteusVariable(
        dim_name="multiplier",
        values=[1.5, 2.0, 0.8, 1.2],
    )

    result = x.mean()

    # Verify the structure
    assert result.dim_name == "multiplier"
    assert isinstance(result.values, list)
    assert len(result.values) == 4

    # Scalar values should be unchanged
    assert result.values[0] == 1.5
    assert result.values[1] == 2.0
    assert result.values[2] == 0.8
    assert result.values[3] == 1.2


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
    x = ProteusVariable(dim_name="empty", values=[])

    result = x.mean()

    # Should return ProteusVariable with empty list
    assert result.dim_name == "empty"
    assert result.values == []


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
