import numpy as np

from pal.variables import ProteusVariable as pv, StochasticScalar, FreqSevSims


def test_empty():
    x = pv(dim_name="dim1", values=[])
    assert x.values == []


def test_variable():
    x = pv(dim_name="dim1", values=[1, 2, 3])
    y = x + 1
    assert y.values == [2, 3, 4]


def test_variable2():
    """Test that a variable can be created with a list of StochasticScalars."""
    x = pv(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = x + 2.2
    assert (
        y
        == pv(
            dim_name="dim1",
            values=[
                StochasticScalar([3.2, 4.2, 5.2]),
                StochasticScalar([4.2, 5.2, 6.2]),
            ],
        )
    ).all()


def test_variable3():
    """Test that a variable can be created with a dictionary, that label matching works, and that the variable can be summed."""
    x = pv(
        dim_name="dim1",
        values={"a": 1, "b": 2},
    )
    y = pv(
        dim_name="dim1",
        values={"b": 5, "a": 8},
    )
    z = x + y
    assert z.values == {"a": 9, "b": 7}


def test_dict_variable_dereferencing():
    x = pv(
        dim_name="dim1",
        values={"a": 1, "b": 2},
    )
    assert x["a"] == 1
    assert x["b"] == 2
    assert x[0] == 1
    assert x[1] == 2


def test_array_variable_dereferencing():
    x = pv(
        dim_name="dim1",
        values=[1, 2],
    )
    assert x[0] == 1
    assert x[1] == 2


def test_sum():
    x = pv(dim_name="dim1", values=[1, 2])
    y = sum(x)
    assert y == 3


def test_sum_stochastic():
    x = pv(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = sum(x)
    assert (y == StochasticScalar([3, 5, 7])).all()


def test_sum_dict_stochastic():
    x = pv(
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
    x = pv(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = x / 2.0
    assert (
        pv(
            dim_name="dim1",
            values=[StochasticScalar([0.5, 1, 3 / 2]), StochasticScalar([1, 3 / 2, 2])],
        )
        == y
    ).all()


def test_divide_two():
    x = pv(dim_name="dim1", values=[1, 2, 3])
    y = x / pv(dim_name="dim1", values=[2, 4, 6])
    assert y.values == [0.5, 0.5, 0.5]


def test_rdivide():
    x = pv(
        dim_name="dim1",
        values=[1, 2, 3],
    )
    y = 2 / x
    assert y.values == [2, 1, 2 / 3]


def test_multiply_stochastic():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = StochasticScalar([2, 3, 4])
    z = y * x
    assert (
        pv(
            dim_name="dim1",
            values={
                "a": StochasticScalar([2, 6, 12]),
                "b": StochasticScalar([4, 9, 16]),
            },
        )
        == z
    ).all()


def test_rmultiply_stochastic():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = StochasticScalar([2, 3, 4])
    z = x * y
    assert (
        pv(
            dim_name="dim1",
            values={
                "a": StochasticScalar([2, 6, 12]),
                "b": StochasticScalar([4, 9, 16]),
            },
        )
        == z
    ).all()


def test_subtract():
    x = pv(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = x - 1
    assert (
        pv(
            dim_name="dim1",
            values=[StochasticScalar([0, 1, 2]), StochasticScalar([1, 2, 3])],
        )
        == y
    ).all()


def test_rsubtract():
    x = pv(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([2, 3, 4])],
    )
    y = 1 - x
    assert (
        pv(
            dim_name="dim1",
            values=[StochasticScalar([0, -1, -2]), StochasticScalar([-1, -2, -3])],
        )
        == y
    ).all()


def test_subtract_two():
    x = pv(
        dim_name="dim1",
        values=[1, 2, 3],
    )
    y = x - pv(dim_name="dim1", values=[2, 4, 6])
    assert y.values == [-1, -2, -3]


def test_sub_stochastic():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = StochasticScalar([2, 3, 4])
    z = y - x
    assert (
        pv(
            dim_name="dim1",
            values={
                "a": StochasticScalar([1, 1, 1]),
                "b": StochasticScalar([0, 0, 0]),
            },
        )
        == z
    ).all()


def test_rsub_stochastic():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = StochasticScalar([2, 3, 4])
    z = x - y
    assert (
        pv(
            dim_name="dim1",
            values={
                "a": StochasticScalar([-1, -1, -1]),
                "b": StochasticScalar([0, 0, 0]),
            },
        )
        == z
    ).all()


def test_sub_stochastic_scalar():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = pv(
        dim_name="dim1",
        values={"a": 1, "b": 2},
    )
    z = x - y
    assert (
        pv(
            dim_name="dim1",
            values={
                "a": StochasticScalar([0, 1, 2]),
                "b": StochasticScalar([0, 1, 2]),
            },
        )
        == z
    ).all()


def test_rsub_stochastic_scalar():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = pv(
        dim_name="dim1",
        values={"a": 1, "b": 2},
    )
    z = y - x
    assert (
        pv(
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
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = pv(
        dim_name="dim1",
        values={
            "a": FreqSevSims([0, 0, 1, 2], [1, 2, 3, 4], 3),
            "b": FreqSevSims([0, 1, 1, 2], [1, 2, 3, 4], 3),
        },
    )
    z = x - y
    assert (
        pv(
            dim_name="dim1",
            values={
                "a": FreqSevSims([0, 0, 1, 2], [0, -1, -1, -1], 3),
                "b": FreqSevSims([0, 1, 1, 2], [1, 1, 0, 0], 3),
            },
        )
        == z
    ).all()


def test_corr():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 10, 2]), "b": StochasticScalar([2, 3, 4])},
    )
    matrix = x.correlation_matrix()
    assert (np.array(matrix) == np.array([[1, 0.5], [0.5, 1]])).all()


def test_get_value_at_sim():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    assert x.get_value_at_sim(0).values == {"a": 1, "b": 2}
    assert x.get_value_at_sim(1).values == {"a": 2, "b": 3}


def test_get_value_at_sim_stochastic():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    assert (
        x.get_value_at_sim(StochasticScalar([0, 2]))
        == pv("dim1", {"a": StochasticScalar([1, 3]), "b": StochasticScalar([2, 4])})
    ).all()


def test_array_ufunc():
    x = pv(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3])],
    )
    y = np.exp(x)
    assert (y.values[0] == StochasticScalar([np.exp(1), np.exp(2), np.exp(3)])).all()


def test_array_func2():
    x = pv(
        dim_name="dim1",
        values=[StochasticScalar([1, 2, 3]), StochasticScalar([1, 2, 3])],
    )
    y = np.cumsum(x)
    assert (
        y == pv("dim1", [StochasticScalar([1, 2, 3]), StochasticScalar([2, 4, 6])])
    ).all()


def test_from_csv():
    x = pv.from_csv("tests/data/variable.csv", "class", "value")
    expected = pv(
        dim_name="class",
        values={
            "Motor": StochasticScalar([0.1, 0.4]),
            "Property": StochasticScalar([0.2, 0.5]),
            "Liability": StochasticScalar([0.3, 0.6]),
        },
    )
    assert (x == expected).all()
