from pcm.variables import ProteusVariable as pv, StochasticScalar


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
    assert y.values == [
        StochasticScalar([3.2, 4.2, 5.2]),
        StochasticScalar([5.2, 6.2, 7.2]),
    ]


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
    assert y == StochasticScalar([3, 5, 7])


def test_sum_dict_stochastic():
    x = pv(
        dim_name="dim1",
        values={"a": StochasticScalar([1, 2, 3]), "b": StochasticScalar([2, 3, 4])},
    )
    y = sum(x)
    assert y == StochasticScalar([3, 5, 7])
    assert (
        y.coupled_variable_group
        == x[0].coupled_variable_group
        == x[1].coupled_variable_group
    )


import numpy as np


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


my_variable = pv(
    dim_name="dim1",
    values={"a": 1, "b": 2},
)
