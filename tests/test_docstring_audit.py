"""Test the documentation contract independently of PAL runtime imports."""

from scripts.check_docstrings import scan


def test_missing_docstrings_arguments_and_attributes():
    source = '''
class Model:
    """A model."""

    def __init__(self, rate):
        self.rate = rate

    def loss(self, x, /, *, limit):
        """Calculate loss.

        Args:
            x: Ground-up loss.
        """
'''
    assert scan("test.py", source) == {
        "test.py:Model: missing attribute rate",
        "test.py:Model.__init__: missing docstring",
        "test.py:Model.__init__: missing argument rate",
        "test.py:Model.loss: missing argument limit",
    }


def test_variadic_async_and_multiline_descriptions():
    source = '''
async def run(*args, **kwargs):
    """Run the calculation.

    Args:
        *args:
            Inputs passed to the model.
        **kwargs: Keyword configuration.
    """
'''
    assert scan("test.py", source) == set()


def test_overloads_and_enum_attributes():
    source = '''
from enum import Enum
from typing import overload

class Method(Enum):
    """Sampling methods.

    Attributes:
        MC: Ordinary Monte Carlo.
    """
    MC = "mc"
    RQMC = "rqmc"

@overload
def run(x: int) -> int: ...

def run(x):
    """Run.

    Args:
        x: Input.
    """
    return x
'''
    assert scan("test.py", source) == {"test.py:Method: missing attribute RQMC"}


def test_empty_description_is_not_documentation():
    source = '''
def f(x):
    """Calculate.

    Args:
        x:

    Returns:
        The result.
    """
'''
    assert scan("test.py", source) == {"test.py:f: missing argument x"}
