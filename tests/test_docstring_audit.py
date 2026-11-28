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


def test_explicit_cls_and_static_self_are_not_receivers():
    source = '''
def construct(cls):
    """Construct an instance."""

class Model:
    """A model."""

    @staticmethod
    def f(self):
        """An ordinary input named self."""
'''
    assert scan("test.py", source) == {
        "test.py:construct: missing argument cls",
        "test.py:Model.f: missing argument self",
    }


def test_stale_and_duplicate_entries():
    source = '''
class Model:
    """A model.

    Attributes:
        rate: Claim rate.
        rate: Claim rate again.
    """

    def __init__(self, rate):
        """Create a model.

        Args:
            rate: Claim rate.
            limit: Removed from the signature.
            rate: Claim rate again.
        """
        self.rate = rate
'''
    assert scan("test.py", source) == {
        "test.py:Model: duplicate attribute rate",
        "test.py:Model.__init__: unknown argument limit",
        "test.py:Model.__init__: duplicate argument rate",
    }


def test_numpy_style_and_bulleted_entries_are_malformed():
    source = '''
def f(n_sims, rng):
    """Simulate.

    Args:
        n_sims : int
            Number of simulations.
        - rng (RandomGenerator): Random generator.
    """
'''
    assert scan("test.py", source) == {
        "test.py:f: malformed Args entry 'n_sims : int'",
        "test.py:f: malformed Args entry '- rng (RandomGenerator): Random generator.'",
        "test.py:f: missing argument n_sims",
        "test.py:f: missing argument rng",
    }


def test_google_type_annotations_and_continuations_are_accepted():
    source = '''
def f(x, y, *args):
    """Calculate.

    Args:
        x (float): Ground-up loss, which may continue
            onto an indented second line.
        y:
            Description starting on the next line.
        *args: Further inputs.

    Returns:
        The result.

    This trailing paragraph is not part of the Args section.
    """
'''
    assert scan("test.py", source) == set()


def test_unindented_section_bodies_are_reported():
    source = '''
def f(x):
    """Calculate.

    Args:
    x: Unindented argument.

    References:
    Author (2020). Title.
    """
'''
    assert scan("test.py", source) == {
        "test.py:f: unindented Args section",
        "test.py:f: unindented References section",
        "test.py:f: missing argument x",
    }
