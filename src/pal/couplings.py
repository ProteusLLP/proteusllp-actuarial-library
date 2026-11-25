"""Stochastic variable coupling and dependency management.

Provides coupling mechanisms for stochastic variables, allowing them to maintain
dependency relationships during reordering and copula applications. Key classes
include CouplingGroup for managing variable groups and ProteusStochasticVariable
as the base class for all stochastic types.
"""

from __future__ import annotations

import typing as t
import weakref
from abc import ABC

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin

from pal._compat import Self
from pal._maths import asnumpy, to_backend
from pal.config import Config


def _is_scalar_constant(variable: ProteusStochasticVariable) -> bool:
    """A length-one StochasticScalar is a constant, not a one-path random factor.

    Args:
        variable: Stochastic variable whose coupling or data is being processed.
    """
    from .stochastic_scalar import StochasticScalar

    return isinstance(variable, StochasticScalar) and variable.n_sims == 1


def validate_stochastic_axes(variables: t.Iterable[ProteusStochasticVariable], n_sims: int | None = None) -> int | None:
    """Validate full simulation axes while permitting length-one scalar constants.

    Only StochasticScalar objects with one element can broadcast. Frequency-
    severity results and nonconstant vectors must always match the full
    scenario count. This validation does not mutate variables or coupling
    groups, including when a later input is incompatible.

    Args:
        n_sims: Number of simulations on the scenario axis.
        variables: Stochastic variables to process in their existing order.
    """
    has_constant = False
    for variable in variables:
        if _is_scalar_constant(variable):
            has_constant = True
            continue
        if n_sims is None:
            n_sims = variable.n_sims
        elif variable.n_sims != n_sims:
            raise ValueError(f"Stochastic inputs must have matching n_sims (expected {n_sims}, got {variable.n_sims})")
    return n_sims if n_sims is not None else (1 if has_constant else None)


def couple_stochastic_results(
    results: t.Iterable[ProteusStochasticVariable],
    inputs: t.Iterable[ProteusStochasticVariable] = (),
    *,
    n_sims: int | None = None,
) -> None:
    """Link scenario-indexed result leaves to all upstream stochastic inputs.

    The caller must finish numerical calculations and construct *all* results
    before calling this function. Scenario lengths are validated across every
    input and result before any coupling group changes. In particular, expanded
    claim-level parameter temporaries must not be passed as inputs: a
    FreqSevSims result has the original simulation axis, not the claim axis.

    The first input's existing group is retained when possible. Groups hold
    weak references, so linking outputs does not prolong temporary lifetimes.

    Args:
        inputs: Input operands supplied to the NumPy universal function.
        n_sims: Number of simulations on the scenario axis.
        results: New stochastic results to associate with the input coupling groups.
    """
    output_leaves = tuple(results)
    sources = tuple(inputs)
    if not output_leaves:
        raise ValueError("At least one stochastic result is required")
    target_n = validate_stochastic_axes((*sources, *output_leaves), n_sims=n_sims)
    if target_n is None:
        raise ValueError("Cannot determine stochastic simulation axis")
    # A full-length result must be materialised before coupling. Constant
    # inputs are not attached to a scenario coupling group: reordering them
    # would incorrectly change their one-element shape.
    if target_n > 1 and any(variable.n_sims != target_n for variable in output_leaves):
        raise ValueError("Stochastic result has incompatible n_sims")
    participants = tuple(
        variable for variable in (*sources, *output_leaves) if target_n == 1 or not _is_scalar_constant(variable)
    )
    anchor = participants[0].coupled_variable_group
    for variable in participants:
        anchor.merge(variable.coupled_variable_group)


class CouplingGroup:
    """A class to represent a group of variables that are coupled together."""

    def __init__(self, variable: ProteusStochasticVariable):
        """Initialize coupling group with a single variable.

        Args:
            variable: The initial variable to add to the group.
        """
        self._refs: weakref.WeakValueDictionary[int, ProteusStochasticVariable] = weakref.WeakValueDictionary(
            {variable._uid: variable}  # type: ignore[misc]  # pyright: ignore[reportPrivateUsage]
        )

    def add(self, obj: ProteusStochasticVariable) -> None:
        """Add a variable to the coupling group.

        Args:
            obj: Candidate object to test or wrap.
        """
        self._refs[obj._uid] = obj  # type: ignore[misc]  # pyright: ignore[reportPrivateUsage]

    def discard(self, obj: ProteusStochasticVariable) -> None:
        """Remove a variable from the coupling group if it exists.

        Args:
            obj: Candidate object to test or wrap.
        """
        self._refs.pop(obj._uid, None)  # type: ignore[misc]  # pyright: ignore[reportPrivateUsage]

    def __contains__(self, obj: ProteusStochasticVariable) -> bool:
        """Test whether the object is a member of the collection.

        Args:
            obj: Candidate object to test or wrap.
        """
        return obj._uid in self._refs  # type: ignore[misc]  # pyright: ignore[reportPrivateUsage]

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._refs)

    def __iter__(self) -> t.Iterator[ProteusStochasticVariable]:
        """Iterate over the entries in their stored order."""
        # yield live objects only
        yield from self._refs.values()

    def merge(self, other: CouplingGroup) -> None:
        """Merge another coupling group into this one.

        Args:
            other: The other coupling group to merge.
        """
        if self is other:
            return
        # Merge the other group's variables into this one, updating their pointer.
        for var in list(other):
            var.coupled_variable_group = self
            self.add(var)
        return


class ProteusStochasticVariable(NDArrayOperatorsMixin, ABC):
    """A class to represent a stochastic variable in a simulation.

    Attributes:
        coupled_variable_group: Coupling group identifying simulations that must remain aligned.
        n_sims: Number of simulations represented by the object.
        values: Stored values in simulation or dimension order.
    """

    n_sims: int
    values: npt.NDArray[np.number[t.Any]]
    _uid: int

    # ===================
    # DUNDER METHODS
    # ===================

    def __init__(self) -> None:
        """Initialize stochastic variable with new coupling group."""
        self._uid = next(Config._uid_counter)  # type: ignore[misc]
        self.coupled_variable_group = CouplingGroup(self)

    def __array__(self, dtype: t.Any = None, copy: bool | None = None) -> npt.NDArray[t.Any]:
        """Return the underlying numpy array for compatibility with numpy functions.

        Args:
            copy: Whether to return a copy rather than sharing existing storage.
            dtype: Requested data type for the converted or reduced array.
        """
        result = asnumpy(self.values, dtype=dtype)
        if copy is True:
            return result.copy()
        return result

    # Override NDArrayOperatorsMixin comparison operators with proper return
    # type annotations.
    # NDArrayOperatorsMixin provides comparison operations but returns Any/object types.
    # Since our __array_ufunc__ correctly returns Self, we override these methods
    # to provide accurate type information to static type checkers.
    def __gt__(self, other: t.Any) -> Self:
        """Greater than comparison returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__gt__(other)  # type: ignore[return-value]

    def __ge__(self, other: t.Any) -> Self:
        """Greater than or equal comparison returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__ge__(other)  # type: ignore[return-value]

    def __lt__(self, other: t.Any) -> Self:
        """Less than comparison returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__lt__(other)  # type: ignore[return-value]

    def __le__(self, other: t.Any) -> Self:
        """Less than or equal comparison returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__le__(other)  # type: ignore[return-value]

    def __eq__(self, other: t.Any) -> Self:  # type: ignore[override]
        """Equality comparison returning instance of same type.

        Args:
            other: Other operand of the comparison.
        """
        return super().__eq__(other)  # type: ignore[return-value]

    def __ne__(self, other: t.Any) -> Self:  # type: ignore[override]
        """Not equal comparison returning instance of same type.

        Args:
            other: Other operand of the comparison.
        """
        return super().__ne__(other)  # type: ignore[return-value]

    # Override NDArrayOperatorsMixin arithmetic operators with proper return
    # type annotations for direct arithmetic operations and ufuncs.
    def __add__(self, other: t.Any) -> Self:
        """Add operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__add__(other)  # type: ignore[return-value]

    def __radd__(self, other: t.Any) -> Self:
        """Right add operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__radd__(other)  # type: ignore[return-value]

    def __sub__(self, other: t.Any) -> Self:
        """Subtract operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__sub__(other)  # type: ignore[return-value]

    def __rsub__(self, other: t.Any) -> Self:
        """Right subtract operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__rsub__(other)  # type: ignore[return-value]

    def __mul__(self, other: t.Any) -> Self:
        """Multiply operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__mul__(other)  # type: ignore[return-value]

    def __rmul__(self, other: t.Any) -> Self:
        """Right multiply operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__rmul__(other)  # type: ignore[return-value]

    def __truediv__(self, other: t.Any) -> Self:
        """Division operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__truediv__(other)  # type: ignore[return-value]

    def __rtruediv__(self, other: t.Any) -> Self:
        """Right division operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__rtruediv__(other)  # type: ignore[return-value]

    def __pow__(self, other: t.Any) -> Self:
        """Power operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__pow__(other)  # type: ignore[return-value]

    def __rpow__(self, other: t.Any) -> Self:
        """Right power operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__rpow__(other)  # type: ignore[return-value]

    def __floordiv__(self, other: t.Any) -> Self:
        """Floor division operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__floordiv__(other)  # type: ignore[return-value]

    def __rfloordiv__(self, other: t.Any) -> Self:
        """Right floor division operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__rfloordiv__(other)  # type: ignore[return-value]

    def __mod__(self, other: t.Any) -> Self:
        """Modulo operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__mod__(other)  # type: ignore[return-value]

    def __rmod__(self, other: t.Any) -> Self:
        """Right modulo operation returning instance of same type.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        return super().__rmod__(other)  # type: ignore[return-value]

    def __divmod__(self, other: t.Any) -> tuple[Self, Self]:
        """Combined quotient and remainder returning tuple of same types.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        # Validate both scenario axes before the operation or any group merge.
        if isinstance(other, ProteusStochasticVariable):
            validate_stochastic_axes((self, other))
        other_values = other.values if isinstance(other, ProteusStochasticVariable) else to_backend(other)
        q_values, r_values = np.divmod(self.values, other_values)

        # Create result objects
        quotient = type(self)(q_values)  # type: ignore[assignment]
        remainder = type(self)(r_values)  # type: ignore[assignment]

        sources = (self, other) if isinstance(other, ProteusStochasticVariable) else (self,)
        couple_stochastic_results((quotient, remainder), sources)

        return quotient, remainder  # type: ignore[return-value]

    def __rdivmod__(self, other: t.Any) -> tuple[Self, Self]:
        """Right combined quotient and remainder returning tuple of same types.

        Args:
            other: Other operand of the arithmetic or comparison operation.
        """
        # Validate both scenario axes before the operation or any group merge.
        if isinstance(other, ProteusStochasticVariable):
            validate_stochastic_axes((self, other))
        other_values = other.values if isinstance(other, ProteusStochasticVariable) else to_backend(other)
        q_values, r_values = np.divmod(other_values, self.values)

        # Create result objects
        quotient = type(self)(q_values)  # type: ignore[assignment]
        remainder = type(self)(r_values)  # type: ignore[assignment]

        sources = (self, other) if isinstance(other, ProteusStochasticVariable) else (self,)
        couple_stochastic_results((quotient, remainder), sources)

        return quotient, remainder  # type: ignore[return-value]

    def __pos__(self) -> Self:
        """Unary positive operation returning instance of same type."""
        # Unary positive is identity - use numpy's positive ufunc
        return np.positive(self)  # type: ignore[return-value]

    def __abs__(self) -> Self:
        """Absolute value operation returning instance of same type."""
        # Use numpy ufunc to ensure proper handling through __array_ufunc__
        return np.abs(self)  # type: ignore[return-value]

    def __round__(self, ndigits: int | None = None) -> Self:
        """Round to specified number of digits returning instance of same type.

        Args:
            ndigits: Number of decimal places to round to.
        """
        # Use numpy function to ensure proper handling through __array_ufunc__
        if ndigits is None:
            return np.round(self)  # type: ignore[return-value]
        return np.round(self, ndigits)  # type: ignore[return-value]

    def __floor__(self) -> Self:
        """Floor operation returning instance of same type."""
        # Use numpy ufunc to ensure proper handling through __array_ufunc__
        return np.floor(self)  # type: ignore[return-value]

    def __ceil__(self) -> Self:
        """Ceiling operation returning instance of same type."""
        # Use numpy ufunc to ensure proper handling through __array_ufunc__
        return np.ceil(self)  # type: ignore[return-value]

    def __trunc__(self) -> Self:
        """Truncate to integer returning instance of same type."""
        # Use numpy ufunc to ensure proper handling through __array_ufunc__
        return np.trunc(self)  # type: ignore[return-value]

    # Explicitly mark as unhashable (mutable type)
    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        """Prevent ambiguous truth value evaluation."""
        raise TypeError(
            f"Ambiguous truth value for {self.__class__.__name__}. Use .any() or .all()."  # noqa: E501
        )

    def __array_function__(
        self,
        func: t.Any,
        _: tuple[type, ...],
        args: tuple[t.Any, ...],
        kwargs: dict[str, t.Any],
    ) -> t.Any:
        """Handle numpy array functions by delegating to numpy after array conversion.

        This implementation satisfies the SupportsArray protocol requirement while
        maintaining backward compatibility. Array functions like np.sum(), np.mean()
        will work by converting to array first, returning numpy scalars/arrays.

        For type-preserving operations:
        - Use pal.maths functions (pnp.sum, pnp.mean) for explicit type preservation
        - Element-wise ufuncs (np.exp, +, -, etc.) preserve types via __array_ufunc__

        This approach is intentionally simple: we convert our custom types to arrays
        and let numpy handle the function, which matches numpy's default behavior.

        Args:
            _: Types participating in NumPy dispatch; unused by PAL's implementation.
            args: Positional arguments supplied by the caller or NumPy dispatch.
            func: Array function to dispatch.
            kwargs: Keyword arguments supplied by the caller or NumPy dispatch.
        """
        # Convert arguments that are our type to arrays
        converted_args: list[t.Any] = []
        for arg in args:
            if isinstance(arg, ProteusStochasticVariable):
                # Use self.__array__() since super() doesn't have __array__
                converted_args.append(arg.__array__())
            else:
                converted_args.append(arg)

        # Let numpy handle the function with array arguments
        return func(*converted_args, **kwargs)

    # ===================
    # PUBLIC METHODS
    # ===================

    def upsample(self, n_sims: int) -> Self:
        """Upsample the variable to match the specified number of simulations.

        Args:
            n_sims: The number of simulations to upsample to.

        Returns:
            A new instance of self with the upsampled values.
        """
        raise NotImplementedError

    def astype(self, dtype: np.dtype[t.Any] | type[t.Any]) -> npt.NDArray[t.Any]:
        """Convert the underlying values to a specified dtype.

        Args:
            dtype: The data type to convert to.

        Returns:
            A new numpy array with the specified dtype.
        """
        return self.values.astype(dtype)

    def any(self) -> bool:
        """Check if any value in the variable is True (non-zero).

        Returns:
            True if any value is non-zero, False otherwise.
        """
        return bool(np.any(self.values))

    def all(self) -> bool:
        """Check if all values in the variable are True (non-zero).

        Returns:
            True if all values are non-zero, False otherwise.
        """
        return bool(np.all(self.values))

    # ===================
    # PRIVATE METHODS
    # ===================

    # Private methods for internal use
    def _reorder_sims(self, new_order: t.Sequence[int]) -> None:
        """Reorder the simulations using the supplied permutation.

        Args:
            new_order: Permutation of simulation positions to apply.
        """
        raise NotImplementedError
