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


class CouplingGroup:
    """A class to represent a group of variables that are coupled together."""

    variables: weakref.WeakSet[ProteusStochasticVariable]

    def __init__(self, variable: ProteusStochasticVariable):
        """Initialize coupling group with a single variable.

        Args:
            variable: The initial variable to add to the group.
        """
        # Start the group with a single variable, stored as a weak reference.
        self.variables: weakref.WeakSet[ProteusStochasticVariable] = weakref.WeakSet(
            [variable]
        )

    @property
    def id(self) -> int:
        """Get the unique identifier for this coupling group."""
        return id(self)

    def merge(self, other: CouplingGroup) -> None:
        """Merge another coupling group into this one.

        Args:
            other: The other coupling group to merge.
        """
        if self is other:
            return
        # Merge the other group's variables into this one, updating their pointer.
        for var in list(other.variables):
            var.coupled_variable_group = self
            self.variables.add(var)
        return


class ProteusStochasticVariable(NDArrayOperatorsMixin, ABC):
    """A class to represent a stochastic variable in a simulation."""

    n_sims: int | None = None
    values: npt.NDArray[np.floating]

    # ===================
    # DUNDER METHODS
    # ===================

    def __init__(self) -> None:
        """Initialize stochastic variable with new coupling group."""
        self.coupled_variable_group = CouplingGroup(self)

    def __array__(self, dtype: t.Any = None) -> npt.NDArray[np.floating]:
        """Return the underlying numpy array for compatibility with numpy functions."""
        return self.values if dtype is None else np.asarray(self.values, dtype=dtype)

    # Override NDArrayOperatorsMixin comparison operators with proper return
    # type annotations.
    # NDArrayOperatorsMixin provides comparison operations but returns Any/object types.
    # Since our __array_ufunc__ correctly returns Self, we override these methods
    # to provide accurate type information to static type checkers.
    def __gt__(self, other: t.Any) -> t.Self:
        """Greater than comparison returning instance of same type."""
        return super().__gt__(other)  # type: ignore[return-value]

    def __ge__(self, other: t.Any) -> t.Self:
        """Greater than or equal comparison returning instance of same type."""
        return super().__ge__(other)  # type: ignore[return-value]

    def __lt__(self, other: t.Any) -> t.Self:
        """Less than comparison returning instance of same type."""
        return super().__lt__(other)  # type: ignore[return-value]

    def __le__(self, other: t.Any) -> t.Self:
        """Less than or equal comparison returning instance of same type."""
        return super().__le__(other)  # type: ignore[return-value]

    def __eq__(self, other: t.Any) -> t.Self:  # type: ignore[override]
        """Equality comparison returning instance of same type."""
        return super().__eq__(other)  # type: ignore[return-value]

    def __ne__(self, other: t.Any) -> t.Self:  # type: ignore[override]
        """Not equal comparison returning instance of same type."""
        return super().__ne__(other)  # type: ignore[return-value]

    # Override NDArrayOperatorsMixin arithmetic operators with proper return
    # type annotations for direct arithmetic operations and ufuncs.
    def __add__(self, other: t.Any) -> t.Self:
        """Add operation returning instance of same type."""
        return super().__add__(other)  # type: ignore[return-value]

    def __radd__(self, other: t.Any) -> t.Self:
        """Right add operation returning instance of same type."""
        return super().__radd__(other)  # type: ignore[return-value]

    def __sub__(self, other: t.Any) -> t.Self:
        """Subtract operation returning instance of same type."""
        return super().__sub__(other)  # type: ignore[return-value]

    def __rsub__(self, other: t.Any) -> t.Self:
        """Right subtract operation returning instance of same type."""
        return super().__rsub__(other)  # type: ignore[return-value]

    def __mul__(self, other: t.Any) -> t.Self:
        """Multiply operation returning instance of same type."""
        return super().__mul__(other)  # type: ignore[return-value]

    def __rmul__(self, other: t.Any) -> t.Self:
        """Right multiply operation returning instance of same type."""
        return super().__rmul__(other)  # type: ignore[return-value]

    def __truediv__(self, other: t.Any) -> t.Self:
        """Division operation returning instance of same type."""
        return super().__truediv__(other)  # type: ignore[return-value]

    def __rtruediv__(self, other: t.Any) -> t.Self:
        """Right division operation returning instance of same type."""
        return super().__rtruediv__(other)  # type: ignore[return-value]

    def __pow__(self, other: t.Any) -> t.Self:
        """Power operation returning instance of same type."""
        return super().__pow__(other)  # type: ignore[return-value]

    def __rpow__(self, other: t.Any) -> t.Self:
        """Right power operation returning instance of same type."""
        return super().__rpow__(other)  # type: ignore[return-value]

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

    def upsample(self, n_sims: int) -> t.Self:
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
        raise NotImplementedError
