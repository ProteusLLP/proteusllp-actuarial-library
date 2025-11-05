# PAL Type System Architecture

## Design Principles

### Dependency Inversion: Types Don't Import Implementations
**Critical Rule**: Protocol definitions in `types.py` must NEVER import from concrete implementation modules (like `stochastic_scalar.py`, `variables.py`, etc.).

**Why**: This prevents circular dependencies and maintains clean architecture where:
- Protocols define interfaces (abstractions)
- Implementations depend on protocols (not vice versa)
- Type system remains independent of specific implementations

**Example**:
```python
import typing as t

# CORRECT: types.py defines protocol without importing implementations
class VectorLikeProtocol(t.Protocol):
    def __add__(self, other) -> t.Self: ...

# INCORRECT: types.py importing implementation
from pal.stochastic_scalar import StochasticScalar  # ❌ NEVER DO THIS
def generate() -> StochasticScalar: ...  # ❌ Protocol depends on implementation

# CORRECT: Protocol-based return type
from pal.types import VectorLike
def generate() -> VectorLike: ...  # ✅ Depends only on abstraction
```

### Protocols Are for Structural Typing, Not Inheritance
**Critical Rule**: Protocols in `types.py` are designed for structural subtyping and type hints ONLY. Classes should NOT inherit from protocols.

**Why**:
- Protocols define "shape" contracts for static type checking
- Inheritance creates runtime dependencies and coupling
- Structural typing allows flexibility - any object with the right methods satisfies the protocol
- Cleaner separation between type system and implementation hierarchy

**Pattern**:

<!--pytest-codeblocks:cont-->

```python
from typing import Self
from numpy.typing import NDArray
from pal.types import VectorLike
# CORRECT: Structural typing - no inheritance
class StochasticScalar:  # ← Does NOT inherit from VectorLikeProtocol
    def __add__(self, other) -> Self: ...  # ← But conforms to the protocol
    def __array__(self) -> NDArray: ...    # ← Type checker recognizes it as VectorLike

def process_vector(v: VectorLike) -> VectorLike:  # ← Uses protocol for typing
    return v + 1  # ← Works with StochasticScalar due to structural typing

# INCORRECT: Protocol inheritance
class StochasticScalar(VectorLikeProtocol):  # ❌ Don't inherit protocols
    ...
```


## Type Hierarchy

### Protocol Structure
```
ArithmeticProtocol
├── __add__, __sub__, __mul__, etc. (return t.Self)
│
├── NumericProtocol (scalar semantics)
│   ├── inherits ArithmeticProtocol
│   └── __lt__, __eq__ return bool
│
└── VectorLikeProtocol (vector semantics)
    ├── inherits ArithmeticProtocol
    ├── __lt__, __eq__ return t.Self (element-wise)
    ├── __array__, __len__ (numpy compatibility)
    └── __array_ufunc__ (numpy integration)

ProteusLike[T] (Generic Protocol)
├── inherits VectorLikeProtocol
├── inherits SequenceLike[T]
├── T: ScalarOrVector (bounded type parameter)
└── defines: n_sims, values, mean(), upsample()
```

### Type Aliases

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from typing import Any
from pal.types import NumericProtocol, VectorLike

Numeric = float | int | np.number[Any]
NumericLike = Numeric | NumericProtocol  # Scalars
VectorLike = VectorLikeProtocol           # Vectors
ScalarOrVector = NumericLike | VectorLike # Either
```

### Implementation Classes
```
ProteusVariable[T: ScalarOrVector] (generic, homogeneous container)
├── conforms to ProteusLike[T] protocol
└── values: Mapping[str, T] (all same type)

ProteusStochasticVariable (base class)
├── conforms to VectorLikeProtocol
├── provides __array__, __len__ for numpy
└── concrete implementations:
    ├── StochasticScalar (conforms to VectorLikeProtocol)
    └── FreqSevSims (conforms to VectorLikeProtocol)
```

## Module Dependencies

**Fix Order: Work from bottom to top to avoid cascading type errors**

```
Layer 0: Foundation
├── _maths/
├── config.py
└── types.py (protocols only, no imports from other pal modules)

Layer 1: Base Classes
├── couplings.py (depends on: types)
└── stochastic_scalar.py (depends on: couplings, types)

Layer 2: Core Components
├── distributions.py (depends on: stochastic_scalar, types)
└── frequency_severity.py (depends on: couplings, stochastic_scalar)

Layer 3: Containers & Analysis
├── variables.py (depends on: couplings, frequency_severity, stochastic_scalar, types)
└── stats.py (depends on: frequency_severity, types)

Layer 4: Applications
├── contracts.py (depends on: frequency_severity, variables)
└── copulas.py (depends on: variables, stochastic_scalar, types)
```

**Dependency Rules:**
- **types.py**: No imports from other pal modules (dependency inversion)
- **Higher layers**: Can import from lower layers only
- **Same layer**: Minimize cross-dependencies
