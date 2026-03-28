"""Compatibility shim for typing features across Python versions.

Provides `Self` (3.11+) and `override` (3.12+) from `typing` when
available, falling back to `typing_extensions` for older Pythons.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

__all__ = ["Self", "override"]
