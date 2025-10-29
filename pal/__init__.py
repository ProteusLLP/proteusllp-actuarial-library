"""
Proteus Actuarial Library (PAL)
==========================
http:://github.com/ProteusLLP/proteus-actuarial-library

A simple, fast and lightweight framework for building simulation-based actuarial and financial models.


PAL is designed to look after the complicated stuff, such as copulas and simulation re-ordering, providing easy to use objects and clear syntax.

PAL is based on the scientific python stack of numpy and scipy for fast performance. It can optionally run on a GPU using the cupy package for extremely fast performance. It is designed for interoperability with numpy and ndarrays.

"""

from .contracts import *
from .distributions import *
from .frequency_severity import *
from .config import *
from .stats import *
