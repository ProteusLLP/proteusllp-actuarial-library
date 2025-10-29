from pal import FreqSevSims
import numpy as np
import pytest

x = FreqSevSims(np.array([0, 0, 0, 0]), np.array([100000, 800000, 500000, 200000]), 0)

print(x > 100000)

print(np.where(x > 100000, 0, 1))
