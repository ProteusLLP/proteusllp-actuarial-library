"""GPU residency tests for nearest correlation matrix repair."""

import pytest

from pal._maths import xp
from pal.correlation import nearest_correlation

pytestmark = pytest.mark.skipif(xp.__name__ != "cupy", reason="requires a CuPy backend")


@pytest.mark.parametrize("method", ["higham", "spectral"])
def test_correlation_repair_remains_on_gpu(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    matrix = xp.asarray(
        [[1.0, 0.8, 0.8, 0.6], [0.8, 1.0, -0.4, 0.7],
         [0.8, -0.4, 1.0, -0.9], [0.6, 0.7, -0.9, 1.0]]
    )

    def reject_host_copy(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected GPU-to-CPU copy")

    monkeypatch.setattr(xp, "asnumpy", reject_host_copy)
    repaired = nearest_correlation(matrix, method=method, minimum_eigenvalue=1e-5)

    assert isinstance(repaired, xp.ndarray)
    assert bool(xp.allclose(xp.diag(repaired), 1.0))
    assert bool(xp.allclose(repaired, repaired.T))
    assert float(xp.linalg.eigvalsh(repaired)[0].item()) >= 1e-5 - 1e-8
