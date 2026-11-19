"""Smoke tests for isolated memory profiling and portfolio scaling."""

import json
import sys

import numpy as np
import pytest
from benchmarks import profile_memory
from benchmarks.profile_memory import markdown_report, run_isolated, worker
from benchmarks.profile_portfolio import run_portfolio

from pal import set_random_seed


@pytest.mark.skipif(sys.platform == "win32", reason="RSS profiling requires Unix resource or Linux /proc")
def test_memory_worker_records_process_and_backend_allocations():
    """Even tiny workloads should report a coherent memory high-water mark."""
    record = worker("aggregate", 128, n_lobs=2, n_layers=2)
    assert record["status"] == "ok"
    assert record["rss_peak_mb"] >= record["rss_after_setup_mb"] >= record["rss_baseline_mb"]
    assert record["rss_increment_mb"] >= 0
    assert record["python_tracemalloc_peak_mb"] >= 0
    if record["gpu_pool_peak_mb"] is not None:
        assert record["gpu_pool_peak_mb"] >= record["gpu_pool_after_setup_mb"]


@pytest.mark.skipif(sys.platform == "win32", reason="RSS profiling requires Unix resource or Linux /proc")
def test_isolated_worker_returns_machine_readable_results():
    record = run_isolated("event_indexing", 64, n_lobs=2, n_layers=2)
    json.loads(json.dumps(record))
    assert record["status"] == "ok", record.get("error")
    assert record["n_sims"] == 64
    report = markdown_report(
        {
            "environment": {"backend": "cpu", "cpu": "test", "gpu": None, "python": "3", "commit": "x"},
            "results": [record],
        }
    )
    assert "Peak process RSS" in report


def test_portfolio_workload_small_and_reproducible():
    set_random_seed(123456)
    first = float(run_portfolio(n_sims=24, n_lobs=2, n_layers=3))
    set_random_seed(123456)
    second = float(run_portfolio(n_sims=24, n_lobs=2, n_layers=3))
    assert np.isfinite(first)
    assert first >= 0.0
    assert first == second


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="VmHWM is Linux-specific")
def test_linux_peak_rss_is_read_from_process_status(monkeypatch: pytest.MonkeyPatch):
    """Read the post-exec process peak, not an inherited getrusage high-water mark."""

    def fake_status(path, encoding="utf-8"):
        assert str(path) == "/proc/self/status"
        return "Name:\tpython\nVmHWM:\t8192 kB\nVmRSS:\t1024 kB\n"

    monkeypatch.setattr(profile_memory.Path, "read_text", fake_status)
    assert profile_memory._peak_rss_bytes() == 8192 * 1024
