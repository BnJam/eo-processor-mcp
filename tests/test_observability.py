"""Tests for observability module."""

from __future__ import annotations

import contextlib
import os

from eo_processor_mcp.observability import (
    MetricsRegistry,
    init_logging,
    instrument_tool_execution,
    new_correlation_id,
)


def test_metrics_registry_inc():
    r = MetricsRegistry()
    r.inc("test.counter")
    r.inc("test.counter", 5)
    assert r.snapshot()["test.counter"] == 6


def test_metrics_registry_gauge():
    r = MetricsRegistry()
    r.set_gauge("test.gauge", 42.0)
    assert r.gauge_snapshot()["test.gauge"] == 42.0


def test_metrics_registry_latency():
    r = MetricsRegistry()
    r.observe_latency("test.latency", 50.0)
    r.observe_latency("test.latency", 150.0)
    snap = r.latency_snapshot()
    assert "test.latency" in snap
    assert snap["test.latency"]["count"] == 2


def test_correlation_id():
    cid = new_correlation_id()
    assert isinstance(cid, str)
    assert len(cid) == 36


def test_instrument_tool_execution():
    def dummy_handler(_args):
        return "ok"

    result = instrument_tool_execution("test_tool", dummy_handler, {})
    assert result.value == "ok"
    assert result.correlation_id is not None
    assert result.duration_ms >= 0


def test_instrument_tool_execution_error():
    def failing_handler(_args):
        raise RuntimeError("boom")

    with contextlib.suppress(RuntimeError):
        instrument_tool_execution("failing_tool", failing_handler, {})


def test_init_logging():
    os.environ["EO_PROCESSOR_MCP_LOG_LEVEL"] = "DEBUG"
    init_logging()
    del os.environ["EO_PROCESSOR_MCP_LOG_LEVEL"]
