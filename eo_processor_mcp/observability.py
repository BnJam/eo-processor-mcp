"""Observability primitives for EO Processor MCP.

This module provides:
* Structured logging initialization (stderr only)
* Correlation ID generation per request
* Minimal in-process metrics counters
* Timing utilities and a no-op trace span abstraction

Design goals:
- Zero external dependencies.
- Safe to import early (lazy initialization where possible).
- Does not write to stdout (stdin/stdout reserved for MCP protocol).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

LOG_LEVEL_ENV = "EO_PROCESSOR_MCP_LOG_LEVEL"
LOG_FORMAT_ENV = "EO_PROCESSOR_MCP_LOG_FORMAT"
ENABLE_METRICS_ENV = "EO_PROCESSOR_MCP_ENABLE_METRICS"
ENABLE_TRACE_ENV = "EO_PROCESSOR_MCP_ENABLE_TRACE"
LATENCY_BUCKETS_ENV = "EO_PROCESSOR_MCP_LATENCY_BUCKETS_MS"

_logger_state = {"initialized": False}
_logger_initialized = False
_init_lock = RLock()


def _get_bool(env: str, default: bool) -> bool:
    val = os.getenv(env)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def init_logging() -> None:
    """Configure the library logger (re-initializable for tests)."""
    if _logger_state["initialized"] and _logger_initialized:
        return
    with _init_lock:
        if _logger_state["initialized"] and _logger_initialized:
            return
        level_name = os.getenv(LOG_LEVEL_ENV, "WARNING")
        level: int | None
        if level_name is None:
            level = logging.WARNING
        else:
            normalized = level_name.upper()
            level = getattr(logging, normalized, None)
            if not isinstance(level, int):
                level = logging.INFO
        log_format = os.getenv(LOG_FORMAT_ENV, "text").lower()
        handler = logging.StreamHandler(stream=sys.stderr)
        if log_format == "json":
            handler.setFormatter(JSONLogFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(levelname)s %(name)s: %(message)s"),
            )
        logger = logging.getLogger("eo_processor_mcp")
        logger.setLevel(level)
        logger.handlers = [handler]
        logger.propagate = False
    _logger_state["initialized"] = True
    globals()["_logger_initialized"] = True


_LOG_RECORD_BASE_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process",
}


class JSONLogFormatter(logging.Formatter):
    """Serialize log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created),
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attr in [
            "event", "tool_name", "duration_ms", "error_type",
            "correlation_id",
        ]:
            if hasattr(record, attr):
                base[attr] = getattr(record, attr)
        if record.exc_info:
            try:
                base["exc_info"] = self.formatException(record.exc_info)
            except Exception:
                base["exc_info"] = str(record.exc_info)
        for key, value in record.__dict__.items():
            if key in base or key in _LOG_RECORD_BASE_KEYS or key.startswith("_"):
                continue
            base[key] = value
        return json.dumps(base, separators=(",", ":"))


class MetricsRegistry:
    """In-process metrics counters + latency histograms (thread-safe)."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._counters: dict[str, int] = {}
        self._latency_buckets = self._parse_buckets()
        self._histograms: dict[str, list[int]] = {}
        self._latency_stats: dict[str, dict[str, float]] = {}
        self._gauges: dict[str, float] = {}

    def _parse_buckets(self) -> list[float]:
        raw = os.getenv(LATENCY_BUCKETS_ENV)
        if raw:
            try:
                buckets = sorted(
                    {float(x.strip()) for x in raw.split(",") if x.strip()},
                )
                return [b for b in buckets if b > 0]
            except (ValueError, TypeError):
                pass
        return [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000]

    def inc(self, name: str, amount: int = 1) -> None:
        if not _get_bool(ENABLE_METRICS_ENV, True):
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe_latency(self, name: str, value_ms: float) -> None:
        if not _get_bool(ENABLE_METRICS_ENV, True):
            return
        with self._lock:
            hist = self._histograms.get(name)
            if hist is None:
                hist = [0] * (len(self._latency_buckets) + 1)
                self._histograms[name] = hist
            stats = self._latency_stats.setdefault(
                name,
                {
                    "count": 0,
                    "sum": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                },
            )
            stats["count"] += 1
            stats["sum"] += value_ms
            stats["min"] = min(stats["min"], value_ms)
            stats["max"] = max(stats["max"], value_ms)
            placed = False
            for idx, upper in enumerate(self._latency_buckets):
                if value_ms <= upper:
                    hist[idx] += 1
                    placed = True
                    break
            if not placed:
                hist[-1] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def set_gauge(self, name: str, value: float) -> None:
        if not _get_bool(ENABLE_METRICS_ENV, True):
            return
        with self._lock:
            self._gauges[name] = float(value)

    def latency_snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            snap: dict[str, dict[str, Any]] = {}
            for name, counts in self._histograms.items():
                stats = self._latency_stats.get(
                    name, {"count": 0, "sum": 0.0, "min": 0.0, "max": 0.0},
                )
                bucket_map = {}
                for idx, upper in enumerate(self._latency_buckets):
                    bucket_map[str(int(upper))] = counts[idx]
                bucket_map["overflow"] = counts[-1]
                snap[name] = {
                    "count": int(stats["count"]),
                    "sum": stats["sum"],
                    "min": 0.0 if stats["count"] == 0 else stats["min"],
                    "max": 0.0 if stats["count"] == 0 else stats["max"],
                    "buckets": bucket_map,
                }
            return snap

    def gauge_snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._gauges)


metrics = MetricsRegistry()


def _metric_name(*parts: str) -> str:
    return ".".join(parts)


@contextmanager
def trace_span(name: str, **_attrs: Any) -> Generator[None, None, None]:
    """No-op span context manager placeholder."""
    enabled = _get_bool(ENABLE_TRACE_ENV, False)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            logging.getLogger("eo_processor_mcp").debug(
                "trace_span",
                extra={
                    "event": "trace_span",
                    "span": name,
                    "duration_ms": round(duration_ms, 2),
                },
            )


def new_correlation_id() -> str:
    return str(uuid.uuid4())


@dataclass
class ToolExecutionResult:
    """Container for instrumented tool execution output."""
    value: Any
    correlation_id: str
    duration_ms: float
    error_type: str | None = None


def instrument_tool_execution(
    tool_name: str,
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> ToolExecutionResult:
    """Execute a tool handler with logging, timing, metrics, and correlation id."""
    init_logging()
    correlation_id = new_correlation_id()
    logger = logging.getLogger("eo_processor_mcp")
    invocation_metric = _metric_name("tool_invocations_total", tool_name)
    global_invocation_metric = _metric_name("tool_invocations_total", "_all")
    inflight_metric = _metric_name("tool_inflight_current", tool_name)
    global_inflight_metric = _metric_name("tool_inflight_current", "_all")
    metrics.inc(invocation_metric)
    metrics.inc(global_invocation_metric)
    metrics.inc(inflight_metric)
    metrics.inc(global_inflight_metric)
    t0 = time.perf_counter()
    error_type: str | None = None
    duration_ms = 0.0
    try:
        with trace_span(f"tool.{tool_name}"):
            result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return ToolExecutionResult(
            value=result,
            correlation_id=correlation_id,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        etype = type(exc).__name__
        if "timeout" in etype.lower():
            error_type = "TimeoutError"
        elif "memory" in etype.lower():
            error_type = "MemoryError"
        else:
            error_type = "UnknownError"
        metrics.inc(_metric_name("tool_errors_total", tool_name, error_type))
        duration_ms = (time.perf_counter() - t0) * 1000.0
        logger.warning(
            "tool_error",
            extra={
                "event": "tool_error",
                "tool_name": tool_name,
                "error_type": error_type,
                "correlation_id": correlation_id,
                "duration_ms": round(duration_ms, 2),
            },
        )
        raise
    finally:
        if duration_ms == 0.0:
            duration_ms = (time.perf_counter() - t0) * 1000.0
        tool_latency_metric = _metric_name("tool_latency_ms", tool_name)
        global_latency_metric = _metric_name("tool_latency_ms", "_all")
        metrics.observe_latency(tool_latency_metric, duration_ms)
        metrics.observe_latency(global_latency_metric, duration_ms)
        metrics.set_gauge(_metric_name("tool_last_duration_ms", tool_name), duration_ms)
        metrics.set_gauge(_metric_name("tool_last_duration_ms", "_all"), duration_ms)
        metrics.inc(inflight_metric, -1)
        metrics.inc(global_inflight_metric, -1)
        if error_type is None:
            metrics.inc(_metric_name("tool_success_total", tool_name))
            metrics.inc(_metric_name("tool_success_total", "_all"))
            logger.info(
                "tool_complete",
                extra={
                    "event": "tool_complete",
                    "tool_name": tool_name,
                    "duration_ms": round(duration_ms, 2),
                    "correlation_id": correlation_id,
                },
            )
        else:
            metrics.inc(_metric_name("tool_failure_total", tool_name))
            metrics.inc(_metric_name("tool_failure_total", "_all"))


def metrics_snapshot() -> dict[str, int]:
    """Return a copy of current counter values."""
    return metrics.snapshot()


def metrics_latency_snapshot() -> dict[str, dict[str, Any]]:
    """Return current latency histogram snapshots."""
    return metrics.latency_snapshot()


def record_tool_result_size(tool_name: str, size_bytes: int) -> None:
    """Record aggregate byte size metrics for tool results."""
    metrics.inc(_metric_name("tool_result_bytes_total", tool_name), size_bytes)
    metrics.inc(_metric_name("tool_result_bytes_total", "_all"), size_bytes)
    metrics.set_gauge(
        _metric_name("tool_last_result_bytes", tool_name), float(size_bytes)
    )
    metrics.set_gauge(_metric_name("tool_last_result_bytes", "_all"), float(size_bytes))
