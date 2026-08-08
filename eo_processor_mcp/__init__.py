"""EO Processor MCP Server - An MCP Server for Earth Observation processing."""

__version__ = "0.1.0"

from .observability import metrics_latency_snapshot, metrics_snapshot

__all__ = ["__version__", "metrics_latency_snapshot", "metrics_snapshot"]
