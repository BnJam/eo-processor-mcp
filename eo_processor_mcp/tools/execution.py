"""Tool execution engine.

Central dispatch for all tool handlers. Each handler is a sync function
that receives (arguments: dict) and returns a result (str or dict).
Execution is offloaded to a thread to avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, NoReturn

from mcp.types import TextContent

from eo_processor_mcp.observability import (
    instrument_tool_execution,
    record_tool_result_size,
)
from eo_processor_mcp.tools.analyze_trends import handle_analyze_trends
from eo_processor_mcp.tools.apply_mask import handle_apply_mask
from eo_processor_mcp.tools.bfast_monitor import handle_bfast_monitor
from eo_processor_mcp.tools.classify import handle_classify
from eo_processor_mcp.tools.compute_change_index import handle_compute_change_index
from eo_processor_mcp.tools.compute_distances import handle_compute_distances
from eo_processor_mcp.tools.compute_spectral_index import handle_compute_spectral_index
from eo_processor_mcp.tools.list_capabilities import handle_list_capabilities
from eo_processor_mcp.tools.morphological_op import handle_morphological_operation
from eo_processor_mcp.tools.moving_average import handle_moving_average
from eo_processor_mcp.tools.pixelwise_transform import handle_pixelwise_transform
from eo_processor_mcp.tools.temporal_composite import handle_temporal_composite
from eo_processor_mcp.tools.temporal_stats import handle_temporal_statistics
from eo_processor_mcp.tools.texture_features import handle_texture_features
from eo_processor_mcp.tools.zonal_statistics import handle_zonal_statistics

logger = logging.getLogger(__name__)

Handler = Any

_TOOL_HANDLERS: dict[str, Handler] = {
    "compute_spectral_index": handle_compute_spectral_index,
    "compute_change_index": handle_compute_change_index,
    "temporal_statistics": handle_temporal_statistics,
    "temporal_composite": handle_temporal_composite,
    "moving_average": handle_moving_average,
    "apply_mask": handle_apply_mask,
    "morphological_operation": handle_morphological_operation,
    "compute_distances": handle_compute_distances,
    "analyze_trends": handle_analyze_trends,
    "bfast_monitor": handle_bfast_monitor,
    "classify": handle_classify,
    "texture_features": handle_texture_features,
    "zonal_statistics": handle_zonal_statistics,
    "pixelwise_transform": handle_pixelwise_transform,
    "list_capabilities": handle_list_capabilities,
}


def _raise_unknown_tool(name: str) -> NoReturn:
    tools = list(_TOOL_HANDLERS.keys())
    msg = f"Unknown tool: {name}. Available tools: {tools}"
    raise ValueError(msg)


def _as_text_content_list(result: Any) -> list[TextContent]:
    def _single(value: Any) -> TextContent:
        if isinstance(value, TextContent):
            return value
        if isinstance(value, str):
            return TextContent(type="text", text=value)
        try:
            serialized = json.dumps(value, separators=(",", ":"))
        except TypeError:
            serialized = str(value)
        return TextContent(type="text", text=serialized)

    if result is None:
        return []
    if isinstance(result, TextContent):
        return [result]
    if isinstance(result, list):
        return [_single(item) for item in result]
    return [_single(result)]


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    handler: Handler | None = None,
) -> list[TextContent]:
    """Execute a tool handler with observability instrumentation."""
    arguments = dict(arguments or {})

    if handler is None:
        handler = _TOOL_HANDLERS.get(tool_name)
        if handler is None:
            _raise_unknown_tool(tool_name)

    instrumented = await asyncio.to_thread(
        instrument_tool_execution,
        tool_name,
        handler,
        arguments,
    )
    raw_result = instrumented.value

    output_format = arguments.get("output_format", "text")
    if output_format == "json":
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
                payload = {"mode": "json", "data": parsed}
            except json.JSONDecodeError:
                payload = {"mode": "text_fallback", "content": raw_result}
        else:
            payload = {"mode": "json", "data": raw_result}
        payload_text = json.dumps(payload, separators=(",", ":"))
        record_tool_result_size(tool_name, len(payload_text.encode("utf-8")))
        return [TextContent(type="text", text=payload_text)]

    normalized = _as_text_content_list(raw_result)
    total_bytes = sum(len(item.text.encode("utf-8")) for item in normalized)
    record_tool_result_size(tool_name, total_bytes)
    return normalized
