"""List capabilities tool handler."""

from __future__ import annotations

import json
from typing import Any

from eo_processor_mcp.tools import (
    ALL_TOOLS,
    CHANGE_INDICES,
    DISTANCE_METRICS,
    MASK_METHODS,
    MORPHOLOGY_OPS,
    SPECTRAL_INDICES_2BAND,
    SPECTRAL_INDICES_3BAND,
    TEMPORAL_METHODS,
)


def handle_list_capabilities(_args: dict[str, Any]) -> str:
    """List all available indices, operations, and tools."""
    return json.dumps({
        "tools": ALL_TOOLS,
        "spectral_indices_2band": SPECTRAL_INDICES_2BAND,
        "spectral_indices_3band": SPECTRAL_INDICES_3BAND,
        "change_indices": CHANGE_INDICES,
        "temporal_methods": TEMPORAL_METHODS,
        "mask_methods": MASK_METHODS,
        "morphology_operations": MORPHOLOGY_OPS,
        "distance_metrics": DISTANCE_METRICS,
    }, indent=2)
