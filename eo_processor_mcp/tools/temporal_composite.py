"""Temporal composite tool handler."""

from __future__ import annotations

from typing import Any

import numpy as np

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_temporal_composite(args: dict[str, Any]) -> str:
    """Compute a weighted temporal composite of a 4D array."""
    import eo_processor as eop

    arr = load_array(args["input"])
    weights = load_array(args["weights"])
    skip_na = args.get("skip_na", True)
    output_path = args.get("output_path")

    result = eop.temporal_composite(arr, weights, skip_na=skip_na)
    if np.ndim(result) == 0:
        result = np.array(result)

    out = save_array(result, output_path)
    return format_result(out, result, {"method": "temporal_composite"})
