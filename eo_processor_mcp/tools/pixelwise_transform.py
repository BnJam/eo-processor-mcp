"""Pixelwise transform tool handler."""

from __future__ import annotations

from typing import Any

import numpy as np

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_pixelwise_transform(args: dict[str, Any]) -> str:
    """Apply a linear pixelwise transform with optional clamping."""
    import eo_processor as eop

    arr = load_array(args["input"])
    scale = float(args.get("scale", 1.0))
    offset = float(args.get("offset", 0.0))
    clamp_min = args.get("clamp_min")
    clamp_max = args.get("clamp_max")
    output_path = args.get("output_path")

    if clamp_min is not None:
        clamp_min = float(clamp_min)
    if clamp_max is not None:
        clamp_max = float(clamp_max)

    result = eop.pixelwise_transform(
        arr,
        scale=scale,
        offset=offset,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

    if np.ndim(result) == 0:
        result = np.array(result)

    out = save_array(result, output_path)
    return format_result(out, result, {
        "method": "pixelwise_transform",
        "scale": scale,
        "offset": offset,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
    })
