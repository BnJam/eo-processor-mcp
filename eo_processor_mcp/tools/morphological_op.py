"""Morphological operations tool handler."""

from __future__ import annotations

from typing import Any

import numpy as np

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_morphological_operation(args: dict[str, Any]) -> str:
    """Apply morphological operations (dilation, erosion, opening, closing)."""
    import eo_processor as eop

    operation = args["operation"].lower()
    arr = load_array(args["input"])
    kernel_size = int(args.get("kernel_size", 3))
    output_path = args.get("output_path")

    arr_uint8 = arr.astype(np.uint8)

    func_map = {
        "dilation": eop.binary_dilation,
        "erosion": eop.binary_erosion,
        "opening": eop.binary_opening,
        "closing": eop.binary_closing,
    }

    if operation not in func_map:
        msg = f"Unknown operation: {operation}. Available: {list(func_map)}"
        raise ValueError(msg)

    result = func_map[operation](arr_uint8, kernel_size=kernel_size)

    out = save_array(result, output_path)
    return format_result(
        out, result, {"operation": operation, "kernel_size": kernel_size}
    )
