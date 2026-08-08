"""Temporal statistics tool handler."""

from __future__ import annotations

from typing import Any

import numpy as np

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_temporal_statistics(args: dict[str, Any]) -> str:
    """Compute temporal statistics (median, mean, std, sum) along the time axis."""
    import eo_processor as eop

    method = args["method"].lower()
    arr = load_array(args["input"])
    skip_na = args.get("skip_na", True)
    output_path = args.get("output_path")

    func_map = {
        "median": lambda a: eop.median(a, skip_na=skip_na),
        "mean": lambda a: eop.temporal_mean(a, skip_na=skip_na),
        "std": lambda a: eop.temporal_std(a, skip_na=skip_na),
        "sum": lambda a: eop.temporal_sum(a, skip_na=skip_na),
    }

    if method not in func_map:
        msg = f"Unknown method: {method}. Available: {list(func_map)}"
        raise ValueError(msg)

    result = func_map[method](arr)
    if np.ndim(result) == 0:
        result = np.array(result)

    out = save_array(result, output_path)
    return format_result(out, result, {"method": method})
