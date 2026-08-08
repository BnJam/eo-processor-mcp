"""Moving average tool handler."""

from __future__ import annotations

from typing import Any

import numpy as np

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_moving_average(args: dict[str, Any]) -> str:
    """Compute a moving average along the time axis."""
    import eo_processor as eop

    arr = load_array(args["input"])
    window = int(args["window"])
    skip_na = args.get("skip_na", True)
    mode = args.get("mode", "valid")
    output_path = args.get("output_path")
    stride = args.get("stride")

    if stride is not None:
        result = eop.moving_average_temporal_stride(
            arr, window=window, stride=int(stride), skip_na=skip_na, mode=mode,
        )
    else:
        result = eop.moving_average_temporal(
            arr, window=window, skip_na=skip_na, mode=mode,
        )

    if np.ndim(result) == 0:
        result = np.array(result)

    out = save_array(result, output_path)
    extra = {"window": window, "mode": mode}
    if stride is not None:
        extra["stride"] = int(stride)
    return format_result(out, result, extra)
