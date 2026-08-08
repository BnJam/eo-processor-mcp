"""Trend analysis tool handler."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from eo_processor_mcp.utils import load_array


def handle_analyze_trends(args: dict[str, Any]) -> str:
    """Perform trend analysis and/or linear regression on a 1D array."""
    import eo_processor as eop

    method = args.get("method", "linear_regression").lower()
    arr = load_array(args["input"])

    if method == "linear_regression":
        result = eop.linear_regression(arr)
        slope, intercept, residuals = result
        return json.dumps({
            "method": "linear_regression",
            "slope": float(slope),
            "intercept": float(intercept),
            "residuals_shape": list(np.array(residuals).shape),
        }, separators=(",", ":"))

    if method == "trend_analysis":
        threshold = float(args.get("threshold", 1.0))
        from eo_processor._core import trend_analysis as _trend_analysis
        segments = _trend_analysis(arr, threshold=threshold)
        seg_list = []
        for seg in segments:
            seg_list.append({
                "start_index": int(seg.start_index),
                "end_index": int(seg.end_index),
                "slope": float(seg.slope),
                "intercept": float(seg.intercept),
            })
        return json.dumps({
            "method": "trend_analysis",
            "threshold": threshold,
            "segments": seg_list,
        }, separators=(",", ":"))

    msg = f"Unknown method: {method}. Available: [linear_regression, trend_analysis]"
    raise ValueError(msg)
