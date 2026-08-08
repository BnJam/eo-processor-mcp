"""Pairwise distance computation tool handler."""

from __future__ import annotations

from typing import Any

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_compute_distances(args: dict[str, Any]) -> str:
    """Compute pairwise distances between point sets."""
    import eo_processor as eop

    metric = args["metric"].lower()
    points_a = load_array(args["points_a"])
    points_b = load_array(args["points_b"])
    output_path = args.get("output_path")

    if metric == "minkowski":
        p = float(args.get("p", 2.0))
        result = eop.minkowski_distance(points_a, points_b, p)
    else:
        func_map = {
            "euclidean": eop.euclidean_distance,
            "manhattan": eop.manhattan_distance,
            "chebyshev": eop.chebyshev_distance,
        }
        if metric not in func_map:
            msg = f"Unknown metric: {metric}. Available: {list(func_map)}"
            raise ValueError(msg)
        result = func_map[metric](points_a, points_b)

    out = save_array(result, output_path)
    extra = {"metric": metric}
    if metric == "minkowski":
        extra["p"] = float(args.get("p", 2.0))
    return format_result(out, result, extra)
