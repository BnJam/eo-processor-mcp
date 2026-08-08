"""Zonal statistics tool handler."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from eo_processor_mcp.utils import load_array


def handle_zonal_statistics(args: dict[str, Any]) -> str:
    """Compute zonal statistics for values grouped by zone labels."""
    import eo_processor as eop

    values = load_array(args["values"])
    zones = load_array(args["zones"])

    if zones.dtype != np.int64:
        zones = zones.astype(np.int64)

    result = eop.zonal_stats(values, zones)

    zone_dict = {}
    for zone_id, stats in result.items():
        zone_dict[str(int(zone_id))] = {
            "count": int(stats.count),
            "sum": float(stats.sum),
            "mean": float(stats.mean),
            "min": float(stats.min),
            "max": float(stats.max),
            "std": float(stats.std),
        }

    return json.dumps({
        "method": "zonal_statistics",
        "zones": zone_dict,
    }, separators=(",", ":"))
