"""BFAST Monitor tool handler."""

from __future__ import annotations

from typing import Any

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_bfast_monitor(args: dict[str, Any]) -> str:
    """Run BFAST Monitor change detection on a time series stack."""
    import eo_processor as eop

    stack = load_array(args["stack"])
    dates = args["dates"]
    history_start_date = int(args["history_start_date"])
    monitor_start_date = int(args["monitor_start_date"])
    order = int(args.get("order", 3))
    h = float(args.get("h", 0.25))
    alpha = float(args.get("alpha", 0.05))
    output_path = args.get("output_path")

    result = eop.bfast_monitor(
        stack,
        dates=dates,
        history_start_date=history_start_date,
        monitor_start_date=monitor_start_date,
        order=order,
        h=h,
        alpha=alpha,
    )

    out = save_array(result, output_path)
    return format_result(out, result, {
        "method": "bfast_monitor",
        "order": order,
        "h": h,
        "alpha": alpha,
    })
