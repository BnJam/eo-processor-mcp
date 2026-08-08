"""Unified masking tool handler."""

from __future__ import annotations

from typing import Any

import numpy as np

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_apply_mask(args: dict[str, Any]) -> str:
    """Apply masking operations to an array."""
    import eo_processor as eop

    method = args["method"].lower()
    arr = load_array(args["input"])
    output_path = args.get("output_path")

    if method == "vals":
        values = args.get("values", [])
        fill_value = args.get("fill_value", np.nan)
        nan_to = args.get("nan_to")
        result = eop.mask_vals(arr, values=values, fill_value=fill_value, nan_to=nan_to)

    elif method == "replace_nans":
        value = float(args.get("value", 0.0))
        result = eop.replace_nans(arr, value=value)

    elif method == "out_range":
        min_val = args.get("min_val")
        max_val = args.get("max_val")
        fill_value = args.get("fill_value", np.nan)
        result = eop.mask_out_range(
            arr, min_val=min_val, max_val=max_val, fill_value=fill_value
        )

    elif method == "in_range":
        min_val = args.get("min_val")
        max_val = args.get("max_val")
        fill_value = args.get("fill_value", np.nan)
        result = eop.mask_in_range(
            arr, min_val=min_val, max_val=max_val, fill_value=fill_value
        )

    elif method == "invalid":
        invalid_values = args.get("invalid_values", [])
        fill_value = args.get("fill_value", np.nan)
        result = eop.mask_invalid(
            arr, invalid_values=invalid_values, fill_value=fill_value
        )

    elif method == "scl":
        scl_path = args.get("scl", args.get("input"))
        if scl_path is None:
            msg = "method='scl' requires either 'scl' or 'input' parameter"
            raise ValueError(msg)
        scl = load_array(scl_path)
        keep_codes = args.get("keep_codes", [4, 5, 6])
        fill_value = args.get("fill_value", np.nan)
        original_shape = scl.shape
        scl_flat = scl.flatten()
        result_flat = eop.mask_scl(
            scl_flat, keep_codes=keep_codes, fill_value=fill_value,
        )
        result = result_flat.reshape(original_shape)

    elif method == "with_scl":
        scl = load_array(args["scl"])
        mask_codes = args.get("mask_codes", [0, 1, 3, 8, 9, 10, 11])
        fill_value = args.get("fill_value", np.nan)
        result = eop.mask_with_scl(
            arr, scl, mask_codes=mask_codes, fill_value=fill_value
        )

    else:
        msg = (
            f"Unknown mask method: {method}. "
            "Available: [vals, replace_nans, out_range,"
            " in_range, invalid, scl, with_scl]"
        )
        raise ValueError(msg)

    out = save_array(result, output_path)
    return format_result(out, result, {"method": method})
