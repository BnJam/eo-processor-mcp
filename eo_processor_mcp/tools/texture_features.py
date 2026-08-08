"""Texture features (Haralick/GLCM) tool handler."""

from __future__ import annotations

from typing import Any

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_texture_features(args: dict[str, Any]) -> str:
    """Compute Haralick/GLCM texture features."""
    import eo_processor as eop

    data = load_array(args["input"])
    window_size = int(args.get("window_size", 3))
    levels = int(args.get("levels", 16))
    output_path = args.get("output_path")

    result = eop.haralick_features(data, window_size=window_size, levels=levels)

    out = save_array(result, output_path)
    return format_result(out, result, {
        "method": "haralick_features",
        "window_size": window_size,
        "levels": levels,
    })
