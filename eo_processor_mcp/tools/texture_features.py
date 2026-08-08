"""Texture features (Haralick/GLCM) tool handler."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from eo_processor_mcp.utils import format_result, load_array, save_array

_DEFAULT_FEATURES = ["contrast", "dissimilarity", "homogeneity", "entropy"]


def handle_texture_features(args: dict[str, Any]) -> str:
    """Compute Haralick/GLCM texture features."""
    import eo_processor as eop

    data = load_array(args["input"])
    if data.ndim != 2:
        msg = f"texture_features expects a 2D array, got {data.ndim}D"
        raise ValueError(msg)

    window_size = int(args.get("window_size", 3))
    levels = int(args.get("levels", 8))
    features = args.get("features") or list(_DEFAULT_FEATURES)
    output_path = args.get("output_path")

    da = xr.DataArray(np.asarray(data, dtype=np.float64), dims=("y", "x"))
    result = eop.haralick_features(
        da,
        window_size=window_size,
        levels=levels,
        features=list(features),
    )

    arr = np.asarray(result.values)
    out = save_array(arr, output_path)
    return format_result(
        out,
        arr,
        {
            "method": "haralick_features",
            "window_size": window_size,
            "levels": levels,
            "features": list(features),
        },
    )
