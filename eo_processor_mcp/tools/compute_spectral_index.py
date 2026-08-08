"""Compute spectral index tool handler."""

from __future__ import annotations

from typing import Any

from eo_processor_mcp.utils import format_result, load_array, save_array

_TWO_BAND = {
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),
    "ndsi": ("green", "swir1"),
    "evi2": ("nir", "red"),
    "savi": ("nir", "red"),
    "osavi": ("nir", "red"),
    "msavi": ("nir", "red"),
    "gndvi": ("nir", "green"),
    "ndre": ("nir", "rededge"),
    "nbr": ("nir", "swir2"),
    "ndmi": ("nir", "swir1"),
    "nbr2": ("swir1", "swir2"),
    "gci": ("nir", "green"),
    "ci_re": ("nir", "rededge"),
}

_THREE_BAND = {
    "evi": ("nir", "red", "blue"),
    "lai": ("nir", "red", "blue"),
    "ndvi_re2": ("nir", "rededge", "red"),
    "mtci": ("rededge", "red", "green"),
}


def handle_compute_spectral_index(args: dict[str, Any]) -> str:
    """Compute a spectral index from band .npy files."""
    import eo_processor as eop

    index_name = args["index"].lower()
    output_path = args.get("output_path")

    if index_name in _TWO_BAND:
        band_names = _TWO_BAND[index_name]
        bands = {}
        for b in band_names:
            if b not in args:
                msg = f"Missing required band '{b}' for index '{index_name}'"
                raise ValueError(msg)
            bands[b] = load_array(args[b])

        kwargs: dict[str, Any] = {}
        if index_name == "savi" and "savi_l" in args:
            kwargs["L"] = float(args["savi_l"])

        func = getattr(eop, index_name)
        result = func(*[bands[b] for b in band_names], **kwargs)

    elif index_name in _THREE_BAND:
        band_names = _THREE_BAND[index_name]
        bands = {}
        for b in band_names:
            if b not in args:
                msg = f"Missing required band '{b}' for index '{index_name}'"
                raise ValueError(msg)
            bands[b] = load_array(args[b])

        func_name = "enhanced_vegetation_index" if index_name == "evi" else index_name
        func = getattr(eop, func_name)
        result = func(*[bands[b] for b in band_names])

    elif index_name == "normalized_difference":
        if "a" not in args or "b" not in args:
            raise ValueError("normalized_difference requires 'a' and 'b' band paths")
        a = load_array(args["a"])
        b = load_array(args["b"])
        result = eop.normalized_difference(a, b)

    else:
        available = list(_TWO_BAND) + list(_THREE_BAND) + ["normalized_difference"]
        msg = f"Unknown spectral index: {index_name}. Available: {available}"
        raise ValueError(msg)

    out = save_array(result, output_path)
    return format_result(out, result, {"index": index_name})
