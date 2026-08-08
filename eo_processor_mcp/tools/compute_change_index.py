"""Compute change/delta index tool handler."""

from __future__ import annotations

from typing import Any

from eo_processor_mcp.utils import format_result, load_array, save_array

_CHANGE_INDICES = {
    "delta_ndvi": ("pre_nir", "pre_red", "post_nir", "post_red"),
    "delta_nbr": ("pre_nir", "pre_swir2", "post_nir", "post_swir2"),
    "dnbr": ("pre_nir", "pre_swir2", "post_nir", "post_swir2"),
    "rbr": ("pre_nir", "pre_swir2", "post_nir", "post_swir2"),
}


def handle_compute_change_index(args: dict[str, Any]) -> str:
    """Compute a change detection index from pre/post band .npy files."""
    import eo_processor as eop

    index_name = args["index"].lower()
    output_path = args.get("output_path")

    if index_name not in _CHANGE_INDICES:
        msg = f"Unknown change index: {index_name}. Available: {list(_CHANGE_INDICES)}"
        raise ValueError(msg)

    band_names = _CHANGE_INDICES[index_name]
    bands = {}
    for b in band_names:
        if b not in args:
            msg = f"Missing required band '{b}' for index '{index_name}'"
            raise ValueError(msg)
        bands[b] = load_array(args[b])

    func = getattr(eop, index_name)
    result = func(*[bands[b] for b in band_names])

    out = save_array(result, output_path)
    return format_result(out, result, {"index": index_name})
