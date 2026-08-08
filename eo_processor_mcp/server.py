from __future__ import annotations

import logging
from typing import Any

from fastmcp.server.server import FastMCP

from eo_processor_mcp.prompts import register_prompts
from eo_processor_mcp.tools import execution

app = FastMCP()

_LOGGER = logging.getLogger(__name__)

register_prompts(app)


@app.tool
async def compute_spectral_index(
    index: str,
    output_path: str | None = None,
    nir: str | None = None,
    red: str | None = None,
    green: str | None = None,
    blue: str | None = None,
    swir1: str | None = None,
    swir2: str | None = None,
    rededge: str | None = None,
    a: str | None = None,
    b: str | None = None,
    savi_l: float | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Compute a spectral index from band .npy files.

    Args:
        index: Index name (ndvi, ndwi, ndsi, evi, evi2, savi, osavi, msavi,
               gndvi, ndre, ndvi_re2, lai, nbr, ndmi, nbr2, gci, ci_re, mtci,
               normalized_difference).
        output_path: Optional path to save result .npy. Auto-generated if omitted.
        nir: Path to NIR band .npy file.
        red: Path to Red band .npy file.
        green: Path to Green band .npy file.
        blue: Path to Blue band .npy file.
        swir1: Path to SWIR1 band .npy file.
        swir2: Path to SWIR2 band .npy file.
        rededge: Path to Red Edge band .npy file.
        a: Path to first input for normalized_difference.
        b: Path to second input for normalized_difference.
        savi_l: Soil brightness factor for SAVI (default 0.5).
        output_format: Output format ("text" or "json").
    """
    arguments = {
        "index": index,
        "output_path": output_path,
        "output_format": output_format,
    }
    for band_name, band_val in [
        ("nir", nir),
        ("red", red),
        ("green", green),
        ("blue", blue),
        ("swir1", swir1),
        ("swir2", swir2),
        ("rededge", rededge),
        ("a", a),
        ("b", b),
    ]:
        if band_val is not None:
            arguments[band_name] = band_val
    if savi_l is not None:
        arguments["savi_l"] = savi_l
    return await execution.execute_tool("compute_spectral_index", arguments=arguments)


@app.tool
async def compute_change_index(
    index: str,
    pre_nir: str,
    post_nir: str,
    pre_red: str | None = None,
    pre_swir2: str | None = None,
    post_red: str | None = None,
    post_swir2: str | None = None,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Compute a change detection index from pre/post band .npy files.

    Args:
        index: Change index name (delta_ndvi, delta_nbr, dnbr, rbr).
        pre_nir: Path to pre-event NIR band .npy.
        post_nir: Path to post-event NIR band .npy.
        pre_red: Path to pre-event Red band .npy (for delta_ndvi).
        pre_swir2: Path to pre-event SWIR2 band .npy (for delta_nbr, dnbr, rbr).
        post_red: Path to post-event Red band .npy (for delta_ndvi).
        post_swir2: Path to post-event SWIR2 band .npy (for delta_nbr, dnbr, rbr).
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments: dict[str, Any] = {
        "index": index,
        "pre_nir": pre_nir,
        "post_nir": post_nir,
        "output_path": output_path,
        "output_format": output_format,
    }
    for band_name, band_val in [
        ("pre_red", pre_red),
        ("pre_swir2", pre_swir2),
        ("post_red", post_red),
        ("post_swir2", post_swir2),
    ]:
        if band_val is not None:
            arguments[band_name] = band_val
    return await execution.execute_tool("compute_change_index", arguments=arguments)


@app.tool
async def temporal_statistics(
    input: str,
    method: str,
    skip_na: bool | None = True,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Compute temporal statistics along the time axis of a 1D-4D array.

    Args:
        input: Path to input .npy file (time-first, 1D-4D).
        method: Statistic to compute (median, mean, std, sum).
        skip_na: If True, NaNs are excluded (default True).
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments = {
        "input": input,
        "method": method,
        "skip_na": skip_na,
        "output_path": output_path,
        "output_format": output_format,
    }
    return await execution.execute_tool("temporal_statistics", arguments=arguments)


@app.tool
async def temporal_composite(
    input: str,
    weights: str,
    skip_na: bool | None = True,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Compute a weighted temporal composite of a 4D array.

    Args:
        input: Path to 4D input .npy (time, bands, y, x).
        weights: Path to 1D weights .npy (same length as time dim).
        skip_na: If True, NaNs are excluded (default True).
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments = {
        "input": input,
        "weights": weights,
        "skip_na": skip_na,
        "output_path": output_path,
        "output_format": output_format,
    }
    return await execution.execute_tool("temporal_composite", arguments=arguments)


@app.tool
async def moving_average(
    input: str,
    window: int,
    stride: int | None = None,
    skip_na: bool | None = True,
    mode: str | None = "valid",
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Compute a moving average along the time axis.

    Args:
        input: Path to input .npy file (time-first, 1D-4D).
        window: Window size for the moving average.
        stride: Optional stride for downsampling (uses strided variant).
        skip_na: If True, NaNs are excluded (default True).
        mode: Convolution mode ("valid" or "same", default "valid").
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments = {
        "input": input,
        "window": window,
        "stride": stride,
        "skip_na": skip_na,
        "mode": mode,
        "output_path": output_path,
        "output_format": output_format,
    }
    return await execution.execute_tool("moving_average", arguments=arguments)


@app.tool
async def apply_mask(
    input: str,
    method: str,
    values: list[float] | None = None,
    fill_value: float | None = None,
    nan_to: float | None = None,
    value: float | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    invalid_values: list[float] | None = None,
    scl: str | None = None,
    keep_codes: list[int] | None = None,
    mask_codes: list[int] | None = None,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Apply masking operations to an array.

    Args:
        input: Path to input .npy file.
        method: Masking method (vals, replace_nans, out_range, in_range,
                invalid, scl, with_scl).
        values: Values to mask (for method=vals).
        fill_value: Fill value for masked pixels (default NaN).
        nan_to: Convert NaNs to this value before masking (for method=vals).
        value: Replacement value (for method=replace_nans).
        min_val: Minimum of range to mask (for out_range/in_range).
        max_val: Maximum of range to mask (for out_range/in_range).
        invalid_values: List of invalid sentinel values (for method=invalid).
        scl: Path to SCL band .npy (for method=scl or with_scl).
        keep_codes: SCL codes to keep (for method=scl, default [4,5,6]).
        mask_codes: SCL codes to mask out (for method=with_scl).
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments: dict[str, Any] = {
        "input": input,
        "method": method,
        "output_path": output_path,
        "output_format": output_format,
    }
    for k, v in [
        ("values", values),
        ("fill_value", fill_value),
        ("nan_to", nan_to),
        ("value", value),
        ("min_val", min_val),
        ("max_val", max_val),
        ("invalid_values", invalid_values),
        ("scl", scl),
        ("keep_codes", keep_codes),
        ("mask_codes", mask_codes),
    ]:
        if v is not None:
            arguments[k] = v
    return await execution.execute_tool("apply_mask", arguments=arguments)


@app.tool
async def morphological_operation(
    input: str,
    operation: str,
    kernel_size: int | None = 3,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Apply morphological operations to a 2D binary array.

    Args:
        input: Path to input .npy file (2D, will be cast to uint8).
        operation: Operation (dilation, erosion, opening, closing).
        kernel_size: Kernel size (default 3).
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments = {
        "input": input,
        "operation": operation,
        "kernel_size": kernel_size,
        "output_path": output_path,
        "output_format": output_format,
    }
    return await execution.execute_tool("morphological_operation", arguments=arguments)


@app.tool
async def compute_distances(
    points_a: str,
    points_b: str,
    metric: str,
    p: float | None = None,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Compute pairwise distances between two point sets.

    Args:
        points_a: Path to first point set .npy (N, D).
        points_b: Path to second point set .npy (N, D).
        metric: Distance metric (euclidean, manhattan, chebyshev, minkowski).
        p: Minkowski p parameter (only for metric=minkowski).
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments: dict[str, Any] = {
        "points_a": points_a,
        "points_b": points_b,
        "metric": metric,
        "output_path": output_path,
        "output_format": output_format,
    }
    if p is not None:
        arguments["p"] = p
    return await execution.execute_tool("compute_distances", arguments=arguments)


@app.tool
async def analyze_trends(
    input: str,
    method: str | None = "linear_regression",
    threshold: float | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Perform trend analysis or linear regression on a 1D array.

    Args:
        input: Path to input .npy file (1D time series).
        method: Analysis method (linear_regression, trend_analysis).
        threshold: Break detection threshold (for trend_analysis).
        output_format: Output format ("text" or "json").
    """
    arguments: dict[str, Any] = {
        "input": input,
        "method": method,
        "output_format": output_format,
    }
    if threshold is not None:
        arguments["threshold"] = threshold
    return await execution.execute_tool("analyze_trends", arguments=arguments)


@app.tool
async def bfast_monitor(
    stack: str,
    dates: list[int],
    history_start_date: int,
    monitor_start_date: int,
    order: int | None = 3,
    h: float | None = 0.25,
    alpha: float | None = 0.05,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Run BFAST Monitor change detection on a time series stack.

    Args:
        stack: Path to 3D time series stack .npy (time, y, x).
        dates: Sequence of integer dates (e.g. YYYYMMDD or ordinal).
        history_start_date: Start of the stable history period.
        monitor_start_date: Start of the monitoring period.
        order: Harmonic model order (default 3).
        h: Bandwidth parameter (default 0.25).
        alpha: Significance level (default 0.05).
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments = {
        "stack": stack,
        "dates": dates,
        "history_start_date": history_start_date,
        "monitor_start_date": monitor_start_date,
        "order": order,
        "h": h,
        "alpha": alpha,
        "output_path": output_path,
        "output_format": output_format,
    }
    return await execution.execute_tool("bfast_monitor", arguments=arguments)


@app.tool
async def classify(
    method: str,
    features: str,
    labels: str | None = None,
    model: str | None = None,
    n_estimators: int | None = 100,
    min_samples_split: int | None = 2,
    max_depth: int | None = None,
    max_features: int | None = None,
    blue: str | None = None,
    green: str | None = None,
    red: str | None = None,
    nir: str | None = None,
    swir1: str | None = None,
    swir2: str | None = None,
    temp: str | None = None,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Train/predict Random Forest or run complex multi-band classification.

    Args:
        method: Classification method (train, predict, complex).
        features: Path to features .npy (for train/predict).
        labels: Path to labels .npy (for train).
        model: JSON model string (for predict).
        n_estimators: Number of estimators (for train, default 100).
        min_samples_split: Min samples to split a node (for train, default 2).
        max_depth: Max tree depth (for train, default None).
        max_features: Max features per split (for train, default None).
        blue: Path to Blue band .npy (for complex).
        green: Path to Green band .npy (for complex).
        red: Path to Red band .npy (for complex).
        nir: Path to NIR band .npy (for complex).
        swir1: Path to SWIR1 band .npy (for complex).
        swir2: Path to SWIR2 band .npy (for complex).
        temp: Path to Thermal band .npy (for complex).
        output_path: Optional path to save result .npy (for predict/complex).
        output_format: Output format ("text" or "json").
    """
    arguments: dict[str, Any] = {
        "method": method,
        "features": features,
        "output_path": output_path,
        "output_format": output_format,
    }
    for k, v in [
        ("labels", labels),
        ("model", model),
        ("n_estimators", n_estimators),
        ("min_samples_split", min_samples_split),
        ("max_depth", max_depth),
        ("max_features", max_features),
        ("blue", blue),
        ("green", green),
        ("red", red),
        ("nir", nir),
        ("swir1", swir1),
        ("swir2", swir2),
        ("temp", temp),
    ]:
        if v is not None:
            arguments[k] = v
    return await execution.execute_tool("classify", arguments=arguments)


@app.tool
async def texture_features(
    input: str,
    window_size: int | None = 3,
    levels: int | None = 8,
    features: list[str] | None = None,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Compute Haralick/GLCM texture features.

    Args:
        input: Path to input .npy file (2D).
        window_size: GLCM window size (default 3).
        levels: Quantization levels (default 8).
        features: Feature subset (default all: contrast, dissimilarity,
                  homogeneity, entropy).
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments: dict[str, Any] = {
        "input": input,
        "window_size": window_size,
        "levels": levels,
        "output_path": output_path,
        "output_format": output_format,
    }
    if features is not None:
        arguments["features"] = features
    return await execution.execute_tool("texture_features", arguments=arguments)


@app.tool
async def zonal_statistics(
    values: str,
    zones: str,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Compute zonal statistics for values grouped by zone labels.

    Args:
        values: Path to values .npy file.
        zones: Path to zones .npy file (integer labels, same shape as values).
        output_format: Output format ("text" or "json").
    """
    arguments = {
        "values": values,
        "zones": zones,
        "output_format": output_format,
    }
    return await execution.execute_tool("zonal_statistics", arguments=arguments)


@app.tool
async def pixelwise_transform(
    input: str,
    scale: float | None = 1.0,
    offset: float | None = 0.0,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    output_path: str | None = None,
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """Apply a linear pixelwise transform: output = input * scale + offset.

    Args:
        input: Path to input .npy file.
        scale: Scale factor (default 1.0).
        offset: Offset value (default 0.0).
        clamp_min: Optional minimum clamp value.
        clamp_max: Optional maximum clamp value.
        output_path: Optional path to save result .npy.
        output_format: Output format ("text" or "json").
    """
    arguments = {
        "input": input,
        "scale": scale,
        "offset": offset,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "output_path": output_path,
        "output_format": output_format,
    }
    return await execution.execute_tool("pixelwise_transform", arguments=arguments)


@app.tool
async def list_capabilities(
    output_format: str | None = "text",
) -> list[dict[str, Any]]:
    """List all available spectral indices, operations, and tools."""
    arguments = {"output_format": output_format}
    return await execution.execute_tool("list_capabilities", arguments=arguments)
