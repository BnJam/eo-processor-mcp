"""Real-world scenario tests to surface usability issues.

These simulate what an AI agent would actually do when calling the MCP tools
in realistic EO workflows, exercising edge cases and multi-step pipelines.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import numpy as np

from eo_processor_mcp.tools.execution import execute_tool


def _make_bands(tmp: str, shape: tuple[int, ...] = (64, 64), seed: int = 42):
    """Create realistic Sentinel-2-like band arrays (surface reflectance 0-1)."""
    rng = np.random.default_rng(seed)
    bands = {}
    for name in ["blue", "green", "red", "nir", "rededge", "swir1", "swir2"]:
        arr = rng.random(shape).astype(np.float64) * 0.4 + 0.05
        path = os.path.join(tmp, f"{name}.npy")
        np.save(path, arr)
        bands[name] = path
    return bands


def _make_scl(tmp: str, shape: tuple[int, ...] = (64, 64)):
    """Create a realistic Sentinel-2 SCL mask."""
    rng = np.random.default_rng(99)
    codes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    probs = np.array(
        [0.02, 0.01, 0.05, 0.03, 0.35, 0.25, 0.15, 0.02, 0.05, 0.03, 0.03, 0.01],
    )
    scl = rng.choice(codes, size=shape, p=probs).astype(np.float64)
    path = os.path.join(tmp, "scl.npy")
    np.save(path, scl)
    return path


def _make_time_stack(tmp: str, n_time: int = 12, shape: tuple[int, ...] = (32, 32)):
    """Create a time-series stack (time, y, x)."""
    rng = np.random.default_rng(42)
    stack = rng.random((n_time, *shape)).astype(np.float64) * 0.5 + 0.2
    path = os.path.join(tmp, "time_stack.npy")
    np.save(path, stack)
    return path


def _make_4d_stack(
    tmp: str, n_time: int = 6, n_bands: int = 3, shape: tuple[int, ...] = (16, 16),
):
    """Create a 4D stack (time, bands, y, x)."""
    rng = np.random.default_rng(42)
    stack = rng.random((n_time, n_bands, *shape)).astype(np.float64)
    path = os.path.join(tmp, "stack_4d.npy")
    np.save(path, stack)
    return path


async def scenario_1_basic_ndvi():
    """Scenario: Agent computes NDVI from NIR and Red bands."""
    print("\n=== Scenario 1: Basic NDVI computation ===")
    with tempfile.TemporaryDirectory() as tmp:
        bands = _make_bands(tmp)
        out = os.path.join(tmp, "ndvi.npy")

        result = await execute_tool("compute_spectral_index", arguments={
            "index": "ndvi",
            "nir": bands["nir"],
            "red": bands["red"],
            "output_path": out,
        })
        data = json.loads(result[0].text)
        print(f"  Result: {data['index']}, shape={data['stats']['shape']}")
        arr = np.load(out)
        print(f"  NDVI range: [{arr.min():.3f}, {arr.max():.3f}]")
        print(f"  NaN count: {np.sum(np.isnan(arr))}")
        assert arr.shape == (64, 64)
        print("  PASS")


async def scenario_2_multi_index():
    """Scenario: Agent computes multiple indices from the same bands."""
    print("\n=== Scenario 2: Multiple indices from same bands ===")
    with tempfile.TemporaryDirectory() as tmp:
        bands = _make_bands(tmp)

        indices = {
            "ndvi": {"nir": bands["nir"], "red": bands["red"]},
            "ndwi": {"green": bands["green"], "nir": bands["nir"]},
            "nbr": {"nir": bands["nir"], "swir2": bands["swir2"]},
            "evi": {"nir": bands["nir"], "red": bands["red"], "blue": bands["blue"]},
            "savi": {"nir": bands["nir"], "red": bands["red"], "savi_l": 0.5},
        }

        for idx_name, kwargs in indices.items():
            out = os.path.join(tmp, f"{idx_name}.npy")
            result = await execute_tool("compute_spectral_index", arguments={
                "index": idx_name,
                "output_path": out,
                **kwargs,
            })
            json.loads(result[0].text)
            arr = np.load(out)
            print(
                f"  {idx_name}: shape={arr.shape}, "
                f"range=[{arr.min():.3f}, {arr.max():.3f}]",
            )

        print("  PASS")


async def scenario_3_cloud_mask_pipeline():
    """Scenario: Agent masks clouds with SCL, then computes NDVI."""
    print("\n=== Scenario 3: Cloud mask + NDVI pipeline ===")
    with tempfile.TemporaryDirectory() as tmp:
        bands = _make_bands(tmp)
        scl_path = _make_scl(tmp)

        np.load(bands["nir"])
        masked_nir_path = os.path.join(tmp, "nir_masked.npy")
        result = await execute_tool("apply_mask", arguments={
            "input": bands["nir"],
            "method": "with_scl",
            "scl": scl_path,
            "output_path": masked_nir_path,
        })
        data = json.loads(result[0].text)
        print(f"  Masked NIR: {data['stats']['nan_count']} NaN pixels")

        np.load(bands["red"])
        masked_red_path = os.path.join(tmp, "red_masked.npy")
        await execute_tool("apply_mask", arguments={
            "input": bands["red"],
            "method": "with_scl",
            "scl": scl_path,
            "output_path": masked_red_path,
        })

        ndvi_path = os.path.join(tmp, "ndvi_masked.npy")
        result = await execute_tool("compute_spectral_index", arguments={
            "index": "ndvi",
            "nir": masked_nir_path,
            "red": masked_red_path,
            "output_path": ndvi_path,
        })
        data = json.loads(result[0].text)
        arr = np.load(ndvi_path)
        print(
            f"  Masked NDVI: {np.sum(np.isnan(arr))} NaN, "
            f"valid range=[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]",
        )
        print("  PASS")


async def scenario_4_temporal_composite():
    """Scenario: Agent computes median composite of a time series."""
    print("\n=== Scenario 4: Temporal median composite ===")
    with tempfile.TemporaryDirectory() as tmp:
        stack_path = _make_time_stack(tmp, n_time=12)

        result = await execute_tool("temporal_statistics", arguments={
            "input": stack_path,
            "method": "median",
            "output_path": os.path.join(tmp, "median.npy"),
        })
        data = json.loads(result[0].text)
        print(f"  Input: (12, 32, 32) -> Median: {data['stats']['shape']}")
        assert data["stats"]["shape"] == [32, 32]
        print("  PASS")


async def scenario_5_change_detection():
    """Scenario: Agent detects burn severity from pre/post fire imagery."""
    print("\n=== Scenario 5: Change detection (dNBR) ===")
    with tempfile.TemporaryDirectory() as tmp:
        pre = _make_bands(tmp, seed=10)
        post = _make_bands(tmp, seed=20)

        result = await execute_tool("compute_change_index", arguments={
            "index": "dnbr",
            "pre_nir": pre["nir"],
            "pre_swir2": pre["swir2"],
            "post_nir": post["nir"],
            "post_swir2": post["swir2"],
            "output_path": os.path.join(tmp, "dnbr.npy"),
        })
        json.loads(result[0].text)
        arr = np.load(os.path.join(tmp, "dnbr.npy"))
        print(f"  dNBR: shape={arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}]")
        print("  PASS")


async def scenario_6_morphological_cleanup():
    """Scenario: Agent cleans up a binary classification mask."""
    print("\n=== Scenario 6: Morphological cleanup of mask ===")
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(42)
        mask = (rng.random((32, 32)) > 0.7).astype(np.uint8)
        mask_path = os.path.join(tmp, "mask.npy")
        np.save(mask_path, mask)
        print(f"  Original mask: {np.sum(mask)} positive pixels")

        for op in ["opening", "closing"]:
            out = os.path.join(tmp, f"mask_{op}.npy")
            await execute_tool("morphological_operation", arguments={
                "input": mask_path,
                "operation": op,
                "kernel_size": 3,
                "output_path": out,
            })
            arr = np.load(out)
            print(f"  After {op}: {np.sum(arr > 0)} positive pixels")

        print("  PASS")


async def scenario_7_zonal_stats():
    """Scenario: Agent computes per-region statistics."""
    print("\n=== Scenario 7: Zonal statistics ===")
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(42)
        values = rng.random((32, 32)) * 100
        zones = np.zeros((32, 32))
        zones[:16, :16] = 1
        zones[:16, 16:] = 2
        zones[16:, :16] = 3
        zones[16:, 16:] = 1

        vp = os.path.join(tmp, "values.npy")
        zp = os.path.join(tmp, "zones.npy")
        np.save(vp, values)
        np.save(zp, zones)

        result = await execute_tool("zonal_statistics", arguments={
            "values": vp,
            "zones": zp,
        })
        data = json.loads(result[0].text)
        for zid, stats in data["zones"].items():
            print(f"  Zone {zid}: mean={stats['mean']:.1f}, count={stats['count']}")
        print("  PASS")


async def scenario_8_moving_average():
    """Scenario: Agent smooths a time series with moving average."""
    print("\n=== Scenario 8: Moving average smoothing ===")
    with tempfile.TemporaryDirectory() as tmp:
        stack_path = _make_time_stack(tmp, n_time=20)

        result = await execute_tool("moving_average", arguments={
            "input": stack_path,
            "window": 5,
            "mode": "same",
            "output_path": os.path.join(tmp, "smoothed.npy"),
        })
        data = json.loads(result[0].text)
        print(f"  Smoothed: shape={data['stats']['shape']}, window={data['window']}")
        print("  PASS")


async def scenario_9_pixelwise_scaling():
    """Scenario: Agent scales NDVI to 0-255 uint8 range."""
    print("\n=== Scenario 9: Pixelwise scaling ===")
    with tempfile.TemporaryDirectory() as tmp:
        bands = _make_bands(tmp)
        ndvi_path = os.path.join(tmp, "ndvi.npy")
        await execute_tool("compute_spectral_index", arguments={
            "index": "ndvi",
            "nir": bands["nir"],
            "red": bands["red"],
            "output_path": ndvi_path,
        })

        scaled_path = os.path.join(tmp, "ndvi_scaled.npy")
        result = await execute_tool("pixelwise_transform", arguments={
            "input": ndvi_path,
            "scale": 127.5,
            "offset": 127.5,
            "clamp_min": 0.0,
            "clamp_max": 255.0,
            "output_path": scaled_path,
        })
        json.loads(result[0].text)
        arr = np.load(scaled_path)
        print(f"  Scaled: range=[{arr.min():.1f}, {arr.max():.1f}]")
        assert arr.min() >= 0.0
        assert arr.max() <= 255.0
        print("  PASS")


async def scenario_10_error_handling():
    """Scenario: Agent hits various error conditions."""
    print("\n=== Scenario 10: Error handling ===")

    try:
        await execute_tool("compute_spectral_index", arguments={"index": "ndvi"})
        print("  FAIL: should have raised for missing bands")
    except (ValueError, FileNotFoundError) as e:
        print(f"  Missing band error: {type(e).__name__}: {e}")

    try:
        await execute_tool("compute_spectral_index", arguments={"index": "fake_index"})
        print("  FAIL: should have raised for unknown index")
    except ValueError as e:
        print(f"  Unknown index error: {type(e).__name__}: {e}")

    try:
        await execute_tool("nonexistent_tool", arguments={})
        print("  FAIL: should have raised for unknown tool")
    except ValueError as e:
        print(f"  Unknown tool error: {type(e).__name__}: {e}")

    try:
        await execute_tool("compute_change_index", arguments={
            "index": "delta_ndvi",
            "pre_nir": "/nonexistent.npy",
            "post_nir": "/nonexistent.npy",
        })
        print("  FAIL: should have raised for missing file")
    except FileNotFoundError as e:
        print(f"  Missing file error: {type(e).__name__}: {e}")

    print("  PASS")


async def scenario_11_shape_mismatch():
    """Scenario: Agent passes bands with different shapes."""
    print("\n=== Scenario 11: Shape mismatch detection ===")
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(42)
        nir = rng.random((64, 64))
        red = rng.random((32, 32))
        nir_path = os.path.join(tmp, "nir.npy")
        red_path = os.path.join(tmp, "red.npy")
        np.save(nir_path, nir)
        np.save(red_path, red)

        try:
            await execute_tool("compute_spectral_index", arguments={
                "index": "ndvi",
                "nir": nir_path,
                "red": red_path,
            })
            print("  Result: no error (Rust layer may handle or crash)")
        except Exception as e:
            print(f"  Shape mismatch error: {type(e).__name__}: {e}")
        print("  DONE")


async def scenario_12_list_capabilities():
    """Scenario: Agent discovers available tools."""
    print("\n=== Scenario 12: Capability discovery ===")
    result = await execute_tool("list_capabilities", arguments={
        "output_format": "json",
    })
    data = json.loads(result[0].text)
    caps = data["data"]
    print(f"  Tools: {len(caps['tools'])}")
    print(f"  2-band indices: {caps['spectral_indices_2band']}")
    print(f"  3-band indices: {caps['spectral_indices_3band']}")
    print(f"  Change indices: {caps['change_indices']}")
    print(f"  Temporal methods: {caps['temporal_methods']}")
    print(f"  Mask methods: {caps['mask_methods']}")
    print("  PASS")


async def scenario_13_json_output():
    """Scenario: Agent requests JSON output for programmatic parsing."""
    print("\n=== Scenario 13: JSON output format ===")
    with tempfile.TemporaryDirectory() as tmp:
        bands = _make_bands(tmp)
        result = await execute_tool("compute_spectral_index", arguments={
            "index": "ndvi",
            "nir": bands["nir"],
            "red": bands["red"],
            "output_format": "json",
        })
        envelope = json.loads(result[0].text)
        print(f"  Envelope mode: {envelope['mode']}")
        print(f"  Has output_path: {'output_path' in envelope['data']}")
        print(f"  Has stats: {'stats' in envelope['data']}")
        print("  PASS")


async def scenario_14_auto_output_path():
    """Scenario: Agent doesn't specify output_path (auto-generated)."""
    print("\n=== Scenario 14: Auto-generated output path ===")
    with tempfile.TemporaryDirectory() as tmp:
        bands = _make_bands(tmp)
        result = await execute_tool("compute_spectral_index", arguments={
            "index": "ndvi",
            "nir": bands["nir"],
            "red": bands["red"],
        })
        data = json.loads(result[0].text)
        print(f"  Auto path: {data['output_path']}")
        assert os.path.exists(data["output_path"])
        arr = np.load(data["output_path"])
        print(f"  Shape: {arr.shape}")
        os.unlink(data["output_path"])
        print("  PASS")


async def scenario_15_replace_nans():
    """Scenario: Agent replaces NaN values after masking."""
    print("\n=== Scenario 15: NaN replacement ===")
    with tempfile.TemporaryDirectory() as tmp:
        arr = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]])
        path = os.path.join(tmp, "nan_array.npy")
        np.save(path, arr)

        result = await execute_tool("apply_mask", arguments={
            "input": path,
            "method": "replace_nans",
            "value": -9999.0,
            "output_path": os.path.join(tmp, "filled.npy"),
        })
        data = json.loads(result[0].text)
        filled = np.load(data["output_path"])
        print(f"  Before: {np.sum(np.isnan(arr))} NaNs")
        print(
            f"  After: {np.sum(np.isnan(filled))} NaNs, "
            f"{np.sum(filled == -9999.0)} filled",
        )
        assert np.sum(np.isnan(filled)) == 0
        print("  PASS")


async def scenario_16_distances():
    """Scenario: Agent computes distances between point sets."""
    print("\n=== Scenario 16: Distance computation ===")
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(42)
        a = rng.random((100, 2))
        b = rng.random((100, 2))
        pa = os.path.join(tmp, "a.npy")
        pb = os.path.join(tmp, "b.npy")
        np.save(pa, a)
        np.save(pb, b)

        for metric in ["euclidean", "manhattan", "chebyshev"]:
            result = await execute_tool("compute_distances", arguments={
                "points_a": pa,
                "points_b": pb,
                "metric": metric,
                "output_path": os.path.join(tmp, f"dist_{metric}.npy"),
            })
            data = json.loads(result[0].text)
            arr = np.load(data["output_path"])
            print(f"  {metric}: shape={arr.shape}, mean={np.mean(arr):.3f}")

        result = await execute_tool("compute_distances", arguments={
            "points_a": pa,
            "points_b": pb,
            "metric": "minkowski",
            "p": 3.0,
            "output_path": os.path.join(tmp, "dist_mink.npy"),
        })
        data = json.loads(result[0].text)
        print(f"  minkowski (p=3): shape={np.load(data['output_path']).shape}")
        print("  PASS")


async def scenario_17_trend_analysis():
    """Scenario: Agent analyzes a vegetation time series trend."""
    print("\n=== Scenario 17: Trend analysis ===")
    with tempfile.TemporaryDirectory() as tmp:
        t = np.arange(100, dtype=np.float64)
        series = 0.5 + 0.003 * t + np.random.default_rng(42).normal(0, 0.1, 100)
        path = os.path.join(tmp, "series.npy")
        np.save(path, series)

        result = await execute_tool("analyze_trends", arguments={
            "input": path,
            "method": "linear_regression",
        })
        data = json.loads(result[0].text)
        print(
            f"  Linear regression: slope={data['slope']:.4f}, "
            f"intercept={data['intercept']:.4f}",
        )
        assert abs(data["slope"] - 0.003) < 0.001
        print("  PASS")


async def scenario_18_scl_keep_codes():
    """Scenario: Agent uses SCL mask with custom keep codes."""
    print("\n=== Scenario 18: SCL mask with custom keep codes ===")
    with tempfile.TemporaryDirectory() as tmp:
        scl_path = _make_scl(tmp)

        result = await execute_tool("apply_mask", arguments={
            "input": scl_path,
            "method": "scl",
            "keep_codes": [4, 5, 6, 7],
            "output_path": os.path.join(tmp, "scl_filtered.npy"),
        })
        data = json.loads(result[0].text)
        arr = np.load(data["output_path"])
        nan_count = np.sum(np.isnan(arr))
        print(
            f"  SCL filtered (keep 4,5,6,7): {nan_count} NaN pixels out of {arr.size}",
        )
        print("  PASS")


async def main():
    scenarios = [
        scenario_1_basic_ndvi,
        scenario_2_multi_index,
        scenario_3_cloud_mask_pipeline,
        scenario_4_temporal_composite,
        scenario_5_change_detection,
        scenario_6_morphological_cleanup,
        scenario_7_zonal_stats,
        scenario_8_moving_average,
        scenario_9_pixelwise_scaling,
        scenario_10_error_handling,
        scenario_11_shape_mismatch,
        scenario_12_list_capabilities,
        scenario_13_json_output,
        scenario_14_auto_output_path,
        scenario_15_replace_nans,
        scenario_16_distances,
        scenario_17_trend_analysis,
        scenario_18_scl_keep_codes,
    ]

    passed = 0
    failed = 0
    issues = []

    for scenario in scenarios:
        try:
            await scenario()
            passed += 1
        except Exception as e:
            failed += 1
            issues.append((scenario.__name__, type(e).__name__, str(e)))
            print(f"  FAIL: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(scenarios)}")
    if issues:
        print("\nIssues found:")
        for name, etype, msg in issues:
            print(f"  {name}: {etype}: {msg}")


if __name__ == "__main__":
    asyncio.run(main())
