"""Tests for the EO Processor MCP server."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eo_processor_mcp.tools.execution import execute_tool


@pytest.mark.asyncio
async def test_list_capabilities():
    result = await execute_tool(
        "list_capabilities", arguments={"output_format": "json"}
    )
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert data["mode"] == "json"
    assert "tools" in data["data"]
    assert "compute_spectral_index" in data["data"]["tools"]


@pytest.mark.asyncio
async def test_compute_spectral_index_ndvi(sample_bands, tmp_path):
    output_path = str(tmp_path / "ndvi_out.npy")
    result = await execute_tool("compute_spectral_index", arguments={
        "index": "ndvi",
        "nir": sample_bands["nir"],
        "red": sample_bands["red"],
        "output_path": output_path,
    })
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert data["stats"]["shape"] == [10, 10]
    assert data["index"] == "ndvi"
    arr = np.load(output_path)
    assert arr.shape == (10, 10)


@pytest.mark.asyncio
async def test_compute_spectral_index_missing_band():
    with pytest.raises(ValueError, match="Missing required band"):
        await execute_tool("compute_spectral_index", arguments={
            "index": "ndvi",
        })


@pytest.mark.asyncio
async def test_compute_spectral_index_unknown():
    with pytest.raises(ValueError, match="Unknown spectral index"):
        await execute_tool("compute_spectral_index", arguments={
            "index": "fake_index",
        })


@pytest.mark.asyncio
async def test_temporal_statistics_mean(tmp_npy, sample_3d):
    input_path = tmp_npy(sample_3d, "temporal_input")
    result = await execute_tool("temporal_statistics", arguments={
        "input": input_path,
        "method": "mean",
    })
    data = json.loads(result[0].text)
    assert data["method"] == "mean"
    assert data["stats"]["shape"] == [10, 10]


@pytest.mark.asyncio
async def test_apply_mask_replace_nans(tmp_npy):
    arr = np.array([[1.0, np.nan], [3.0, 4.0]])
    input_path = tmp_npy(arr, "mask_input")
    result = await execute_tool("apply_mask", arguments={
        "input": input_path,
        "method": "replace_nans",
        "value": 0.0,
    })
    data = json.loads(result[0].text)
    assert data["method"] == "replace_nans"
    out = np.load(data["output_path"])
    assert not np.any(np.isnan(out))
    assert out[0, 1] == 0.0


@pytest.mark.asyncio
async def test_pixelwise_transform(tmp_npy):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_path = tmp_npy(arr, "transform_input")
    result = await execute_tool("pixelwise_transform", arguments={
        "input": input_path,
        "scale": 2.0,
        "offset": 1.0,
        "clamp_min": 0.0,
        "clamp_max": 7.0,
    })
    data = json.loads(result[0].text)
    out = np.load(data["output_path"])
    np.testing.assert_array_equal(out, [[3.0, 5.0], [7.0, 7.0]])


@pytest.mark.asyncio
async def test_compute_distances(tmp_npy):
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    pa = tmp_npy(a, "points_a")
    pb = tmp_npy(b, "points_b")
    result = await execute_tool("compute_distances", arguments={
        "points_a": pa,
        "points_b": pb,
        "metric": "euclidean",
    })
    data = json.loads(result[0].text)
    assert data["metric"] == "euclidean"


@pytest.mark.asyncio
async def test_zonal_statistics(tmp_npy):
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    zones = np.array([[1, 1], [2, 2]])
    vp = tmp_npy(values, "values")
    zp = tmp_npy(zones, "zones")
    result = await execute_tool("zonal_statistics", arguments={
        "values": vp,
        "zones": zp,
    })
    data = json.loads(result[0].text)
    assert data["method"] == "zonal_statistics"
    assert "1" in data["zones"]
    assert "2" in data["zones"]
    assert data["zones"]["1"]["mean"] == 1.5
    assert data["zones"]["2"]["mean"] == 3.5


@pytest.mark.asyncio
async def test_analyze_trends_linear_regression(tmp_npy):
    arr = np.arange(10, dtype=np.float64)
    input_path = tmp_npy(arr, "trend_input")
    result = await execute_tool("analyze_trends", arguments={
        "input": input_path,
        "method": "linear_regression",
    })
    data = json.loads(result[0].text)
    assert data["method"] == "linear_regression"
    assert abs(data["slope"] - 1.0) < 1e-10


@pytest.mark.asyncio
async def test_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        await execute_tool("nonexistent_tool", arguments={})


@pytest.mark.asyncio
async def test_json_output_format(sample_bands, tmp_path):
    output_path = str(tmp_path / "ndvi_json.npy")
    result = await execute_tool("compute_spectral_index", arguments={
        "index": "ndvi",
        "nir": sample_bands["nir"],
        "red": sample_bands["red"],
        "output_path": output_path,
        "output_format": "json",
    })
    data = json.loads(result[0].text)
    assert data["mode"] == "json"
    assert "output_path" in data["data"]


@pytest.mark.asyncio
async def test_texture_features(tmp_npy):
    arr = np.random.default_rng(7).integers(0, 7, (8, 8)).astype(np.float64)
    input_path = tmp_npy(arr, "texture_input")
    result = await execute_tool("texture_features", arguments={
        "input": input_path,
        "window_size": 3,
        "levels": 8,
    })
    data = json.loads(result[0].text)
    assert data["method"] == "haralick_features"
    assert data["stats"]["shape"][0] == 4
    out = np.load(data["output_path"])
    assert out.shape == (4, 8, 8)


@pytest.mark.asyncio
async def test_texture_features_subset(tmp_npy):
    arr = np.random.default_rng(7).integers(0, 7, (8, 8)).astype(np.float64)
    input_path = tmp_npy(arr, "texture_subset")
    result = await execute_tool("texture_features", arguments={
        "input": input_path,
        "features": ["contrast", "entropy"],
    })
    data = json.loads(result[0].text)
    assert data["features"] == ["contrast", "entropy"]
    out = np.load(data["output_path"])
    assert out.shape == (2, 8, 8)


@pytest.mark.asyncio
async def test_classify_train_new_params(tmp_npy):
    features = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    labels = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    fp = tmp_npy(features, "clf_features")
    lp = tmp_npy(labels, "clf_labels")
    result = await execute_tool("classify", arguments={
        "method": "train",
        "features": fp,
        "labels": lp,
        "n_estimators": 8,
        "min_samples_split": 2,
        "max_depth": 2,
    })
    data = json.loads(result[0].text)
    assert data["method"] == "train"
    assert data["n_estimators"] == 8
    assert data["min_samples_split"] == 2
    assert data["max_depth"] == 2
    assert isinstance(data["model"], str)


@pytest.mark.asyncio
async def test_classify_predict(tmp_npy):
    features = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    labels = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    fp = tmp_npy(features, "clf_features")
    lp = tmp_npy(labels, "clf_labels")
    train = await execute_tool("classify", arguments={
        "method": "train",
        "features": fp,
        "labels": lp,
        "n_estimators": 5,
        "max_depth": 2,
    })
    model_json = json.loads(train[0].text)["model"]
    result = await execute_tool("classify", arguments={
        "method": "predict",
        "features": fp,
        "model": model_json,
    })
    data = json.loads(result[0].text)
    assert data["method"] == "predict"
    out = np.load(data["output_path"])
    assert out.shape == (6,)


@pytest.mark.asyncio
async def test_classify_complex(tmp_npy):
    bands = {name: tmp_npy(np.ones((4, 4)) * val, f"cc_{name}")
            for name, val in
            {"blue": 0.1, "green": 0.2, "red": 0.3,
             "nir": 0.5, "swir1": 0.4, "swir2": 0.3,
             "temp": 30.0}.items()}
    result = await execute_tool("classify", arguments={
        "method": "complex",
        "features": bands["blue"],
        "blue": bands["blue"],
        "green": bands["green"],
        "red": bands["red"],
        "nir": bands["nir"],
        "swir1": bands["swir1"],
        "swir2": bands["swir2"],
        "temp": bands["temp"],
    })
    data = json.loads(result[0].text)
    assert data["method"] == "complex_classification"
    out = np.load(data["output_path"])
    assert out.shape == (4, 4)


@pytest.mark.asyncio
async def test_classify_unknown_method(tmp_npy):
    fp = tmp_npy(np.ones((2, 2)), "feat")
    with pytest.raises(ValueError, match="Unknown classify method"):
        await execute_tool("classify", arguments={
            "method": "bogus",
            "features": fp,
        })


@pytest.mark.asyncio
async def test_bfast_monitor(tmp_npy):
    stack = np.random.default_rng(42).random((20, 5, 5)).astype(np.float64)
    stack_path = tmp_npy(stack, "bfast_stack")
    result = await execute_tool("bfast_monitor", arguments={
        "stack": stack_path,
        "dates": list(range(20100101, 20100101 + 20 * 30, 30)),
        "history_start_date": 20100101,
        "monitor_start_date": 20100601,
        "order": 1,
    })
    data = json.loads(result[0].text)
    assert data["method"] == "bfast_monitor"
    out = np.load(data["output_path"])
    assert out.ndim == 3


@pytest.mark.asyncio
async def test_temporal_composite(tmp_npy):
    arr = np.random.default_rng(42).random((6, 3, 4, 4)).astype(np.float64)
    weights = np.array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2])
    input_path = tmp_npy(arr, "tc_input")
    weights_path = tmp_npy(weights, "tc_weights")
    result = await execute_tool("temporal_composite", arguments={
        "input": input_path,
        "weights": weights_path,
    })
    data = json.loads(result[0].text)
    assert data["method"] == "temporal_composite"
    out = np.load(data["output_path"])
    assert out.shape == (3, 4, 4)


@pytest.mark.asyncio
async def test_analyze_trends_trend_analysis(tmp_npy):
    arr = np.arange(20, dtype=np.float64)
    input_path = tmp_npy(arr, "ta_input")
    result = await execute_tool("analyze_trends", arguments={
        "input": input_path,
        "method": "trend_analysis",
        "threshold": 1.0,
    })
    data = json.loads(result[0].text)
    assert data["method"] == "trend_analysis"
    assert "segments" in data


@pytest.mark.asyncio
async def test_analyze_trends_unknown_method(tmp_npy):
    input_path = tmp_npy(np.ones(5), "ta_bad")
    with pytest.raises(ValueError, match="Unknown method"):
        await execute_tool("analyze_trends", arguments={
            "input": input_path,
            "method": "bogus",
        })
