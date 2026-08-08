"""Tests for array I/O utilities."""

from __future__ import annotations

import json

import numpy as np

from eo_processor_mcp.utils import array_stats, format_result, load_array, save_array


def test_save_and_load(tmp_path):
    arr = np.array([1.0, 2.0, 3.0])
    path = str(tmp_path / "test.npy")
    out = save_array(arr, path)
    assert out == path
    loaded = load_array(path)
    np.testing.assert_array_equal(arr, loaded)


def test_save_auto_path():
    arr = np.array([1.0, 2.0])
    path = save_array(arr)
    assert path.endswith(".npy")
    loaded = load_array(path)
    np.testing.assert_array_equal(arr, loaded)
    import os
    os.unlink(path)


def test_array_stats():
    arr = np.array([[1.0, 2.0], [3.0, np.nan]])
    stats = array_stats(arr)
    assert stats["shape"] == [2, 2]
    assert stats["dtype"] == "float64"
    assert stats["size"] == 4
    assert stats["nan_count"] == 1
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_format_result(tmp_path):
    arr = np.array([1.0, 2.0])
    path = str(tmp_path / "result.npy")
    save_array(arr, path)
    text = format_result(path, arr, {"extra": "info"})
    data = json.loads(text)
    assert data["output_path"] == path
    assert data["stats"]["shape"] == [2]
    assert data["extra"] == "info"
