"""Array I/O utilities for MCP tool handlers.

Since MCP is a text-based protocol, tools accept .npy file paths as input
and write results to .npy files, returning metadata (shape, dtype, stats)
as text/JSON.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np


def load_array(path: str) -> np.ndarray:
    """Load a numpy array from a .npy file."""
    return np.load(path, allow_pickle=False)


def save_array(arr: np.ndarray, output_path: str | None = None) -> str:
    """Save a numpy array to a .npy file.

    If output_path is None, saves to a temp file.
    Returns the actual path written.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".npy", prefix="eoproc_")
        os.close(fd)
    np.save(output_path, arr)
    return output_path


def array_stats(arr: np.ndarray) -> dict[str, Any]:
    """Compute summary statistics for an array (NaN-aware)."""
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "ndim": int(arr.ndim),
    }
    if finite.size > 0:
        stats["min"] = float(np.min(finite))
        stats["max"] = float(np.max(finite))
        stats["mean"] = float(np.mean(finite))
        stats["std"] = float(np.std(finite))
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None
        stats["std"] = None
    if np.issubdtype(arr.dtype, np.floating):
        stats["nan_count"] = int(np.sum(np.isnan(arr)))
    return stats


def format_result(
    output_path: str,
    arr: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> str:
    """Format a tool result as JSON text for MCP response."""
    result = {
        "output_path": output_path,
        "stats": array_stats(arr),
    }
    if extra:
        result.update(extra)
    return json.dumps(result, separators=(",", ":"))
