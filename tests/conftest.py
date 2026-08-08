"""Shared test fixtures."""


import numpy as np
import pytest


@pytest.fixture
def tmp_npy(tmp_path):
    """Helper to create a temporary .npy file."""
    def _make(arr, name="test"):
        path = str(tmp_path / f"{name}.npy")
        np.save(path, arr)
        return path
    return _make


@pytest.fixture
def sample_2d():
    """A 10x10 float64 array."""
    rng = np.random.default_rng(42)
    return rng.random((10, 10))


@pytest.fixture
def sample_3d():
    """A 5x10x10 float64 array (time-first)."""
    rng = np.random.default_rng(42)
    return rng.random((5, 10, 10))


@pytest.fixture
def sample_bands(tmp_path):
    """Create NIR and Red band .npy files for spectral index tests."""
    rng = np.random.default_rng(42)
    nir = rng.random((10, 10)) * 0.5 + 0.3
    red = rng.random((10, 10)) * 0.3 + 0.1
    nir_path = str(tmp_path / "nir.npy")
    red_path = str(tmp_path / "red.npy")
    np.save(nir_path, nir)
    np.save(red_path, red)
    return {"nir": nir_path, "red": red_path}
