# eo-processor-mcp

![Coverage](https://raw.githubusercontent.com/BnJam/eo-processor-mcp/main/coverage-badge.svg)

An MCP (Model Context Protocol) server for Earth Observation processing, powered by [eo-processor](https://github.com/BnJam/eo-processor).

Similar to [stac-mcp](https://github.com/BnJam/stac-mcp) for STAC catalog operations, this server exposes eo-processor's high-performance Rust-accelerated EO computation functions as MCP tools that AI agents can invoke.

## Features

- **15 MCP tools** covering spectral indices, change detection, temporal statistics, masking, morphology, distances, trend analysis, classification, texture features, zonal statistics, and more
- **Rust-accelerated** computation via eo-processor's PyO3 bindings (GIL-released, multi-core parallel)
- **stdio and HTTP transport** support
- **Observability** built in (structured logging, metrics, correlation IDs, latency histograms)
- **Prompts** to guide AI agents in using the tools effectively

## Tools

| Tool | Description |
|------|-------------|
| `compute_spectral_index` | Compute spectral indices (NDVI, NDWI, EVI, SAVI, etc.) from band .npy files |
| `compute_change_index` | Change detection (delta_ndvi, delta_nbr, dnbr, rbr) from pre/post imagery |
| `temporal_statistics` | Median, mean, std, sum along the time axis |
| `temporal_composite` | Weighted temporal composite of 4D arrays |
| `moving_average` | Moving average with optional stride/downsampling |
| `apply_mask` | Unified masking (by values, range, SCL, NaN replacement) |
| `morphological_operation` | Binary dilation, erosion, opening, closing |
| `compute_distances` | Pairwise distances (euclidean, manhattan, chebyshev, minkowski) |
| `analyze_trends` | Linear regression and trend analysis with break detection |
| `bfast_monitor` | BFAST Monitor change detection on time series |
| `classify` | Random Forest train/predict, complex multi-band classification |
| `texture_features` | Haralick/GLCM texture features |
| `zonal_statistics` | Zonal statistics (count, sum, mean, min, max, std per zone) |
| `pixelwise_transform` | Linear transform with optional clamping |
| `list_capabilities` | List all available indices, operations, and tools |

## Installation

```bash
pip install -e .
# or with uv
uv pip install -e .
```

## Usage

### stdio (default)

```bash
eo-processor-mcp
# or
python -m eo_processor_mcp
```

### HTTP/SSE

```bash
EO_PROCESSOR_MCP_TRANSPORT=http python -m eo_processor_mcp
```

### MCP Client Configuration

```json
{
  "eo-processor": {
    "command": "uvx",
    "args": ["--from", "git+https://github.com/BnJam/eo-processor-mcp", "eo-processor-mcp"],
    "transport": "stdio"
  }
}
```

## Data Format

All tools accept **NumPy .npy file paths** as input and write results to .npy files. Tool responses include:
- `output_path`: Path to the result .npy file
- `stats`: Summary statistics (shape, dtype, min, max, mean, std, NaN count)

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EO_PROCESSOR_MCP_TRANSPORT` | `stdio` | Transport mode (`stdio`, `http`, `streamable-http`, `sse`) |
| `EO_PROCESSOR_MCP_HOST` | `127.0.0.1` | HTTP host |
| `EO_PROCESSOR_MCP_PORT` | `8000` | HTTP port |
| `EO_PROCESSOR_MCP_LOG_LEVEL` | `WARNING` | Logging level |
| `EO_PROCESSOR_MCP_LOG_FORMAT` | `text` | Log format (`text` or `json`) |
| `EO_PROCESSOR_MCP_ENABLE_METRICS` | `true` | Enable in-process metrics |
| `EO_PROCESSOR_MCP_ENABLE_TRACE` | `false` | Enable trace span logging |

## Development

```bash
make install-dev    # Install with dev dependencies
make test           # Run tests
make lint           # Run ruff
make format         # Auto-format
make coverage       # Coverage report
```

## License

Apache-2.0
