# Agent File

## Commands

- Install: `make install-dev`
- Test: `make test`
- Lint: `make lint`
- Format: `make format`
- Run: `python -m eo_processor_mcp`

## Architecture

This is an MCP server wrapping [eo-processor](https://github.com/BnJam/eo-processor) (a Rust+Python EO computation library) using FastMCP. It follows the same patterns as [stac-mcp](https://github.com/BnJam/stac-mcp).

### Key patterns
- Tool handlers are sync functions in `eo_processor_mcp/tools/` that accept `dict[str, Any]` and return `str`
- Execution is dispatched via `tools/execution.py` which handles threading, observability, and output format normalization
- Array I/O uses .npy file paths (MCP is text-based, so binary data goes through files)
- Observability module provides logging, metrics, correlation IDs (stderr only, never stdout)
