"""Prompt registrations for the EO Processor MCP server."""

from __future__ import annotations

import json
from typing import Any

from fastmcp.prompts.base import Message
from mcp.types import TextContent


def register_prompts(app: Any) -> None:
    """Register all prompt definitions on the provided FastMCP app."""

    @app.prompt(
        name="eo_processor_overview_prompt",
        description="Overview of EO Processor tools available",
        meta={},
    )
    def _prompt_overview() -> list[Message]:
        human = (
            "Available EO Processor tools: compute_spectral_index, "
            "compute_change_index, temporal_statistics, temporal_composite, "
            "moving_average, apply_mask, morphological_operation, "
            "compute_distances, analyze_trends, bfast_monitor, classify, "
            "texture_features, zonal_statistics, pixelwise_transform, "
            "list_capabilities. All tools accept .npy file paths as input "
            "and return result .npy files with metadata."
        )
        payload = {
            "name": "eo_processor_overview",
            "description": "Overview of EO Processor tools available",
            "parameters": {},
            "example": {},
        }
        return [
            Message(
                role="user",
                content=TextContent(
                    type="text",
                    text=human,
                    _meta={"machine_payload": payload},
                ),
            )
        ]

    @app.prompt(
        name="tool_compute_spectral_index_prompt",
        description="Usage for compute_spectral_index tool",
        meta={
            "schema": {
                "type": "object",
                "properties": {
                    "index": {"type": "string"},
                    "nir": {"type": "string"},
                    "red": {"type": "string"},
                    "green": {"type": "string"},
                    "blue": {"type": "string"},
                    "swir1": {"type": "string"},
                    "swir2": {"type": "string"},
                    "rededge": {"type": "string"},
                    "output_path": {"type": "string"},
                },
                "required": ["index"],
            },
            "example": {
                "index": "ndvi",
                "nir": "/path/to/nir.npy",
                "red": "/path/to/red.npy",
            },
        },
    )
    def _prompt_spectral_index() -> list[Message]:
        schema = {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "description": "Index name",
                    "enum": [
                        "ndvi", "ndwi", "ndsi", "evi", "evi2", "savi", "osavi",
                        "msavi", "gndvi", "ndre", "ndvi_re2", "lai", "nbr",
                        "ndmi", "nbr2", "gci", "ci_re", "mtci",
                        "normalized_difference",
                    ],
                },
                "nir": {"type": "string"},
                "red": {"type": "string"},
                "green": {"type": "string"},
                "blue": {"type": "string"},
                "swir1": {"type": "string"},
                "swir2": {"type": "string"},
                "rededge": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["index"],
        }
        payload = {
            "name": "compute_spectral_index",
            "description": "Compute a spectral index from band .npy files.",
            "parameters": schema,
            "example": {
                "index": "ndvi",
                "nir": "/path/to/nir.npy",
                "red": "/path/to/red.npy",
            },
        }
        human = (
            f"Tool: compute_spectral_index\nDescription: {payload['description']}\n\n"
            "Parameters:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Example:\n"
            f"{json.dumps(payload['example'], indent=2)}\n\n"
            "Notes:\n"
            "- All inputs must be .npy files with matching shapes.\n"
            "- Required bands depend on the index chosen.\n"
            "- Result is saved as .npy with metadata returned as JSON."
        )
        return [
            Message(
                role="user",
                content=TextContent(
                    type="text",
                    text=human,
                    _meta={"machine_payload": payload},
                ),
            )
        ]

    @app.prompt(
        name="tool_apply_mask_prompt",
        description="Usage for apply_mask tool",
        meta={
            "schema": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": [
                            "vals", "replace_nans", "out_range", "in_range",
                            "invalid", "scl", "with_scl",
                        ],
                    },
                    "scl": {"type": "string"},
                    "output_path": {"type": "string"},
                },
                "required": ["input", "method"],
            },
            "example": {
                "input": "/path/to/data.npy",
                "method": "scl",
                "scl": "/path/to/scl.npy",
            },
        },
    )
    def _prompt_apply_mask() -> list[Message]:
        schema = {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": [
                        "vals", "replace_nans", "out_range", "in_range",
                        "invalid", "scl", "with_scl",
                    ],
                },
                "values": {"type": "array", "items": {"type": "number"}},
                "fill_value": {"type": "number"},
                "scl": {"type": "string"},
                "keep_codes": {"type": "array", "items": {"type": "integer"}},
                "mask_codes": {"type": "array", "items": {"type": "integer"}},
                "output_path": {"type": "string"},
            },
            "required": ["input", "method"],
        }
        payload = {
            "name": "apply_mask",
            "description": "Apply masking operations to an array.",
            "parameters": schema,
            "example": {
                "input": "/path/to/data.npy",
                "method": "scl",
                "scl": "/path/to/scl.npy",
            },
        }
        human = (
            f"Tool: apply_mask\nDescription: {payload['description']}\n\n"
            "Parameters:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Example:\n"
            f"{json.dumps(payload['example'], indent=2)}\n\n"
            "Notes:\n"
            "- method=scl: mask SCL band, keeping only keep_codes (default 4,5,6).\n"
            "- method=with_scl: apply SCL mask to data array.\n"
            "- method=vals: mask specific values.\n"
            "- method=out_range/in_range: mask outside/inside a value range."
        )
        return [
            Message(
                role="user",
                content=TextContent(
                    type="text",
                    text=human,
                    _meta={"machine_payload": payload},
                ),
            )
        ]

    @app.prompt(
        name="workflow_ndvi_pipeline_prompt",
        description="Typical NDVI processing workflow",
        meta={},
    )
    def _prompt_ndvi_workflow() -> list[Message]:
        human = (
            "Typical NDVI workflow:\n"
            "1) Ensure NIR and Red band .npy files are available.\n"
            "2) Call compute_spectral_index with index=ndvi, nir=<path>, red=<path>.\n"
            "3) Optionally call apply_mask with method=scl to filter clouds.\n"
            "4) Optionally call temporal_statistics with method=median"
            " for compositing.\n"
            "5) Optionally call pixelwise_transform to scale/clamp results.\n"
            "6) Use the output_path from each step as input to the next."
        )
        payload = {
            "name": "ndvi_workflow",
            "description": "Typical NDVI processing workflow",
            "parameters": {},
            "example": {},
        }
        return [
            Message(
                role="user",
                content=TextContent(
                    type="text",
                    text=human,
                    _meta={"machine_payload": payload},
                ),
            )
        ]

    @app.prompt(
        name="workflow_change_detection_prompt",
        description="Change detection workflow using pre/post imagery",
        meta={},
    )
    def _prompt_change_detection() -> list[Message]:
        human = (
            "Change detection workflow:\n"
            "1) Ensure pre-event and post-event band .npy files are available.\n"
            "2) Call compute_change_index with the appropriate index:\n"
            "   - delta_ndvi: vegetation change (needs pre/post NIR + Red)\n"
            "   - delta_nbr/dnbr/rbr: burn severity (needs pre/post NIR + SWIR2)\n"
            "3) Optionally apply morphological operations to clean the result.\n"
            "4) Optionally compute zonal_statistics for affected areas."
        )
        payload = {
            "name": "change_detection_workflow",
            "description": "Change detection workflow",
            "parameters": {},
            "example": {},
        }
        return [
            Message(
                role="user",
                content=TextContent(
                    type="text",
                    text=human,
                    _meta={"machine_payload": payload},
                ),
            )
        ]

    @app.prompt(
        name="data_format_info_prompt",
        description="Information about expected data formats",
        meta={},
    )
    def _prompt_data_format() -> list[Message]:
        human = (
            "Data format requirements:\n"
            "- All inputs must be NumPy .npy files.\n"
            "- Band arrays must have matching shapes for index computation.\n"
            "- Temporal arrays should be time-first (time, [bands,] y, x).\n"
            "- Distance inputs should be (N, D) point sets.\n"
            "- Morphology inputs are cast to uint8 internally.\n"
            "- Results are saved as .npy with stats"
            " (shape, dtype, min, max, mean, std)."
        )
        payload = {
            "name": "data_format_info",
            "description": "Expected data formats for EO Processor tools",
            "parameters": {},
            "example": {},
        }
        return [
            Message(
                role="user",
                content=TextContent(
                    type="text",
                    text=human,
                    _meta={"machine_payload": payload},
                ),
            )
        ]
