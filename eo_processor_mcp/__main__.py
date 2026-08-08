"""Entry point for running the EO Processor MCP server."""

import os

from eo_processor_mcp.server import app


def main() -> None:
    """Launch the EO Processor MCP server CLI.

    Defaults to stdio, matching every previously documented invocation.
    Set EO_PROCESSOR_MCP_TRANSPORT=http (or streamable-http/sse) to serve over HTTP
    instead, configurable via EO_PROCESSOR_MCP_HOST/EO_PROCESSOR_MCP_PORT.
    """
    transport = os.environ.get("EO_PROCESSOR_MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        app.run()
    else:
        app.run(
            transport=transport,
            host=os.environ.get("EO_PROCESSOR_MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("EO_PROCESSOR_MCP_PORT", "8000")),
        )


if __name__ == "__main__":
    main()
