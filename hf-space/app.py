"""
scomp-link MCP Server — Hugging Face Spaces entrypoint.

Runs the MCP server with SSE transport on port 7860
so it can be used as a remote MCP tool from any client.
"""

from scomp_link.mcp_server import mcp

if __name__ == "__main__":
    mcp.run(transport="sse")
# scomp-link MCP
