"""
scomp-link MCP Server — Hugging Face Spaces entrypoint.

Runs the MCP server with SSE transport on 0.0.0.0:7860 (HF Spaces requirement)
and serves .well-known/mcp/server-card.json for discovery on a background thread.
"""

import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

from scomp_link.mcp_server import mcp

SERVER_CARD = {
    "$schema": "https://modelcontextprotocol.io/schemas/server-card/v1.0",
    "version": "1.2.15",
    "protocolVersion": "2025-06-18",
    "serverInfo": {
        "name": "scomp-link",
        "version": "1.2.15",
        "description": "End-to-end ML toolkit: 15 MCP tools for zero-code machine learning.",
        "homepage": "https://github.com/GiacomoSaccaggi/scomp_link",
    },
    "transport": {"type": "streamable-http", "url": "https://Euribor512-scomp-link.hf.space/sse"},
    "capabilities": {"tools": True, "resources": True, "prompts": True},
}


class DiscoveryHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/.well-known/mcp/server-card.json", "/.well-known/mcp"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(json.dumps(SERVER_CARD).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "server": "scomp-link"}).encode())
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"scomp-link MCP server is running. Connect via /sse")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logs


def start_discovery_server():
    """Serve discovery on port 7861 (internal, proxied by HF)."""
    server = HTTPServer(("0.0.0.0", 7861), DiscoveryHandler)
    server.serve_forever()


if __name__ == "__main__":
    # Start discovery endpoint in background
    threading.Thread(target=start_discovery_server, daemon=True).start()

    # Configure MCP server to listen on 0.0.0.0:7860 (HF Spaces requirement)
    # The mcp instance was created with default host="127.0.0.1" which auto-enables
    # DNS rebinding protection restricted to localhost. We must disable it for public access.
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 7860
    if hasattr(mcp.settings, "transport_security"):
        mcp.settings.transport_security = None

    # Start MCP server with SSE transport
    mcp.run(transport="sse")
