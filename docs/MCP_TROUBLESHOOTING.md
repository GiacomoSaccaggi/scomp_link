# MCP Server Troubleshooting Guide

This guide helps diagnose and fix common issues when setting up the scomp-link MCP server in AI agents like Junie, Claude Desktop, Kiro, or Cursor.

## Quick Setup (Recommended)

For most users installing from PyPI:

```json
{
  "mcpServers": {
    "scomp-link": {
      "command": "uvx",
      "args": ["--python", "3.12", "--from", "scomp-link[mcp]", "scomp-link", "mcp"]
    }
  }
}
```

**Key points:**
- `--python 3.12` — Required because some dependencies (numba via shap) don't support Python 3.13+
- `--from scomp-link[mcp]` — The `[mcp]` extra is **mandatory** to install MCP dependencies

---

## Common Issues

### Issue 1: "MCP dependencies not installed"

**Symptom:**
```
Error: MCP dependencies not installed. Install with: pip install scomp-link[mcp]
```

**Cause:** The `[mcp]` extra was not specified in the `--from` argument.

**Wrong:**
```json
"args": ["--from", "scomp-link", "scomp-link", "mcp"]
```

**Correct:**
```json
"args": ["--from", "scomp-link[mcp]", "scomp-link", "mcp"]
```

---

### Issue 2: "Cannot install on Python version 3.13"

**Symptom:**
```
RuntimeError: Cannot install on Python version 3.13.12; only versions >=3.6,<3.10 are supported.
```

This error comes from `numba` (a transitive dependency via `shap`).

**Cause:** `uvx` defaults to the system Python, which may be 3.13+.

**Solution:** Pin Python 3.12:
```json
"args": ["--python", "3.12", "--from", "scomp-link[mcp]", "scomp-link", "mcp"]
```

---

### Issue 3: "does not appear to be a Python project"

**Symptom:**
```
× Failed to resolve `--with` requirement
╰─▶ /some/path does not appear to be a Python project
```

**Cause:** Using a relative path like `.[mcp]` but the agent runs the command from a different working directory.

**Solution for local development:** Use the absolute path:
```json
"args": ["--python", "3.12", "--from", "/full/path/to/scomp_link[mcp]", "scomp-link", "mcp"]
```

**Solution for installed package:** Use the PyPI package name:
```json
"args": ["--python", "3.12", "--from", "scomp-link[mcp]", "scomp-link", "mcp"]
```

---

### Issue 4: Server shows "Failed" but no error message

**Cause:** The MCP stdio transport requires clean stdout. Any print statements, warnings, or logging to stdout breaks the JSON-RPC protocol.

**Diagnosis:** Test the server manually:
```bash
# Should show help without errors
uvx --python 3.12 --from scomp-link[mcp] scomp-link mcp --help

# Test JSON-RPC handshake
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | uvx --python 3.12 --from scomp-link[mcp] scomp-link mcp
```

If you see any output before the JSON response, that's stdout pollution.

---

## Configuration Locations

### Junie

- **Global config:** `~/.junie/mcp/mcp.json`
- **Project config:** `<project>/.junie/mcp.json` (if exists)

### Claude Desktop

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

### Kiro

- **Project config:** `<project>/.kiro/mcp.json`

### Cursor

- **Project config:** `<project>/.cursor/mcp.json`

---

## Alternative: Local Python Installation

If `uvx` doesn't work in your environment (offline, corporate proxy, etc.), install directly in a virtual environment:

```bash
# Create and activate venv
python3.12 -m venv ~/.scomp-link-venv
source ~/.scomp-link-venv/bin/activate

# Install with MCP extras
pip install scomp-link[mcp]
```

Then configure the MCP server to use that Python:

```json
{
  "mcpServers": {
    "scomp-link": {
      "command": "/Users/yourname/.scomp-link-venv/bin/python",
      "args": ["-m", "scomp_link.mcp_server"]
    }
  }
}
```

---

## Verification Checklist

1. ✅ Python 3.12 is available (`python3.12 --version`)
2. ✅ `uvx` is installed (`uvx --version`)
3. ✅ Config uses `--python 3.12`
4. ✅ Config uses `--from scomp-link[mcp]` (with `[mcp]` extra)
5. ✅ Manual test works: `uvx --python 3.12 --from scomp-link[mcp] scomp-link mcp --help`
6. ✅ Agent reloaded after config change

---

## Still Not Working?

1. Check the agent's MCP logs (usually in the agent's log directory)
2. Run the command manually in terminal to see the actual error
3. Ensure no firewall/proxy blocks PyPI downloads
4. Try the local Python installation method as fallback

---

## Auto-Update System

scomp-link includes an automatic update system that checks PyPI every 30 days:

### How It Works

1. On first tool use in a session, scomp-link checks if 30 days have passed since the last update check
2. If yes, it queries PyPI for the latest version
3. If a newer version exists, it automatically runs `pip install --upgrade scomp-link[mcp]`
4. The update timestamp is saved to `~/.scomp-link/last_update_check.json`

### Manual Update

You can also trigger an update manually using the MCP tools:

```
# Check current version and update availability
check_version()

# Update to latest version
update_scomp_link()

# Force update even if already on latest
update_scomp_link(force=true)
```

### After Updating

After an update, **restart the MCP server** to use the new version. The running server process still has the old code in memory.

### Disabling Auto-Update

To disable auto-update, you can manually edit `~/.scomp-link/last_update_check.json` and set the `last_check` date to a future date.
