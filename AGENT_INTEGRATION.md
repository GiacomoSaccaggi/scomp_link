# Agent Integration Guide

scomp-link integrates with AI agents in two ways:

## 1. SKILL.md — Agent Skills (Zero Dependencies)

The `skills/scomp-link/SKILL.md` file teaches any agent how to use scomp-link via CLI commands. No server required — the agent reads the skill and executes commands directly.

### Setup for Kiro

```bash
# Symlink into your project's .kiro/skills/
ln -s /path/to/scomp_link/skills/scomp-link ~/.kiro/skills/scomp-link

# Or copy it
cp -r /path/to/scomp_link/skills/scomp-link ~/.kiro/skills/
```

### Setup for Claude Code

```bash
# Claude Code reads skills from .claude/skills/ in your project
mkdir -p .claude/skills
cp -r /path/to/scomp_link/skills/scomp-link .claude/skills/
```

### Setup for Cursor

```bash
# Cursor reads from .cursor/skills/
mkdir -p .cursor/skills
cp -r /path/to/scomp_link/skills/scomp-link .cursor/skills/
```

### Setup for VS Code Copilot

```bash
# Copilot reads from .github/copilot/skills/
mkdir -p .github/copilot/skills
cp -r /path/to/scomp_link/skills/scomp-link .github/copilot/skills/
```

---

## 2. MCP Server — Model Context Protocol (Structured Tools)

The MCP server exposes 15 tools, 3 resources, and 4 prompts over the standard MCP protocol. Any MCP-compatible client can discover and call them with typed inputs.

### Install

```bash
pip install scomp-link[mcp]
```

### Setup for Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "scomp-link": {
      "command": "scomp-link",
      "args": ["mcp"]
    }
  }
}
```

Restart Claude Desktop. You'll see scomp-link tools in the 🔌 menu.

### Setup for Kiro

Add to `.kiro/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "scomp-link": {
      "command": "scomp-link",
      "args": ["mcp"],
      "env": {}
    }
  }
}
```

### Setup for Cursor

**Option 1 — Marketplace plugin** (installs MCP + skill in one click):

Visit [cursor.com/marketplace/scomp-link](https://cursor.com/marketplace/scomp-link) and click "Add to Cursor".

**Option 2 — Manual MCP config** (Cursor Settings → MCP Servers):

```json
{
  "scomp-link": {
    "command": "scomp-link",
    "args": ["mcp"]
  }
}
```

**Option 3 — Remote (no install):**

```json
{
  "scomp-link": {
    "url": "https://Euribor512-scomp-link.hf.space/sse"
  }
}
```

### Setup for VS Code (Copilot + MCP extension)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "scomp-link": {
      "command": "scomp-link",
      "args": ["mcp"]
    }
  }
}
```

### Remote MCP Server (no local install needed)

scomp-link is available as a hosted MCP server on Hugging Face Spaces. No local installation required — connect directly from any MCP client:

**🔗 Live endpoint:** `https://Euribor512-scomp-link.hf.space/sse`

**🌐 Space page:** https://huggingface.co/spaces/Euribor512/scomp-link

#### Claude Desktop / Kiro / Cursor (remote)

```json
{
  "mcpServers": {
    "scomp-link": {
      "url": "https://Euribor512-scomp-link.hf.space/sse"
    }
  }
}
```

#### VS Code (remote)

```json
{
  "servers": {
    "scomp-link": {
      "url": "https://Euribor512-scomp-link.hf.space/sse"
    }
  }
}
```

> **Note:** The remote server runs on HF Spaces free tier. For heavy workloads or low-latency needs, use the local MCP server instead (`pip install scomp-link[mcp]`).

---

### Running Standalone (for testing)

```bash
# Start server (stdio transport — for MCP clients)
scomp-link mcp

# Or run directly
python -m scomp_link.mcp_server

# Or via Docker
docker run -i jack15121/scomp-link mcp
```

---

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `describe_data` | Profile a dataset (columns, types, stats) |
| `train_model` | Train a model (regression/classification) with optional tuning |
| `predict` | Generate predictions from a .scomp artifact |
| `validate_model` | Evaluate model on test data with metrics + report |
| `detect_drift` | Check distribution drift between datasets |
| `detect_anomalies` | Multi-method anomaly detection |
| `check_fairness` | Fairness and bias metrics |
| `forecast_series` | Time series forecasting |
| `engineer_features` | Automated feature engineering |
| `cluster_data` | KMeans/MeanShift clustering |
| `generate_report` | Create interactive HTML report |
| `create_visualization` | Generate a single chart (39 types available) |
| `compare_models` | Side-by-side model comparison |
| `export_model` | Convert .scomp to pickle/joblib/ONNX |
| `tune_model` | Hyperparameter optimization (via train_model with tune=true) |

## Available MCP Resources

| URI | Description |
|-----|-------------|
| `scomp://artifact/{path}` | Inspect a .scomp artifact (model type, metrics, schema) |
| `scomp://data/{path}` | Get schema + sample rows from a dataset |
| `scomp://models` | List all available model types |

## Available MCP Prompts

| Prompt | Use When |
|--------|----------|
| `ml_workflow` | Starting a new ML project from scratch |
| `debug_model` | Model has poor performance, need diagnosis |
| `monitor_production` | Setting up production data monitoring |
| `create_dashboard` | Building an analytical HTML dashboard |

---

## Which Integration to Choose?

| Criterion | Remote MCP | Local MCP | Cursor Plugin | SKILL.md |
|-----------|------------|-----------|---------------|----------|
| Setup | Zero (just URL) | `pip install` + config | One-click | Copy folder |
| Dependencies | Internet only | `mcp` package | Cursor | None |
| How it works | Remote function calls | Local function calls | Bundled MCP + skill | Agent reads docs |
| Best for | Quick start, demos | Production, speed | Cursor users | Flexibility |

**Recommendation**: Use the Cursor Plugin if you use Cursor. Otherwise, start with Remote MCP for zero setup, switch to Local MCP for production. Add SKILL.md for extra context.

---

## Server Discovery

scomp-link supports the emerging `.well-known/mcp/server-card.json` standard (SEP-1649) for automatic server discovery. AI clients can probe the hosted endpoint to auto-detect capabilities:

```
GET https://Euribor512-scomp-link.hf.space/.well-known/mcp/server-card.json
```

The server card advertises all 15 tools, transport type, and connection URL — enabling auto-configuration without manual setup.

---

## Available On

| Platform | Link |
|----------|------|
| 📦 PyPI | [pypi.org/project/scomp-link](https://pypi.org/project/scomp-link/) |
| 🐙 GitHub | [github.com/GiacomoSaccaggi/scomp_link](https://github.com/GiacomoSaccaggi/scomp_link) |
| 🐳 Docker Hub | [hub.docker.com/r/jack15121/scomp-link](https://hub.docker.com/r/jack15121/scomp-link) |
| 📦 GHCR | [ghcr.io/giacomosaccaggi/scomp-link](https://github.com/GiacomoSaccaggi/scomp_link/pkgs/container/scomp-link) |
| 🤗 HF Space | [huggingface.co/spaces/Euribor512/scomp-link](https://huggingface.co/spaces/Euribor512/scomp-link) |
| 🔧 Smithery | [smithery.ai/servers/giacomosaccaggi/scomp-link](https://smithery.ai/servers/giacomosaccaggi/scomp-link) |
| 🖱️ Cursor | [cursor.com/marketplace/scomp-link](https://cursor.com/marketplace/scomp-link) |
