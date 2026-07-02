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

Add to Cursor Settings → MCP Servers:

```json
{
  "scomp-link": {
    "command": "scomp-link",
    "args": ["mcp"]
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

### Running Standalone (for testing)

```bash
# Start server (stdio transport — for MCP clients)
scomp-link mcp

# Or run directly
python -m scomp_link.mcp_server
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

| Criterion | SKILL.md | MCP Server |
|-----------|----------|------------|
| Setup complexity | Copy one folder | Install + config MCP |
| Dependencies | None (uses CLI) | `mcp` package |
| Reliability | Agent interprets instructions | Deterministic function calls |
| Speed | CLI subprocess overhead | Direct function calls |
| Flexibility | Agent can adapt creatively | Fixed tool signatures |
| Best for | Quick setup, any agent | Production workflows, structured outputs |

**Recommendation**: Start with SKILL.md for immediate use. Add MCP server when you need deterministic, structured tool calls in production agent workflows.
