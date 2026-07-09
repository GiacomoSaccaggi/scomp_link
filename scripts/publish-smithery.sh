#!/usr/bin/env bash
# Publish scomp-link to Smithery using the official CLI
# Usage: SMITHERY_API_KEY=your_key ./scripts/publish-smithery.sh
set -euo pipefail

SMITHERY_API_KEY="${SMITHERY_API_KEY:?Set SMITHERY_API_KEY env var}"

echo "═══════════════════════════════════════════════════════"
echo "  scomp-link → Smithery Publisher"
echo "═══════════════════════════════════════════════════════"

# Check smithery CLI is installed
if ! command -v smithery &>/dev/null; then
    echo "Installing Smithery CLI..."
    npm install -g @anthropic-ai/smithery
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "▶ Publishing to Smithery (reads smithery.yaml)..."
cd "$REPO_ROOT"
smithery mcp publish . -n giacomosaccaggi/scomp-link --yes

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Done! Check: https://smithery.ai/servers/giacomosaccaggi/scomp-link"
echo "═══════════════════════════════════════════════════════"
