#!/usr/bin/env bash
# Publish scomp-link to Smithery (both URL and MCPB bundle)
# Usage: SMITHERY_API_KEY=your_key ./scripts/publish-smithery.sh
set -euo pipefail

SMITHERY_API_KEY="${SMITHERY_API_KEY:?Set SMITHERY_API_KEY env var}"
SERVER_NAME="giacomosaccaggi%2Fscomp-link"
HF_SPACE_URL="https://Euribor512-scomp-link.hf.space"
API_URL="https://api.smithery.ai/servers/${SERVER_NAME}/releases"

echo "═══════════════════════════════════════════════════════"
echo "  scomp-link → Smithery Publisher"
echo "═══════════════════════════════════════════════════════"

# ─── Step 1: Wake up HF Space ────────────────────────────
echo ""
echo "▶ Waking up HF Space..."
for i in 1 2 3 4 5 6 7 8; do
    status=$(curl -s -o /dev/null -w "%{http_code}" "${HF_SPACE_URL}/" 2>/dev/null || echo "000")
    echo "  Attempt $i: HTTP $status"
    if [ "$status" = "200" ]; then
        echo "  ✅ Space is awake"
        break
    fi
    if [ "$i" = "8" ]; then
        echo "  ⚠️  Space still not responding — continuing anyway (Smithery may use server-card.json cache)"
    fi
    sleep 15
done

# ─── Step 2: Publish external URL ────────────────────────
echo ""
echo "▶ Publishing external URL release..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "${API_URL}" \
    -H "Authorization: Bearer ${SMITHERY_API_KEY}" \
    -H "Content-Type: multipart/form-data" \
    -F 'payload={"type":"external","url":"https://Euribor512-scomp-link.hf.space/sse"}')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
echo "  HTTP $HTTP_CODE"
if [ "$HTTP_CODE" -lt 400 ]; then
    echo "  ✅ External URL published"
    echo "  $BODY"
else
    echo "  ⚠️  URL publish returned HTTP $HTTP_CODE (Space may be sleeping)"
    echo "  $BODY"
fi

# ─── Step 3: Build MCPB bundle ───────────────────────────
echo ""
echo "▶ Building MCPB bundle..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MCPB_DIR="${REPO_ROOT}/mcpb"
BUNDLE_FILE="${MCPB_DIR}/scomp-link.mcpb"

# Check if mcpb CLI is available
if command -v mcpb &>/dev/null; then
    echo "  Using mcpb CLI..."
    (cd "$MCPB_DIR" && mcpb pack)
    BUNDLE_FILE=$(ls "${MCPB_DIR}"/*.mcpb 2>/dev/null | head -1)
else
    echo "  mcpb CLI not found — building zip manually..."
    (cd "$MCPB_DIR" && zip -r scomp-link.mcpb manifest.json server/ pyproject.toml)
fi

if [ ! -f "$BUNDLE_FILE" ]; then
    echo "  ❌ Bundle file not created"
    exit 1
fi
echo "  ✅ Bundle created: $(basename "$BUNDLE_FILE") ($(du -h "$BUNDLE_FILE" | cut -f1))"

# ─── Step 4: Publish MCPB bundle ─────────────────────────
echo ""
echo "▶ Publishing MCPB bundle..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "${API_URL}" \
    -H "Authorization: Bearer ${SMITHERY_API_KEY}" \
    -F 'payload={"type":"stdio"}' \
    -F "bundle=@${BUNDLE_FILE}")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
echo "  HTTP $HTTP_CODE"
if [ "$HTTP_CODE" -lt 400 ]; then
    echo "  ✅ MCPB bundle published"
    echo "  $BODY"
else
    echo "  ❌ Bundle publish failed"
    echo "  $BODY"
    exit 1
fi

# ─── Done ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Done! Check: https://smithery.ai/servers/giacomosaccaggi/scomp-link"
echo "═══════════════════════════════════════════════════════"
