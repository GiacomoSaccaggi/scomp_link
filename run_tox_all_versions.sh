#!/bin/bash
# Setup and run tox across all Python versions (3.7 - 3.13)
# Requires: pyenv, tox

set -e

# Initialize pyenv in this shell
export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)" 2>/dev/null || true
eval "$(pyenv init -)" 2>/dev/null || true

echo "=========================================="
echo "scomp-link Multi-Version Test Runner"
echo "=========================================="

# Python versions to test
VERSIONS=("3.10.13" "3.11.8" "3.12.2" "3.13.0")

# 1. Install missing Python versions via pyenv
echo ""
echo "📦 Checking/installing Python versions via pyenv..."
for VERSION in "${VERSIONS[@]}"; do
    if pyenv versions --bare | grep -q "^${VERSION}$"; then
        echo "  ✅ Python $VERSION already installed"
    else
        echo "  ⬇️  Installing Python $VERSION..."
        pyenv install "$VERSION"
    fi
done

# 2. Set pyenv local for this project
echo ""
echo "📌 Setting pyenv local versions..."
pyenv local "${VERSIONS[@]}"
echo "  ✅ .python-version created"

# 3. Verify all versions are accessible
echo ""
echo "🔍 Verifying Python executables..."
for VERSION in "${VERSIONS[@]}"; do
    MAJOR_MINOR=$(echo "$VERSION" | cut -d. -f1,2)
    if command -v "python${MAJOR_MINOR}" &>/dev/null; then
        echo "  ✅ python${MAJOR_MINOR} → $(python${MAJOR_MINOR} --version)"
    else
        echo "  ❌ python${MAJOR_MINOR} not found in PATH"
        echo "     Make sure pyenv shims are in your PATH:"
        echo "     export PATH=\"\$HOME/.pyenv/shims:\$PATH\""
        exit 1
    fi
done

# 4. Install tox if not present
if ! command -v tox &>/dev/null; then
    echo ""
    echo "📦 Installing tox..."
    pip install tox
fi

# 5. Run tox
echo ""
echo "=========================================="
echo "🚀 Running tox across all Python versions"
echo "=========================================="
echo ""
tox

echo ""
echo "=========================================="
echo "✅ All done!"
echo "=========================================="
