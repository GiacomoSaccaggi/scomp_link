#!/usr/bin/env python3
"""
Bump version across ALL files from a single source.

Usage:
    python scripts/bump_version.py 1.3.0
    python scripts/bump_version.py patch   # 1.2.9 -> 1.2.10
    python scripts/bump_version.py minor   # 1.2.9 -> 1.3.0
    python scripts/bump_version.py major   # 1.2.9 -> 2.0.0
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
INIT_FILE = ROOT / "scomp_link" / "__init__.py"

# Files where ALL version-like strings should be updated to the new version.
# Format: (path, pattern, replacement_template, flags)
# Use count=0 to replace ALL occurrences in a file.
VERSION_FILES = [
    # Core package
    (ROOT / "pyproject.toml", r'^version = "[^"]+"', 'version = "{v}"', re.MULTILINE),
    (ROOT / "scomp_link" / "__init__.py", r'__version__ = "[^"]+"', '__version__ = "{v}"', 0),
    (ROOT / "scomp_link" / "cli.py", r'version="%\(prog\)s [^"]+"', 'version="%(prog)s {v}"', 0),
    # MCP / Discovery
    (ROOT / "server.json", r'"version": "[^"]+"', '"version": "{v}"', 0),
    (ROOT / ".well-known" / "mcp.json", r'"version": "[^"]+"', '"version": "{v}"', 0),
    (ROOT / ".well-known" / "mcp" / "server-card.json", r'"version": "\d+\.\d+\.\d+"', '"version": "{v}"', 0),
    # Docker / HF Space
    (ROOT / "Dockerfile", r'org\.opencontainers\.image\.version="[^"]+"', 'org.opencontainers.image.version="{v}"', 0),
    (ROOT / "hf-space" / "Dockerfile", r'scomp-link\[mcp\]>=[^"]+', 'scomp-link[mcp]>={v}', 0),
    (ROOT / "hf-space" / "app.py", r'"version": "\d+\.\d+\.\d+"', '"version": "{v}"', 0),
    # Plugins
    (ROOT / ".cursor-plugin" / "plugin.json", r'"version": "[^"]+"', '"version": "{v}"', 0),
    # Skills
    (ROOT / "skills" / "scomp-link" / "SKILL.md", r'version: "[^"]+"', 'version: "{v}"', 0),
    # Wiki (What's New header)
    (ROOT / "wiki" / "Home.md", r"What's New \(v[^)]+\)", "What's New (v{v})", 0),
]


def get_current_version():
    content = INIT_FILE.read_text()
    match = re.search(r'__version__ = "([^"]+)"', content)
    return match.group(1) if match else None


def bump(current, part):
    major, minor, patch = map(int, current.split("."))
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "major":
        return f"{major + 1}.0.0"
    return part


def update_files(new_version):
    for filepath, pattern, replacement, flags in VERSION_FILES:
        if not filepath.exists():
            print(f"  skip  {filepath.relative_to(ROOT)} (not found)")
            continue
        content = filepath.read_text()
        new_content = re.sub(pattern, replacement.format(v=new_version), content, flags=flags)
        if content != new_content:
            filepath.write_text(new_content)
            count = len(re.findall(pattern, content, flags=flags))
            print(f"  done  {filepath.relative_to(ROOT)} ({count} replacement{'s' if count > 1 else ''})")
        else:
            print(f"  same  {filepath.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py <version|patch|minor|major>")
        sys.exit(1)

    current = get_current_version()
    target = sys.argv[1]
    new_version = bump(current, target) if target in ("patch", "minor", "major") else target

    print(f"\n  {current} -> {new_version}\n")
    update_files(new_version)
    print(f"\n  Next: git add -A && git commit -m 'chore: bump to {new_version}' && git tag v{new_version} && git push --follow-tags")
