#!/usr/bin/env python3
"""
Bump version across all files from a single source.

Usage:
    python scripts/bump_version.py 1.3.0
    python scripts/bump_version.py patch   # 1.2.2 -> 1.2.3
    python scripts/bump_version.py minor   # 1.2.2 -> 1.3.0
    python scripts/bump_version.py major   # 1.2.2 -> 2.0.0
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
INIT_FILE = ROOT / "scomp_link" / "__init__.py"

VERSION_FILES = [
    (ROOT / "pyproject.toml", r'^version = "[^"]+"', 'version = "{v}"'),
    (ROOT / "scomp_link" / "__init__.py", r'__version__ = "[^"]+"', '__version__ = "{v}"'),
    (ROOT / "scomp_link" / "cli.py", r'version="%\(prog\)s [^"]+"', 'version="%(prog)s {v}"'),
    (ROOT / "server.json", r'"version": "[^"]+"', '"version": "{v}"'),
    (ROOT / ".well-known" / "mcp.json", r'"version": "[^"]+"', '"version": "{v}"'),
    (ROOT / ".cursor-plugin" / "plugin.json", r'"version": "[^"]+"', '"version": "{v}"'),
    (ROOT / "hf-space" / "app.py", r'"version": "[^"]+"', '"version": "{v}"'),
    (ROOT / "Dockerfile", r'org\.opencontainers\.image\.version="[^"]+"', 'org.opencontainers.image.version="{v}"'),
    (ROOT / "hf-space" / "Dockerfile", r'scomp-link\[mcp\]>=[^"]+', 'scomp-link[mcp]>={v}'),
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
    for filepath, pattern, replacement in VERSION_FILES:
        if not filepath.exists():
            print(f"  skip  {filepath.relative_to(ROOT)} (not found)")
            continue
        content = filepath.read_text()
        new_content = re.sub(pattern, replacement.format(v=new_version), content, count=1, flags=re.MULTILINE)
        if content != new_content:
            filepath.write_text(new_content)
            print(f"  done  {filepath.relative_to(ROOT)}")
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
    print(
        f"\n  Next: git add -A && git commit -m 'chore: bump to {new_version}' && git tag v{new_version} && git push --follow-tags"
    )
