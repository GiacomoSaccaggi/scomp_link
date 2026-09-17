# -*- coding: utf-8 -*-
"""
 ██████╗██╗  ██╗██╗   ██╗███╗   ██╗██╗  ██╗██╗███╗   ██╗ ██████╗
██╔════╝██║  ██║██║   ██║████╗  ██║██║ ██╔╝██║████╗  ██║██╔════╝
██║     ███████║██║   ██║██╔██╗ ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗
██║     ██╔══██║██║   ██║██║╚██╗██║██╔═██╗ ██║██║╚██╗██║██║   ██║
╚██████╗██║  ██║╚██████╔╝██║ ╚████║██║  ██╗██║██║ ╚████║╚██████╔╝
 ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝

Code-aware chunking for RAG: Python (AST), YAML, Markdown, SQL, plain text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_type: str  # python_function, python_class, yaml_section, markdown_section, sql_statement, paragraph, comment
    start_line: int = 0
    end_line: int = 0
    metadata: dict = field(default_factory=dict)


def chunk_python(content: str, filepath: str) -> list[Chunk]:
    import ast

    chunks: list[Chunk] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Can't parse it — just treat the whole thing as one chunk
        return [
            Chunk(
                text=content,
                source_file=filepath,
                chunk_type="paragraph",
                start_line=1,
                end_line=content.count("\n") + 1,
            )
        ]

    lines = content.splitlines()
    top_level_ranges: list[tuple[int, int]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno or node.lineno
            top_level_ranges.append((start, end))
            source = "\n".join(lines[start - 1 : end])
            meta: dict[str, Any] = {"name": node.name}
            docstring = ast.get_docstring(node)
            if docstring:
                meta["docstring"] = docstring
            chunks.append(
                Chunk(
                    text=source,
                    source_file=filepath,
                    chunk_type="python_function",
                    start_line=start,
                    end_line=end,
                    metadata=meta,
                )
            )

        elif isinstance(node, ast.ClassDef):
            start = node.lineno
            end = node.end_lineno or node.lineno
            top_level_ranges.append((start, end))
            source = "\n".join(lines[start - 1 : end])
            meta = {"name": node.name}
            docstring = ast.get_docstring(node)
            if docstring:
                meta["docstring"] = docstring
            chunks.append(
                Chunk(
                    text=source,
                    source_file=filepath,
                    chunk_type="python_class",
                    start_line=start,
                    end_line=end,
                    metadata=meta,
                )
            )

    # Module-level code: lines not inside any function/class
    module_lines: list[str] = []
    module_start = 0
    for i, line in enumerate(lines, 1):
        inside = any(s <= i <= e for s, e in top_level_ranges)
        if not inside:
            if not module_lines:
                module_start = i
            module_lines.append(line)

    module_text = "\n".join(module_lines).strip()
    if module_text:
        chunks.append(
            Chunk(
                text=module_text,
                source_file=filepath,
                chunk_type="python_module",
                start_line=module_start,
                end_line=module_start + len(module_lines) - 1,
                metadata={"name": "__module__"},
            )
        )

    return chunks


def chunk_yaml(content: str, filepath: str) -> list[Chunk]:
    lines = content.splitlines()
    sections: list[tuple[str, int, list[str]]] = []
    current_key = ""
    current_start = 0
    current_lines: list[str] = []

    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        # Top-level key: starts at column 0, not a comment, contains ':'
        if stripped and not stripped.startswith("#") and not line[0:1].isspace() and ":" in stripped:
            if current_lines:
                sections.append((current_key, current_start, current_lines))
            current_key = stripped.split(":")[0].strip()
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_key, current_start, current_lines))

    chunks: list[Chunk] = []
    for key, start, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    source_file=filepath,
                    chunk_type="yaml_section",
                    start_line=start,
                    end_line=start + len(sec_lines) - 1,
                    metadata={"name": key},
                )
            )
    return chunks


def chunk_markdown(content: str, filepath: str) -> list[Chunk]:
    lines = content.splitlines()
    sections: list[tuple[str, int, list[str]]] = []
    current_header = ""
    current_start = 1
    current_lines: list[str] = []

    for i, line in enumerate(lines, 1):
        if line.startswith("#"):
            if current_lines:
                sections.append((current_header, current_start, current_lines))
            current_header = line.lstrip("#").strip()
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, current_start, current_lines))

    chunks: list[Chunk] = []
    for header, start, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    source_file=filepath,
                    chunk_type="markdown_section",
                    start_line=start,
                    end_line=start + len(sec_lines) - 1,
                    metadata={"name": header},
                )
            )
    return chunks


def chunk_sql(content: str, filepath: str) -> list[Chunk]:
    statements = [s.strip() for s in content.split(";") if s.strip()]
    chunks: list[Chunk] = []
    offset = 1
    for stmt in statements:
        line_count = stmt.count("\n") + 1
        chunks.append(
            Chunk(
                text=stmt,
                source_file=filepath,
                chunk_type="sql_statement",
                start_line=offset,
                end_line=offset + line_count - 1,
            )
        )
        offset += line_count
    return chunks


def chunk_text(content: str, filepath: str, max_tokens: int = 500, overlap: int = 50) -> list[Chunk]:
    paragraphs = re.split(r"\n\s*\n", content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[Chunk] = []
    current_words: list[str] = []
    current_start = 1

    for para in paragraphs:
        words = para.split()
        if len(current_words) + len(words) > max_tokens and current_words:
            text = " ".join(current_words)
            end_line = current_start + text.count("\n")
            chunks.append(
                Chunk(
                    text=text, source_file=filepath, chunk_type="paragraph", start_line=current_start, end_line=end_line
                )
            )
            # Keep overlap words for context continuity
            current_words = current_words[-overlap:] if overlap > 0 else []
            current_start = end_line + 1
        current_words.extend(words)

    if current_words:
        text = " ".join(current_words)
        end_line = current_start + text.count("\n")
        chunks.append(
            Chunk(text=text, source_file=filepath, chunk_type="paragraph", start_line=current_start, end_line=end_line)
        )

    return chunks


def chunk_file(filepath: str | Path) -> list[Chunk]:
    filepath = Path(filepath)
    try:
        content = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    if not content.strip():
        return []

    ext = filepath.suffix.lower()
    name = str(filepath)

    if ext == ".py":
        return chunk_python(content, name)
    elif ext in (".yaml", ".yml"):
        return chunk_yaml(content, name)
    elif ext == ".md":
        return chunk_markdown(content, name)
    elif ext == ".sql":
        return chunk_sql(content, name)
    else:
        return chunk_text(content, name)


_SKIP_DIRS = {"__pycache__", ".git", "physical_output", ".ipynb_checkpoints"}
_BINARY_EXTS = {".pyc", ".db", ".zip", ".png", ".jpg", ".gif", ".ico", ".whl", ".tar", ".gz"}


def chunk_directory(path: str | Path) -> list[Chunk]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    all_chunks: list[Chunk] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Walk the directory, skip junk like __pycache__ and hidden files
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]

        for fname in filenames:
            if fname.startswith("."):
                continue
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() in _BINARY_EXTS:
                continue
            rel = str(fpath.relative_to(root))
            file_chunks = chunk_file(fpath)
            for c in file_chunks:
                c.source_file = rel
            all_chunks.extend(file_chunks)

    return all_chunks


if __name__ == "__main__":
    # Chunk some Python code
    code = '''
import os

def greet(name):
    """Say hello to someone."""
    return f"Hello {name}"

class Calculator:
    """Basic math."""
    def add(self, a, b):
        return a + b

x = 42
'''
    chunks = chunk_python(code, "example.py")
    print(f"Python ({len(chunks)} chunks):")
    for c in chunks:
        print(f"  [{c.chunk_type}] {c.metadata.get('name', '?')} L{c.start_line}-{c.end_line}")

    # Chunk some YAML
    yaml = "name: myproject\nversion: 1.0\ndeps:\n  - numpy\n  - pandas\n"
    for c in chunk_yaml(yaml, "config.yaml"):
        print(f"  [yaml] {c.metadata['name']}")

    # Chunk some markdown
    md = "# API\nEndpoints here\n## Auth\nToken based\n## Users\nCRUD ops\n"
    for c in chunk_markdown(md, "docs.md"):
        print(f"  [md] {c.metadata['name']}")

    # Chunk SQL
    sql = "CREATE TABLE users (id INT); INSERT INTO users VALUES (1); SELECT * FROM users;"
    for c in chunk_sql(sql, "init.sql"):
        print(f"  [sql] {c.text[:40]}...")
