# -*- coding: utf-8 -*-
"""
███████╗ ██████╗ ██████╗ ███╗   ███╗ █████╗ ████████╗
██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔══██╗╚══██╔══╝
█████╗  ██║   ██║██████╔╝██╔████╔██║███████║   ██║
██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║██╔══██║   ██║
██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║██║  ██║   ██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝

Convert between instruction-tuning formats: Alpaca, ShareGPT, OpenAI → ChatML, Llama, plain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from scomp_link.exceptions import DataValidationError

_ROLE_MAP_SHAREGPT = {"human": "user", "gpt": "assistant", "system": "system"}
_VALID_ROLES = {"system", "user", "assistant"}

_SOURCE_FORMATS = {"alpaca", "sharegpt", "openai"}
_TARGET_FORMATS = {"chatml", "llama", "alpaca", "plain"}


@dataclass
class Conversation:
    """Universal conversation representation."""

    messages: list[dict] = field(default_factory=list)


class DatasetFormatter:
    """Convert between instruction-tuning dataset formats."""

    # ── Parsing (format → Conversation) ──────────────────────────────────

    @staticmethod
    def from_alpaca(record: dict) -> Conversation:
        if "instruction" not in record:
            raise DataValidationError("Alpaca record missing required key 'instruction'")
        if "output" not in record:
            raise DataValidationError("Alpaca record missing required key 'output'")

        instruction = record["instruction"]
        inp = record.get("input", "")
        if inp:
            instruction = f"{instruction}\n{inp}"

        return Conversation(
            messages=[
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": record["output"]},
            ]
        )

    @staticmethod
    def from_sharegpt(record: dict) -> Conversation:
        if "conversations" not in record:
            raise DataValidationError("ShareGPT record missing required key 'conversations'")

        messages: list[dict] = []
        for turn in record["conversations"]:
            sender = turn.get("from", "")
            role = _ROLE_MAP_SHAREGPT.get(sender)
            if role is None:
                raise DataValidationError(
                    f"Unknown ShareGPT role '{sender}', " f"expected one of {sorted(_ROLE_MAP_SHAREGPT)}"
                )
            messages.append({"role": role, "content": turn.get("value", "")})

        return Conversation(messages=messages)

    @staticmethod
    def from_openai(record: dict) -> Conversation:
        if "messages" not in record:
            raise DataValidationError("OpenAI record missing required key 'messages'")

        messages: list[dict] = []
        for msg in record["messages"]:
            role = msg.get("role", "")
            if role not in _VALID_ROLES:
                raise DataValidationError(f"Unknown OpenAI role '{role}', expected one of {sorted(_VALID_ROLES)}")
            messages.append({"role": role, "content": msg.get("content", "")})

        return Conversation(messages=messages)

    # ── Rendering (Conversation → string) ────────────────────────────────

    @staticmethod
    def to_chatml(conv: Conversation) -> str:
        parts: list[str] = []
        for msg in conv.messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        return "\n".join(parts)

    @staticmethod
    def to_llama(conv: Conversation) -> str:
        parts = ["<|begin_of_text|>"]
        for msg in conv.messages:
            parts.append(f"<|start_header_id|>{msg['role']}<|end_header_id|>\n" f"{msg['content']}<|eot_id|>")
        return "".join(parts)

    @staticmethod
    def to_alpaca(conv: Conversation) -> str:
        instruction = ""
        response = ""
        for msg in conv.messages:
            if msg["role"] == "user":
                instruction = msg["content"]
            elif msg["role"] == "assistant":
                response = msg["content"]

        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    @staticmethod
    def to_plain(conv: Conversation) -> str:
        parts = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in conv.messages]
        return "\n".join(parts)

    # ── Batch conversion ─────────────────────────────────────────────────

    @classmethod
    def _get_parser(cls, source_format: str):
        parsers = {
            "alpaca": cls.from_alpaca,
            "sharegpt": cls.from_sharegpt,
            "openai": cls.from_openai,
        }
        if source_format not in parsers:
            raise DataValidationError(
                f"Unknown source format '{source_format}', " f"expected one of {sorted(_SOURCE_FORMATS)}"
            )
        return parsers[source_format]

    @classmethod
    def _get_renderer(cls, target_format: str):
        renderers = {
            "chatml": cls.to_chatml,
            "llama": cls.to_llama,
            "alpaca": cls.to_alpaca,
            "plain": cls.to_plain,
        }
        if target_format not in renderers:
            raise DataValidationError(
                f"Unknown target format '{target_format}', " f"expected one of {sorted(_TARGET_FORMATS)}"
            )
        return renderers[target_format]

    @classmethod
    def convert_records(
        cls,
        records: list[dict],
        source_format: str,
        target_format: str,
    ) -> list[str]:
        parse = cls._get_parser(source_format)
        render = cls._get_renderer(target_format)
        return [render(parse(r)) for r in records]

    @classmethod
    def convert_file(
        cls,
        input_path: str | Path,
        output_path: str | Path,
        source_format: str,
        target_format: str,
        text_field: str = "text",
    ) -> int:
        input_path = Path(input_path)
        output_path = Path(output_path)

        raw = input_path.read_text(encoding="utf-8")

        # Auto-detect JSON vs JSONL
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            data = []
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        rendered = cls.convert_records(data, source_format, target_format)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for text in rendered:
                f.write(json.dumps({text_field: text}, ensure_ascii=False) + "\n")

        return len(rendered)


if __name__ == "__main__":
    # Round-trip: Alpaca → ChatML → Llama → Plain
    record = {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
    conv = DatasetFormatter.from_alpaca(record)
    print("Messages:", conv.messages)
    print("\n--- ChatML ---")
    print(DatasetFormatter.to_chatml(conv))
    print("\n--- Llama ---")
    print(DatasetFormatter.to_llama(conv))
    print("\n--- Plain ---")
    print(DatasetFormatter.to_plain(conv))

    # Batch convert
    records = [
        {"instruction": "Say hi", "input": "", "output": "Hi!"},
        {"instruction": "Count", "input": "1,2,3", "output": "Three numbers"},
    ]
    rendered = DatasetFormatter.convert_records(records, "alpaca", "chatml")
    print(f"\nBatch: converted {len(rendered)} records")
