# -*- coding: utf-8 -*-
"""
███████╗ █████╗  ██████╗████████╗ ██████╗ ██████╗ ██╗   ██╗
██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
█████╗  ███████║██║        ██║   ██║   ██║██████╔╝ ╚████╔╝
██╔══╝  ██╔══██║██║        ██║   ██║   ██║██╔══██╗  ╚██╔╝
██║     ██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║   ██║
╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝

Unified factory for creating LLM capabilities by name.
"""

from __future__ import annotations

from scomp_link.llm.core.registry import _CAPABILITIES, get_capability


class LLMFactory:
    @classmethod
    def register(cls, name: str, module_path: str, class_name: str) -> None:
        _CAPABILITIES[name] = (module_path, class_name)

    @classmethod
    def list_capabilities(cls) -> list[str]:
        return sorted(_CAPABILITIES)

    @classmethod
    def create(cls, task: str, **kwargs):
        if task not in _CAPABILITIES:
            registered = ", ".join(sorted(_CAPABILITIES))
            raise ValueError(f"Unknown LLM capability {task!r}. " f"Registered capabilities: {registered}")
        try:
            klass = get_capability(task)
        except ImportError:
            raise
        except AttributeError as exc:
            module_path, class_name = _CAPABILITIES[task]
            raise ImportError(
                f"Capability {task!r}: module {module_path!r} does not " f"contain class {class_name!r}"
            ) from exc
        return klass(**kwargs)


if __name__ == "__main__":
    # List what's available, then try to create something that doesn't exist
    print("Capabilities:", LLMFactory.list_capabilities())

    # Register a custom capability and verify it shows up
    LLMFactory.register("custom", ".core.configs", "TransformerConfig")
    print("After register:", LLMFactory.list_capabilities())

    # Create from registry — TransformerConfig doesn't need torch
    obj = LLMFactory.create("custom", d_model=128, n_heads=4)
    print(f"Created: {type(obj).__name__} with d_model={obj.d_model}")

    try:
        LLMFactory.create("does_not_exist")
    except ValueError as e:
        print(f"Unknown capability: {e}")
