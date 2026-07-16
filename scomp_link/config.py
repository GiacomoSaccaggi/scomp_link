# -*- coding: utf-8 -*-
"""
 ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝

Persistent configuration for scomp-link.
Loads defaults from ~/.scomp-link/config.yaml (global) and .scomp-link.yaml (local).
Precedence: local > global > hardcoded defaults.
"""

import os
from pathlib import Path
from typing import Any

_HARDCODED_DEFAULTS = {
    "report": {
        "font_family": "Baloo 2",
        "url_img_logo": "",
        "url_background_header": "https://giacomosaccaggi.github.io/deep-dives/sfondo.png",
        "description": "Automatic Report",
        "author": "scomp-link toolkit",
        "language": "en",
        "main_color": "#6E37FA",
        "light_color": "#9682FF",
        "dark_color": "#4614B4",
        "footer_html": None,
    }
}

_CONFIG_TEMPLATE = """\
# scomp-link configuration
# These values are used as defaults when creating reports via MCP or Python API.
# Precedence: .scomp-link.yaml (local) > ~/.scomp-link/config.yaml (global) > hardcoded defaults.

report:
  font_family: "Baloo 2"
  url_img_logo: ""
  url_background_header: "https://giacomosaccaggi.github.io/deep-dives/sfondo.png"
  description: "Automatic Report"
  author: "scomp-link toolkit"
  language: "en"
  main_color: "#6E37FA"
  light_color: "#9682FF"
  dark_color: "#4614B4"
  footer_html: null  # Custom HTML wrapped in <footer>...</footer>, null = scomp-link default

# Example for corporate branding (Pirelli):
# report:
#   font_family: "Arial"
#   url_img_logo: "https://www.pirelli.com/corporate/media/logo.png"
#   url_background_header: "https://cdn.pirelli.com/assets/report-header.jpg"
#   description: "Pirelli Data Analytics Report"
#   author: "Pirelli Digital Team"
#   main_color: "#CC0000"
#   light_color: "#FF3333"
#   dark_color: "#990000"
#   footer_html: "<footer><strong>Pirelli S.p.A.</strong><br>Internal use only. Confidential.<br>Copyright &copy; 2026 Pirelli. All rights reserved.</footer>"
"""

_GLOBAL_CONFIG_DIR = Path.home() / ".scomp-link"
_GLOBAL_CONFIG_PATH = _GLOBAL_CONFIG_DIR / "config.yaml"
_LOCAL_CONFIG_PATH = Path(".scomp-link.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base (recursively for nested dicts)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, return empty dict if not found or invalid."""
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_config() -> dict:
    """Load configuration with precedence: local > global > hardcoded.

    Returns a dict with all report defaults resolved.
    """
    config = _HARDCODED_DEFAULTS.copy()
    config["report"] = _HARDCODED_DEFAULTS["report"].copy()

    # Global: ~/.scomp-link/config.yaml
    global_cfg = _load_yaml(_GLOBAL_CONFIG_PATH)
    if global_cfg:
        config = _deep_merge(config, global_cfg)

    # Local: .scomp-link.yaml in working directory
    local_cfg = _load_yaml(_LOCAL_CONFIG_PATH)
    if local_cfg:
        config = _deep_merge(config, local_cfg)

    return config


def get_report_defaults() -> dict:
    """Get report defaults from config. Convenience wrapper."""
    return load_config().get("report", _HARDCODED_DEFAULTS["report"])


def init_config(path: str = None, force: bool = False) -> str:
    """Create a config file with the default template.

    Args:
        path: Where to create the file. None = global (~/.scomp-link/config.yaml).
              Pass ".scomp-link.yaml" for local config.
        force: Overwrite existing file if True.

    Returns:
        The path of the created file.
    """
    if path is None:
        target = _GLOBAL_CONFIG_PATH
    else:
        target = Path(path)

    if target.exists() and not force:
        raise FileExistsError(f"Config file already exists at {target}. Use force=True to overwrite.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_CONFIG_TEMPLATE, encoding="utf-8")
    return str(target)
