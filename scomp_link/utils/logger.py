# -*- coding: utf-8 -*-
"""
██╗      ██████╗  ██████╗  ██████╗ ███████╗██████╗ 
██║     ██╔═══██╗██╔════╝ ██╔════╝ ██╔════╝██╔══██╗
██║     ██║   ██║██║  ██╗ ██║  ██╗ █████╗  ██████╔╝
██║     ██║   ██║██║  ╚██╗██║  ╚██╗██╔══╝  ██╔══██╗
███████╗╚██████╔╝╚██████╔╝╚██████╔╝███████╗██║  ██║
╚══════╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝
"""
import logging
import sys

_LOGGER_NAME = "scomp_link"
_handler = None


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Get a logger for scomp_link submodule. All loggers are children of 'scomp_link'."""
    return logging.getLogger(name)


def set_verbosity(level: str = "info"):
    """
    Set global verbosity for scomp_link.

    PARAMETERS:
     1. level: 'silent', 'warning', 'info', 'debug'

    Usage example:
        import scomp_link
        scomp_link.set_verbosity('silent')   # suppress all output
        scomp_link.set_verbosity('info')     # default (like print)
        scomp_link.set_verbosity('debug')    # verbose
    """
    global _handler
    logger = logging.getLogger(_LOGGER_NAME)

    level_map = {
        "silent": logging.CRITICAL + 1,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    log_level = level_map.get(level.lower(), logging.INFO)
    logger.setLevel(log_level)

    # Ensure handler exists
    if _handler is None:
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(_handler)

    _handler.setLevel(log_level)


# Initialize with INFO by default (behaves like print)
set_verbosity("info")
