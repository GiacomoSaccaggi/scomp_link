# -*- coding: utf-8 -*-
"""
pytest configuration for the scomp-link test suite.

Disables the SCOMP_WARN_ON_LOAD pickle security warning globally for all tests.
In real usage the warning is enabled by default; in the test environment we
suppress it because tests load artifacts they just created (trusted by definition)
and the warning noise makes it harder to spot real failures.
"""

import pytest


def pytest_configure(config):
    """Disable pickle security warning for test-created artifacts."""
    try:
        from scomp_link.persistence import artifact as _artifact_module

        _artifact_module.SCOMP_WARN_ON_LOAD = False
    except ImportError:
        pass  # mcp extra not installed, nothing to configure
