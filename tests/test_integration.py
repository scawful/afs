"""Placeholder for extension-owned end-to-end training-system tests.

The former cost, continuous-learning, and model-deployment integration suite now
lives with the companion extension modules that provide those features. Core AFS
keeps this explicit module-level skip so pytest collection documents the move
without carrying stale imports for modules that are no longer shipped here.
"""

import pytest

pytest.skip(
    "extension-owned cost/continuous-learning integration suite moved to a companion repo",
    allow_module_level=True,
)
