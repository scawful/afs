"""AFS package stub."""

__version__ = "0.0.0"

from .config import load_config, load_config_model
from .core import find_root, resolve_context_root
from .plugins import discover_plugins, load_plugins

__all__ = [
    "load_config",
    "load_config_model",
    "discover_plugins",
    "load_plugins",
    "find_root",
    "resolve_context_root",
]
