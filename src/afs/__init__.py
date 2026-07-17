"""Agentic File System core package."""

from .config import load_config, load_config_model
from .core import find_root, resolve_context_root
from .discovery import discover_contexts, get_project_stats
from .graph import build_graph, default_graph_path, write_graph
from .manager import AFSManager
from .models import ContextCategory, ContextRoot, MountPoint, MountType, ProjectMetadata
from .plugins import discover_plugins, load_plugins
from .project_registry import COMMON_SCOPE_ID, ProjectRecord, ProjectRegistry
from .schema import DirectoryConfig, PolicyType
from .validator import AFSValidator
from .version import __version__, __version_info__

__all__ = [
    "__version__",
    "__version_info__",
    "load_config",
    "load_config_model",
    "discover_plugins",
    "load_plugins",
    "find_root",
    "resolve_context_root",
    "discover_contexts",
    "get_project_stats",
    "build_graph",
    "default_graph_path",
    "write_graph",
    "AFSManager",
    "AFSValidator",
    "MountType",
    "ContextCategory",
    "MountPoint",
    "ProjectMetadata",
    "ContextRoot",
    "DirectoryConfig",
    "PolicyType",
    "ProjectRegistry",
    "ProjectRecord",
    "COMMON_SCOPE_ID",
]

# Registry and model management (optional import)
try:
    from . import registry
    __all__.append("registry")
except ImportError:
    pass
