"""AFS package stub."""

__version__ = "0.0.0"

from .config import load_config, load_config_model
from .core import find_root, resolve_context_root
from .discovery import discover_contexts, get_project_stats
from .graph import build_graph, default_graph_path, write_graph
from .manager import AFSManager
from .models import ContextRoot, MountPoint, MountType, ProjectMetadata
from .plugins import discover_plugins, load_plugins
from .schema import DirectoryConfig, PolicyType
from .validator import AFSValidator

__all__ = [
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
    "MountPoint",
    "ProjectMetadata",
    "ContextRoot",
    "DirectoryConfig",
    "PolicyType",
]

# Registry and model management (optional import)
try:
    from . import registry
    __all__.append("registry")
except ImportError:
    pass
