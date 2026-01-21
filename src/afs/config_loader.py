"""
Configuration loader for AFS.
Handles loading TOML configuration with fallback for standard library support.
"""
import sys
from pathlib import Path
from typing import Dict, Any

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
except ImportError:
    # Fallback for environments without tomli/tomllib
    # We will use a simple parser for our specific use case or fail gracefully
    tomllib = None

def load_toml(path: Path) -> Dict[str, Any]:
    """Load TOML file with fallback."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        if tomllib:
            return tomllib.load(f)
        else:
            # Very basic fallback or error
            raise ImportError("Python 3.11+ or 'tomli' package required to parse TOML configuration.")

def get_chat_registry(root_path: Path = None) -> Dict[str, Any]:
    """Load the chat registry configuration."""
    if root_path is None:
        # Try to find from common locations relative to this file
        # src/afs/config_loader.py -> ... -> src/lab/afs-scawful/config/chat_registry.toml
        # This assumes standard layout
        current_file = Path(__file__).resolve()
        # Navigate up to find 'src'
        src_root = current_file.parent.parent.parent.parent
        config_path = src_root / "lab" / "afs-scawful" / "config" / "chat_registry.toml"
    else:
        config_path = root_path

    if not config_path.exists():
         # Fallback absolute path for scawful user
        config_path = Path("/Users/scawful/src/lab/afs-scawful/config/chat_registry.toml")
    
    return load_toml(config_path)
