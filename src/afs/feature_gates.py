"""Optional-feature gates for training / embeddings surfaces.

Core AFS (afs-lite) stays importable without torch/transformers or embedding
provider stacks. CLI and MCP entry points call :func:`require_feature` so those
surfaces fail closed unless an extra is installed or ``AFS_ENABLE_*`` is set.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Literal

FeatureName = Literal["training", "embeddings"]

_ENV_KEYS: dict[FeatureName, str] = {
    "training": "AFS_ENABLE_TRAINING",
    "embeddings": "AFS_ENABLE_EMBEDDINGS",
}

# Marker modules that indicate the matching optional extra is present.
_MARKERS: dict[FeatureName, tuple[str, ...]] = {
    "training": ("torch", "transformers", "datasets"),
    # embeddings extra is intentionally thin (httpx); treat common provider
    # SDKs as evidence the operator opted into embedding workflows.
    "embeddings": ("google.genai", "openai", "sentence_transformers"),
}

_INSTALL_HINTS: dict[FeatureName, str] = {
    "training": "pip install 'afs[training]' (or set AFS_ENABLE_TRAINING=1)",
    "embeddings": "pip install 'afs[embeddings]' / a provider SDK, or set AFS_ENABLE_EMBEDDINGS=1",
}


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _marker_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def feature_enabled(feature: FeatureName) -> bool:
    """Return True when the feature env flag or an optional marker is present."""
    if _truthy(os.environ.get(_ENV_KEYS[feature])):
        return True
    return any(_marker_available(name) for name in _MARKERS[feature])


def require_feature(feature: FeatureName) -> None:
    """Raise ``RuntimeError`` when the optional feature is not enabled."""
    if feature_enabled(feature):
        return
    env_key = _ENV_KEYS[feature]
    raise RuntimeError(
        f"afs {feature} is disabled in afs-lite. Enable with {env_key}=1 "
        f"or install the optional extra: {_INSTALL_HINTS[feature]}."
    )


def feature_gate_message(feature: FeatureName) -> str:
    """Human-readable one-liner for CLI help / doctor."""
    state = "enabled" if feature_enabled(feature) else "disabled"
    return f"{feature}: {state} ({_INSTALL_HINTS[feature]})"


def gated_cli(feature: FeatureName):
    """Decorator: print a clear error and exit 1 when the feature is off."""

    def decorator(func):
        def wrapped(args):
            try:
                require_feature(feature)
            except RuntimeError as exc:
                print(str(exc))
                return 1
            return func(args)

        wrapped.__name__ = getattr(func, "__name__", "gated_command")
        wrapped.__doc__ = getattr(func, "__doc__", None)
        return wrapped

    return decorator
