"""Load personal context for personalized agent prompts.

This module is a thin loader for a personal-context directory containing a
``profile.toml`` and a ``manifest.toml`` declaring named conversation modes.
It is consumed by the AFS MCP server (``afs.personal.load`` prompt) and the
``afs personal`` CLI subcommand.

Path resolution (highest precedence first):
    1. ``context_root`` argument passed by the caller
    2. ``AFS_PERSONAL_CONTEXT_ROOT`` environment variable
    3. ``~/.config/afs/personal``

The default is intentionally inside ``$XDG_CONFIG_HOME``-style territory so
AFS does not assume the user organizes personal docs under ``~/src/`` or
any other workspace layout. Users who keep personal context elsewhere
(e.g. a writing folder) should export ``AFS_PERSONAL_CONTEXT_ROOT``.

Privacy: this is opt-in personal data. Callers must explicitly request a
mode; nothing here is auto-loaded by general AFS sessions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


_DEFAULT_PERSONAL_CONTEXT = Path("~/.config/afs/personal").expanduser()
_ENV_VAR = "AFS_PERSONAL_CONTEXT_ROOT"
PROFILE_FILENAME = "profile.toml"
MANIFEST_FILENAME = "manifest.toml"


def default_context_root() -> Path:
    """Resolve the default personal-context root.

    Order of precedence: ``AFS_PERSONAL_CONTEXT_ROOT`` env var, then the
    XDG-style fallback ``~/.config/afs/personal``.
    """
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return _DEFAULT_PERSONAL_CONTEXT


# Generic fallback when manifest.toml is missing or doesn't declare modes.
# The authoritative list always comes from the user's own manifest.toml.
KNOWN_MODES: tuple[str, ...] = ()


@dataclass
class PersonalContextPayload:
    """Result of loading personal context for a given mode."""

    mode: str
    profile_text: str
    files: list[tuple[str, str]]  # (relative_path, content)
    tone: str
    bias_warning: str | None
    missing: list[str]

    def render_markdown(self) -> str:
        """Render as a single markdown blob suitable for prompt injection."""
        lines = [
            f"# Personal context — mode: {self.mode}",
            "",
            f"_Tone:_ {self.tone or '(none)'}",
        ]
        if self.bias_warning:
            lines.append(f"_Bias warning:_ {self.bias_warning}")
        lines.append("")

        if self.profile_text:
            lines.append("## Profile")
            lines.append("```toml")
            lines.append(self.profile_text.strip())
            lines.append("```")
            lines.append("")

        for rel, content in self.files:
            lines.append(f"## {rel}")
            lines.append(content.rstrip())
            lines.append("")

        if self.missing:
            lines.append("## Missing files (skipped)")
            for rel in self.missing:
                lines.append(f"- {rel}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def _load_profile(context_root: Path) -> str:
    profile = context_root / PROFILE_FILENAME
    if not profile.exists():
        return ""
    return profile.read_text(encoding="utf-8")


def _load_manifest(context_root: Path) -> dict:
    manifest = context_root / MANIFEST_FILENAME
    if not manifest.exists():
        return {}
    with manifest.open("rb") as fh:
        return tomllib.load(fh)


def list_modes(context_root: Path | None = None) -> list[str]:
    """Return the available modes declared in manifest.toml."""
    root = (context_root or default_context_root()).expanduser()
    data = _load_manifest(root)
    modes = data.get("modes", {})
    if isinstance(modes, dict):
        return sorted(modes.keys())
    return list(KNOWN_MODES)


def load_personal_context(
    mode: str,
    context_root: Path | None = None,
) -> PersonalContextPayload:
    """Load profile + mode-specific files from a personal-context directory."""
    root = (context_root or default_context_root()).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"personal context root not found: {root}. "
            f"Set {_ENV_VAR} or pass context_root explicitly."
        )

    profile_text = _load_profile(root)
    manifest = _load_manifest(root)
    modes_data = manifest.get("modes", {}) if isinstance(manifest, dict) else {}

    if mode not in modes_data:
        available = ", ".join(sorted(modes_data.keys())) or ", ".join(KNOWN_MODES)
        raise ValueError(f"unknown mode '{mode}'. available: {available}")

    mode_entry = modes_data[mode] or {}
    raw_files = mode_entry.get("load", []) or []
    if not isinstance(raw_files, list):
        raise ValueError(f"mode '{mode}' load must be a list of strings")

    files: list[tuple[str, str]] = []
    missing: list[str] = []
    for rel in raw_files:
        if not isinstance(rel, str):
            continue
        rel_clean = rel.strip().lstrip("/")
        # Sanitize path traversal — must stay under context_root
        candidate = (root / rel_clean).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            missing.append(f"{rel_clean} (outside context root, skipped)")
            continue
        if not candidate.exists():
            missing.append(rel_clean)
            continue
        files.append((rel_clean, candidate.read_text(encoding="utf-8")))

    tone = mode_entry.get("tone", "") if isinstance(mode_entry.get("tone", ""), str) else ""
    bias_warning_raw = mode_entry.get("bias_warning")
    bias_warning = bias_warning_raw if isinstance(bias_warning_raw, str) else None

    return PersonalContextPayload(
        mode=mode,
        profile_text=profile_text,
        files=files,
        tone=tone,
        bias_warning=bias_warning,
        missing=missing,
    )


def render_personal_context(
    mode: str,
    context_root: Path | None = None,
) -> str:
    """Convenience: load + render in one call."""
    return load_personal_context(mode, context_root=context_root).render_markdown()
