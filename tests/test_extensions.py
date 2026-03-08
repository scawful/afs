from __future__ import annotations

from pathlib import Path

from afs.extensions import discover_extension_manifests, load_extensions
from afs.schema import ExtensionsConfig


def test_discover_and_load_extension_manifest(tmp_path: Path) -> None:
    root = tmp_path / "extensions"
    ext = root / "afs_google_test"
    ext.mkdir(parents=True)
    (ext / "skills").mkdir()
    (ext / "knowledge").mkdir()

    (ext / "extension.toml").write_text(
        "name = \"afs_google_test\"\n"
        "description = \"work adapter\"\n"
        "knowledge_mounts = [\"knowledge\"]\n"
        "skill_roots = [\"skills\"]\n"
        "model_registries = []\n"
        "\n"
        "[mcp_tools]\n"
        "module = \"afs_google_test.mcp\"\n"
        "factory = \"register_mcp_tools\"\n",
        encoding="utf-8",
    )

    config = ExtensionsConfig(enabled_extensions=["afs_google_test"], extension_dirs=[root])
    discovered = discover_extension_manifests(config)
    assert "afs_google_test" in discovered

    loaded = load_extensions(config)
    assert "afs_google_test" in loaded
    manifest = loaded["afs_google_test"]
    assert manifest.description == "work adapter"
    assert (ext / "knowledge").resolve() in manifest.knowledge_mounts
    assert (ext / "skills").resolve() in manifest.skill_roots
    assert manifest.mcp_tools_module == "afs_google_test.mcp"
    assert manifest.mcp_tools_factory == "register_mcp_tools"
