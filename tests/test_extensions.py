from __future__ import annotations

from pathlib import Path

from afs.extensions import discover_extension_manifests, load_extensions
from afs.schema import ExtensionsConfig


def _clear_extension_env(monkeypatch) -> None:
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)
    monkeypatch.delenv("AFS_EXTENSION_REPO_ROOTS", raising=False)
    monkeypatch.delenv("AFS_EXTENSION_REPO_PREFIXES", raising=False)
    monkeypatch.delenv("AFS_EXTENSION_MANIFEST_FILENAMES", raising=False)


def test_discover_and_load_extension_manifest(tmp_path: Path) -> None:
    root = tmp_path / "extensions"
    ext = root / "workspace_adapter_test"
    ext.mkdir(parents=True)
    (ext / "skills").mkdir()
    (ext / "knowledge").mkdir()

    (ext / "extension.toml").write_text(
        "name = \"workspace_adapter_test\"\n"
        "description = \"work adapter\"\n"
        "knowledge_mounts = [\"knowledge\"]\n"
        "skill_roots = [\"skills\"]\n"
        "model_registries = []\n"
        "agent_modules = [\"workspace_adapter_test.agents\"]\n"
        "\n"
        "[manager]\n"
        "actions = [\"afs status\"]\n"
        "\n"
        "[mcp_tools]\n"
        "module = \"workspace_adapter_test.mcp\"\n"
        "factory = \"register_mcp_tools\"\n"
        "\n"
        "[mcp_server]\n"
        "module = \"workspace_adapter_test.server\"\n"
        "factory = \"register_mcp_server\"\n",
        encoding="utf-8",
    )

    config = ExtensionsConfig(enabled_extensions=["workspace_adapter_test"], extension_dirs=[root])
    discovered = discover_extension_manifests(config)
    assert "workspace_adapter_test" in discovered

    loaded = load_extensions(config)
    assert "workspace_adapter_test" in loaded
    manifest = loaded["workspace_adapter_test"]
    assert manifest.description == "work adapter"
    assert (ext / "knowledge").resolve() in manifest.knowledge_mounts
    assert (ext / "skills").resolve() in manifest.skill_roots
    assert manifest.agent_modules == ["workspace_adapter_test.agents"]
    assert manifest.manager_actions == ["afs status"]
    assert manifest.mcp_tools_module == "workspace_adapter_test.mcp"
    assert manifest.mcp_tools_factory == "register_mcp_tools"
    assert manifest.mcp_server_module == "workspace_adapter_test.server"
    assert manifest.mcp_server_factory == "register_mcp_server"


def test_discover_extension_prefers_earlier_directories(tmp_path: Path) -> None:
    work_root = tmp_path / "work-exts"
    user_root = tmp_path / "user-exts"
    work_ext = work_root / "shared"
    user_ext = user_root / "shared"
    work_ext.mkdir(parents=True)
    user_ext.mkdir(parents=True)
    (work_ext / "extension.toml").write_text(
        "name = \"shared\"\n"
        "description = \"work\"\n"
        "cli_modules = [\"work.cli\"]\n",
        encoding="utf-8",
    )
    (user_ext / "extension.toml").write_text(
        "name = \"shared\"\n"
        "description = \"user\"\n"
        "cli_modules = [\"user.cli\"]\n",
        encoding="utf-8",
    )

    config = ExtensionsConfig(
        enabled_extensions=["shared"],
        extension_dirs=[work_root, user_root],
    )
    loaded = load_extensions(config)
    assert loaded["shared"].root == work_ext.resolve()
    assert loaded["shared"].cli_modules == ["work.cli"]


def test_discovers_companion_afs_name_repo_and_implicit_src_python_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _clear_extension_env(monkeypatch)
    workspace_root = tmp_path / "lab"
    repo = workspace_root / "afs_example"
    package = repo / "src" / "afs_example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repo / "extension.toml").write_text(
        "name = \"afs_example\"\n"
        "description = \"Example context adapter\"\n"
        "cli_modules = [\"afs_example.cli\"]\n",
        encoding="utf-8",
    )

    config = ExtensionsConfig(
        enabled_extensions=["afs_example"],
        extension_dirs=[],
        extension_repo_roots=[workspace_root],
        auto_discover=False,
    )

    discovered = discover_extension_manifests(config)
    assert discovered["afs_example"] == (repo / "extension.toml").resolve()

    manifest = load_extensions(config)["afs_example"]
    assert manifest.root == repo.resolve()
    assert manifest.python_paths == [(repo / "src").resolve()]
    assert (repo / "src").resolve() in manifest.import_roots


def test_companion_repo_discovery_honors_custom_prefixes_and_manifest_names(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _clear_extension_env(monkeypatch)
    workspace_root = tmp_path / "workspaces"
    repo = workspace_root / "company_google"
    repo.mkdir(parents=True)
    (repo / "afs-extension.toml").write_text(
        "name = \"company_google\"\n"
        "python_paths = [\"lib\"]\n"
        "agent_modules = [\"company_google.agents\"]\n",
        encoding="utf-8",
    )
    (repo / "lib").mkdir()

    config = ExtensionsConfig(
        enabled_extensions=["company_google"],
        extension_dirs=[],
        extension_repo_roots=[workspace_root],
        extension_repo_prefixes=["company_"],
        manifest_filenames=["afs-extension.toml"],
        auto_discover=False,
    )

    manifest = load_extensions(config)["company_google"]
    assert manifest.manifest_path == (repo / "afs-extension.toml").resolve()
    assert manifest.python_paths == [(repo / "lib").resolve()]
    assert manifest.agent_modules == ["company_google.agents"]


def test_companion_repo_discovery_can_be_configured_with_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _clear_extension_env(monkeypatch)
    workspace_root = tmp_path / "repos"
    repo = workspace_root / "custom_google"
    repo.mkdir(parents=True)
    (repo / "extension.toml").write_text(
        "name = \"custom_google\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AFS_EXTENSION_REPO_ROOTS", str(workspace_root))
    monkeypatch.setenv("AFS_EXTENSION_REPO_PREFIXES", "custom_")
    monkeypatch.setenv("AFS_ENABLED_EXTENSIONS", "custom_google")

    loaded = load_extensions(ExtensionsConfig(extension_dirs=[], auto_discover=False))
    assert loaded["custom_google"].root == repo.resolve()
