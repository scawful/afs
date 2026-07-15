"""Tests for extension manifest validation and surfaced load errors."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from afs.diagnostics import check_extensions
from afs.extensions import (
    EXTENSION_API_VERSION,
    ExtensionManifestError,
    discover_extension_manifests,
    extension_load_report,
    load_extension_manifest,
    load_extensions,
)
from afs.schema import ExtensionsConfig


def _write_manifest(tmp_path: Path, name: str, body: str) -> Path:
    ext_dir = tmp_path / name
    ext_dir.mkdir(parents=True)
    manifest = ext_dir / "extension.toml"
    manifest.write_text(body, encoding="utf-8")
    return manifest


def _config(tmp_path: Path) -> ExtensionsConfig:
    return ExtensionsConfig(
        auto_discover=True,
        extension_dirs=[tmp_path],
    )


# ---------------------------------------------------------------------------
# load_extension_manifest validation
# ---------------------------------------------------------------------------


def test_api_version_defaults_and_accepts_supported(tmp_path: Path) -> None:
    implicit = load_extension_manifest(_write_manifest(tmp_path, "implicit", 'name = "implicit"\n'))
    assert implicit.api_version == EXTENSION_API_VERSION
    assert implicit.warnings == []

    explicit = load_extension_manifest(
        _write_manifest(tmp_path, "explicit", 'name = "explicit"\napi_version = 1\n')
    )
    assert explicit.api_version == 1

    legacy = load_extension_manifest(
        _write_manifest(
            tmp_path,
            "legacy",
            'schema_version = "0.1"\nname = "legacy"\n',
        )
    )
    assert legacy.api_version == 1
    assert legacy.warnings == []


def test_unsupported_api_version_is_actionable(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "future", 'name = "future"\napi_version = 99\n')
    with pytest.raises(ExtensionManifestError) as excinfo:
        load_extension_manifest(manifest)
    message = str(excinfo.value)
    assert "api_version 99" in message
    assert "supports: 1" in message
    assert str(manifest) in message


def test_non_integer_api_version_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "bad", 'name = "bad"\napi_version = "one"\n')
    with pytest.raises(ExtensionManifestError, match="api_version must be an integer"):
        load_extension_manifest(manifest)


def test_invalid_toml_is_wrapped(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "broken", 'name = "broken\n')
    with pytest.raises(ExtensionManifestError, match="invalid TOML"):
        load_extension_manifest(manifest)


@pytest.mark.parametrize(
    "payload",
    [
        b'name = "\xff"\n',
        b"api_version = " + b"9" * 5000 + b"\n",
    ],
    ids=("invalid-utf8", "huge-integer"),
)
def test_parser_failures_are_bounded_manifest_errors(tmp_path: Path, payload: bytes) -> None:
    manifest = _write_manifest(tmp_path, "parser", 'name = "parser"\n')
    manifest.write_bytes(payload)

    with pytest.raises(ExtensionManifestError) as excinfo:
        load_extension_manifest(manifest)

    assert "invalid TOML" in str(excinfo.value)
    assert len(str(excinfo.value)) < 1024


def test_unexpected_parser_failure_is_a_structured_manifest_error(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path, "recursive", 'name = "recursive"\n')
    manifest.write_text("x = " + "[" * 500 + "0" + "]" * 500, encoding="utf-8")

    with pytest.raises(
        ExtensionManifestError,
        match=r"internal manifest loader error \(RecursionError\)",
    ):
        load_extension_manifest(manifest)


def test_oversized_manifest_is_rejected_before_parsing(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "large", 'name = "large"\n')
    manifest.write_bytes(b"#" * (1024 * 1024 + 1))

    with pytest.raises(ExtensionManifestError, match="manifest exceeds 1048576 bytes"):
        load_extension_manifest(manifest)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO files are POSIX-specific")
def test_manifest_must_be_a_regular_file(tmp_path: Path) -> None:
    extension = tmp_path / "fifo"
    extension.mkdir()
    manifest = extension / "extension.toml"
    os.mkfifo(manifest)

    with pytest.raises(ExtensionManifestError, match="must be a regular file"):
        load_extension_manifest(manifest)


def test_wrong_typed_fields_are_errors(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "typed",
        'name = "typed"\nskill_roots = "skills"\ndescription = 3\n',
    )
    with pytest.raises(ExtensionManifestError) as excinfo:
        load_extension_manifest(manifest)
    message = str(excinfo.value)
    assert "skill_roots must be a list" in message
    assert "description must be a string" in message


def test_nested_fields_and_list_members_are_validated(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "nested",
        """name = "nested"
skill_roots = ["skills", 3]
cli_modules = ["valid.module", "not a module"]

[hooks]
before_context_read = "script.sh"

[manager]
actions = [3]

[mcp_tools]
module = 3
factory = "bad factory"
catalog = "wide"

[mcp_server]
module = "bad module"

[[context_sources]]
name = "tickets"
module = 4
kinds = ["ticket", 5]
""",
    )

    with pytest.raises(ExtensionManifestError) as excinfo:
        load_extension_manifest(manifest)

    message = str(excinfo.value)
    assert "skill_roots[1] must be a string" in message
    assert "cli_modules[1] must be a dotted Python module" in message
    assert "hooks.before_context_read must be a list" in message
    assert "manager.actions[0] must be a string" in message
    assert "mcp_tools.module must be a string" in message
    assert "mcp_tools.factory must be a Python identifier" in message
    assert "mcp_tools.catalog must be 'full' or 'slim'" in message
    assert "mcp_server.module must be a dotted Python module" in message
    assert "context_sources[0].module must be a string" in message
    assert "context_sources[0].kinds[1] must be a string" in message


def test_unknown_key_warns_with_suggestion(tmp_path: Path) -> None:
    manifest = load_extension_manifest(
        _write_manifest(tmp_path, "typo", 'name = "typo"\nskil_roots = ["skills"]\n')
    )
    assert any(
        "unknown key 'skil_roots'" in warning and "skill_roots" in warning
        for warning in manifest.warnings
    )


def test_invalid_mcp_tools_section_is_a_structured_error(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "toolless",
        'name = "toolless"\n[mcp_tools]\ncatalog = "wide"\n',
    )

    with pytest.raises(ExtensionManifestError) as excinfo:
        load_extension_manifest(manifest)

    assert "mcp_tools.module is required" in str(excinfo.value)
    assert "mcp_tools.catalog must be 'full' or 'slim'" in str(excinfo.value)


@pytest.mark.parametrize("catalog", ['""', '"wide"', "7"])
def test_mcp_catalog_is_fail_closed(tmp_path: Path, catalog: str) -> None:
    manifest = _write_manifest(
        tmp_path,
        f"catalog-{len(catalog)}",
        "name = \"catalog\"\n"
        "[mcp_tools]\n"
        "module = \"example.tools\"\n"
        f"catalog = {catalog}\n",
    )

    with pytest.raises(ExtensionManifestError, match="catalog must be 'full' or 'slim'"):
        load_extension_manifest(manifest)


def test_mcp_catalog_defaults_to_full(tmp_path: Path) -> None:
    manifest = load_extension_manifest(
        _write_manifest(
            tmp_path,
            "catalog-default",
            'name = "catalog-default"\n[mcp_tools]\nmodule = "example.tools"\n',
        )
    )

    assert manifest.mcp_tools_catalog == "full"


def test_manifest_warnings_are_bounded(tmp_path: Path) -> None:
    unknowns = "".join(f"unknown_{index} = {index}\n" for index in range(75))
    manifest = load_extension_manifest(
        _write_manifest(tmp_path, "warnings", f'name = "warnings"\n{unknowns}')
    )

    assert len(manifest.warnings) == 51
    assert manifest.warnings[-1] == "25 additional issue(s) omitted"


def test_manifest_diagnostics_escape_control_characters(tmp_path: Path) -> None:
    manifest = load_extension_manifest(
        _write_manifest(
            tmp_path,
            "controls",
            'name = "controls"\n"bad\\u001bkey" = 1\n',
        )
    )

    assert "\x1b" not in manifest.warnings[0]
    assert "\\x1b" in manifest.warnings[0]

    unsafe_description = _write_manifest(
        tmp_path,
        "description-controls",
        'name = "description-controls"\ndescription = "safe\\u001b[31mred"\n',
    )
    with pytest.raises(ExtensionManifestError, match="control characters"):
        load_extension_manifest(unsafe_description)


def test_factory_must_be_a_single_python_identifier(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        "factory",
        'name = "factory"\n[mcp_tools]\nmodule = "example.tools"\nfactory = "Factory.build"\n',
    )

    with pytest.raises(ExtensionManifestError, match="factory must be a Python identifier"):
        load_extension_manifest(manifest)


# ---------------------------------------------------------------------------
# Discovery resilience + report
# ---------------------------------------------------------------------------


def _isolate_discovery(monkeypatch) -> None:
    """Keep discovery inside tmp dirs: no env, default, or repo-root leakage."""
    import afs.extensions as extensions_module

    monkeypatch.setattr(extensions_module, "_default_extension_dirs", lambda: [])
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_EXTENSION_REPO_ROOTS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)


def test_broken_manifest_does_not_hide_good_ones(tmp_path: Path, monkeypatch) -> None:
    _isolate_discovery(monkeypatch)
    _write_manifest(tmp_path, "good", 'name = "good"\n')
    _write_manifest(tmp_path, "future", 'name = "future"\napi_version = 99\n')

    discovered = discover_extension_manifests(_config(tmp_path))
    assert set(discovered) == {"good"}

    loaded = load_extensions(_config(tmp_path))
    assert set(loaded) == {"good"}


def test_discovery_logs_escape_manifest_paths(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    _isolate_discovery(monkeypatch)
    _write_manifest(
        tmp_path,
        "bad\n\x1b[31mpath",
        'name = "bad"\napi_version = 9\n',
    )

    assert discover_extension_manifests(_config(tmp_path)) == {}
    assert "\x1b" not in caplog.text
    assert "\\x1b" in caplog.text


def test_extension_load_report_surfaces_errors_and_warnings(tmp_path: Path, monkeypatch) -> None:
    _isolate_discovery(monkeypatch)
    _write_manifest(tmp_path, "good", 'name = "good"\nskil_roots = ["skills"]\n')
    _write_manifest(tmp_path, "future", 'name = "future"\napi_version = 99\n')

    report = extension_load_report(_config(tmp_path))
    assert [entry["name"] for entry in report["extensions"]] == ["good"]
    assert report["extensions"][0]["api_version"] == 1
    assert report["extensions"][0]["warnings"]
    assert len(report["errors"]) == 1
    assert "api_version 99" in report["errors"][0]["error"]


def test_extension_load_report_sanitizes_unexpected_loader_errors(
    tmp_path: Path, monkeypatch
) -> None:
    import afs.extensions as extensions_module

    _isolate_discovery(monkeypatch)
    _write_manifest(tmp_path, "internal", 'name = "internal"\n')

    def fail_loader(_path: Path):
        raise RuntimeError("unexpected\n\x1b[31m" + "x" * 2000)

    monkeypatch.setattr(extensions_module, "load_extension_manifest", fail_loader)

    report = extension_load_report(_config(tmp_path))

    assert len(report["errors"]) == 1
    assert "RuntimeError" in report["errors"][0]["error"]
    assert "unexpected" not in report["errors"][0]["error"]
    assert "\x1b" not in report["errors"][0]["error"]


def test_extension_load_report_matches_explicit_enablement(tmp_path: Path, monkeypatch) -> None:
    _isolate_discovery(monkeypatch)
    _write_manifest(tmp_path, "selected", 'name = "selected"\n')
    _write_manifest(tmp_path, "other", 'name = "other"\n')
    config = ExtensionsConfig(
        auto_discover=True,
        enabled_extensions=["selected", "missing"],
        extension_dirs=[tmp_path],
    )

    report = extension_load_report(config)
    enabled = {entry["name"]: entry["enabled"] for entry in report["extensions"]}

    assert enabled == {"other": False, "selected": True}
    assert len(report["errors"]) == 1
    assert (
        "enabled extension 'missing' has no discoverable manifest" in report["errors"][0]["error"]
    )


def test_extension_load_report_deduplicates_manifest_paths(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_discovery(monkeypatch)
    manifest = _write_manifest(tmp_path, "broken", 'name = "broken"\napi_version = 9\n')
    config = ExtensionsConfig(
        auto_discover=True,
        extension_dirs=[manifest.parent],
        extension_repo_roots=[manifest.parent],
        extension_repo_prefixes=["broken"],
    )

    report = extension_load_report(config)

    assert len(report["errors"]) == 1


def test_extension_load_report_deduplicates_symlink_aliases(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_discovery(monkeypatch)
    actual = tmp_path / "actual"
    actual.mkdir()
    _write_manifest(actual, "broken", 'name = "broken"\napi_version = 9\n')
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    config = ExtensionsConfig(
        auto_discover=True,
        extension_dirs=[actual, alias],
    )

    report = extension_load_report(config)

    assert len(report["errors"]) == 1


def test_doctor_check_reports_broken_manifest(tmp_path: Path, monkeypatch) -> None:
    _isolate_discovery(monkeypatch)
    _write_manifest(tmp_path, "future", 'name = "future"\napi_version = 99\n')
    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        f'[extensions]\nauto_discover = true\nextension_dirs = ["{tmp_path}"]\n',
        encoding="utf-8",
    )

    result = check_extensions(config_path=config_path)
    assert result.status == "warn"
    assert "api_version 99" in result.message
    assert "afs plugins" in result.message


def test_doctor_does_not_call_disabled_manifests_loaded(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_discovery(monkeypatch)
    _write_manifest(tmp_path, "disabled", 'name = "disabled"\n')
    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        "[extensions]\n"
        "auto_discover = false\n"
        f'extension_dirs = ["{tmp_path.as_posix()}"]\n',
        encoding="utf-8",
    )

    result = check_extensions(config_path=config_path)

    assert result.status == "ok"
    assert "none enabled" in result.message
    assert "loaded" not in result.message


def test_doctor_manifest_summary_is_bounded(tmp_path: Path, monkeypatch) -> None:
    _isolate_discovery(monkeypatch)
    for index in range(75):
        _write_manifest(
            tmp_path,
            f"broken-{index}",
            f'name = "broken-{index}"\napi_version = 9\n',
        )
    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        "[extensions]\n"
        "auto_discover = true\n"
        f'extension_dirs = ["{tmp_path.as_posix()}"]\n',
        encoding="utf-8",
    )

    result = check_extensions(config_path=config_path)

    assert result.status == "warn"
    assert len(result.message) < 4096
    assert "70 more omitted" in result.message


@pytest.mark.parametrize(
    "relative_path",
    [
        "extensions/example_work/extension.toml",
        "examples/extension_hello_world/extension.toml",
    ],
)
def test_repository_extension_examples_validate_without_warnings(
    relative_path: str,
) -> None:
    root = Path(__file__).resolve().parents[1]

    manifest = load_extension_manifest(root / relative_path)

    assert manifest.warnings == []
