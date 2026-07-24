from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import afs.model_retention as model_retention
from afs.model_retention import (
    ActiveModelScan,
    ModelRetentionError,
    audit_model_retention,
    default_active_reader,
)

NOW_NS = 1_800_000_000_000_000_000
OLD_NS = NOW_NS - 30 * 24 * 60 * 60 * 1_000_000_000


def _gguf(path: Path, payload: bytes = b"GGUFmodel") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.utime(path, ns=(OLD_NS, OLD_NS), follow_symlinks=False)
    return path


def _mlx(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    for child in path.iterdir():
        os.utime(child, ns=(OLD_NS, OLD_NS), follow_symlinks=False)
    os.utime(path, ns=(OLD_NS, OLD_NS), follow_symlinks=False)
    return path


def _policy(
    path: Path,
    *,
    keep: Path,
    review: Path,
    replacement: Path | None,
) -> Path:
    replacement_line = f'superseded_by = "{replacement}"\n' if replacement is not None else ""
    path.write_text(
        "\n".join(
            [
                'schema = "afs.model-retention.v1"',
                "",
                "[[artifacts]]",
                f'path = "{keep}"',
                'decision = "keep"',
                'because = "Current evaluated deployment artifact."',
                "",
                "[[artifacts]]",
                f'path = "{review}"',
                'decision = "review"',
                'because = "Superseded after measured evaluation."',
                replacement_line.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _clear() -> ActiveModelScan:
    return ActiveModelScan(available=True)


def _empty_registry(root: Path) -> Path:
    path = root / "empty-chat-registry.toml"
    if not path.exists():
        path.write_text("models = []\n", encoding="utf-8")
    return path


def _records(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    return {str(record["path"]): record for record in artifacts}


def test_explicit_review_requires_protected_existing_replacement(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    old = _gguf(root / "old.gguf")
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    records = _records(payload)
    assert records[str(current)]["status"] == "keep"
    assert records[str(old)]["status"] == "review"
    assert records[str(old)]["blocked_reasons"] == []
    assert payload["schema"] == "afs.storage.models.v1"
    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert summary["review"] == 1


@pytest.mark.parametrize(
    ("registry_text", "expected_detail"),
    [
        (None, "no_registry_sources"),
        (
            '[[model]]\nname = "typo"\nmodel_id = "old.gguf"\n',
            "registry_top_level_keys_invalid",
        ),
        (
            '[[models]]\nname = "typo"\nmodelid = "old.gguf"\n',
            "registry_model_keys_invalid",
        ),
    ],
)
def test_review_requires_an_explicit_valid_registry_source(
    tmp_path: Path,
    registry_text: str | None,
    expected_detail: str,
) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    old = _gguf(root / "old.gguf")
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)
    registries: list[Path] = []
    if registry_text is not None:
        registry = tmp_path / "typo-registry.toml"
        registry.write_text(registry_text, encoding="utf-8")
        registries.append(registry)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=registries,
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["registry_scan_unavailable"]
    assert payload["registry_scan"]["detail"] == expected_detail


def test_active_and_registry_references_override_review_policy(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    active = _gguf(root / "active.gguf")
    registered = _gguf(root / "registered.gguf")
    policy = tmp_path / "policy.toml"
    policy.write_text(
        "\n".join(
            [
                'schema = "afs.model-retention.v1"',
                "",
                "[[artifacts]]",
                f'path = "{current}"',
                'decision = "keep"',
                'because = "Current."',
                "",
                "[[artifacts]]",
                f'path = "{active}"',
                'decision = "review"',
                'because = "Old but still active."',
                f'superseded_by = "{current}"',
                "",
                "[[artifacts]]",
                f'path = "{registered}"',
                'decision = "review"',
                'because = "Old but still registered."',
                f'superseded_by = "{current}"',
            ]
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "chat_registry.toml"
    registry.write_text(
        "\n".join(
            [
                "[[models]]",
                'name = "registered"',
                'provider = "llama-cpp"',
                f'model_id = "{registered}"',
            ]
        ),
        encoding="utf-8",
    )

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[registry],
        now_ns=NOW_NS,
        active_reader=lambda: ActiveModelScan(True, (str(active),)),
    )

    records = _records(payload)
    assert records[str(active)]["status"] == "keep"
    assert records[str(active)]["evidence"] == ["active_runtime_reference"]
    assert records[str(registered)]["status"] == "keep"
    assert records[str(registered)]["evidence"] == ["registry_reference"]


def test_router_references_override_review_policy(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    default = _gguf(root / "default.gguf")
    ensemble = _gguf(root / "ensemble.gguf")
    rule = _gguf(root / "rule.gguf")
    policy = tmp_path / "policy.toml"
    entries = [
        'schema = "afs.model-retention.v1"',
        "",
        "[[artifacts]]",
        f'path = "{current}"',
        'decision = "keep"',
        'because = "Current."',
    ]
    for artifact in (default, ensemble, rule):
        entries.extend(
            [
                "",
                "[[artifacts]]",
                f'path = "{artifact}"',
                'decision = "review"',
                'because = "Still referenced by a live router."',
                f'superseded_by = "{current}"',
            ]
        )
    policy.write_text("\n".join(entries), encoding="utf-8")
    registry = tmp_path / "chat_registry.toml"
    registry.write_text(
        "\n".join(
            [
                "models = []",
                "",
                "[[routers]]",
                'name = "live-router"',
                f'default_model = "{default}"',
                f'models = ["{ensemble}"]',
                "",
                "[[routers.rules]]",
                'keywords = ["special"]',
                f'model = "{rule}"',
            ]
        ),
        encoding="utf-8",
    )

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[registry],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    records = _records(payload)
    for artifact in (default, ensemble, rule):
        assert records[str(artifact)]["status"] == "keep"
        assert records[str(artifact)]["evidence"] == ["registry_reference"]


def test_unlisted_artifact_is_unknown_without_filename_or_age_inference(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    artifact = _gguf(root / "obviously-old-v1.gguf")

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(artifact)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["no_explicit_policy"]


def test_missing_or_unprotected_replacement_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    old = _gguf(root / "old.gguf")
    policy = _policy(
        tmp_path / "policy.toml",
        keep=root / "not-present.gguf",
        review=old,
        replacement=root / "not-present.gguf",
    )

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["replacement_not_discovered"]


def test_empty_replacement_cannot_make_an_artifact_reviewable(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    replacement = _gguf(root / "replacement.gguf", b"")
    old = _gguf(root / "old.gguf")
    policy = _policy(
        tmp_path / "policy.toml",
        keep=replacement,
        review=old,
        replacement=replacement,
    )

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["replacement_empty_artifact"]


def test_runtime_scan_unavailable_blocks_review(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    old = _gguf(root / "old.gguf")
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=lambda: ActiveModelScan(False, detail="injected"),
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["active_scan_unavailable"]
    assert payload["active_scan"] == {
        "status": "unavailable",
        "detail": "injected",
        "reference_count": 0,
        "matched_artifacts": [],
    }


def test_default_active_reader_fails_closed_on_unreadable_same_user_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnreadableProcess:
        def uids(self) -> SimpleNamespace:
            return SimpleNamespace(effective=os.getuid())

        def cmdline(self) -> list[str]:
            raise model_retention.psutil.AccessDenied(pid=123)

    monkeypatch.setattr(
        model_retention.psutil,
        "process_iter",
        lambda: iter([UnreadableProcess()]),
    )

    scan = default_active_reader()

    assert scan.available is False
    assert scan.detail == "1_processes_inaccessible"


def test_default_active_reader_fails_closed_when_process_owner_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnknownOwnerProcess:
        def uids(self) -> SimpleNamespace:
            raise model_retention.psutil.AccessDenied(pid=123)

    monkeypatch.setattr(
        model_retention.psutil,
        "process_iter",
        lambda: iter([UnknownOwnerProcess()]),
    )

    scan = default_active_reader()

    assert scan.available is False
    assert scan.detail == "1_processes_inaccessible"


@pytest.mark.parametrize("returncode", [0, 1])
def test_default_active_reader_rejects_ambiguous_lsof_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    artifact = _gguf(tmp_path / "model.gguf")
    monkeypatch.setattr(model_retention.psutil, "process_iter", lambda: iter(()))
    monkeypatch.setattr(
        model_retention.shutil,
        "which",
        lambda name: "/usr/sbin/lsof" if name == "lsof" else None,
    )
    monkeypatch.setattr(
        model_retention.subprocess,
        "run",
        lambda *_args, **_kwargs: model_retention.subprocess.CompletedProcess(
            args=["lsof"],
            returncode=returncode,
            stdout="",
            stderr="permission denied",
        ),
    )

    scan = default_active_reader([artifact])

    assert scan.available is False
    assert scan.detail == "open_file_scan_failed"


def test_default_active_reader_reports_an_open_nested_mlx_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _mlx(tmp_path / "model")
    weight = artifact / "model-00001-of-00001.safetensors"
    monkeypatch.setattr(model_retention.psutil, "process_iter", lambda: iter(()))
    monkeypatch.setattr(
        model_retention.shutil,
        "which",
        lambda name: "/usr/sbin/lsof" if name == "lsof" else None,
    )

    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> model_retention.subprocess.CompletedProcess[str]:
        assert command == ["/usr/sbin/lsof", "-F0n", "+D", str(artifact)]
        return model_retention.subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=f"p123\0n{weight}\0",
            stderr="",
        )

    monkeypatch.setattr(model_retention.subprocess, "run", fake_run)

    scan = default_active_reader([artifact])

    assert scan.available is True
    assert str(weight) in scan.references


def test_recent_grace_forces_keep_even_when_policy_says_review(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    recent = _gguf(root / "recent.gguf")
    os.utime(recent, ns=(NOW_NS, NOW_NS), follow_symlinks=False)
    policy = _policy(
        tmp_path / "policy.toml",
        keep=current,
        review=recent,
        replacement=current,
    )

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(recent)]
    assert record["status"] == "keep"
    assert record["evidence"] == ["recent_within_7_days"]


def test_nested_active_mlx_file_keeps_the_directory_artifact(tmp_path: Path) -> None:
    root = tmp_path / "models"
    current = _gguf(root / "current.gguf")
    old = _mlx(root / "old-mlx")
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=lambda: ActiveModelScan(
            True,
            (str(old / "model-00001-of-00001.safetensors"),),
        ),
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "keep"
    assert record["evidence"] == ["active_runtime_reference"]


def test_external_hardlink_blocks_review(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    old = _gguf(root / "old.gguf")
    outside = tmp_path / "other-link.gguf"
    try:
        os.link(old, outside)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["external_hardlink"]


def test_hardlink_shared_with_another_artifact_blocks_review(tmp_path: Path) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    old = _gguf(root / "old.gguf")
    peer = root / "retained-peer.gguf"
    try:
        os.link(old, peer)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["shared_with_other_artifact"]


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("live-00001-of-00002.gguf", "live-00002-of-00002.gguf"),
        ("live-1-of-2.gguf", "live-2-of-2.gguf"),
    ],
)
def test_split_gguf_shard_cannot_be_reviewed_independently(
    tmp_path: Path,
    first_name: str,
    second_name: str,
) -> None:
    root = tmp_path / "models" / "gguf"
    current = _gguf(root / "current.gguf")
    first = _gguf(root / first_name)
    second = _gguf(root / second_name)
    policy = _policy(
        tmp_path / "policy.toml",
        keep=current,
        review=second,
        replacement=current,
    )

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=lambda: ActiveModelScan(True, (str(first),)),
    )

    record = _records(payload)[str(second)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["sharded_gguf_requires_bundle_policy"]


def test_policy_rejects_unknown_keys_instead_of_ignoring_typos(tmp_path: Path) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text(
        "\n".join(
            [
                'schema = "afs.model-retention.v1"',
                'rootz = ["~/models"]',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelRetentionError, match="unknown.*rootz"):
        audit_model_retention(
            tmp_path,
            policy_path=policy,
            registry_paths=[],
            now_ns=NOW_NS,
            active_reader=_clear,
        )


def test_incomplete_mlx_shard_index_blocks_review(tmp_path: Path) -> None:
    root = tmp_path / "models"
    current = _gguf(root / "current.gguf")
    old = _mlx(root / "old-mlx")
    (old / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"missing-00002.safetensors"}}\n',
        encoding="utf-8",
    )
    os.utime(old / "model.safetensors.index.json", ns=(OLD_NS, OLD_NS))
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["missing_mlx_shard"]


@pytest.mark.parametrize(
    "shard_name",
    [
        "model-00001-of-00002.safetensors",
        "model-1-of-2.safetensors",
        "model-1-of-999999999999999999999999999999999999.safetensors",
    ],
)
def test_incomplete_mlx_filename_shard_set_blocks_review(
    tmp_path: Path,
    shard_name: str,
) -> None:
    root = tmp_path / "models"
    current = _gguf(root / "current.gguf")
    old = root / "old-mlx"
    old.mkdir(parents=True)
    (old / "config.json").write_text("{}\n", encoding="utf-8")
    (old / shard_name).write_bytes(b"weights")
    for child in old.iterdir():
        os.utime(child, ns=(OLD_NS, OLD_NS), follow_symlinks=False)
    os.utime(old, ns=(OLD_NS, OLD_NS), follow_symlinks=False)
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == ["incomplete_mlx_shard_set"]


def test_policy_path_must_stay_inside_audited_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside-policy.toml"
    outside.write_text('schema = "afs.model-retention.v1"\n', encoding="utf-8")

    with pytest.raises(ModelRetentionError, match="escapes the audited home"):
        audit_model_retention(
            home,
            policy_path=outside,
            registry_paths=[],
            now_ns=NOW_NS,
            active_reader=_clear,
        )


def test_discovery_entry_limit_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    for index in range(4):
        (root / f"entry-{index}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(model_retention, "MAX_DISCOVERY_ENTRIES", 2)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        registry_paths=[],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    assert payload["roots"][0]["status"] == "partial"
    assert payload["roots"][0]["issues"] == ["discovery_entry_limit_exceeded"]
    assert any("discovery_entry_limit_exceeded" in issue for issue in payload["issues"])


def test_partial_root_inventory_blocks_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models"
    current = _gguf(root / "current.gguf")
    old = _gguf(root / "old.gguf")
    large = root / "large"
    large.mkdir()
    for index in range(4):
        (large / f"entry-{index}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(model_retention, "MAX_DIRECTORY_ENTRIES", 3)
    policy = _policy(tmp_path / "policy.toml", keep=current, review=old, replacement=current)

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        policy_path=policy,
        registry_paths=[_empty_registry(tmp_path)],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    record = _records(payload)[str(old)]
    assert record["status"] == "unknown"
    assert record["blocked_reasons"] == [
        "replacement_root_inventory_incomplete",
        "root_inventory_incomplete",
    ]


def test_symlink_root_is_blocked_and_outside_root_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    _gguf(real_root / "outside.gguf")
    linked_root = tmp_path / "models" / "gguf"
    linked_root.parent.mkdir(parents=True)
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    payload = audit_model_retention(
        tmp_path,
        roots=[linked_root],
        registry_paths=[],
        now_ns=NOW_NS,
        active_reader=_clear,
    )
    assert payload["artifacts"] == []
    assert payload["roots"] == [
        {
            "path": str(linked_root),
            "status": "blocked",
            "issues": ["root_contains_link"],
        }
    ]

    outside = tmp_path.parent / "outside-model-root"
    with pytest.raises(ModelRetentionError, match="escapes the audited home"):
        audit_model_retention(
            tmp_path,
            roots=[outside],
            registry_paths=[],
            now_ns=NOW_NS,
            active_reader=_clear,
        )


def test_mlx_discovery_is_bounded_and_reports_directory_as_one_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models" / "mlx"
    artifact = _mlx(root / "oracle-v2")
    outside = tmp_path / "outside"
    _gguf(outside / "escaped.gguf")
    try:
        (artifact / "escape").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    payload = audit_model_retention(
        tmp_path,
        roots=[root],
        registry_paths=[],
        now_ns=NOW_NS,
        active_reader=_clear,
    )

    records = _records(payload)
    assert list(records) == [str(artifact)]
    assert records[str(artifact)]["kind"] == "mlx"
    assert records[str(artifact)]["status"] == "unknown"


def test_audit_is_deterministic_with_fixed_readers_and_does_not_modify_models(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models" / "gguf"
    second = _gguf(root / "b.gguf", b"second")
    first = _gguf(root / "a.gguf", b"first")
    before = {path: (path.read_bytes(), os.lstat(path).st_mtime_ns) for path in (first, second)}

    kwargs = {
        "roots": [root],
        "registry_paths": [],
        "now_ns": NOW_NS,
        "active_reader": _clear,
    }
    first_payload = audit_model_retention(tmp_path, **kwargs)
    second_payload = audit_model_retention(tmp_path, **kwargs)

    assert first_payload == second_payload
    assert [record["path"] for record in first_payload["artifacts"]] == [
        str(first),
        str(second),
    ]
    assert before == {
        path: (path.read_bytes(), os.lstat(path).st_mtime_ns) for path in (first, second)
    }


def test_bounded_children_discards_an_order_dependent_truncated_subset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "many"
    root.mkdir()
    for name in ("z", "a", "m"):
        (root / name).write_text(name, encoding="utf-8")

    children, truncated = model_retention._bounded_children(root, limit=2)

    assert children == []
    assert truncated is True


def test_bounded_children_has_a_verified_path_fallback_without_fd_scandir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "portable"
    root.mkdir()
    (root / "b").write_text("b", encoding="utf-8")
    (root / "a").write_text("a", encoding="utf-8")
    monkeypatch.setattr(model_retention.os, "supports_fd", set())

    children, truncated = model_retention._bounded_children(root, limit=2)

    assert children == [root / "a", root / "b"]
    assert truncated is False
