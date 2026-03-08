from __future__ import annotations

import argparse

from afs.cli.profile import profile_switch_command
from afs.config import load_config_model


def test_profile_switch_updates_active_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[profiles]\n"
        "active_profile = \"default\"\n"
        "\n"
        "[profiles.default]\n"
        "knowledge_mounts = []\n"
        "skill_roots = []\n"
        "model_registries = []\n"
        "enabled_extensions = []\n"
        "policies = []\n"
        "\n"
        "[profiles.work]\n"
        "knowledge_mounts = []\n"
        "skill_roots = []\n"
        "model_registries = []\n"
        "enabled_extensions = []\n"
        "policies = []\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(name="work", config=str(config_path), json=False)
    assert profile_switch_command(args) == 0

    reloaded = load_config_model(config_path=config_path, merge_user=False)
    assert reloaded.profiles.active_profile == "work"


def test_profile_switch_rejects_unknown_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[profiles]\n"
        "active_profile = \"default\"\n"
        "\n"
        "[profiles.default]\n"
        "knowledge_mounts = []\n"
        "skill_roots = []\n"
        "model_registries = []\n"
        "enabled_extensions = []\n"
        "policies = []\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(name="work", config=str(config_path), json=False)
    assert profile_switch_command(args) == 1
