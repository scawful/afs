"""Tests for ZeldaToolkit and mission_runner zelda integration."""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from afs.agents.zelda_tools import ZeldaToolkit, _extract_labels_from_asm
from afs.agents.mission_runner import (
    _is_zelda_mission,
    load_mission,
    DEFAULT_OOS_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toolkit() -> ZeldaToolkit:
    return ZeldaToolkit()


@pytest.fixture
def fake_oos(tmp_path: Path) -> Path:
    """Create a minimal Oracle-of-Secrets-like directory."""
    repo = tmp_path / "oracle-of-secrets"
    repo.mkdir()

    # Create some ASM files
    core = repo / "Core"
    core.mkdir()
    (core / "hardware.asm").write_text(textwrap.dedent("""\
        ; Hardware registers
        ; TODO: verify DMA timing
        HardwareInit:
            SEI
            STZ $4200
            RTL

        SetupPPU:
            LDA #$80
            STA $2100
            RTS
    """))

    sprites = repo / "Sprites" / "Enemies"
    sprites.mkdir(parents=True)
    (sprites / "goriya.asm").write_text(textwrap.dedent("""\
        ; Goriya enemy sprite
        ; FIXME: boomerang return logic
        incsrc "Core/hardware.asm"
        GoriyaMain:
            JSL HardwareInit
            RTL
        GoriyaAI:
            JSR SetupPPU
            RTS
    """))

    music = repo / "Music"
    music.mkdir()
    (music / "dungeon_theme.asm").write_text(textwrap.dedent("""\
        ; Dungeon music
        ; XXX: tempo needs tuning
        incbin "Data/dungeon.bin"
        DungeonTheme:
            RTS
    """))

    # Fake git repo
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo), capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(repo), capture_output=True,
    )

    return repo


@pytest.fixture
def fake_yaze(tmp_path: Path) -> Path:
    """Create a minimal YAZE-like directory."""
    repo = tmp_path / "yaze"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n")

    cli_dir = repo / "src" / "cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "cli.cc").write_text("int main() { return 0; }\n")

    zelda3 = repo / "src" / "zelda3"
    zelda3.mkdir(parents=True)
    (zelda3 / "overworld.cc").write_text("// overworld editor\n")
    (zelda3 / "overworld.h").write_text("// header\n")

    build = repo / "build"
    build.mkdir()

    # Fake git repo
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo), capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init yaze"],
        cwd=str(repo), capture_output=True,
    )

    return repo


@pytest.fixture
def fake_mesen2(tmp_path: Path) -> Path:
    """Create a minimal Mesen2 fork-like directory."""
    repo = tmp_path / "mesen2"
    repo.mkdir()

    tools = repo / "tools"
    tools.mkdir()
    ctl = tools / "mesen2ctl"
    ctl.write_text("#!/usr/bin/env python3\nprint('mesen2ctl')\n")

    # Socket-related file
    core = repo / "Core"
    core.mkdir()
    (core / "SocketServer.cpp").write_text("// socket server\n")

    lua = repo / "Lua"
    lua.mkdir()
    (lua / "debug.lua").write_text("-- debug script\n")

    build = repo / "build"
    build.mkdir()

    # Fake git repo
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo), capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init mesen2"],
        cwd=str(repo), capture_output=True,
    )

    return repo


@pytest.fixture
def fake_disasm(tmp_path: Path) -> Path:
    """Create a minimal disassembly directory with shared labels."""
    disasm = tmp_path / "USDASM"
    disasm.mkdir()
    (disasm / "main.asm").write_text(textwrap.dedent("""\
        ; ALTTP disassembly
        HardwareInit:
            SEI
            RTL
        Module_MainRouting:
            JSR SubModule
            RTS
        SubModule:
            RTS
    """))
    return disasm


# ---------------------------------------------------------------------------
# ZeldaToolkit.scan_oos_codebase
# ---------------------------------------------------------------------------

class TestScanOoSCodebase:
    def test_scans_asm_files(self, toolkit: ZeldaToolkit, fake_oos: Path) -> None:
        result = toolkit.scan_oos_codebase(fake_oos)
        assert result["available"] is True
        assert result["asm_file_count"] == 3
        assert result["label_count"] > 0
        # Should find TODO, FIXME, XXX
        assert result["todo_count"] >= 3

    def test_counts_routine_instructions(self, toolkit: ZeldaToolkit, fake_oos: Path) -> None:
        result = toolkit.scan_oos_codebase(fake_oos)
        # RTL x2, RTS x3, JSL x1, JSR x1 = 7
        assert result["routine_instructions"] == 7

    def test_finds_include_directives(self, toolkit: ZeldaToolkit, fake_oos: Path) -> None:
        result = toolkit.scan_oos_codebase(fake_oos)
        includes = result["include_directives"]
        assert len(includes) >= 2  # incsrc and incbin
        assert any("incsrc" in inc for inc in includes)
        assert any("incbin" in inc for inc in includes)

    def test_git_info_present(self, toolkit: ZeldaToolkit, fake_oos: Path) -> None:
        result = toolkit.scan_oos_codebase(fake_oos)
        assert result["git_branch"] is not None
        assert result["git_last_commit"] is not None

    def test_missing_directory(self, toolkit: ZeldaToolkit, tmp_path: Path) -> None:
        result = toolkit.scan_oos_codebase(tmp_path / "nonexistent")
        assert result["available"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# ZeldaToolkit.scan_yaze_status
# ---------------------------------------------------------------------------

class TestScanYazeStatus:
    def test_detects_z3ed(self, toolkit: ZeldaToolkit, fake_yaze: Path) -> None:
        result = toolkit.scan_yaze_status(fake_yaze)
        assert result["available"] is True
        assert result["z3ed_cli_present"] is True
        assert result["has_cmake"] is True

    def test_detects_zelda3_module(self, toolkit: ZeldaToolkit, fake_yaze: Path) -> None:
        result = toolkit.scan_yaze_status(fake_yaze)
        assert result["zelda3_module_present"] is True

    def test_file_counts(self, toolkit: ZeldaToolkit, fake_yaze: Path) -> None:
        result = toolkit.scan_yaze_status(fake_yaze)
        assert result["cpp_file_count"] >= 1
        assert result["header_file_count"] >= 1

    def test_missing_directory(self, toolkit: ZeldaToolkit, tmp_path: Path) -> None:
        result = toolkit.scan_yaze_status(tmp_path / "nonexistent")
        assert result["available"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# ZeldaToolkit.scan_mesen2_status
# ---------------------------------------------------------------------------

class TestScanMesen2Status:
    def test_detects_tools(self, toolkit: ZeldaToolkit, fake_mesen2: Path) -> None:
        result = toolkit.scan_mesen2_status(fake_mesen2)
        assert result["available"] is True
        assert result["mesen2ctl_present"] is True
        assert result["tools_dir_exists"] is True

    def test_finds_socket_files(self, toolkit: ZeldaToolkit, fake_mesen2: Path) -> None:
        result = toolkit.scan_mesen2_status(fake_mesen2)
        assert len(result["socket_api_files"]) >= 1
        assert any("Socket" in f for f in result["socket_api_files"])

    def test_counts_lua_scripts(self, toolkit: ZeldaToolkit, fake_mesen2: Path) -> None:
        result = toolkit.scan_mesen2_status(fake_mesen2)
        assert result["lua_script_count"] >= 1

    def test_missing_directory(self, toolkit: ZeldaToolkit, tmp_path: Path) -> None:
        result = toolkit.scan_mesen2_status(tmp_path / "nonexistent")
        assert result["available"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# ZeldaToolkit.cross_reference_symbols
# ---------------------------------------------------------------------------

class TestCrossReferenceSymbols:
    def test_finds_shared_labels(
        self,
        toolkit: ZeldaToolkit,
        fake_oos: Path,
        fake_disasm: Path,
    ) -> None:
        result = toolkit.cross_reference_symbols(fake_oos, fake_disasm)
        assert result["available"] is True
        # HardwareInit exists in both fake_oos and fake_disasm
        assert "HardwareInit" in result["shared_labels"]
        assert result["shared_count"] >= 1

    def test_unique_labels(
        self,
        toolkit: ZeldaToolkit,
        fake_oos: Path,
        fake_disasm: Path,
    ) -> None:
        result = toolkit.cross_reference_symbols(fake_oos, fake_disasm)
        assert result["oos_only_count"] > 0
        assert result["disasm_only_count"] > 0
        # GoriyaMain is only in OoS
        assert "GoriyaMain" in result["oos_only_sample"]
        # Module_MainRouting is only in disasm
        assert "Module_MainRouting" in result["disasm_only_sample"]

    def test_missing_both_dirs(self, toolkit: ZeldaToolkit, tmp_path: Path) -> None:
        result = toolkit.cross_reference_symbols(
            tmp_path / "a",
            tmp_path / "b",
        )
        assert result["available"] is False
        assert "error" in result

    def test_one_dir_missing(
        self,
        toolkit: ZeldaToolkit,
        fake_oos: Path,
        tmp_path: Path,
    ) -> None:
        result = toolkit.cross_reference_symbols(fake_oos, tmp_path / "missing")
        assert result["available"] is True
        assert result["oos_exists"] is True
        assert result["disasm_exists"] is False
        # All OoS labels should be oos_only
        assert result["shared_count"] == 0


# ---------------------------------------------------------------------------
# _extract_labels_from_asm helper
# ---------------------------------------------------------------------------

class TestExtractLabels:
    def test_extracts_basic_labels(self, tmp_path: Path) -> None:
        f = tmp_path / "test.asm"
        f.write_text("Foo:\n  NOP\nBar:\n  RTS\n")
        labels = _extract_labels_from_asm([f])
        assert labels == {"Foo", "Bar"}

    def test_ignores_comments(self, tmp_path: Path) -> None:
        f = tmp_path / "test.asm"
        f.write_text("; not_a_label:\nActualLabel:\n  RTS\n")
        labels = _extract_labels_from_asm([f])
        # The regex matches lines starting with label pattern, so
        # "; not_a_label:" won't match because ; is not [A-Za-z_]
        assert "ActualLabel" in labels

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.asm"
        labels = _extract_labels_from_asm([missing])
        assert labels == set()


# ---------------------------------------------------------------------------
# Mission runner integration
# ---------------------------------------------------------------------------

class TestMissionDetection:
    def test_zelda_by_name(self) -> None:
        assert _is_zelda_mission({"mission": {"name": "zelda-research"}})

    def test_zelda_by_owner(self) -> None:
        assert _is_zelda_mission({"mission": {"owner": "zelda-researcher"}})

    def test_zelda_by_description(self) -> None:
        assert _is_zelda_mission({
            "mission": {"description": "Scan Oracle of Secrets codebase"}
        })

    def test_not_zelda(self) -> None:
        assert not _is_zelda_mission({"mission": {"name": "code-review"}})

    def test_yaze_detected(self) -> None:
        assert _is_zelda_mission({"mission": {"name": "yaze-scan"}})

    def test_mesen_detected(self) -> None:
        assert _is_zelda_mission({"mission": {"name": "mesen-debug"}})


class TestZeldaScanIntegration:
    """Test zelda scanning via ZeldaToolkit end-to-end."""

    def test_non_zelda_not_detected(self) -> None:
        mission = {"mission": {"name": "generic-task", "owner": "bot"}}
        assert not _is_zelda_mission(mission)

    def test_zelda_full_scan(
        self,
        toolkit: ZeldaToolkit,
        fake_oos: Path,
        fake_yaze: Path,
        fake_mesen2: Path,
        fake_disasm: Path,
    ) -> None:
        """Simulate what a zelda mission observe phase would do."""
        scans = {
            "oos": toolkit.scan_oos_codebase(fake_oos),
            "yaze": toolkit.scan_yaze_status(fake_yaze),
            "mesen2": toolkit.scan_mesen2_status(fake_mesen2),
            "cross_reference": toolkit.cross_reference_symbols(fake_oos, fake_disasm),
        }
        assert scans["oos"]["available"] is True
        assert scans["yaze"]["available"] is True
        assert scans["mesen2"]["available"] is True
        assert scans["cross_reference"]["available"] is True

    def test_zelda_scan_missing_paths(
        self,
        toolkit: ZeldaToolkit,
        tmp_path: Path,
    ) -> None:
        scans = {
            "oos": toolkit.scan_oos_codebase(tmp_path / "nope1"),
            "yaze": toolkit.scan_yaze_status(tmp_path / "nope2"),
            "mesen2": toolkit.scan_mesen2_status(tmp_path / "nope3"),
            "cross_reference": toolkit.cross_reference_symbols(tmp_path / "a", tmp_path / "b"),
        }
        assert scans["oos"]["available"] is False
        assert scans["yaze"]["available"] is False
        assert scans["mesen2"]["available"] is False
        assert scans["cross_reference"]["available"] is False


class TestLoadMission:
    def test_load_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(textwrap.dedent("""\
            [mission]
            name = "test-mission"
            owner = "tester"
        """))
        data = load_mission(toml_file)
        assert data["mission"]["name"] == "test-mission"
