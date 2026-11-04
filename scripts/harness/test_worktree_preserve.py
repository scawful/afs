"""Focused loss-prevention tests. Uses only disposable repositories and files."""

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sqlite3
import tempfile
import unittest
from unittest import mock


LOADER = importlib.machinery.SourceFileLoader("preserve", str(Path(__file__).with_name("worktree-preserve")))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
p = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(p)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="worktree-preserve-test-")
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "source"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "fixture")
        self.git("config", "user.email", "fixture@invalid")
        (self.root / ".gitignore").write_text("Roms/\ncache/\n")
        (self.root / "tracked.py").write_text("tracked source\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        (self.root / "Roms").mkdir()
        (self.root / "Roms/candidate.sfc").write_bytes(bytes(range(256)) * 32)
        (self.root / "Roms/save.srm").write_bytes(b"unique save\x00")
        (self.root / "Roms/helper.py").write_text("print('helper')\n")
        (self.root / "Roms/helper.py").chmod(0o751)
        (self.root / "Roms/empty").mkdir()
        (self.root / "evidence with spaces.json").write_text('{"pass":true}\n')
        self.external = self.base / "outside.sfc"
        self.external.write_bytes(b"must not read or restore target")
        (self.root / "Roms/link").symlink_to(self.external)
        self.store = self.base / "store"

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def inv(self):
        return p.inventory([self.root])

    def cap(self, data=None):
        return p.capture(data or self.inv(), self.store, 100000, reserve=0)

    def test_real_idle_capture_verify_restore_binary_save_helper_evidence_and_symlink(self):
        # Real ps + lsof check; this process owns no open source descriptor/CWD.
        self.assertEqual(p.active_users(self.root), [])
        result = self.cap()
        receipt = json.loads(Path(result["receipt"]).read_text())
        p.verify_receipt(receipt)
        restored = p.restore(receipt, self.base / "restore")
        target = Path(restored["restored"][0]["restored"])
        for name in ["Roms/candidate.sfc", "Roms/save.srm", "Roms/helper.py", "evidence with spaces.json"]:
            self.assertEqual((target / name).read_bytes(), (self.root / name).read_bytes())
        self.assertEqual(stat.S_IMODE((target / "Roms/helper.py").stat().st_mode), 0o751)
        self.assertTrue((target / "Roms/empty").is_dir())
        self.assertTrue((target / "Roms/link").is_symlink())
        self.assertEqual(os.readlink(target / "Roms/link"), str(self.external))
        self.assertNotIn(b"must not read", b"".join(path.read_bytes() for path in (self.store / "objects").glob("*/*")))
        self.assertFalse((target / "tracked.py").exists())
        self.assertEqual(self.git("status", "--porcelain").stdout, b'?? "evidence with spaces.json"\n')

    def test_refuses_active_process_without_stopping_it(self):
        child = subprocess.Popen(
            ["python3", "-u", "-c", "import sys,time; f=open(sys.argv[1],'rb'); print('ready'); time.sleep(30)",
             str(self.root / "Roms/save.srm")], stdout=subprocess.PIPE, text=True,
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            with self.assertRaisesRegex(p.Hold, "active worktree"):
                self.cap()
            self.assertIsNone(child.poll())
            self.assertFalse(self.store.exists())
        finally:
            child.terminate()
            child.wait(timeout=5)
            child.stdout.close()

    def test_source_change_since_inventory_blocks_before_copy(self):
        data = self.inv()
        (self.root / "Roms/save.srm").write_bytes(b"changed save")
        with self.assertRaisesRegex(p.Hold, "source changed"):
            self.cap(data)
        self.assertFalse(self.store.exists())

    def test_source_change_during_read_never_creates_receipt(self):
        data = self.inv()
        fsync = os.fsync
        changed = False

        def mutate(fd):
            nonlocal changed
            fsync(fd)
            if not changed:
                changed = True
                (self.root / "Roms/candidate.sfc").write_bytes(b"changed during capture")

        with mock.patch.object(p.os, "fsync", side_effect=mutate):
            with self.assertRaisesRegex(p.Hold, "source changed"):
                self.cap(data)
        self.assertEqual(list((self.store / "manifests").glob("*.json")), [])

    def test_capacity_and_byte_budget_fail_closed(self):
        with self.assertRaisesRegex(p.Hold, "byte budget"):
            p.capture(self.inv(), self.store, 1, reserve=0)
        with mock.patch.object(p.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()):
            with self.assertRaisesRegex(p.Hold, "insufficient capacity"):
                self.cap()
        self.assertFalse(self.store.exists())

    def test_missing_lsof_is_hold(self):
        with mock.patch.object(p.shutil, "which", return_value=None):
            with self.assertRaisesRegex(p.Hold, "lsof"):
                self.cap()

    def test_socket_or_fifo_and_nested_git_are_hold(self):
        os.mkfifo(self.root / "Roms/fifo")
        with self.assertRaisesRegex(p.Hold, "FIFO"):
            self.inv()
        (self.root / "Roms/fifo").unlink()
        (self.root / "Roms/nested").mkdir()
        (self.root / "Roms/nested/.git").mkdir()
        with self.assertRaisesRegex(p.Hold, "embedded Git"):
            self.inv()

    def test_store_inside_source_and_symlink_store_are_hold(self):
        with self.assertRaisesRegex(p.Hold, "outside every"):
            p.capture(self.inv(), self.root / "backup", 100000, reserve=0)
        self.store.symlink_to(self.base)
        with self.assertRaisesRegex(p.Hold, "symlink path component"):
            self.cap()

    def test_corruption_and_overwrite_and_traversal_block(self):
        result = self.cap()
        receipt = json.loads(Path(result["receipt"]).read_text())
        destination = self.base / "already-exists"
        destination.mkdir()
        with self.assertRaisesRegex(p.Hold, "must be absent"):
            p.restore(receipt, destination)
        for bad in ["../outside", "/absolute", ".git/config", "sub/../../outside"]:
            with self.assertRaises(p.Hold):
                p.safe_relative(bad)
        first = next((self.store / "objects").glob("*/*"))
        first.write_bytes(b"corrupt")
        with self.assertRaisesRegex(p.Hold, "hash mismatch"):
            p.verify_receipt(receipt)

    def test_inventory_and_receipt_tampering_block(self):
        data = self.inv()
        data["worktrees"][0]["bytes"] = 0
        with self.assertRaisesRegex(p.Hold, "checksum"):
            self.cap(data)
        result = self.cap()
        receipt = json.loads(Path(result["receipt"]).read_text())
        receipt["bytes"] = 0
        with self.assertRaisesRegex(p.Hold, "checksum"):
            p.verify_receipt(receipt)

    def chat_database(self):
        database = self.base / "state.sqlite"
        connection = sqlite3.connect(database)
        connection.executescript("CREATE TABLE threads(id TEXT,cwd TEXT); CREATE TABLE thread_attachments(thread_id TEXT,attachment_type TEXT,identity_key TEXT,payload TEXT);")
        connection.execute("INSERT INTO threads VALUES(?,?)", ("fixture-chat", str(self.root)))
        connection.execute("INSERT INTO thread_attachments VALUES(?,?,?,?)",
                           ("fixture-chat", "worktree", str(self.root), json.dumps({"root": str(self.root)})))
        connection.commit()
        connection.close()
        return database

    def test_fresh_chat_gate_and_changed_attachments_new_missing_or_changed_files(self):
        database = self.chat_database()
        self.assertEqual(p.attachment_roots("fixture-chat", database), [str(self.root)])
        data = p.inventory([self.root], "fixture-chat", str(database))
        result = self.cap(data)
        receipt = json.loads(Path(result["receipt"]).read_text())
        self.assertEqual(p.check_archive(receipt)["thread"], "fixture-chat")
        (self.root / "Roms/new-evidence").write_bytes(b"new")
        with self.assertRaisesRegex(p.Hold, "source changed"):
            p.check_archive(receipt)
        (self.root / "Roms/new-evidence").unlink()
        # Directory timestamps change too; use a fresh admitted capture.
        result = self.cap(p.inventory([self.root], "fixture-chat", str(database)))
        receipt = json.loads(Path(result["receipt"]).read_text())
        (self.root / "Roms/save.srm").unlink()
        with self.assertRaisesRegex(p.Hold, "source changed"):
            p.check_archive(receipt)
        connection = sqlite3.connect(database)
        connection.execute("DELETE FROM thread_attachments")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(p.Hold, "inventory changed"):
            p.check_archive(receipt)

    def test_archive_gate_refuses_capture_without_chat_inventory(self):
        result = self.cap()
        receipt = json.loads(Path(result["receipt"]).read_text())
        with self.assertRaisesRegex(p.Hold, "attachment inventory"):
            p.check_archive(receipt)

    def test_xattr_binary_round_trip(self):
        path = self.root / "Roms/save.srm"
        key = "com.scawful.worktree-preserve-fixture" if p.sys.platform == "darwin" else "user.fixture"
        value = b"binary\x00resource\xff"
        if p.sys.platform == "darwin":
            p.darwin_xattr(path, key, value)
        else:
            os.setxattr(path, key, value)
        result = self.cap()
        receipt = json.loads(Path(result["receipt"]).read_text())
        restored = p.restore(receipt, self.base / "xattr-restore")
        target = Path(restored["restored"][0]["restored"]) / "Roms/save.srm"
        self.assertEqual(p.attrs(path), p.attrs(target))

    def test_symlink_parent_swap_cannot_open_external_file(self):
        moved = self.root / "Roms-moved"
        (self.root / "Roms").rename(moved)
        (self.root / "Roms").symlink_to(self.base)
        with self.assertRaises(OSError):
            p.open_source(self.root, "Roms/outside.sfc")

    def test_tracked_path_replaced_by_directory_retains_directory_metadata(self):
        path = self.root / "tracked.py"
        path.unlink()
        path.mkdir(mode=0o711)
        result = self.cap()
        receipt = json.loads(Path(result["receipt"]).read_text())
        restored = p.restore(receipt, self.base / "replaced-directory")
        target = Path(restored["restored"][0]["restored"]) / "tracked.py"
        self.assertTrue(target.is_dir())
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o711)

    def test_lsof_permission_warning_is_hold(self):
        actual_run = p.run

        def warning(args, **kwargs):
            if str(args[0]).endswith("lsof"):
                return actual_run(["python3", "-c", "import sys; print('Permission denied',file=sys.stderr); sys.exit(1)"], **kwargs)
            return actual_run(args, **kwargs)

        with mock.patch.object(p, "run", side_effect=warning):
            with self.assertRaisesRegex(p.Hold, "Permission denied"):
                self.cap()


if __name__ == "__main__":
    unittest.main(verbosity=2)
