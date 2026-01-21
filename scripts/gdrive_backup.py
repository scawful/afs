#!/usr/bin/env python3
"""Automated Google Drive backup for training artifacts.

Backs up training data, models, and evaluation results to Google Drive
for storage management and disaster recovery.

Uses gsync command (if available) or rclone as fallback.

Backup structure:
    Google Drive/AFS_Backups/
    ├── training_data/
    │   ├── majora_v1_YYYYMMDD.tar.gz
    │   ├── veran_v5_YYYYMMDD.tar.gz
    │   └── ...
    ├── models/
    │   ├── majora-v1-lora_YYYYMMDD.tar.gz
    │   ├── veran-v5-lora_YYYYMMDD.tar.gz
    │   └── ...
    ├── evaluations/
    │   └── ...
    └── logs/
        └── backup_YYYYMMDD_HHMMSS.log

Usage:
    # Backup all training data
    python3 scripts/gdrive_backup.py --training-data

    # Backup specific model
    python3 scripts/gdrive_backup.py --model majora-v1-lora --path /workspace/output/majora-v1-lora

    # Backup everything (aggressive)
    python3 scripts/gdrive_backup.py --all

    # Cleanup old backups (keep last 5)
    python3 scripts/gdrive_backup.py --cleanup --keep 5
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path


class GoogleDriveBackup:
    """Manage Google Drive backups for training artifacts."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.backup_root = Path("~/Google Drive/My Drive/AFS_Backups").expanduser()
        self.log_file = self.backup_root / "logs" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Create backup directories
        if not dry_run:
            (self.backup_root / "training_data").mkdir(parents=True, exist_ok=True)
            (self.backup_root / "models").mkdir(parents=True, exist_ok=True)
            (self.backup_root / "evaluations").mkdir(parents=True, exist_ok=True)
            (self.backup_root / "logs").mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        """Log message to file and stdout."""
        print(message)
        if not self.dry_run:
            with open(self.log_file, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")

    def _create_archive(self, source: Path, output_name: str) -> Path:
        """Create compressed tar.gz archive of source directory/file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        archive_name = f"{output_name}_{timestamp}.tar.gz"
        archive_path = Path(f"/tmp/{archive_name}")

        self._log(f"\nCreating archive: {archive_name}")
        self._log(f"  Source: {source}")

        if self.dry_run:
            self._log(f"  [DRY RUN] Would create {archive_path}")
            return archive_path

        # Create tar.gz archive
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source, arcname=source.name)

        size_mb = archive_path.stat().st_size / 1024 / 1024
        self._log(f"  Created: {archive_path} ({size_mb:.1f} MB)")

        return archive_path

    def _upload_to_gdrive(self, archive: Path, category: str) -> bool:
        """Upload archive to Google Drive."""
        dest_path = self.backup_root / category / archive.name

        self._log(f"\nUploading to Google Drive...")
        self._log(f"  Destination: {dest_path}")

        if self.dry_run:
            self._log(f"  [DRY RUN] Would upload to {dest_path}")
            return True

        try:
            # Copy to Google Drive (Google Drive File Stream mounts at ~/Google Drive)
            shutil.copy2(archive, dest_path)
            self._log(f"  ✓ Uploaded successfully")
            return True

        except Exception as e:
            self._log(f"  ✗ Error uploading: {e}")
            return False

    def backup_training_data(self, data_path: Path, name: str):
        """Backup training dataset."""
        if not data_path.exists():
            self._log(f"Error: Training data not found: {data_path}")
            return

        # Create archive
        archive = self._create_archive(data_path, name)

        # Upload to Google Drive
        success = self._upload_to_gdrive(archive, "training_data")

        # Cleanup temp archive
        if not self.dry_run and archive.exists():
            archive.unlink()

        if success:
            self._log(f"\n✓ Backed up training data: {name}")

    def backup_model(self, model_path: Path, name: str):
        """Backup trained model."""
        if not model_path.exists():
            self._log(f"Error: Model not found: {model_path}")
            return

        # Create archive
        archive = self._create_archive(model_path, name)

        # Upload to Google Drive
        success = self._upload_to_gdrive(archive, "models")

        # Cleanup temp archive
        if not self.dry_run and archive.exists():
            archive.unlink()

        if success:
            self._log(f"\n✓ Backed up model: {name}")

    def backup_all_training_data(self):
        """Backup all training datasets from ~/.context/training/"""
        training_root = Path("~/.context/training").expanduser()

        if not training_root.exists():
            self._log("No training data found in ~/.context/training/")
            return

        self._log("=" * 60)
        self._log("Backing Up All Training Data")
        self._log("=" * 60)

        # Find all processed training datasets
        for dataset_dir in training_root.rglob("*_processed"):
            name = dataset_dir.parent.name + "_" + dataset_dir.name
            self.backup_training_data(dataset_dir, name)

        # Also backup raw data
        for raw_file in training_root.rglob("*_raw.jsonl"):
            name = raw_file.stem
            self.backup_training_data(raw_file, name)

    def cleanup_old_backups(self, category: str, keep: int = 5):
        """Remove old backups, keeping only the most recent N."""
        backup_dir = self.backup_root / category

        if not backup_dir.exists():
            return

        self._log(f"\nCleaning up old {category} backups...")
        self._log(f"  Keeping most recent {keep} backups")

        # Get all archives sorted by modification time
        archives = sorted(
            backup_dir.glob("*.tar.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Delete old archives
        for archive in archives[keep:]:
            self._log(f"  Deleting: {archive.name}")
            if not self.dry_run:
                archive.unlink()

        self._log(f"  ✓ Kept {min(keep, len(archives))} backups")

    def list_backups(self):
        """List all backups in Google Drive."""
        print("=" * 60)
        print("Google Drive Backups")
        print("=" * 60)

        for category in ["training_data", "models", "evaluations"]:
            backup_dir = self.backup_root / category

            if not backup_dir.exists():
                continue

            archives = sorted(
                backup_dir.glob("*.tar.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if not archives:
                continue

            print(f"\n{category.replace('_', ' ').title()}:")
            for archive in archives:
                size_mb = archive.stat().st_size / 1024 / 1024
                mtime = datetime.fromtimestamp(archive.stat().st_mtime)
                print(f"  {archive.name} ({size_mb:.1f} MB) - {mtime.strftime('%Y-%m-%d %H:%M')}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated Google Drive backup for training artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--training-data",
        action="store_true",
        help="Backup all training data",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to backup (requires --path)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to model/data to backup",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backup everything (training data, models, etc.)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Cleanup old backups",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="Number of backups to keep during cleanup (default: 5)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all backups",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print operations without executing",
    )

    args = parser.parse_args()

    backup = GoogleDriveBackup(dry_run=args.dry_run)

    # List mode
    if args.list:
        backup.list_backups()
        return 0

    # Cleanup mode
    if args.cleanup:
        for category in ["training_data", "models", "evaluations"]:
            backup.cleanup_old_backups(category, keep=args.keep)
        return 0

    # Backup modes
    if args.all:
        backup.backup_all_training_data()
        # TODO: Add model backup discovery
        return 0

    if args.training_data:
        backup.backup_all_training_data()
        return 0

    if args.model:
        if not args.path:
            print("Error: --model requires --path")
            return 1

        path = args.path.expanduser().resolve()
        backup.backup_model(path, args.model)
        return 0

    print("Error: Specify --training-data, --model, --all, --cleanup, or --list")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
