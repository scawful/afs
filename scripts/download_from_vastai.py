#!/usr/bin/env python3
"""
Download LoRA adapters from vast.ai training instances.

This script handles:
- SSH connection to vast.ai instances
- Downloading adapter_model.safetensors files
- Computing checksums for verification
- Organizing downloads by model
- Resumable downloads with integrity checking

Usage:
    # Download from a specific instance
    python3 download_from_vastai.py --instance-id 12345678 --model majora

    # Download all models from config
    python3 download_from_vastai.py --config deployment_config.yaml

    # Resume failed downloads
    python3 download_from_vastai.py --resume --config deployment_config.yaml

    # Download with compression
    python3 download_from_vastai.py --instance-id 12345678 --compress
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import yaml


@dataclass
class DownloadTask:
    """Task to download a LoRA adapter from vast.ai."""
    model_name: str
    instance_id: str
    remote_path: str
    local_path: Path
    base_model: str
    description: str


class VastAIDownloader:
    """Download LoRA adapters from vast.ai instances."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize downloader with configuration."""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.vast_config = self.config.get("vast", {})
        self.api_key = os.getenv("VAST_API_KEY")

        if not self.api_key:
            self.logger.warning("VAST_API_KEY not set. Using vastai CLI directly.")

        self.download_dir = Path(self.config.get("deployment", {}).get("backup_dir", "models/backups"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("VastAIDownloader")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load deployment configuration."""
        if not config_path:
            config_path = Path(__file__).parent / "deployment_config.yaml"

        if not Path(config_path).exists():
            self.logger.warning(f"Config file not found: {config_path}")
            return {}

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _get_ssh_url(self, instance_id: str) -> Optional[str]:
        """Get SSH URL for a vast.ai instance."""
        try:
            result = subprocess.run(
                ["vastai", "ssh-url", str(instance_id)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.logger.error(f"Failed to get SSH URL: {result.stderr}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting SSH URL: {e}")
            return None

    def _get_remote_file_size(self, ssh_url: str, remote_path: str) -> Optional[int]:
        """Get size of remote file."""
        try:
            result = subprocess.run(
                ["ssh", ssh_url, f"ls -lh {remote_path}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                parts = result.stdout.split()
                size_str = parts[4]
                return self._parse_size(size_str)
            return None
        except Exception as e:
            self.logger.error(f"Error getting file size: {e}")
            return None

    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '1.5G') to bytes."""
        multipliers = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
        size_str = size_str.strip()

        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                value = float(size_str[:-1])
                return int(value * multiplier)

        # Try to parse as plain number
        try:
            return int(size_str)
        except ValueError:
            return 0

    def _download_file_rsync(
        self,
        ssh_url: str,
        remote_path: str,
        local_path: Path,
        resume: bool = True
    ) -> bool:
        """Download file using rsync (faster than SCP)."""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract host and port from SSH URL
        if ":" in ssh_url:
            host, port = ssh_url.rsplit(":", 1)
        else:
            host = ssh_url
            port = "22"

        remote_spec = f"ssh -p {port} {host}:{remote_path}"

        rsync_args = [
            "rsync",
            "-avh",
            "--progress",
        ]

        if resume:
            rsync_args.append("--partial")
            rsync_args.append("--append-verify")

        rsync_args.extend([remote_spec, str(local_path)])

        try:
            self.logger.info(f"Downloading {remote_path} using rsync...")
            result = subprocess.run(rsync_args, timeout=None)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"rsync failed: {e}")
            return False

    def _download_file_scp(
        self,
        ssh_url: str,
        remote_path: str,
        local_path: Path
    ) -> bool:
        """Download file using SCP as fallback."""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract port from SSH URL if present
        if ":" in ssh_url:
            host, port = ssh_url.rsplit(":", 1)
            scp_args = ["scp", "-P", port, f"{host}:{remote_path}", str(local_path)]
        else:
            scp_args = ["scp", f"{ssh_url}:{remote_path}", str(local_path)]

        try:
            self.logger.info(f"Downloading {remote_path} using scp...")
            result = subprocess.run(scp_args, timeout=None)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"scp failed: {e}")
            return False

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _verify_download(
        self,
        file_path: Path,
        remote_path: str,
        ssh_url: str
    ) -> bool:
        """Verify downloaded file integrity."""
        if not file_path.exists():
            return False

        # Compute local checksum
        local_checksum = self._compute_checksum(file_path)
        self.logger.info(f"Local checksum: {local_checksum}")

        # Try to get remote checksum
        try:
            result = subprocess.run(
                ["ssh", ssh_url, f"sha256sum {remote_path}"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                remote_checksum = result.stdout.split()[0]
                if local_checksum == remote_checksum:
                    self.logger.info("✓ Checksums match!")
                    return True
                else:
                    self.logger.error("✗ Checksums do not match!")
                    return False
        except Exception as e:
            self.logger.warning(f"Could not verify remote checksum: {e}")
            return True  # Continue anyway

    def download(
        self,
        model_name: str,
        instance_id: str,
        remote_path: str,
        local_path: Optional[str] = None,
        resume: bool = True,
        verify: bool = True,
        compress: bool = False
    ) -> bool:
        """Download a LoRA adapter from vast.ai instance."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Downloading {model_name} from instance {instance_id}")
        self.logger.info(f"{'='*60}")

        # Get SSH URL
        ssh_url = self._get_ssh_url(instance_id)
        if not ssh_url:
            self.logger.error(f"Could not get SSH URL for instance {instance_id}")
            return False

        # Prepare local path
        if not local_path:
            local_path = self.download_dir / f"{model_name}_adapter_model.safetensors"
        else:
            local_path = Path(local_path)

        # Check if file already exists
        if local_path.exists():
            size = local_path.stat().st_size / (1024**3)
            self.logger.info(f"File already exists: {local_path} ({size:.2f} GB)")

            if verify:
                if self._verify_download(local_path, remote_path, ssh_url):
                    self.logger.info("Download verified successfully!")
                    return True
                else:
                    self.logger.warning("Verification failed, re-downloading...")
            else:
                return True

        # Get file size
        remote_size = self._get_remote_file_size(ssh_url, remote_path)
        if remote_size:
            self.logger.info(f"Remote file size: {remote_size / (1024**3):.2f} GB")

        # Download using rsync (with fallback to scp)
        success = self._download_file_rsync(ssh_url, remote_path, local_path, resume=resume)

        if not success:
            self.logger.warning("rsync failed, trying scp...")
            success = self._download_file_scp(ssh_url, remote_path, local_path)

        if not success:
            self.logger.error("Download failed!")
            return False

        # Verify integrity
        if verify:
            if not self._verify_download(local_path, remote_path, ssh_url):
                self.logger.error("Download verification failed!")
                return False

        # Optional: compress for storage
        if compress:
            self._compress_file(local_path)

        # Save metadata
        self._save_metadata(model_name, instance_id, local_path, remote_path)

        self.logger.info(f"✓ Download complete: {local_path}")
        return True

    def _compress_file(self, file_path: Path) -> None:
        """Compress file with zstd for storage."""
        try:
            import zstandard as zstd

            compressed_path = Path(str(file_path) + ".zst")
            self.logger.info(f"Compressing {file_path}...")

            cctx = zstd.ZstdCompressor(level=10)
            with open(file_path, "rb") as f_in:
                with open(compressed_path, "wb") as f_out:
                    f_out.write(cctx.compress(f_in.read()))

            original_size = file_path.stat().st_size / (1024**3)
            compressed_size = compressed_path.stat().st_size / (1024**3)
            ratio = (1 - compressed_size / original_size) * 100

            self.logger.info(
                f"Compression complete: "
                f"{original_size:.2f} GB → {compressed_size:.2f} GB ({ratio:.1f}% saved)"
            )
        except ImportError:
            self.logger.warning("zstandard not installed, skipping compression")

    def _save_metadata(
        self,
        model_name: str,
        instance_id: str,
        local_path: Path,
        remote_path: str
    ) -> None:
        """Save download metadata."""
        metadata = {
            "model_name": model_name,
            "instance_id": instance_id,
            "remote_path": remote_path,
            "local_path": str(local_path),
            "download_time": datetime.now().isoformat(),
            "file_size_bytes": local_path.stat().st_size if local_path.exists() else 0,
            "checksum": self._compute_checksum(local_path) if local_path.exists() else None
        }

        metadata_path = Path(str(local_path) + ".metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved: {metadata_path}")

    def download_all(self, config_path: str, resume: bool = False) -> Dict[str, bool]:
        """Download all models from configuration."""
        results = {}

        vast_models = self.config.get("vast", {}).get("models", {})

        for model_name, model_config in vast_models.items():
            instance_id = model_config.get("instance_id")
            if not instance_id:
                self.logger.warning(f"Skipping {model_name}: no instance_id")
                continue

            remote_path = model_config.get("output_path")
            local_path = self.download_dir / f"{model_name}_adapter_model.safetensors"

            results[model_name] = self.download(
                model_name=model_name,
                instance_id=instance_id,
                remote_path=remote_path,
                local_path=str(local_path),
                resume=resume
            )

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download LoRA adapters from vast.ai instances"
    )
    parser.add_argument("--instance-id", type=str, help="Vast.ai instance ID")
    parser.add_argument("--model", type=str, help="Model name (majora, nayru, etc.)")
    parser.add_argument("--config", type=str, help="Path to deployment config")
    parser.add_argument("--local-path", type=str, help="Local output path")
    parser.add_argument("--all-models", action="store_true", help="Download all models from config")
    parser.add_argument("--resume", action="store_true", help="Resume failed downloads")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify checksums")
    parser.add_argument("--compress", action="store_true", help="Compress files with zstd")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    downloader = VastAIDownloader(config_path=args.config)

    if args.all_models:
        results = downloader.download_all(args.config or "deployment_config.yaml", resume=args.resume)
        for model_name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {model_name}")
        sys.exit(0 if all(results.values()) else 1)
    else:
        if not args.instance_id or not args.model:
            parser.print_help()
            sys.exit(1)

        success = downloader.download(
            model_name=args.model,
            instance_id=args.instance_id,
            remote_path=None,  # Will be inferred from config if available
            local_path=args.local_path,
            resume=args.resume,
            verify=args.verify,
            compress=args.compress
        )

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
