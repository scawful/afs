#!/usr/bin/env python3
"""
Health check and verification for deployed models.

This script validates:
- Model files exist and are readable
- Model files have correct format (GGUF)
- Models are accessible via LMStudio API
- Models respond correctly to test prompts
- Response quality metrics (latency, token count, etc.)
- System resource availability (disk space, memory)

Usage:
    # Check all models
    python3 health_check.py --all

    # Check specific model
    python3 health_check.py --model majora

    # Detailed diagnostics
    python3 health_check.py --all --detailed

    # Check only file integrity
    python3 health_check.py --files-only

    # Check only API endpoints
    python3 health_check.py --api-only

    # Generate JSON report
    python3 health_check.py --all --json
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml


@dataclass
class FileHealthCheck:
    """File health check result."""
    exists: bool
    path: str
    size_bytes: int = 0
    size_gb: float = 0.0
    is_gguf: bool = False
    readable: bool = False
    error: Optional[str] = None


@dataclass
class APIHealthCheck:
    """API health check result."""
    reachable: bool
    port: int
    response_time_ms: float = 0.0
    model_loaded: bool = False
    error: Optional[str] = None


@dataclass
class ModelTest:
    """Model test result."""
    passed: bool
    model_name: str
    prompt: str
    response: str = ""
    response_length: int = 0
    latency_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    error: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Complete health check result for a model."""
    timestamp: str
    model_name: str
    status: str  # healthy, degraded, unhealthy
    file_check: FileHealthCheck
    api_check: APIHealthCheck
    model_test: Optional[ModelTest] = None
    warnings: List[str] = None
    system_resources: Dict = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.system_resources is None:
            self.system_resources = {}


class ModelHealthChecker:
    """Health check and verification for deployed models."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize health checker."""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.deployment_config = self.config.get("deployment", {})
        self.healthcheck_config = self.config.get("health_check", {})

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("HealthChecker")
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
            self.logger.warning(f"Config not found: {config_path}")
            return {}

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _check_file(self, model_name: str, model_path: str) -> FileHealthCheck:
        """Check if model file exists and is valid."""
        self.logger.info(f"Checking file: {model_name}...")

        path = Path(model_path).expanduser()
        result = FileHealthCheck(exists=path.exists(), path=str(path))

        if not path.exists():
            result.error = f"File not found: {path}"
            return result

        try:
            # Check readability
            with open(path, "rb") as f:
                # Read first bytes to check for GGUF magic number
                header = f.read(4)
                result.readable = True
                result.is_gguf = header == b"GGUF"

            # Get file size
            result.size_bytes = path.stat().st_size
            result.size_gb = result.size_bytes / (1024**3)

            if not result.is_gguf:
                result.error = "File is not in GGUF format (invalid header)"
                return result

        except Exception as e:
            result.error = str(e)
            result.readable = False

        return result

    def _check_api(self, model_name: str, port: int) -> APIHealthCheck:
        """Check if API endpoint is reachable."""
        self.logger.info(f"Checking API for {model_name} on port {port}...")

        result = APIHealthCheck(reachable=False, port=port)
        endpoint = f"http://localhost:{port}/health"

        try:
            start = time.time()
            response = requests.get(
                endpoint,
                timeout=self.healthcheck_config.get("timeout", 10)
            )
            elapsed_ms = (time.time() - start) * 1000

            result.reachable = response.status_code == 200
            result.response_time_ms = elapsed_ms

            if result.reachable:
                result.model_loaded = True
            else:
                result.error = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            result.error = "Request timeout"
        except requests.exceptions.ConnectionError:
            result.error = f"Connection refused on port {port}"
        except Exception as e:
            result.error = str(e)

        return result

    def _test_model(
        self,
        model_name: str,
        port: int,
        prompt: Optional[str] = None
    ) -> Optional[ModelTest]:
        """Test model response quality."""
        if not prompt:
            test_prompts = self.healthcheck_config.get("test_prompts", {})
            prompt = test_prompts.get(model_name, "Hello")

        self.logger.info(f"Testing model response: {model_name}...")

        endpoint = f"http://localhost:{port}/chat"

        try:
            start = time.time()
            response = requests.post(
                endpoint,
                json={"prompt": prompt},
                timeout=self.healthcheck_config.get("timeout", 30)
            )
            elapsed_ms = (time.time() - start) * 1000

            if response.status_code != 200:
                return ModelTest(
                    passed=False,
                    model_name=model_name,
                    prompt=prompt,
                    error=f"HTTP {response.status_code}"
                )

            data = response.json()
            response_text = data.get("response", "")

            # Check response quality
            min_length = self.healthcheck_config.get(
                "min_response_length", 50
            )
            passed = len(response_text) >= min_length

            # Estimate tokens (rough: ~4 chars per token)
            tokens = len(response_text) // 4

            return ModelTest(
                passed=passed,
                model_name=model_name,
                prompt=prompt,
                response=response_text[:100],  # Truncate for display
                response_length=len(response_text),
                latency_ms=elapsed_ms,
                tokens_generated=tokens,
                tokens_per_second=tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            )

        except Exception as e:
            return ModelTest(
                passed=False,
                model_name=model_name,
                prompt=prompt,
                error=str(e)
            )

    def _check_system_resources(self) -> Dict:
        """Check system resource availability."""
        resources = {}

        # Disk space
        try:
            usage = shutil.disk_usage("/")
            resources["disk_total_gb"] = usage.total / (1024**3)
            resources["disk_used_gb"] = usage.used / (1024**3)
            resources["disk_free_gb"] = usage.free / (1024**3)
            resources["disk_usage_percent"] = (usage.used / usage.total) * 100
        except Exception as e:
            resources["disk_error"] = str(e)

        # Memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            resources["memory_total_gb"] = mem.total / (1024**3)
            resources["memory_available_gb"] = mem.available / (1024**3)
            resources["memory_used_gb"] = mem.used / (1024**3)
            resources["memory_percent"] = mem.percent
        except ImportError:
            self.logger.debug("psutil not available, skipping memory check")
        except Exception as e:
            resources["memory_error"] = str(e)

        # GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used",
                 "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                resources["gpu_memory_total_gb"] = int(parts[0]) / 1024
                resources["gpu_memory_used_gb"] = int(parts[1]) / 1024
        except Exception:
            pass  # GPU not available

        return resources

    def check_model(
        self,
        model_name: str,
        model_path: str,
        port: int,
        test: bool = True,
        detailed: bool = False
    ) -> HealthCheckResult:
        """Check health of a single model."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Checking: {model_name}")
        self.logger.info(f"{'='*60}")

        # File check
        file_check = self._check_file(model_name, model_path)

        # API check
        api_check = self._check_api(model_name, port)

        # Model test
        model_test = None
        if test and api_check.reachable:
            model_test = self._test_model(model_name, port)

        # System resources
        system_resources = self._check_system_resources()

        # Determine overall status
        warnings = []
        if not file_check.exists:
            status = "unhealthy"
            warnings.append("Model file not found")
        elif not file_check.is_gguf:
            status = "unhealthy"
            warnings.append("Model file is not GGUF format")
        elif not api_check.reachable:
            status = "degraded"
            warnings.append("API endpoint not reachable")
        elif model_test and not model_test.passed:
            status = "degraded"
            warnings.append("Model test failed")
        else:
            status = "healthy"

        # Additional warnings
        if system_resources.get("disk_usage_percent", 0) > 90:
            warnings.append("Disk usage above 90%")
        if system_resources.get("memory_percent", 0) > 90:
            warnings.append("Memory usage above 90%")

        result = HealthCheckResult(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            status=status,
            file_check=file_check,
            api_check=api_check,
            model_test=model_test,
            warnings=warnings,
            system_resources=system_resources
        )

        # Log results
        self._log_result(result, detailed=detailed)

        return result

    def _log_result(self, result: HealthCheckResult, detailed: bool = False):
        """Log health check result."""
        status_symbol = {
            "healthy": "✓",
            "degraded": "⚠",
            "unhealthy": "✗"
        }.get(result.status, "?")

        self.logger.info(f"{status_symbol} {result.model_name}: {result.status}")

        if result.file_check.error:
            self.logger.info(f"  File: {result.file_check.error}")
        else:
            self.logger.info(f"  File: {result.file_check.size_gb:.2f} GB "
                           f"({'GGUF' if result.file_check.is_gguf else 'unknown'})")

        if result.api_check.error:
            self.logger.info(f"  API: {result.api_check.error}")
        else:
            self.logger.info(f"  API: port {result.api_check.port} "
                           f"({result.api_check.response_time_ms:.0f}ms)")

        if result.model_test:
            if result.model_test.error:
                self.logger.info(f"  Test: {result.model_test.error}")
            else:
                self.logger.info(f"  Test: {result.model_test.tokens_per_second:.1f} tok/s "
                               f"({result.model_test.latency_ms:.0f}ms)")

        if result.warnings:
            for warning in result.warnings:
                self.logger.warning(f"  ⚠ {warning}")

        if detailed:
            self.logger.info(f"  Resources:")
            self.logger.info(f"    Disk: {result.system_resources.get('disk_usage_percent', 0):.1f}%")
            if 'memory_percent' in result.system_resources:
                self.logger.info(f"    Memory: {result.system_resources['memory_percent']:.1f}%")

    def check_all(
        self,
        test: bool = True,
        detailed: bool = False,
        json_output: bool = False
    ) -> Dict[str, HealthCheckResult]:
        """Check all configured models."""
        results = {}

        deployment_models = self.deployment_config.get("models", {})

        for model_name, model_config in deployment_models.items():
            port = model_config.get("port")
            if not port:
                self.logger.warning(f"No port configured for {model_name}")
                continue

            # Try to find model file
            models_dir = Path(
                self.deployment_config.get("lmstudio", {})
                .get("models_dir", "~/.lmstudio/models")
            ).expanduser()

            model_path = None
            for pattern in [f"{model_name}*.gguf", f"{model_name}*"]:
                matches = list(models_dir.glob(pattern))
                if matches:
                    model_path = matches[0]
                    break

            if not model_path:
                self.logger.warning(f"Could not find model file for {model_name}")
                continue

            result = self.check_model(
                model_name=model_name,
                model_path=str(model_path),
                port=port,
                test=test,
                detailed=detailed
            )

            results[model_name] = result

        # Summary
        self._log_summary(results)

        if json_output:
            return self._results_to_json(results)

        return results

    def _log_summary(self, results: Dict[str, HealthCheckResult]):
        """Log summary of all checks."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("SUMMARY")
        self.logger.info(f"{'='*60}")

        healthy = sum(1 for r in results.values() if r.status == "healthy")
        degraded = sum(1 for r in results.values() if r.status == "degraded")
        unhealthy = sum(1 for r in results.values() if r.status == "unhealthy")
        total = len(results)

        self.logger.info(f"✓ Healthy: {healthy}/{total}")
        self.logger.info(f"⚠ Degraded: {degraded}/{total}")
        self.logger.info(f"✗ Unhealthy: {unhealthy}/{total}")

        if unhealthy > 0:
            self.logger.error("Some models are unhealthy. Review logs above.")
        elif degraded > 0:
            self.logger.warning("Some models are degraded. Review warnings above.")
        else:
            self.logger.info("All models are healthy!")

    def _results_to_json(
        self,
        results: Dict[str, HealthCheckResult]
    ) -> Dict:
        """Convert results to JSON-serializable format."""
        return {
            name: {
                "timestamp": result.timestamp,
                "status": result.status,
                "file": asdict(result.file_check),
                "api": asdict(result.api_check),
                "test": asdict(result.model_test) if result.model_test else None,
                "warnings": result.warnings,
                "resources": result.system_resources
            }
            for name, result in results.items()
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Health check for deployed models"
    )
    parser.add_argument("--model", type=str,
                       help="Check specific model")
    parser.add_argument("--all", action="store_true",
                       help="Check all models")
    parser.add_argument("--files-only", action="store_true",
                       help="Only check file integrity")
    parser.add_argument("--api-only", action="store_true",
                       help="Only check API endpoints")
    parser.add_argument("--no-test", action="store_true",
                       help="Skip model response tests")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed diagnostics")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON report")
    parser.add_argument("--config", type=str,
                       help="Path to deployment config")
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    checker = ModelHealthChecker(config_path=args.config)

    if args.all:
        results = checker.check_all(
            test=not args.no_test,
            detailed=args.detailed,
            json_output=args.json
        )

        if args.json:
            print(json.dumps(results, indent=2))

        # Exit with appropriate code
        unhealthy = sum(1 for r in results.values()
                       if isinstance(r, HealthCheckResult) and r.status == "unhealthy")
        sys.exit(0 if unhealthy == 0 else 1)

    elif args.model:
        print(f"Checking model: {args.model}")
        print("Feature not yet implemented. Use --all for now.")
        sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
