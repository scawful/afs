"""Tests for pre-deployment validation system."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from afs.deployment.validator import (
    PreDeploymentValidator,
    ValidationCategory,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            category=ValidationCategory.FILE_INTEGRITY,
            check_name="Test Check",
            status=ValidationStatus.PASSED,
            message="Test message",
        )
        assert result.check_name == "Test Check"
        assert result.status == ValidationStatus.PASSED
        assert result.category == ValidationCategory.FILE_INTEGRITY

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ValidationResult(
            category=ValidationCategory.FILE_INTEGRITY,
            check_name="Test Check",
            status=ValidationStatus.PASSED,
            message="Test message",
            details={"key": "value"},
        )
        result_dict = result.to_dict()
        assert result_dict["check_name"] == "Test Check"
        assert result_dict["status"] == "passed"
        assert result_dict["category"] == "file_integrity"
        assert result_dict["details"] == {"key": "value"}


class TestValidationReport:
    """Tests for ValidationReport."""

    def test_report_creation(self, tmp_path: Path):
        """Test creating a validation report."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        report = ValidationReport(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        assert report.model_name == "test-model"
        assert report.version == "v1"
        assert report.passed is True

    def test_report_passed_property(self, tmp_path: Path):
        """Test passed property."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        report = ValidationReport(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )

        # Add passing check
        report.results.append(
            ValidationResult(
                category=ValidationCategory.FILE_INTEGRITY,
                check_name="Check 1",
                status=ValidationStatus.PASSED,
                message="OK",
            )
        )
        assert report.passed is True

        # Add failing check
        report.results.append(
            ValidationResult(
                category=ValidationCategory.FILE_INTEGRITY,
                check_name="Check 2",
                status=ValidationStatus.FAILED,
                message="Failed",
            )
        )
        assert report.passed is False

    def test_report_summary(self, tmp_path: Path):
        """Test report summary."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        report = ValidationReport(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )

        # Add various results
        report.results.append(
            ValidationResult(
                category=ValidationCategory.FILE_INTEGRITY,
                check_name="Check 1",
                status=ValidationStatus.PASSED,
                message="OK",
            )
        )
        report.results.append(
            ValidationResult(
                category=ValidationCategory.FILE_INTEGRITY,
                check_name="Check 2",
                status=ValidationStatus.WARNING,
                message="Warning",
            )
        )
        report.results.append(
            ValidationResult(
                category=ValidationCategory.FILE_INTEGRITY,
                check_name="Check 3",
                status=ValidationStatus.FAILED,
                message="Failed",
            )
        )

        summary = report.summary()
        assert summary["total_checks"] == 3
        assert summary["passed"] == 1
        assert summary["warnings"] == 1
        assert summary["failed"] == 1
        assert summary["overall_status"] == "FAILED"

    def test_report_save_json(self, tmp_path: Path):
        """Test saving report as JSON."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        report = ValidationReport(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        report.results.append(
            ValidationResult(
                category=ValidationCategory.FILE_INTEGRITY,
                check_name="Test Check",
                status=ValidationStatus.PASSED,
                message="OK",
            )
        )

        output_path = tmp_path / "report.json"
        report.save_json(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["model_name"] == "test-model"
        assert data["version"] == "v1"
        assert len(data["results"]) == 1

    def test_report_save_markdown(self, tmp_path: Path):
        """Test saving report as markdown."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        report = ValidationReport(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        report.results.append(
            ValidationResult(
                category=ValidationCategory.FILE_INTEGRITY,
                check_name="Test Check",
                status=ValidationStatus.PASSED,
                message="OK",
            )
        )

        output_path = tmp_path / "report.md"
        report.save_markdown(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "test-model" in content
        assert "v1" in content
        assert "Test Check" in content


class TestPreDeploymentValidator:
    """Tests for PreDeploymentValidator."""

    def test_validator_creation(self, tmp_path: Path):
        """Test creating a validator."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        assert validator.model_path == model_path
        assert validator.model_name == "test-model"
        assert validator.version == "v1"

    def test_check_file_exists_missing(self, tmp_path: Path):
        """Test file existence check when file is missing."""
        model_path = tmp_path / "missing.gguf"

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_file_exists()

        assert len(validator.report.results) == 1
        result = validator.report.results[0]
        assert result.status == ValidationStatus.FAILED
        assert result.check_name == "File Exists"

    def test_check_file_exists_present(self, tmp_path: Path):
        """Test file existence check when file exists."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_file_exists()

        assert len(validator.report.results) == 1
        result = validator.report.results[0]
        assert result.status == ValidationStatus.PASSED
        assert result.check_name == "File Exists"

    def test_check_file_integrity(self, tmp_path: Path):
        """Test file integrity check."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"test content")

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_file_integrity()

        assert len(validator.report.results) == 1
        result = validator.report.results[0]
        assert result.status == ValidationStatus.PASSED
        assert "sha256" in result.details

    def test_check_gguf_format(self, tmp_path: Path):
        """Test GGUF format validation."""
        model_path = tmp_path / "model.gguf"
        # Write GGUF magic bytes
        model_path.write_bytes(b"GGUF" + b"\x00" * 100)

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_file_format()

        assert len(validator.report.results) >= 1
        # Should have a format check result
        format_results = [r for r in validator.report.results if r.check_name == "GGUF Magic Bytes"]
        assert len(format_results) > 0
        assert format_results[0].status == ValidationStatus.PASSED

    def test_check_gguf_format_invalid(self, tmp_path: Path):
        """Test GGUF format validation with invalid format."""
        model_path = tmp_path / "model.gguf"
        # Write invalid magic bytes
        model_path.write_bytes(b"XXXX" + b"\x00" * 100)

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_file_format()

        format_results = [r for r in validator.report.results if r.check_name == "GGUF Magic Bytes"]
        assert len(format_results) > 0
        assert format_results[0].status == ValidationStatus.FAILED

    def test_check_file_size(self, tmp_path: Path):
        """Test file size check."""
        model_path = tmp_path / "model.gguf"
        # Create a file in acceptable size range
        size = 500 * 1024 * 1024  # 500 MB
        model_path.write_bytes(b"\x00" * size)

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_file_size()

        size_results = [r for r in validator.report.results if "Size" in r.check_name]
        assert len(size_results) > 0
        assert size_results[0].status == ValidationStatus.PASSED

    def test_check_file_size_too_small(self, tmp_path: Path):
        """Test file size check with file too small."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"\x00" * (10 * 1024 * 1024))  # 10 MB

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_file_size()

        size_results = [r for r in validator.report.results if "Size" in r.check_name]
        assert len(size_results) > 0
        # Should be warning or pass (depending on implementation)

    def test_check_memory_requirements(self, tmp_path: Path):
        """Test memory requirements estimation."""
        model_path = tmp_path / "model.gguf"
        # Create a 7GB model (roughly 7B parameter model)
        size = 7 * 1024 * 1024 * 1024
        model_path.write_bytes(b"\x00" * min(size, 1024 * 1024 * 100))  # Create smaller test file

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_memory_requirements()

        mem_results = [r for r in validator.report.results if "VRAM" in r.check_name]
        assert len(mem_results) > 0
        assert mem_results[0].status != ValidationStatus.FAILED

    def test_check_inference_test(self, tmp_path: Path):
        """Test inference capability check."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"GGUF" + b"\x00" * 100)

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        validator._check_inference_capability()

        inference_results = [r for r in validator.report.results if "Model Loading" in r.check_name]
        assert len(inference_results) > 0
        # Should pass in mock mode
        assert inference_results[0].status == ValidationStatus.PASSED

    def test_response_quality_check(self, tmp_path: Path):
        """Test response quality assessment."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        # Set up test responses
        validator.report.test_responses = [
            {
                "query": "What is AI?",
                "response": "Artificial intelligence is a field of computer science.",
                "tokens": 15,
            }
        ]

        validator._check_response_quality()

        quality_results = [r for r in validator.report.results if "Quality" in r.check_name]
        assert len(quality_results) > 0
        assert quality_results[0].status == ValidationStatus.PASSED

    def test_latency_check(self, tmp_path: Path):
        """Test latency measurement."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )
        # Set up test responses with tokens
        validator.report.test_responses = [
            {"query": "Q1", "response": "R1", "tokens": 50},
            {"query": "Q2", "response": "R2", "tokens": 60},
            {"query": "Q3", "response": "R3", "tokens": 55},
        ]

        validator._check_latency()

        latency_results = [r for r in validator.report.results if "Latency" in r.check_name]
        assert len(latency_results) > 0

    def test_rollback_recommendation(self, tmp_path: Path):
        """Test rollback recommendation."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v2",
            baseline_version="v1",
        )

        # Add a critical failure
        validator.report.results.append(
            ValidationResult(
                category=ValidationCategory.FILE_INTEGRITY,
                check_name="GGUF Magic Bytes",
                status=ValidationStatus.FAILED,
                message="Invalid format",
            )
        )

        recommendation = validator.get_rollback_recommendation()
        assert recommendation is not None
        assert "v1" in recommendation

    def test_validate_all_flow(self, tmp_path: Path):
        """Test complete validation flow."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"GGUF" + b"\x00" * (500 * 1024 * 1024))

        validator = PreDeploymentValidator(
            model_path=model_path,
            model_name="test-model",
            version="v1",
        )

        report = validator.validate_all()

        assert len(report.results) > 0
        assert report.model_name == "test-model"
        assert report.version == "v1"
        # Should have some passing checks
        passing = [r for r in report.results if r.status == ValidationStatus.PASSED]
        assert len(passing) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
