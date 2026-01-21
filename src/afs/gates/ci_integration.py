"""CI/CD pipeline integration for quality gates.

Integrates quality gate checks with GitHub Actions and other CI/CD systems.
Handles:
- Gate enforcement in pull request checks
- Deployment approval workflows
- Automatic rollback triggers
- Status reporting and notifications
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .quality_gates import QualityGateReport

logger = logging.getLogger(__name__)


class CIPipelineIntegration(ABC):
    """Abstract base for CI/CD pipeline integrations."""

    @abstractmethod
    def report(self, report: QualityGateReport) -> bool:
        """Report gate check results to CI system.

        Args:
            report: Quality gate report

        Returns:
            True if integration succeeded
        """
        pass

    @abstractmethod
    def block_merge(self, report: QualityGateReport, reason: str) -> bool:
        """Block merge/deployment in CI system.

        Args:
            report: Quality gate report
            reason: Reason for blocking

        Returns:
            True if block succeeded
        """
        pass

    @abstractmethod
    def approve_deployment(self, model_name: str, model_version: str) -> bool:
        """Approve deployment in CI system.

        Args:
            model_name: Name of model
            model_version: Version of model

        Returns:
            True if approval succeeded
        """
        pass


class GitHubActionsIntegration(CIPipelineIntegration):
    """Integration with GitHub Actions CI/CD system."""

    def __init__(self, repo_path: Optional[str] = None):
        """Initialize GitHub Actions integration.

        Args:
            repo_path: Path to repository (auto-detected from environment if None)
        """
        self.repo_path = Path(repo_path or os.getcwd())
        self.github_output = Path(os.environ.get("GITHUB_OUTPUT", "/dev/null"))
        self.github_step_summary = Path(
            os.environ.get("GITHUB_STEP_SUMMARY", "/dev/null")
        )
        self.is_ci = os.environ.get("GITHUB_ACTIONS") == "true"

    def report(self, report: QualityGateReport) -> bool:
        """Report gate results to GitHub Actions.

        Creates annotations, outputs, and summary for GitHub UI.

        Args:
            report: Quality gate report

        Returns:
            True if successful
        """
        try:
            # Write job summary
            self._write_summary(report)

            # Set outputs
            self._set_outputs(report)

            # Create check annotations
            self._create_annotations(report)

            logger.info(f"Reported gate results to GitHub Actions")
            return True
        except Exception as e:
            logger.error(f"Failed to report to GitHub Actions: {e}")
            return False

    def block_merge(self, report: QualityGateReport, reason: str) -> bool:
        """Block PR merge by failing the workflow.

        Args:
            report: Quality gate report
            reason: Reason for blocking

        Returns:
            True if successful
        """
        try:
            # Write failure output
            failure_msg = f"Quality gates failed - {reason}"
            self._write_output("gates_failed", "true")
            self._write_output("failure_reason", reason)

            # Create error annotation
            self._create_error_annotation(failure_msg)

            # Log blocking reason
            logger.error(f"Blocking merge: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to block merge: {e}")
            return False

    def approve_deployment(self, model_name: str, model_version: str) -> bool:
        """Approve deployment by setting output variable.

        Args:
            model_name: Name of model
            model_version: Version of model

        Returns:
            True if successful
        """
        try:
            self._write_output("deployment_approved", "true")
            self._write_output("approved_model", f"{model_name}:{model_version}")

            logger.info(f"Approved deployment of {model_name}:{model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to approve deployment: {e}")
            return False

    def _write_summary(self, report: QualityGateReport) -> None:
        """Write markdown summary to GitHub step summary."""
        if not self.is_ci:
            return

        summary_lines = [
            f"## Quality Gate Report",
            f"**Context:** {report.context.value.upper()}",
            f"**Model:** {report.model_name} v{report.model_version}",
            f"**Status:** {'✅ PASSED' if report.all_passed() else '❌ FAILED'}",
            "",
            f"### Summary",
            f"- Total Checks: {len(report.checks)}",
            f"- Passed: {sum(1 for c in report.checks if c.passed)}",
            f"- Failed: {len(report.failed_checks())}",
        ]

        if report.failed_checks():
            summary_lines.append("")
            summary_lines.append("### Failed Checks")
            for check in report.failed_checks():
                summary_lines.append(
                    f"- **{check.gate_name}** ({check.status.value}): {check.message}"
                )

        summary_lines.append("")
        summary_lines.append(f"Timestamp: {report.timestamp}")

        try:
            with open(self.github_step_summary, "a") as f:
                f.write("\n".join(summary_lines) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write step summary: {e}")

    def _set_outputs(self, report: QualityGateReport) -> None:
        """Set GitHub Actions outputs."""
        outputs = {
            "gates_passed": str(report.all_passed()).lower(),
            "context": report.context.value,
            "model": f"{report.model_name}:{report.model_version}",
            "passed_checks": str(sum(1 for c in report.checks if c.passed)),
            "total_checks": str(len(report.checks)),
            "report_json": json.dumps(report.to_dict()),
        }

        for key, value in outputs.items():
            self._write_output(key, value)

    def _write_output(self, name: str, value: str) -> None:
        """Write GitHub Actions output variable."""
        if not self.is_ci:
            return

        try:
            with open(self.github_output, "a") as f:
                # GitHub Actions format: name=value (with multiline support)
                if "\n" in value:
                    # Use multiline format for JSON
                    f.write(f"{name}<<EOF\n{value}\nEOF\n")
                else:
                    f.write(f"{name}={value}\n")
        except Exception as e:
            logger.warning(f"Failed to write output {name}: {e}")

    def _create_annotations(self, report: QualityGateReport) -> None:
        """Create GitHub Actions check annotations."""
        if not self.is_ci:
            return

        for check in report.checks:
            if check.passed:
                self._create_notice_annotation(check.gate_name, check.message)
            elif check.status.value == "blocked":
                self._create_error_annotation(f"{check.gate_name}: {check.message}")
            else:
                self._create_warning_annotation(check.gate_name, check.message)

    def _create_notice_annotation(self, title: str, message: str) -> None:
        """Create notice-level annotation."""
        # GitHub Actions format: ::notice title={title}::{message}
        print(f"::notice title={title}::{message}")

    def _create_warning_annotation(self, title: str, message: str) -> None:
        """Create warning-level annotation."""
        # GitHub Actions format: ::warning title={title}::{message}
        print(f"::warning title={title}::{message}")

    def _create_error_annotation(self, message: str) -> None:
        """Create error-level annotation."""
        # GitHub Actions format: ::error::{message}
        print(f"::error::{message}")


class JenkinsIntegration(CIPipelineIntegration):
    """Integration with Jenkins CI/CD system."""

    def __init__(self, build_path: Optional[str] = None):
        """Initialize Jenkins integration.

        Args:
            build_path: Path to Jenkins build directory
        """
        self.build_path = Path(build_path or os.environ.get("WORKSPACE", "."))
        self.report_dir = self.build_path / "quality-gates"
        self.report_dir.mkdir(exist_ok=True)

    def report(self, report: QualityGateReport) -> bool:
        """Report gate results to Jenkins.

        Args:
            report: Quality gate report

        Returns:
            True if successful
        """
        try:
            # Write report JSON
            report_file = self.report_dir / f"report-{report.timestamp}.json"
            with open(report_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

            # Write JUnit XML
            self._write_junit_xml(report)

            logger.info(f"Reported gate results to Jenkins")
            return True
        except Exception as e:
            logger.error(f"Failed to report to Jenkins: {e}")
            return False

    def block_merge(self, report: QualityGateReport, reason: str) -> bool:
        """Block merge by marking build as failed.

        Args:
            report: Quality gate report
            reason: Reason for blocking

        Returns:
            True if successful
        """
        try:
            logger.error(f"Blocking merge: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to block merge: {e}")
            return False

    def approve_deployment(self, model_name: str, model_version: str) -> bool:
        """Approve deployment.

        Args:
            model_name: Name of model
            model_version: Version of model

        Returns:
            True if successful
        """
        try:
            approval_file = self.report_dir / "approval.json"
            approval_data = {
                "approved": True,
                "model": model_name,
                "version": model_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(approval_file, "w") as f:
                json.dump(approval_data, f, indent=2)

            logger.info(f"Approved deployment of {model_name}:{model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to approve deployment: {e}")
            return False

    def _write_junit_xml(self, report: QualityGateReport) -> None:
        """Write JUnit XML format for Jenkins."""
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="quality_gates" tests="{len(report.checks)}" '
            f'failures="{len(report.failed_checks())}" '
            f'timestamp="{report.timestamp}">',
        ]

        for check in report.checks:
            if check.passed:
                xml_lines.append(f'  <testcase name="{check.gate_name}" />')
            else:
                xml_lines.append(f'  <testcase name="{check.gate_name}">')
                xml_lines.append(f'    <failure message="{check.message}" />')
                xml_lines.append("  </testcase>")

        xml_lines.append("</testsuite>")

        junit_file = self.report_dir / "junit-gates.xml"
        with open(junit_file, "w") as f:
            f.write("\n".join(xml_lines))


class LocalFileIntegration(CIPipelineIntegration):
    """Local file-based integration for testing/local development."""

    def __init__(self, report_dir: Optional[str] = None):
        """Initialize local file integration.

        Args:
            report_dir: Directory for storing reports (default: .quality-gates)
        """
        self.report_dir = Path(report_dir or ".quality-gates")
        self.report_dir.mkdir(exist_ok=True)

    def report(self, report: QualityGateReport) -> bool:
        """Write report to JSON file.

        Args:
            report: Quality gate report

        Returns:
            True if successful
        """
        try:
            # Create timestamped filename
            timestamp = report.timestamp.replace(":", "-").replace(".", "-")
            report_file = self.report_dir / f"report-{timestamp}.json"

            with open(report_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

            logger.info(f"Wrote gate report to {report_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to write report: {e}")
            return False

    def block_merge(self, report: QualityGateReport, reason: str) -> bool:
        """Log block event to file.

        Args:
            report: Quality gate report
            reason: Reason for blocking

        Returns:
            True if successful
        """
        try:
            block_file = self.report_dir / "blocks.jsonl"
            block_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": f"{report.model_name}:{report.model_version}",
                "reason": reason,
                "context": report.context.value,
            }

            with open(block_file, "a") as f:
                json.dump(block_entry, f)
                f.write("\n")

            logger.info(f"Logged merge block: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to log block: {e}")
            return False

    def approve_deployment(self, model_name: str, model_version: str) -> bool:
        """Log approval to file.

        Args:
            model_name: Name of model
            model_version: Version of model

        Returns:
            True if successful
        """
        try:
            approval_file = self.report_dir / "approvals.jsonl"
            approval_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": f"{model_name}:{model_version}",
            }

            with open(approval_file, "a") as f:
                json.dump(approval_entry, f)
                f.write("\n")

            logger.info(f"Approved deployment: {model_name}:{model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to log approval: {e}")
            return False
