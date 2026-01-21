"""Integration with model registry for version approval and tracking.

Manages model version approval status based on quality gate results.
Coordinates with the model registry to:
- Update approval status
- Track deployment eligibility
- Enable/disable versions
- Audit trail
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .quality_gates import QualityGateReport

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRecord:
    """Record of a version approval decision."""

    model_name: str
    model_version: str
    approved: bool
    context: str  # development, staging, production
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gates_report: dict[str, Any] | None = None
    approved_by: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RegistryIntegration:
    """Integration with model registry for approval management."""

    def __init__(self, registry_path: str | None = None):
        """Initialize registry integration.

        Args:
            registry_path: Path to registry database or config
        """
        self.registry_path = Path(registry_path or "models/registry")
        self.approval_dir = self.registry_path / "approvals"
        self.approval_dir.mkdir(parents=True, exist_ok=True)

    def approve_version(
        self,
        report: QualityGateReport,
        approved_by: str | None = None,
        notes: str = "",
    ) -> bool:
        """Approve a model version based on gate results.

        Args:
            report: Quality gate report
            approved_by: User or system approving
            notes: Additional notes

        Returns:
            True if approval succeeded
        """
        if not report.all_passed():
            logger.warning(
                f"Cannot approve {report.model_name}:{report.model_version} "
                "with failed gates"
            )
            return False

        try:
            record = ApprovalRecord(
                model_name=report.model_name,
                model_version=report.model_version,
                approved=True,
                context=report.context.value,
                gates_report=report.to_dict(),
                approved_by=approved_by or "system",
                notes=notes,
            )

            self._write_approval_record(record)
            self._update_version_status(record)

            logger.info(
                f"Approved {report.model_name}:{report.model_version} "
                f"for {report.context.value}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to approve version: {e}")
            return False

    def reject_version(
        self,
        model_name: str,
        model_version: str,
        context: str,
        reason: str = "",
        rejected_by: str | None = None,
    ) -> bool:
        """Reject a model version.

        Args:
            model_name: Name of model
            model_version: Version of model
            context: Deployment context
            reason: Reason for rejection
            rejected_by: User or system rejecting

        Returns:
            True if rejection succeeded
        """
        try:
            record = ApprovalRecord(
                model_name=model_name,
                model_version=model_version,
                approved=False,
                context=context,
                approved_by=rejected_by or "system",
                notes=f"REJECTED: {reason}",
            )

            self._write_approval_record(record)

            logger.info(
                f"Rejected {model_name}:{model_version} for {context}: {reason}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to reject version: {e}")
            return False

    def get_approval_status(
        self, model_name: str, model_version: str, context: str
    ) -> ApprovalRecord | None:
        """Get approval status for a version.

        Args:
            model_name: Name of model
            model_version: Version of model
            context: Deployment context

        Returns:
            Approval record if found, None otherwise
        """
        try:
            approval_file = self._get_approval_file(
                model_name, model_version, context
            )
            if approval_file.exists():
                with open(approval_file) as f:
                    data = json.load(f)
                return ApprovalRecord(**data)
            return None
        except Exception as e:
            logger.error(f"Failed to get approval status: {e}")
            return None

    def get_approval_history(
        self, model_name: str, model_version: str
    ) -> list[ApprovalRecord]:
        """Get approval history for a version.

        Args:
            model_name: Name of model
            model_version: Version of model

        Returns:
            List of approval records
        """
        try:
            records = []
            for context in ["development", "staging", "production"]:
                approval_file = self._get_approval_file(
                    model_name, model_version, context
                )
                if approval_file.exists():
                    with open(approval_file) as f:
                        data = json.load(f)
                        records.append(ApprovalRecord(**data))
            return sorted(records, key=lambda r: r.timestamp)
        except Exception as e:
            logger.error(f"Failed to get approval history: {e}")
            return []

    def is_deployable(
        self, model_name: str, model_version: str, context: str
    ) -> bool:
        """Check if version is approved for deployment.

        Args:
            model_name: Name of model
            model_version: Version of model
            context: Deployment context

        Returns:
            True if approved for deployment
        """
        record = self.get_approval_status(model_name, model_version, context)
        return record is not None and record.approved

    def list_approved_versions(
        self, model_name: str, context: str
    ) -> list[str]:
        """List all approved versions for a model in context.

        Args:
            model_name: Name of model
            context: Deployment context

        Returns:
            List of version strings
        """
        try:
            approved = []
            model_dir = self.approval_dir / model_name
            if model_dir.exists():
                for approval_file in model_dir.glob(f"*-{context}.json"):
                    with open(approval_file) as f:
                        data = json.load(f)
                        if data.get("approved", False):
                            approved.append(data["model_version"])
            return sorted(approved)
        except Exception as e:
            logger.error(f"Failed to list approved versions: {e}")
            return []

    def _write_approval_record(self, record: ApprovalRecord) -> None:
        """Write approval record to file."""
        approval_file = self._get_approval_file(
            record.model_name, record.model_version, record.context
        )
        approval_file.parent.mkdir(parents=True, exist_ok=True)

        with open(approval_file, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def _get_approval_file(
        self, model_name: str, model_version: str, context: str
    ) -> Path:
        """Get approval file path for a version."""
        return (
            self.approval_dir
            / model_name
            / f"{model_version}-{context}.json"
        )

    def _update_version_status(self, record: ApprovalRecord) -> None:
        """Update version status in registry metadata."""
        try:
            metadata_file = self.registry_path / "metadata" / f"{record.model_name}.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
            else:
                metadata = {"versions": {}}

            if "versions" not in metadata:
                metadata["versions"] = {}

            if record.model_version not in metadata["versions"]:
                metadata["versions"][record.model_version] = {}

            version_meta = metadata["versions"][record.model_version]

            if "approvals" not in version_meta:
                version_meta["approvals"] = {}

            version_meta["approvals"][record.context] = {
                "approved": record.approved,
                "timestamp": record.timestamp,
                "approved_by": record.approved_by,
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to update version status: {e}")


class DeploymentController:
    """Controls deployment eligibility based on quality gates and approvals."""

    def __init__(
        self,
        registry_integration: RegistryIntegration | None = None,
    ):
        """Initialize deployment controller.

        Args:
            registry_integration: Registry integration instance
        """
        self.registry = registry_integration or RegistryIntegration()
        self.deployment_log = Path(".quality-gates/deployments.jsonl")
        self.deployment_log.parent.mkdir(exist_ok=True)

    def can_deploy(
        self, model_name: str, model_version: str, context: str
    ) -> tuple[bool, str]:
        """Check if version can be deployed.

        Args:
            model_name: Name of model
            model_version: Version of model
            context: Deployment context

        Returns:
            (can_deploy, reason) tuple
        """
        # Check approval status
        if not self.registry.is_deployable(model_name, model_version, context):
            return (
                False,
                f"{model_name}:{model_version} not approved for {context}",
            )

        return True, "Version is approved for deployment"

    def pre_deployment_check(
        self, model_name: str, model_version: str, context: str
    ) -> bool:
        """Run pre-deployment checks.

        Args:
            model_name: Name of model
            model_version: Version of model
            context: Deployment context

        Returns:
            True if all checks pass
        """
        can_deploy, reason = self.can_deploy(model_name, model_version, context)

        if can_deploy:
            logger.info(f"Pre-deployment check passed: {reason}")
        else:
            logger.error(f"Pre-deployment check failed: {reason}")

        return can_deploy

    def execute_deployment(
        self,
        model_name: str,
        model_version: str,
        context: str,
        deployment_target: str = "production",
    ) -> bool:
        """Execute deployment and log to audit trail.

        Args:
            model_name: Name of model
            model_version: Version of model
            context: Deployment context
            deployment_target: Target environment

        Returns:
            True if deployment succeeded
        """
        try:
            if not self.pre_deployment_check(model_name, model_version, context):
                return False

            deployment_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": f"{model_name}:{model_version}",
                "context": context,
                "target": deployment_target,
                "status": "deployed",
            }

            with open(self.deployment_log, "a") as f:
                json.dump(deployment_record, f)
                f.write("\n")

            logger.info(
                f"Deployed {model_name}:{model_version} to {deployment_target}"
            )
            return True
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False

    def rollback(
        self,
        model_name: str,
        model_version: str,
        context: str,
        reason: str = "",
    ) -> bool:
        """Rollback deployment.

        Args:
            model_name: Name of model
            model_version: Version of model
            context: Deployment context
            reason: Reason for rollback

        Returns:
            True if rollback succeeded
        """
        try:
            rollback_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": f"{model_name}:{model_version}",
                "context": context,
                "status": "rolled_back",
                "reason": reason,
            }

            with open(self.deployment_log, "a") as f:
                json.dump(rollback_record, f)
                f.write("\n")

            logger.warning(f"Rolled back {model_name}:{model_version}: {reason}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
