"""CLI commands for quality gate management.

Provides command-line interface for:
- Checking gates against models
- Approving/rejecting versions
- Viewing gate reports and history
- Managing deployment permissions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .ci_integration import LocalFileIntegration
from .quality_gates import (
    DeploymentContext,
    ModelMetrics,
    QualityGate,
    SecurityScanResults,
    TestMetrics,
)
from .registry_integration import RegistryIntegration

logger = logging.getLogger(__name__)


def setup_logger(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register gate subcommands.

    Args:
        subparsers: Argument subparsers from parent parser
    """
    # Main gates command
    gates_parser = subparsers.add_parser(
        "gates",
        help="Quality gate checks and approvals",
    )
    gates_subparsers = gates_parser.add_subparsers(dest="gates_command")

    # Check command
    _register_check_command(gates_subparsers)

    # Approve command
    _register_approve_command(gates_subparsers)

    # Reject command
    _register_reject_command(gates_subparsers)

    # Status command
    _register_status_command(gates_subparsers)

    # History command
    _register_history_command(gates_subparsers)

    # Report command
    _register_report_command(gates_subparsers)


def _register_check_command(subparsers: argparse._SubParsersAction) -> None:
    """Register check command."""
    parser = subparsers.add_parser(
        "check",
        help="Run quality gate checks",
    )
    parser.add_argument(
        "--context",
        choices=["development", "staging", "production"],
        default="staging",
        help="Deployment context (default: staging)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name and version (format: name:version or name)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Path to metrics JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline metrics JSON file",
    )
    parser.add_argument(
        "--test-metrics",
        type=str,
        help="Path to test metrics JSON file",
    )
    parser.add_argument(
        "--security-scan",
        type=str,
        help="Path to security scan results JSON file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Apply strict thresholds even for dev/staging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.set_defaults(func=_handle_check_command)


def _register_approve_command(subparsers: argparse._SubParsersAction) -> None:
    """Register approve command."""
    parser = subparsers.add_parser(
        "approve",
        help="Approve a model version for deployment",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name:version to approve",
    )
    parser.add_argument(
        "--context",
        choices=["development", "staging", "production"],
        required=True,
        help="Deployment context",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Path to quality gate report JSON",
    )
    parser.add_argument(
        "--approved-by",
        type=str,
        help="User approving the version",
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes about the approval",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force approval even with failed gates",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.set_defaults(func=_handle_approve_command)


def _register_reject_command(subparsers: argparse._SubParsersAction) -> None:
    """Register reject command."""
    parser = subparsers.add_parser(
        "reject",
        help="Reject a model version",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name:version to reject",
    )
    parser.add_argument(
        "--context",
        choices=["development", "staging", "production"],
        required=True,
        help="Deployment context",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Reason for rejection",
    )
    parser.add_argument(
        "--rejected-by",
        type=str,
        help="User rejecting the version",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.set_defaults(func=_handle_reject_command)


def _register_status_command(subparsers: argparse._SubParsersAction) -> None:
    """Register status command."""
    parser = subparsers.add_parser(
        "status",
        help="Check approval status of a version",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name:version",
    )
    parser.add_argument(
        "--context",
        choices=["development", "staging", "production"],
        help="Deployment context (shows all if omitted)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.set_defaults(func=_handle_status_command)


def _register_history_command(subparsers: argparse._SubParsersAction) -> None:
    """Register history command."""
    parser = subparsers.add_parser(
        "history",
        help="View approval history",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name:version",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.set_defaults(func=_handle_history_command)


def _register_report_command(subparsers: argparse._SubParsersAction) -> None:
    """Register report command."""
    parser = subparsers.add_parser(
        "report",
        help="View quality gate reports",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name:version",
    )
    parser.add_argument(
        "--context",
        choices=["development", "staging", "production"],
        default="production",
        help="Deployment context",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=1,
        help="Show last N reports (default: 1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.set_defaults(func=_handle_report_command)


def _handle_check_command(args: argparse.Namespace) -> int:
    """Handle check command."""
    setup_logger(args.verbose)

    try:
        # Parse model name:version
        if ":" in args.model:
            model_name, model_version = args.model.split(":")
        else:
            model_name = args.model
            model_version = "latest"

        # Create appropriate gate
        context = DeploymentContext(args.context)
        if args.strict:
            gate = QualityGate.production()
        else:
            gate = {
                "development": QualityGate.development,
                "staging": QualityGate.staging,
                "production": QualityGate.production,
            }[args.context]()

        # Load metrics
        test_metrics = None
        model_metrics = None
        security_results = None
        baseline_metrics = None

        if args.test_metrics:
            test_metrics = _load_test_metrics(args.test_metrics)

        if args.metrics:
            model_metrics = _load_model_metrics(args.metrics)

        if args.baseline:
            baseline_metrics = _load_model_metrics(args.baseline)

        if args.security_scan:
            security_results = _load_security_results(args.security_scan)

        # Run checks
        report = gate.check_all(
            model_name=model_name,
            model_version=model_version,
            test_metrics=test_metrics,
            model_metrics=model_metrics,
            security_results=security_results,
            baseline_model_metrics=baseline_metrics,
        )

        # Output report
        print(gate.summary_string(report))

        if args.json or not args.verbose:
            print("\nDetailed report:")
            print(json.dumps(report.to_dict(), indent=2))

        # Save report
        integration = LocalFileIntegration()
        integration.report(report)

        return 0 if report.all_passed() else 1

    except Exception as e:
        logger.error(f"Failed to check gates: {e}")
        return 1


def _handle_approve_command(args: argparse.Namespace) -> int:
    """Handle approve command."""
    setup_logger(args.verbose)

    try:
        # Parse model name:version
        if ":" in args.model:
            model_name, model_version = args.model.split(":")
        else:
            return 1  # Version required for approval

        registry = RegistryIntegration()

        # If report provided, load it
        report_data = None
        if args.report and Path(args.report).exists():
            with open(args.report) as f:
                report_data = json.load(f)

        # Approve
        success = registry.approve_version(
            model_name=model_name,
            model_version=model_version,
            context=args.context,
            approved_by=args.approved_by,
            notes=args.notes or "",
            report=report_data,
        )

        if success:
            print(f"✓ Approved {model_name}:{model_version} for {args.context}")
            return 0
        else:
            print(f"✗ Failed to approve {model_name}:{model_version}")
            return 1

    except Exception as e:
        logger.error(f"Failed to approve: {e}")
        return 1


def _handle_reject_command(args: argparse.Namespace) -> int:
    """Handle reject command."""
    setup_logger(args.verbose)

    try:
        # Parse model name:version
        if ":" in args.model:
            model_name, model_version = args.model.split(":")
        else:
            return 1  # Version required for rejection

        registry = RegistryIntegration()

        # Reject
        success = registry.reject_version(
            model_name=model_name,
            model_version=model_version,
            context=args.context,
            reason=args.reason,
            rejected_by=args.rejected_by,
        )

        if success:
            print(f"✓ Rejected {model_name}:{model_version} for {args.context}")
            return 0
        else:
            print(f"✗ Failed to reject {model_name}:{model_version}")
            return 1

    except Exception as e:
        logger.error(f"Failed to reject: {e}")
        return 1


def _handle_status_command(args: argparse.Namespace) -> int:
    """Handle status command."""
    try:
        # Parse model name:version
        if ":" in args.model:
            model_name, model_version = args.model.split(":")
        else:
            model_name = args.model
            model_version = "latest"

        registry = RegistryIntegration()

        if args.context:
            # Single context
            record = registry.get_approval_status(
                model_name, model_version, args.context
            )
            if args.json:
                if record:
                    print(json.dumps(record.to_dict(), indent=2))
                else:
                    print(json.dumps({"approved": False}))
            else:
                if record:
                    status = "✓ APPROVED" if record.approved else "✗ REJECTED"
                    print(f"{status} - {record.timestamp}")
                    if record.notes:
                        print(f"Notes: {record.notes}")
                else:
                    print("No approval record found")
            return 0
        else:
            # All contexts
            history = registry.get_approval_history(model_name, model_version)
            if args.json:
                print(json.dumps([r.to_dict() for r in history], indent=2))
            else:
                print(f"Approval history for {model_name}:{model_version}:")
                for record in history:
                    status = "✓ APPROVED" if record.approved else "✗ REJECTED"
                    print(f"  {record.context:12} {status} ({record.timestamp})")
            return 0

    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return 1


def _handle_history_command(args: argparse.Namespace) -> int:
    """Handle history command."""
    try:
        # Parse model name:version
        if ":" in args.model:
            model_name, model_version = args.model.split(":")
        else:
            model_name = args.model
            model_version = "latest"

        registry = RegistryIntegration()
        history = registry.get_approval_history(model_name, model_version)

        if args.json:
            print(json.dumps([r.to_dict() for r in history], indent=2))
        else:
            print(f"Approval history for {model_name}:{model_version}:")
            if not history:
                print("  No approval records found")
            else:
                for record in history:
                    status = "✓ APPROVED" if record.approved else "✗ REJECTED"
                    print(f"  {record.context:12} {status} ({record.timestamp})")
                    if record.notes:
                        print(f"              {record.notes}")

        return 0

    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return 1


def _handle_report_command(args: argparse.Namespace) -> int:
    """Handle report command."""
    try:
        report_dir = Path(".quality-gates")
        print(f"Reports for {args.model} ({args.context}):")
        # Implementation would list and display reports from report_dir
        return 0

    except Exception as e:
        logger.error(f"Failed to get report: {e}")
        return 1


def _load_test_metrics(path: str) -> TestMetrics | None:
    """Load test metrics from JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return TestMetrics(**data)
    except Exception as e:
        logger.error(f"Failed to load test metrics: {e}")
        return None


def _load_model_metrics(path: str) -> ModelMetrics | None:
    """Load model metrics from JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return ModelMetrics(**data)
    except Exception as e:
        logger.error(f"Failed to load model metrics: {e}")
        return None


def _load_security_results(path: str) -> SecurityScanResults | None:
    """Load security scan results from JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return SecurityScanResults(**data)
    except Exception as e:
        logger.error(f"Failed to load security results: {e}")
        return None


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="afs-gates")
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
