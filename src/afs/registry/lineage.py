"""Model lineage and version history tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LineageTracker:
    """Track model lineage, training history, and dependencies.

    Maintains a directed acyclic graph (DAG) of model versions showing:
    - Parent-child relationships (fine-tuning chains)
    - Training data dependencies
    - Code versions (git commits)
    - Resource usage over time

    Example:
        ```python
        tracker = LineageTracker()

        # Track a new version
        tracker.add_version(
            model_name="majora",
            version="v2",
            parent_version="v1",
            training_data=["oracle", "toolbench"],
            git_commit="abc123",
        )

        # Get lineage tree
        tree = tracker.get_lineage("majora")

        # Find all descendants of a version
        descendants = tracker.get_descendants("majora", "v1")

        # Get training history
        history = tracker.get_training_history("majora")
        ```
    """

    DEFAULT_PATH = Path.home() / ".context" / "training" / "lineage.json"

    def __init__(self, lineage_path: Path | None = None):
        """Initialize lineage tracker.

        Args:
            lineage_path: Path to lineage JSON file
        """
        self.lineage_path = Path(lineage_path or self.DEFAULT_PATH)
        self.lineages: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load lineage data from disk."""
        if self.lineage_path.exists():
            try:
                with open(self.lineage_path) as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "version" in data:
                        self.lineages = data.get("lineages", {})
                    else:
                        self.lineages = data
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Failed to load lineage: {e}")
                self.lineages = {}

    def _save(self) -> None:
        """Save lineage data to disk."""
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "lineages": self.lineages,
        }
        with open(self.lineage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _init_lineage(self, model_name: str) -> None:
        """Initialize lineage entry if it doesn't exist."""
        if model_name not in self.lineages:
            self.lineages[model_name] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "versions": {},
            }

    def add_version(
        self,
        model_name: str,
        version: str,
        parent_version: str | None = None,
        training_data: list[str] | None = None,
        git_commit: str | None = None,
        base_model: str | None = None,
        notes: str = "",
    ) -> None:
        """Add a version to the lineage.

        Args:
            model_name: Name of the model
            version: Version string (e.g., "v1")
            parent_version: Parent version for fine-tunes
            training_data: List of training data sources
            git_commit: Git commit hash of training code
            base_model: Base model used
            notes: Notes about this version
        """
        self._init_lineage(model_name)

        self.lineages[model_name]["versions"][version] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent_version": parent_version,
            "training_data": training_data or [],
            "git_commit": git_commit,
            "base_model": base_model,
            "notes": notes,
        }

        self._save()

    def get_lineage(self, model_name: str, version: str | None = None) -> dict[str, Any]:
        """Get lineage information for a version.

        Args:
            model_name: Name of the model
            version: Specific version, or None for all

        Returns:
            Lineage dict with parent/child relationships
        """
        if model_name not in self.lineages:
            return {}

        if version is None:
            return self.lineages[model_name].copy()

        if version not in self.lineages[model_name]["versions"]:
            return {}

        return self.lineages[model_name]["versions"][version].copy()

    def get_ancestors(self, model_name: str, version: str) -> list[str]:
        """Get all ancestor versions (parent chain).

        Args:
            model_name: Name of the model
            version: Version to trace back from

        Returns:
            List of ancestor versions in order (nearest first)
        """
        ancestors = []
        current = version

        while current:
            if model_name not in self.lineages:
                break
            if current not in self.lineages[model_name]["versions"]:
                break

            lineage_info = self.lineages[model_name]["versions"][current]
            current = lineage_info.get("parent_version")

            if current:
                ancestors.append(current)

        return ancestors

    def get_descendants(self, model_name: str, version: str) -> list[str]:
        """Get all descendant versions (versions fine-tuned from this one).

        Args:
            model_name: Name of the model
            version: Version to find descendants of

        Returns:
            List of descendant version strings
        """
        descendants = []

        if model_name not in self.lineages:
            return descendants

        for v, lineage_info in self.lineages[model_name]["versions"].items():
            if lineage_info.get("parent_version") == version:
                descendants.append(v)
                # Recursively get descendants of descendants
                descendants.extend(self.get_descendants(model_name, v))

        return descendants

    def get_training_history(self, model_name: str) -> list[dict[str, Any]]:
        """Get chronological training history for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of version records sorted by creation time
        """
        if model_name not in self.lineages:
            return []

        versions = self.lineages[model_name]["versions"]
        history = [
            {"version": v, **info}
            for v, info in versions.items()
        ]

        # Sort by creation time
        history.sort(key=lambda x: x.get("created_at", ""))
        return history

    def get_data_lineage(self, model_name: str, version: str) -> dict[str, list[str]]:
        """Get complete data lineage for a version.

        Traces back through all ancestors to collect all training data sources.

        Args:
            model_name: Name of the model
            version: Version to trace

        Returns:
            Dict mapping version to training data sources
        """
        lineage: dict[str, list[str]] = {}

        # Add current version
        if model_name in self.lineages:
            current_info = self.lineages[model_name]["versions"].get(version, {})
            lineage[version] = current_info.get("training_data", [])

            # Add ancestors
            ancestors = self.get_ancestors(model_name, version)
            for ancestor in ancestors:
                ancestor_info = self.lineages[model_name]["versions"].get(ancestor, {})
                lineage[ancestor] = ancestor_info.get("training_data", [])

        return lineage

    def get_code_versions(self, model_name: str) -> dict[str, str]:
        """Get git commits used for each version.

        Args:
            model_name: Name of the model

        Returns:
            Dict mapping version to git commit hash
        """
        if model_name not in self.lineages:
            return {}

        commits = {}
        versions = self.lineages[model_name]["versions"]

        for version, info in versions.items():
            commit = info.get("git_commit")
            if commit:
                commits[version] = commit

        return commits

    def build_tree(self, model_name: str) -> str:
        """Build a text representation of the version tree.

        Args:
            model_name: Name of the model

        Returns:
            ASCII tree representation
        """
        if model_name not in self.lineages:
            return f"No lineage found for {model_name}"

        lines = [f"{model_name} version tree:"]

        # Find root versions (no parent)
        versions = self.lineages[model_name]["versions"]
        roots = [v for v, info in versions.items() if not info.get("parent_version")]

        def _add_tree(version: str, prefix: str = "", is_last: bool = True) -> None:
            """Recursively add version to tree."""
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{version}")

            descendants = self.get_descendants(model_name, version)
            for i, desc in enumerate(descendants):
                is_last_desc = i == len(descendants) - 1
                next_prefix = prefix + ("    " if is_last else "│   ")
                _add_tree(desc, next_prefix, is_last_desc)

        for i, root in enumerate(sorted(roots)):
            _add_tree(root, "", i == len(roots) - 1)

        return "\n".join(lines)

    def summary(self, model_name: str | None = None) -> str:
        """Generate a summary of lineage information.

        Args:
            model_name: Optional specific model to summarize

        Returns:
            Formatted summary string
        """
        lines = ["Model Lineage Summary", "=" * 60]

        if model_name:
            # Summarize single model
            if model_name not in self.lineages:
                return f"No lineage found for {model_name}"

            lines.append(f"\n{model_name}")
            history = self.get_training_history(model_name)
            lines.append(f"  Total versions: {len(history)}")

            # Group by parent
            by_parent: dict[str | None, list[str]] = {}
            for record in history:
                parent = record.get("parent_version")
                if parent not in by_parent:
                    by_parent[parent] = []
                by_parent[parent].append(record["version"])

            for parent, versions in sorted(by_parent.items()):
                if parent is None:
                    lines.append("  Root versions:")
                else:
                    lines.append(f"  Fine-tuned from {parent}:")
                for v in sorted(versions):
                    lines.append(f"    - {v}")

        else:
            # Summarize all models
            for model_name in sorted(self.lineages.keys()):
                history = self.get_training_history(model_name)
                lines.append(f"\n{model_name}: {len(history)} versions")

        return "\n".join(lines)
