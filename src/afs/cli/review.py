"""Review CLI commands for context-local draft queues."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from ..context_layout import LAYOUT_VERSION, _atomic_write_text, detect_layout_version
from ..history import log_event, resolve_history_root
from ..models import MountType
from ..path_safety import assert_no_linklike_components
from ._utils import load_manager, resolve_context_paths

REVIEW_CATEGORIES = ("plans", "walkthroughs", "automated_reports")


@dataclass(frozen=True)
class ReviewEntry:
    category: str
    path: Path
    queue_root: Path

    @property
    def filename(self) -> str:
        return self.path.name


def register_parsers(subparsers):
    review_parser = subparsers.add_parser(
        "review",
        help="Review and manage context-local draft documents.",
    )
    review_subparsers = review_parser.add_subparsers(dest="review_command")

    list_parser = review_subparsers.add_parser(
        "list",
        help="List pending review documents for the active context.",
    )
    _add_context_args(list_parser)
    list_parser.add_argument(
        "--category",
        choices=REVIEW_CATEGORIES,
        help="Only show one review category.",
    )
    list_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    list_parser.set_defaults(func=handle_list)

    approve_parser = review_subparsers.add_parser(
        "approve",
        help="Approve a queued review document and move it into the current context.",
    )
    _add_context_args(approve_parser)
    approve_parser.add_argument(
        "--category",
        choices=REVIEW_CATEGORIES,
        help="Disambiguate the review category when filenames collide.",
    )
    approve_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    approve_parser.add_argument(
        "target",
        help=(
            "Filename to approve, or a legacy project/path when also passing a filename."
        ),
    )
    approve_parser.add_argument(
        "filename",
        nargs="?",
        help="Legacy compatibility filename when target is a project or project path.",
    )
    approve_parser.set_defaults(func=handle_approve)

    reject_parser = review_subparsers.add_parser(
        "reject",
        help="Reject a queued review document and archive it in context history.",
    )
    _add_context_args(reject_parser)
    reject_parser.add_argument(
        "--category",
        choices=REVIEW_CATEGORIES,
        help="Disambiguate the review category when filenames collide.",
    )
    reject_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    reject_parser.add_argument(
        "target",
        help=(
            "Filename to reject, or a legacy project/path when also passing a filename."
        ),
    )
    reject_parser.add_argument(
        "filename",
        nargs="?",
        help="Legacy compatibility filename when target is a project or project path.",
    )
    reject_parser.add_argument("--reason", "-r", help="Reason for rejection.")
    reject_parser.set_defaults(func=handle_reject)


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config file path.")
    parser.add_argument("--path", help="Project path whose context should be used.")
    parser.add_argument("--context-root", help="Explicit context root override.")
    parser.add_argument("--context-dir", help="Explicit context directory name.")


def _load_review_manager(args: argparse.Namespace):
    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    return load_manager(config_path)


def _resolve_legacy_project_path(target: str, manager) -> Path:
    candidate = Path(target).expanduser()
    if candidate.exists():
        return candidate.resolve()

    matches: list[Path] = []
    for workspace in manager.config.general.workspace_directories:
        root = workspace.path.expanduser().resolve()
        if root.name == target:
            matches.append(root)
        project_path = root / target
        if project_path.exists() and project_path.is_dir():
            matches.append(project_path.resolve())

    unique_matches: list[Path] = []
    seen: set[Path] = set()
    for match in matches:
        if match not in seen:
            unique_matches.append(match)
            seen.add(match)

    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        rendered = ", ".join(str(path) for path in unique_matches)
        raise ValueError(
            f"Ambiguous legacy project '{target}'; use --path explicitly. Matches: {rendered}"
        )
    raise FileNotFoundError(
        f"Legacy project '{target}' not found. Use --path or pass an explicit project path."
    )


def _resolve_context_for_review(
    args: argparse.Namespace,
) -> tuple[object, Path, str | None]:
    manager = _load_review_manager(args)

    filename = None
    if hasattr(args, "target"):
        if getattr(args, "filename", None):
            if not getattr(args, "path", None) and not getattr(args, "context_root", None):
                args.path = str(_resolve_legacy_project_path(args.target, manager))
            filename = args.filename
        else:
            filename = args.target

    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args,
        manager,
    )
    return manager, context_path, filename


def _queue_roots(manager, context_path: Path) -> list[Path]:
    scratchpad_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD)
    scratchpad_review = scratchpad_root / "review"
    legacy_review = context_path / "review"
    if detect_layout_version(context_path) == LAYOUT_VERSION:
        canonical = assert_no_linklike_components(
            context_path / "human" / "common" / "review",
            boundary=context_path,
        )
        migrated = assert_no_linklike_components(
            scratchpad_root / "common" / "review",
            boundary=context_path,
        )
        pre_fix = assert_no_linklike_components(
            scratchpad_review,
            boundary=context_path,
        )
        legacy = assert_no_linklike_components(
            legacy_review,
            boundary=context_path,
        )
        return list(dict.fromkeys((canonical, migrated, pre_fix, legacy)))
    roots = [scratchpad_review]
    if legacy_review != scratchpad_review:
        roots.append(legacy_review)
    return roots


def _collect_review_entries(
    manager,
    context_path: Path,
    *,
    category: str | None = None,
) -> list[ReviewEntry]:
    entries: list[ReviewEntry] = []
    seen: set[tuple[str, str]] = set()
    categories = [category] if category else list(REVIEW_CATEGORIES)
    is_v2 = detect_layout_version(context_path) == LAYOUT_VERSION
    for queue_root in _queue_roots(manager, context_path):
        if not queue_root.exists():
            continue
        if is_v2:
            queue_root = assert_no_linklike_components(
                queue_root,
                boundary=context_path,
                allow_missing=False,
            )
        for review_category in categories:
            category_root = queue_root / review_category
            if not category_root.exists():
                continue
            if is_v2:
                category_root = assert_no_linklike_components(
                    category_root,
                    boundary=context_path,
                    allow_missing=False,
                )
            for path in sorted(category_root.iterdir()):
                if is_v2:
                    path = assert_no_linklike_components(
                        path,
                        boundary=context_path,
                        allow_missing=False,
                    )
                if not path.is_file() or path.name.startswith(".") or path.name.endswith(".reason"):
                    continue
                identity = (review_category, path.name)
                if identity in seen:
                    continue
                seen.add(identity)
                entries.append(
                    ReviewEntry(
                        category=review_category,
                        path=path,
                        queue_root=queue_root,
                    )
                )
    return entries


def _find_review_entry(
    manager,
    context_path: Path,
    filename: str,
    *,
    category: str | None = None,
) -> ReviewEntry:
    matches = [
        entry
        for entry in _collect_review_entries(manager, context_path, category=category)
        if entry.filename == filename
    ]
    if not matches:
        raise FileNotFoundError(f"Review file '{filename}' not found.")
    if len(matches) > 1:
        categories = ", ".join(sorted({entry.category for entry in matches}))
        raise ValueError(
            f"Multiple review files named '{filename}' found across categories: {categories}. "
            "Use --category to disambiguate."
        )
    return matches[0]


def _approved_destination(manager, context_path: Path, entry: ReviewEntry) -> Path:
    if entry.category == "plans":
        mount_root = manager.resolve_mount_root(context_path, MountType.MEMORY)
        if detect_layout_version(context_path) == LAYOUT_VERSION:
            mount_root = mount_root / "common"
    else:
        mount_root = (
            resolve_history_root(context_path, config=manager.config)
            if detect_layout_version(context_path) == LAYOUT_VERSION
            else manager.resolve_mount_root(context_path, MountType.HISTORY)
        )
    destination = mount_root / "reviewed" / entry.category / entry.filename
    if detect_layout_version(context_path) == LAYOUT_VERSION:
        destination = assert_no_linklike_components(
            destination,
            boundary=context_path,
        )
    return destination


def _rejected_destination(manager, context_path: Path, entry: ReviewEntry) -> Path:
    mount_root = (
        resolve_history_root(context_path, config=manager.config)
        if detect_layout_version(context_path) == LAYOUT_VERSION
        else manager.resolve_mount_root(context_path, MountType.HISTORY)
    )
    destination = mount_root / "rejected" / entry.category / entry.filename
    if detect_layout_version(context_path) == LAYOUT_VERSION:
        destination = assert_no_linklike_components(
            destination,
            boundary=context_path,
        )
    return destination


def _move_review_entry(
    entry: ReviewEntry,
    destination: Path,
    *,
    context_path: Path,
) -> None:
    """Move a reviewed file without following v2 queue or destination links."""

    if detect_layout_version(context_path) != LAYOUT_VERSION:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        shutil.move(str(entry.path), str(destination))
        return

    source = assert_no_linklike_components(
        entry.path,
        boundary=context_path,
        allow_missing=False,
    )
    source_stat = os.lstat(source)
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(f"review source is not a safe regular file: {source}")
    assert_no_linklike_components(destination.parent, boundary=context_path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_no_linklike_components(
        destination.parent,
        boundary=context_path,
        allow_missing=False,
    )
    if os.path.lexists(destination):
        raise FileExistsError(f"Destination already exists: {destination}")
    source.replace(destination)


def _render_list_payload(context_path: Path, entries: list[ReviewEntry]) -> dict[str, object]:
    categories: dict[str, list[str]] = {category: [] for category in REVIEW_CATEGORIES}
    for entry in entries:
        categories.setdefault(entry.category, []).append(entry.filename)
    for names in categories.values():
        names.sort()
    return {
        "context_path": str(context_path),
        "pending": sum(len(names) for names in categories.values()),
        "categories": categories,
    }


def handle_list(args):
    manager, context_path, _filename = _resolve_context_for_review(args)
    entries = _collect_review_entries(
        manager,
        context_path,
        category=getattr(args, "category", None),
    )

    if getattr(args, "json", False):
        print(json.dumps(_render_list_payload(context_path, entries), indent=2))
        return 0

    if not entries:
        print("No pending review documents.")
        return 0

    print(f"context_path: {context_path}")
    grouped = _render_list_payload(context_path, entries)["categories"]
    for category in REVIEW_CATEGORIES:
        names = grouped.get(category, [])
        if not names:
            continue
        print(f"\n[{category.upper()}]")
        for name in names:
            print(f"  - {name}")
    return 0


def handle_approve(args):
    manager, context_path, filename = _resolve_context_for_review(args)
    if not filename:
        raise ValueError("filename is required")

    entry = _find_review_entry(
        manager,
        context_path,
        filename,
        category=getattr(args, "category", None),
    )
    destination = _approved_destination(manager, context_path, entry)
    _move_review_entry(entry, destination, context_path=context_path)

    log_event(
        "review",
        "afs.cli.review",
        op="approve",
        context_root=context_path,
        metadata={
            "category": entry.category,
            "filename": entry.filename,
            "source": str(entry.path),
            "destination": str(destination),
        },
    )

    payload = {
        "status": "approved",
        "context_path": str(context_path),
        "category": entry.category,
        "filename": entry.filename,
        "source": str(entry.path),
        "destination": str(destination),
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Approved {entry.filename} -> {destination}")
    return 0


def handle_reject(args):
    manager, context_path, filename = _resolve_context_for_review(args)
    if not filename:
        raise ValueError("filename is required")

    entry = _find_review_entry(
        manager,
        context_path,
        filename,
        category=getattr(args, "category", None),
    )
    destination = _rejected_destination(manager, context_path, entry)
    reason_path = None
    if getattr(args, "reason", None):
        reason_path = destination.with_name(f"{destination.name}.reason.txt")
        if detect_layout_version(context_path) == LAYOUT_VERSION:
            reason_path = assert_no_linklike_components(
                reason_path,
                boundary=context_path,
            )

    _move_review_entry(entry, destination, context_path=context_path)

    if reason_path is not None:
        if detect_layout_version(context_path) == LAYOUT_VERSION:
            _atomic_write_text(reason_path, args.reason)
        else:
            reason_path.write_text(args.reason, encoding="utf-8")

    log_event(
        "review",
        "afs.cli.review",
        op="reject",
        context_root=context_path,
        metadata={
            "category": entry.category,
            "filename": entry.filename,
            "source": str(entry.path),
            "destination": str(destination),
            "reason_file": str(reason_path) if reason_path else None,
        },
    )

    payload = {
        "status": "rejected",
        "context_path": str(context_path),
        "category": entry.category,
        "filename": entry.filename,
        "source": str(entry.path),
        "destination": str(destination),
        "reason_file": str(reason_path) if reason_path else None,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Rejected {entry.filename} -> {destination}")
    if reason_path is not None:
        print(f"Reason saved to {reason_path}")
    return 0
