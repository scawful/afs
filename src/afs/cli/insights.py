"""Scoped local research and human-reviewed insight lifecycles."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Literal

from ..agents.guardrails import ApprovalGate, ApprovalRequest
from ..human_provenance import HumanAuthorization, _broker_for_reader
from ..insights import (
    MAX_INSIGHT_REVIEW_RATIONALE_CHARS,
    InsightContentChangedError,
    InsightRecord,
    InsightStore,
    assert_insight_artifact_reviewable,
    insight_review_gate_binding,
    reflect_evidence,
)
from ..scopes import resolve_scope
from ..search_service import ScopedSearchRequest, search_scoped
from ..sources import (
    ContextSourceRecord,
    ResearchProviderError,
    ResearchRequest,
    execute_research_provider,
)
from ._utils import load_manager, resolve_args_config_path, resolve_context_paths

_TTY_READER = None
_REVIEW_AGENT = "insights"


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--path", help="Project or working path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--context-dir", help="Context directory name.")


def _terminal_text(value: Any, *, limit: int = 500) -> str:
    """Collapse untrusted evidence to one terminal-safe line."""

    printable = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in str(value)
    )
    return " ".join(printable.split())[:limit]


def _terminal_markdown(value: Any) -> str:
    """Return already-validated candidate Markdown without silent truncation."""

    return str(value)


def _manager_context(args: argparse.Namespace):
    config = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config)
    project, context, _override, _directory = resolve_context_paths(args, manager)
    return manager, project, context


def _insight_store(args: argparse.Namespace) -> tuple[Any, Path, InsightStore]:
    manager, project, context = _manager_context(args)
    common = bool(getattr(args, "common", False))
    scoped = resolve_scope(
        context,
        requester_path=project,
        common=common,
    )
    store = InsightStore(
        context,
        scope_id=scoped.scope_id,
        requester_path=project,
        config=manager.config,
    )
    return manager, project, store


def research_command(args: argparse.Namespace) -> int:
    """Research the current codebase and visible context without cross-scope reads."""

    manager, project, context = _manager_context(args)
    query = str(args.query or "").strip()
    if not query:
        print("research query must be non-empty", file=sys.stderr)
        return 2
    internet_request: ResearchRequest | None = None
    if args.internet_provider:
        try:
            internet_request = ResearchRequest(
                query=query,
                network_allowed=True,
                allowed_domains=tuple(args.allow_domain or ()),
                max_results=args.internet_limit,
                timeout_seconds=args.internet_timeout,
                max_bytes=args.internet_max_bytes,
            )
        except ValueError as exc:
            print(f"Internet research failed: {exc}", file=sys.stderr)
            return 2
    elif args.allow_domain:
        print(
            "--allow-domain requires --internet-provider; no provider was loaded.",
            file=sys.stderr,
        )
        return 2
    result = search_scoped(
        ScopedSearchRequest(
            context_root=context,
            project_path=project,
            query=query,
            mode=args.mode,
            limit=args.limit,
            rebuild=not bool(args.reuse_index),
            semantic=bool(args.semantic),
            embedding_provider=args.provider,
            embedding_model=args.model,
            all_projects=False,
        ),
        config=manager.config,
    )
    payload = result.to_dict()
    internet_records: tuple[ContextSourceRecord, ...] = ()
    if internet_request is not None:
        try:
            internet_records = execute_research_provider(
                args.internet_provider,
                internet_request,
                config_path=resolve_args_config_path(args, start_dir=project),
            )
        except (KeyError, ResearchProviderError, ValueError) as exc:
            print(f"Internet research failed: {exc}", file=sys.stderr)
            return 2
    payload.update(
        {
            "research_source": "local_code_and_context",
            "embedding_requested": bool(args.semantic),
            # Both supported embedding backends are reached through provider
            # calls; Gemini may transmit content remotely, while Ollama is
            # normally local. The explicit --semantic flag grants this call.
            "network_requested": bool(args.semantic or args.internet_provider),
            "remote_content_transmission_requested": bool(
                args.internet_provider
                or (args.semantic and args.provider == "gemini")
            ),
            "internet": {
                "provider": args.internet_provider or "",
                "allowed_domains": list(args.allow_domain or ()),
                "records": [record.to_dict() for record in internet_records],
            },
        }
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    response = result.response
    print(
        f"Local research: {len(response.results)} result(s); "
        f"semantic={response.semantic_status}"
    )
    for hit in response.results:
        location = _terminal_text(hit.source_path, limit=2_000)
        if hit.line_start:
            location = f"{location}:{hit.line_start}-{hit.line_end}"
        print(
            f"- {location} [{_terminal_text(hit.scope_id)}] "
            f"score={hit.score:.4f}"
        )
        if hit.text_preview:
            print(f"  {_terminal_text(hit.text_preview, limit=240)}")
    if internet_records:
        print(f"Internet evidence: {len(internet_records)} result(s)")
        for record in internet_records:
            print(
                f"- {_terminal_text(record.title)}: "
                f"{_terminal_text(record.uri, limit=2_000)}"
            )
    if not args.semantic:
        print("Tip: add --semantic to explicitly enable embedding-backed retrieval.")
    return 0


def _evidence_summary(packet: Any) -> dict[str, Any]:
    return {
        "schema_version": packet.schema_version,
        "scope_id": packet.scope_id,
        "event_count": len(packet.events),
        "evidence_ids": list(packet.evidence_ids),
        "evidence_digest": packet.evidence_digest,
    }


def reflect_command(args: argparse.Namespace) -> int:
    """Turn repeated attributed history into one reviewable local candidate."""

    _manager, _project, store = _insight_store(args)
    packet = store.build_evidence_packet(
        limit=args.limit,
        event_types=args.event_type or None,
    )
    candidate_payload = reflect_evidence(packet)
    if candidate_payload is None:
        payload = {
            "status": "no_candidate",
            "evidence": _evidence_summary(packet),
            "reason": "no repeated attributed pattern met the reflection threshold",
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(payload["reason"])
        return 0

    creation = store.create_candidate_result(
        candidate_payload,
        evidence=packet,
        agent_name=str(args.agent_name or "").strip(),
    )
    artifact = creation.artifact
    record = store.show(artifact.metadata.artifact_id)
    if record is None:  # pragma: no cover - store creation contract
        raise RuntimeError("created insight candidate could not be read back")
    payload = {
        "status": "candidate_created" if creation.created else "candidate_existing",
        "candidate_status": record.status,
        "candidate": record.to_dict(),
        "evidence": _evidence_summary(packet),
        "inspected_evidence_digest": packet.evidence_digest,
        "bound_evidence_digest": creation.bound_evidence_digest,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        label = "created" if creation.created else "already exists"
        print(f"Insight candidate ({label}): {artifact.metadata.title}")
        print(f"id: {artifact.metadata.artifact_id}")
        print(f"path: {artifact.path}")
        print("Review with `afs insights show <id>`; accept/reject requires --because.")
    return 0


def _render_record(record: InsightRecord) -> None:
    artifact = record.artifact
    print(
        f"{_terminal_text(artifact.metadata.artifact_id)}  "
        f"{_terminal_text(record.status)}  "
        f"{_terminal_text(artifact.metadata.title)}"
    )
    print(f"  {_terminal_text(artifact.path, limit=2_000)}")


def list_command(args: argparse.Namespace) -> int:
    _manager, _project, store = _insight_store(args)
    status = None if args.status == "all" else args.status
    records = store.list(status=status, limit=args.limit)
    if args.json:
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return 0
    if not records:
        print("No insight candidates.")
        return 0
    for record in records:
        _render_record(record)
    return 0


def show_command(args: argparse.Namespace) -> int:
    _manager, _project, store = _insight_store(args)
    record = store.show(args.identifier)
    if record is None:
        print(f"Insight candidate not found: {args.identifier}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(record.to_dict(), indent=2))
        return 0
    try:
        assert_insight_artifact_reviewable(record.artifact)
    except ValueError as exc:
        print(
            f"Insight candidate cannot be rendered safely: {exc}. "
            "Inspect escaped content with --json, then reject it if appropriate.",
            file=sys.stderr,
        )
        return 2
    _render_record(record)
    print()
    sys.stdout.write(_terminal_markdown(record.artifact.body))
    return 0


def _require_rationale(args: argparse.Namespace, decision: str) -> str | None:
    rationale = str(getattr(args, "because", "") or "").strip()
    if not rationale:
        print(
            f"A rationale is required to {decision} an insight: pass --because "
            '"<why this evidence supports the decision>".',
            file=sys.stderr,
        )
        return None
    if len(rationale) > MAX_INSIGHT_REVIEW_RATIONALE_CHARS:
        print(
            "Insight rationale must be no more than "
            f"{MAX_INSIGHT_REVIEW_RATIONALE_CHARS} characters.",
            file=sys.stderr,
        )
        return None
    if any(
        unicodedata.category(character).startswith("C")
        for character in rationale
    ):
        print(
            "Insight rationale must not contain control or formatting characters.",
            file=sys.stderr,
        )
        return None
    return rationale


def _confirm_decision(
    gate: ApprovalGate,
    request: ApprovalRequest,
    *,
    decision: str,
    rationale: str,
    record: InsightRecord,
) -> HumanAuthorization | None:
    token = f"{request.agent}:{request.action}"
    prompt = "\n".join(
        [
            "",
            "=== HUMAN CONFIRMATION REQUIRED (insight review) ===",
            f"  decision: {_terminal_text(decision)}",
            f"  title:    {_terminal_text(record.artifact.metadata.title)}",
            f"  id:       {_terminal_text(record.artifact.metadata.artifact_id)}",
            f"  scope:    {_terminal_text(record.artifact.metadata.scope_id)}",
            f"  because:  {_terminal_text(rationale, limit=4_096)}",
            f"Type '{token}' to confirm, anything else aborts: ",
        ]
    )
    scope = gate.human_authorization_scope(
        "approve",
        request.request_id,
        rationale,
    )
    return _broker_for_reader(_TTY_READER).confirm_token(token, prompt, scope=scope)


def _approved_request(
    gate: ApprovalGate,
    *,
    action: str,
    detail: str,
    preferred_request_id: str = "",
) -> ApprovalRequest | None:
    matches = [
        request
        for request in gate.all_requests()
        if request.agent == _REVIEW_AGENT
        and request.action == action
        and request.detail == detail
        and request.status == "approved"
    ]
    if preferred_request_id:
        matches = [
            request for request in matches if request.request_id == preferred_request_id
        ]
    if not matches:
        return None
    # Identical crash-retry approvals authorize the same digest-bound action.
    # Prefer the latest durable confirmation deterministically.
    return max(matches, key=lambda request: (request.reviewed_at, request.request_id))


def _review_command(
    args: argparse.Namespace,
    *,
    decision: Literal["accept", "reject"],
) -> int:
    rationale = _require_rationale(args, decision)
    if rationale is None:
        return 2
    _manager, _project, store = _insight_store(args)
    record = store.show(args.identifier)
    if record is None:
        print(f"Insight candidate not found: {args.identifier}", file=sys.stderr)
        return 1
    expected_status = "accepted" if decision == "accept" else "rejected"
    if record.status not in {"pending", expected_status}:
        print(
            f"Insight candidate is already {record.status} and cannot be {decision}ed.",
            file=sys.stderr,
        )
        return 2
    try:
        assert_insight_artifact_reviewable(record.artifact)
    except ValueError as exc:
        if decision == "accept":
            print(
                f"Could not accept insight: {exc}. Inspect escaped content with "
                "`afs insights show --json` and reject the unsafe candidate.",
                file=sys.stderr,
            )
            return 2
        print(
            f"Warning: rejecting a candidate that cannot be rendered safely: {exc}. "
            "Use `afs insights show --json` to inspect its escaped content.",
            file=sys.stderr,
        )

    action, detail = insight_review_gate_binding(
        store,
        record,
        decision=decision,
        rationale=rationale,
    )
    approved_digest = record.content_digest
    gate = ApprovalGate()
    approved_request = _approved_request(gate, action=action, detail=detail)
    authorized = approved_request is not None
    request_id = approved_request.request_id if approved_request is not None else ""
    if not authorized:
        authorized = gate.check(_REVIEW_AGENT, action, detail)
    if not authorized:
        request = gate.find_pending(_REVIEW_AGENT, action)
        if request is None:  # pragma: no cover - ApprovalGate contract
            print("Could not create an insight review request.", file=sys.stderr)
            return 2
        authorization = _confirm_decision(
            gate,
            request,
            decision=decision,
            rationale=rationale,
            record=record,
        )
        if authorization is None:
            print(
                f"{decision} requires interactive human confirmation on a terminal; "
                "refusing in a non-interactive context.",
                file=sys.stderr,
            )
            return 2
        # The gate authorizes execution of the requested review operation.
        # Whether the candidate itself is accepted or rejected is recorded by
        # the immutable insight-decision artifact, not by denying this action.
        authorized = gate.approve_human(
            _REVIEW_AGENT,
            action,
            rationale=rationale,
            authorization=authorization,
        )
        request_id = request.request_id
    if not authorized:
        print(f"Could not record the human {decision} decision.", file=sys.stderr)
        return 2

    approved_request = _approved_request(
        gate,
        action=action,
        detail=detail,
        preferred_request_id=request_id,
    )
    if approved_request is None:
        print(
            "Could not re-read the exact approved insight review request; "
            "refusing to change the candidate.",
            file=sys.stderr,
        )
        return 2
    try:
        artifact = (
            store.accept(
                record.artifact.metadata.artifact_id,
                expected_digest=approved_digest,
                approval_gate=gate,
                approval_request_id=approved_request.request_id,
            )
            if decision == "accept"
            else store.reject(
                record.artifact.metadata.artifact_id,
                expected_digest=approved_digest,
                approval_gate=gate,
                approval_request_id=approved_request.request_id,
            )
        )
    except (InsightContentChangedError, ValueError) as exc:
        print(f"Could not {decision} insight: {exc}", file=sys.stderr)
        return 2
    payload = {
        "decision": decision,
        "candidate_id": record.artifact.metadata.artifact_id,
        "path": str(artifact.path),
        "because": approved_request.rationale,
        "review_request_id": approved_request.request_id,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{decision.title()}ed insight {record.artifact.metadata.artifact_id}")
        print(f"path: {_terminal_text(artifact.path, limit=2_000)}")
    return 0


def accept_command(args: argparse.Namespace) -> int:
    return _review_command(args, decision="accept")


def reject_command(args: argparse.Namespace) -> int:
    return _review_command(args, decision="reject")


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "insights",
        help="Research locally, reflect on attributed history, and review insights.",
    )
    commands = parser.add_subparsers(dest="insights_command")

    research = commands.add_parser(
        "research",
        help="Research the current codebase and visible context.",
    )
    _add_context_args(research)
    research.add_argument("query")
    research.add_argument("--mode", choices=["text", "symbol"], default="text")
    research.add_argument("--limit", type=int, default=10)
    research.add_argument(
        "--reuse-index",
        action="store_true",
        help="Reuse the current index snapshot instead of refreshing code/context first.",
    )
    research.add_argument(
        "--semantic",
        action="store_true",
        help="Explicitly enable embedding-backed retrieval.",
    )
    research.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default="gemini",
        help="Embedding provider used only with --semantic.",
    )
    research.add_argument("--model")
    research.add_argument(
        "--internet-provider",
        help=(
            "Explicitly run one enabled extension research provider in a bounded "
            "subprocess."
        ),
    )
    research.add_argument(
        "--allow-domain",
        action="append",
        help="Allowed HTTPS domain for internet evidence (repeatable and required).",
    )
    research.add_argument("--internet-limit", type=int, default=10)
    research.add_argument("--internet-timeout", type=float, default=20.0)
    research.add_argument("--internet-max-bytes", type=int, default=1_000_000)
    research.add_argument("--json", action="store_true")
    research.set_defaults(func=research_command)

    reflect = commands.add_parser(
        "reflect",
        help="Create a reviewable insight from repeated attributed history.",
    )
    _add_context_args(reflect)
    reflect.add_argument("--common", action="store_true")
    reflect.add_argument("--limit", type=int, default=100)
    reflect.add_argument("--event-type", action="append")
    reflect.add_argument("--agent-name", default="")
    reflect.add_argument("--json", action="store_true")
    reflect.set_defaults(func=reflect_command)

    listing = commands.add_parser("list", help="List insight candidates.")
    _add_context_args(listing)
    listing.add_argument("--common", action="store_true")
    listing.add_argument(
        "--status",
        choices=["pending", "accepted", "rejected", "all"],
        default="pending",
    )
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=list_command)

    show = commands.add_parser("show", help="Show one insight candidate.")
    _add_context_args(show)
    show.add_argument("identifier")
    show.add_argument("--common", action="store_true")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=show_command)

    for name, func in (("accept", accept_command), ("reject", reject_command)):
        review = commands.add_parser(
            name,
            help=f"{name.title()} one insight with a rationale and human confirmation.",
        )
        _add_context_args(review)
        review.add_argument("identifier")
        review.add_argument("--common", action="store_true")
        review.add_argument("--because")
        review.add_argument("--json", action="store_true")
        review.set_defaults(func=func)


__all__ = ["register_parsers"]
