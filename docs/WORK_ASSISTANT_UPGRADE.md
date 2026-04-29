# Work Assistant Upgrade Guide

Use this guide when upgrading an existing AFS checkout or agent harness to use
the work-assistant layer for docs, sheets, planning, tickets, people, review
routes, and approval-gated external writes.

## Upgrade AFS

```bash
cd ~/src/lab/afs
git pull --ff-only
python3.11 -m pip install -e .
./scripts/afs help work
./scripts/afs work --path .
```

No migration command is required. The work-assistant database is created on
demand at:

```text
.context/global/work_assistant.sqlite3
```

## Upgrade A Workspace

For repo-local context:

```bash
cd ~/src/lab/afs
./scripts/afs context ensure-all --path ~/src/project-a
./scripts/afs context repair --path ~/src/project-a --rebuild-index --json
./scripts/afs work --path ~/src/project-a
```

For a workspace that must use global context:

```bash
export AFS_CONTEXT_ROOT="$HOME/.context"
cd ~/src/lab/afs
./scripts/afs context ensure-all --path ~/src/project-a
./scripts/afs work --context-root "$AFS_CONTEXT_ROOT"
```

## Upgrade Agent Instructions

Add this compact contract to harness instructions for Codex, Claude, Gemini,
hcode, companion-repo harnesses, or other agents:

```text
At session start, run `afs session bootstrap --json` if available.
Run `afs work --path . --json` to inspect people, approvals, and activity.
Before editing docs, sheets, tickets, messages, assignments, or due dates in an
external system, create or reuse an AFS work approval request. Do not perform the
external write until the request is approved by the user.
After approval, execute exactly one approved action with
`afs work approvals execute <approval-id> --path . --executor "<connector command>"`.
Log connector reads, drafts, approval requests, and write results through AFS
context/history metadata so future agents can recover the project state.
```

`afs session bootstrap --json` and `afs session prepare-client --json` include a
compact work-assistant summary. Client wrappers also export:

- `AFS_SESSION_WORK_HINT`
- `AFS_SESSION_WORK_APPROVALS_HINT`
- `AFS_SESSION_WORK_COMMUNICATION_HINT`

Keep the default tool surface small. Agents should use AFS context and work
commands first; domain MCPs and connector tools are opt-in for the active task.
For work-context writing, inspect `afs work communication guide --path .` (or
MCP `work.communication.guide`) plus personal-context/scratchpad evidence
before matching the user's tone. Draft locally first and ask for explicit
approval before posting or submitting externally on the user's behalf.

## Approval Flow

Create a request:

```bash
./scripts/afs work approvals request \
  --path ~/src/project-a \
  --target-system zendesk \
  --target-id ticket-123 \
  --action post_ticket_comment \
  --summary "Send drafted reply" \
  --preview "Thanks for the report..." \
  --permission-required "ticket comment approval"
```

For connector-backed actions, pass structured preview data:

```bash
./scripts/afs work approvals request \
  --path ~/src/project-a \
  --target-system gmail \
  --target-id "email:person@example.com" \
  --action send_email \
  --summary "Send approved follow-up email" \
  --permission-required "email send approval" \
  --preview-json '{"to":"person@example.com","subject":"Follow-up","body":"Thanks for the update."}'
```

Review and approve:

```bash
./scripts/afs work approvals list --path ~/src/project-a
./scripts/afs work approvals show <approval-id> --path ~/src/project-a
./scripts/afs work approvals approve <approval-id> --path ~/src/project-a --by human
```

Preview execution payload:

```bash
./scripts/afs work approvals execute <approval-id> --path ~/src/project-a --dry-run --json
```

Smoke-test with the included no-op executor:

```bash
./scripts/afs work approvals execute <approval-id> --path ~/src/project-a \
  --executor "python3 scripts/afs-work-approval-echo.py"
```

Execute supported Google Workspace actions:

```bash
./scripts/setup_gws.sh --dry-run
./scripts/afs work approvals execute <approval-id> --path ~/src/project-a \
  --executor "python3 scripts/afs-work-gws-executor.py"
```

The no-op executor only echoes the approved payload. Real connector commands
should read the final JSON file argument, perform exactly the approved action,
print JSON to stdout, and exit non-zero on failure.

See `docs/WORK_ASSISTANT_CONNECTORS.md` for the supported Google Workspace
actions and payload shapes.

## Connector Payload

Executors receive a JSON file shaped like:

```json
{
  "version": 1,
  "context_root": "/path/to/.context",
  "actor": "agent",
  "approval": {
    "approval_id": "approval_...",
    "status": "approved",
    "target_system": "google-docs",
    "target_id": "doc-123",
    "action": "edit_doc",
    "summary": "Apply approved edits",
    "preview": {"diff": "-old\n+new"}
  }
}
```

AFS also sets these environment variables:

- `AFS_CONTEXT_ROOT`
- `AFS_WORK_APPROVAL_ID`
- `AFS_WORK_APPROVAL_FILE`

## Verification

After upgrading, run:

```bash
./scripts/afs work --path ~/src/project-a --json
./scripts/afs work approvals list --path ~/src/project-a --json
./scripts/afs help work
```

If the context database does not appear, check that the selected path resolves
to the expected `.context` root:

```bash
./scripts/afs context discover --path ~/src/project-a --json
```
