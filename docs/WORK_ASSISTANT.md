# Work Assistant

AFS has a native work-assistant state layer for non-technical workflows:
documents, sheets, planning, task management, ticket replies, and review routing.

The design boundary is intentional:

- AFS stores people, project relationships, review routes, approval requests,
  and activity logs in context-local SQLite state.
- Agents and connectors may draft local changes freely.
- External writes require an explicit approval object before execution.
- MCP should stay thin; it should not become full people, docs, sheets, ticket,
  or permission administration.

## Storage

The native database lives under the active context global mount:

```text
.context/global/work_assistant.sqlite3
```

Tables:

- `people`: work-scoped people and handles
- `relationships`: person-to-project/doc/repo/ticket relationships
- `review_routes`: who should review what, with explainable reasons
- `approvals`: pending/approved/rejected external write approvals
- `communication_samples`: work-writing examples used for tone/style grounding
- `activity`: what AFS read, inferred, drafted, requested, approved, or applied

## Context Logging Enrichment

When AFS writes a history event, it runs a best-effort enrichment hook. The hook
does not block history logging if it fails.

Events are enriched when metadata or payload includes work-assistant fields such
as:

- `people`
- `owner`
- `assignee`
- `requester`
- `reviewers`
- `relationships`
- `review_routes`
- `communication_sample` / `communication_samples`
- `approval_request`
- `requires_approval`
- `target_system`
- `target_type`

Example event metadata:

```json
{
  "target_system": "google-docs",
  "target_type": "docs",
  "target_id": "doc-123",
  "owner": {"display_name": "Doc Owner", "email": "owner@example.com"},
  "reviewers": [{"display_name": "Reviewer", "email": "reviewer@example.com"}],
  "communication_sample": {
    "channel": "doc_comment",
    "purpose": "responding_to_comments",
    "text": "Let's keep this concrete: what changed, what was verified, and what's still risky.",
    "style_notes": ["direct", "evidence-backed"]
  },
  "approval_request": {
    "action": "edit_doc",
    "summary": "Apply suggested doc edit",
    "permission_required": "doc edit approval",
    "preview": {"diff": "-old\n+new"}
  }
}
```

Disable enrichment for debugging:

```bash
AFS_WORK_ASSISTANT_ENRICH_DISABLED=1 ./scripts/afs ...
```

## CLI

Summary:

```bash
./scripts/afs work --path .
./scripts/afs work --path . --json
```

People and relationships:

```bash
./scripts/afs work people list --path .
./scripts/afs work relationships list --path .
```

Reviewer suggestions:

```bash
./scripts/afs work reviewers --path . --target-type docs
./scripts/afs work reviewers --path . --target-type code --scope-type repo --scope-id afs
```

Approvals:

```bash
./scripts/afs work approvals list --path .
./scripts/afs work approvals show <approval-id> --path .
./scripts/afs work approvals request \
  --path . \
  --target-system zendesk \
  --target-id ticket-123 \
  --action post_ticket_comment \
  --summary "Send drafted support reply" \
  --preview "Thanks for the report..." \
  --permission-required "ticket comment approval"
./scripts/afs work approvals request \
  --path . \
  --target-system google-sheets \
  --target-id "<spreadsheet-id>" \
  --action append_sheet_rows \
  --summary "Append approved planning rows" \
  --preview-json '{"range":"Sheet1!A:B","values":[["Task","Owner"],["Draft plan","Dana"]]}'
./scripts/afs work approvals approve <approval-id> --path . --by human
./scripts/afs work approvals reject <approval-id> --path . --by human
./scripts/afs work approvals execute <approval-id> --path . --dry-run
./scripts/afs work approvals execute <approval-id> --path . \
  --executor "python3 scripts/afs-work-approval-echo.py"
./scripts/afs work approvals execute <approval-id> --path . \
  --executor "python3 scripts/afs-work-gws-executor.py"
```

Activity:

```bash
./scripts/afs work activity list --path .
```

Communication samples:

```bash
./scripts/afs work communication list --path .
./scripts/afs work communication list --path . --purpose responding_to_comments --json
./scripts/afs work communication add --path . \
  --purpose responding_to_comments \
  --channel pr_review \
  --style-note "direct" \
  --style-note "evidence-backed" \
  --text "Short, concrete reply with exact file evidence."
./scripts/afs work communication guide --path . --purpose responding_to_comments
```

Session startup:

- `afs session bootstrap --json` includes a compact `work_assistant` block.
- `afs session prepare-client --json` includes work hints under `cli_hints`.
- `afs-client-session` exports `AFS_SESSION_WORK_HINT` and
  `AFS_SESSION_WORK_APPROVALS_HINT` for harness wrappers.
- `AFS_SESSION_WORK_COMMUNICATION_HINT` points editor/harness surfaces at
  captured work communication samples.
- VS Code and Antigravity hosts get `AFS: Work Communication Guidance`,
  `AFS: Work Approvals`, and the `@afs /work` chat command when the bundled
  extension is installed.

## Work Communication Rule

For docs, design docs, technical requirements, and replies/comments, an agent
should first inspect available communication samples (`afs work communication
guide` or MCP `work.communication.guide`), personal-context mode content,
scratchpad, and relevant history before matching the user's voice. If there is
not enough evidence, it should say what is missing instead of inventing a
style.

External posts remain approval-gated: the agent may draft locally, but must ask
for explicit permission before posting, sending, submitting, or editing on the
user's behalf.

## Permission Rule

The work-assistant layer may create drafts and approval requests, but it does
not apply external edits until a request is explicitly approved. Connector write
executors should accept only one approved action at a time and record the result
back into activity.

Actions that should require approval include:

- editing shared docs
- adding doc comments
- changing sheet cells, formulas, protected ranges, rows, or columns
- posting ticket comments
- posting doc, PR, or code-review comments
- submitting reviews
- changing ticket status, priority, or assignee
- sending email or chat messages
- assigning work or changing due dates
- sharing files or changing access

## Approved Action Executors

`afs work approvals execute` is the minimal handoff point between AFS and
external systems. It does not expose a broad MCP CRUD surface. It only passes
one approved request to one explicit local command.

Execution rules:

- the approval must already be `approved`
- the executor is invoked without a shell
- AFS appends a temporary approval JSON file as the final command argument
- AFS also sets `AFS_WORK_APPROVAL_FILE`, `AFS_WORK_APPROVAL_ID`, and
  `AFS_CONTEXT_ROOT`
- exit code `0` marks the approval `applied`
- non-zero exit keeps the approval `approved`, records the failure result, and
  leaves it retryable

Preview the payload before wiring a real connector:

```bash
./scripts/afs work approvals execute <approval-id> --path . --dry-run --json
```

Smoke-test the flow without changing external systems:

```bash
./scripts/afs work approvals execute <approval-id> --path . \
  --executor "python3 scripts/afs-work-approval-echo.py"
```

Execute supported Google Workspace actions after approval:

```bash
./scripts/afs work approvals execute <approval-id> --path . \
  --executor "python3 scripts/afs-work-gws-executor.py"
```

Connector scripts should read the JSON file, perform exactly the approved
action, print a compact JSON result to stdout, and exit non-zero if the external
write fails.

Payload shape:

```json
{
  "version": 1,
  "context_root": "/path/to/.context",
  "actor": "agent",
  "approval": {
    "approval_id": "approval_...",
    "status": "approved",
    "target_system": "zendesk",
    "target_id": "ticket-123",
    "action": "post_ticket_comment",
    "summary": "Send drafted support reply",
    "preview": {"text": "Thanks for the report..."}
  }
}
```

## First Implementation Slice

The current implementation provides:

- native SQLite storage
- context-log enrichment for people, relationships, review routes, approvals,
  and activity
- CLI inspection and approval management under `afs work`
- approval-gated execution through explicit local connector commands
- a Google Workspace executor for approved email send, sheet append, and
  calendar event creation
- no new MCP tool expansion

Future connector executors should remain approval-gated and narrow. See
`docs/WORK_ASSISTANT_CONNECTORS.md`.
