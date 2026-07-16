# Work Assistant Connectors

AFS connectors execute exactly one approved work-assistant action. They are
local commands, not broad MCP CRUD surfaces.

## Google Workspace

The first included connector is:

```bash
python3 scripts/afs-work-gws-executor.py <approval-json>
```

Use it through the AFS approval executor:

```bash
./scripts/afs work approvals execute <approval-id> --path . \
  --executor "python3 scripts/afs-work-gws-executor.py"
```

The script requires the `gws` CLI to be installed and authenticated. Setup:

```bash
./scripts/setup_gws.sh --dry-run
./scripts/setup_gws.sh --scopes gmail,calendar,drive
```

Supported approved actions:

| action | target_system | Required preview JSON |
| --- | --- | --- |
| `send_email` | `gmail`, `google-gmail`, `google-workspace`, `gws` | `to`, `subject`, `body` |
| `append_sheet_rows` | `sheets`, `google-sheets`, `google-workspace`, `gws` | `range`, `values` |
| `create_calendar_event` | `calendar`, `google-calendar`, `google-workspace`, `gws` | `summary`, `start`, `end` |

Unsupported actions fail closed. The connector does not delete files, change
Drive permissions, clear sheets, or run arbitrary `gws` commands.

## Examples

Send an approved email:

```bash
approval_id="$(
  ./scripts/afs work approvals request \
    --path . \
    --target-system gmail \
    --target-id "email:person@example.com" \
    --action send_email \
    --summary "Send approved follow-up email" \
    --permission-required "email send approval" \
    --preview-json '{"to":"person@example.com","subject":"Follow-up","body":"Thanks for the update."}' \
    --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["approval_id"])'
)"

./scripts/afs work approvals approve "$approval_id" --path . --because "email content verified"
./scripts/afs work approvals execute "$approval_id" --path . \
  --executor "python3 scripts/afs-work-gws-executor.py"
```

Append rows to a Google Sheet:

```bash
approval_id="$(
  ./scripts/afs work approvals request \
    --path . \
    --target-system google-sheets \
    --target-id "<spreadsheet-id>" \
    --action append_sheet_rows \
    --summary "Append approved planning rows" \
    --permission-required "sheet append approval" \
    --preview-json '{"range":"Sheet1!A:B","values":[["Task","Owner"],["Draft plan","Dana"]]}' \
    --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["approval_id"])'
)"

./scripts/afs work approvals approve "$approval_id" --path . --because "sheet rows verified"
./scripts/afs work approvals execute "$approval_id" --path . \
  --executor "python3 scripts/afs-work-gws-executor.py"
```

Create a calendar event:

```bash
approval_id="$(
  ./scripts/afs work approvals request \
    --path . \
    --target-system google-calendar \
    --target-id primary \
    --action create_calendar_event \
    --summary "Schedule planning review" \
    --permission-required "calendar event approval" \
    --preview-json '{"summary":"Planning review","start":"2026-04-28T13:00:00-04:00","end":"2026-04-28T13:30:00-04:00","attendees":[{"email":"person@example.com"}]}' \
    --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["approval_id"])'
)"

./scripts/afs work approvals approve "$approval_id" --path . --because "event details verified"
./scripts/afs work approvals execute "$approval_id" --path . \
  --executor "python3 scripts/afs-work-gws-executor.py"
```

## Connector Rules

Connector scripts should:

- read the final JSON file argument
- verify the approval is already `approved`
- support an explicit allow-list of actions
- execute one action only
- print compact JSON to stdout
- exit non-zero on failure

AFS marks the approval `applied` only when the connector exits with code `0`.
