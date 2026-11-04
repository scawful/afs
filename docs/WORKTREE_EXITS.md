# Worktree exits: report first, retain by default

Use the existing entry point:

```sh
afs worktrees report --repo /absolute/repo --all --json
afs worktrees report --repo /absolute/repo --base-ref development --all \
  --mission-record /host/context/items/missions/mission_example.json
```

The report performs no merge, branch deletion, worktree removal or prune.
`--older-than-days` is retained for interface compatibility. Last-commit age is
display/sort information, not inactivity or abandonment evidence. An upstream
branch disappearing is not proof of squash merge. Unattached local branches are
visible and retained; their ownership/adoption must be reconciled separately.

Git observations include full target/head commits, branch, divergence, committed
paths since the common ancestor, status digest, dirty/untracked/ignored counts,
locks, missing directories and inspection failures. Git optional locks and
fsmonitor are disabled. Status, head and branch are rechecked; target movement
invalidates review recommendations. No conflict driver or candidate code is
executed by this report.

Ordinary status alone does not prove content preservation. The report inspects
index flags and sparse/ignoreStat settings, then rechecks their fingerprint with
the final snapshot. Skip-worktree, assume-unchanged, unrecognized index/stage
state, sparse configuration or gitlinks make local content unverified and block
readiness. The reported `dirty` becomes unknown in these cases. Submodule
contents require a separate preservation workflow even when an ignore setting
hides them from status. The report never clears flags, changes sparse state,
initializes submodules or mutates the index to make eligibility pass.

Committed scope inventory disables rename detection so both a rename's source
deletion and destination addition must fall inside the declared scope.

## Reuse the mission store

There is no new lifecycle database. A trusted host can attach the following
contract to an existing `MissionStore.create(metadata=...)` record, or consume a
retained existing mission JSON through `--mission-record` (repeatable).
Do not edit the shared mission file behind the store API. A later host workflow
can version/revise this metadata; this command is read-only.

```json
{
  "mission_id": "mission_example",
  "owner": "named-owner",
  "status": "done",
  "metadata": {
    "worktree_exit": [{
      "repo": "/absolute/repo",
      "worktree": "/absolute/linked-worktree",
      "branch": "task-branch",
      "base_commit": "full-current-target-commit",
      "head_commit": "full-task-head-commit",
      "allowed_scope": ["src/component/*", "tests/test_component.py"],
      "owner_state": "inactive",
      "owner_pid": 12345,
      "lease_path": "/absolute/existing-afs-lease-path",
      "intended_exit": "merge-ready",
      "retention_reason": "Retain until adoption review finishes",
      "review_status": "pending",
      "adoption_status": "pending",
      "artifacts": [{"path": "/retained/handoff.md", "sha256": "full-sha256"}],
      "verification": {"path": "/retained/checks.json", "sha256": "full-sha256"}
    }]
  }
}
```

`base_commit` is the exact intended target for this review, not merely an old
branch creation point. `head_commit` must match the inspected HEAD. For detached
worktrees `branch` is an empty string. Scope uses Python `fnmatchcase` over
committed paths. This is scope visibility, not a write-capability grant.

The verification receipt must be a hash-bound JSON object containing the same
`base_commit`, `head_commit`, `passed: true`, a nonempty `checks` description, and
`conflicts: "none"`. The trusted verifier must actually run relevant tests and an
exact-pair conflict check; the report does not run them or infer a conflict pass
from ancestry. Evidence/receipt producers remain outside agent-writable state.

Artifacts and verification receipts must be retained outside the worktree, be
regular files, and match their hashes. This first contract accepts receipts up
to 8 MiB; large ROM/media assets should be referenced by a retained manifest.
It does not validate those manifest members recursively or infer a handoff's
semantic completeness. Duplicate records for the same worktree fail closed.

## Exit choices and blockers

- `retained`: a handoff or review remains; supply a retention reason.
- `quarantined`: failure or uncertainty needs investigation; preserve evidence
  and supply a reason. Neither state becomes cleanup-ready by aging.
- `merge-ready`: a proposal for adoption review. Dirty/unknown state, missing
  owner/task, an open mission, unknown owner inactivity, a nonfree AFS lease,
  missing retention, stale checks, missing conflict evidence or changes outside
  scope block readiness. Review status is displayed separately.
- `cleanup-ready`: a proposal for disposal review. The same identity/ownership/
  preservation guards apply, and HEAD must be reachable from the pinned target.
  Unique dirty, untracked or ignored content blocks readiness. Unmerged commits
  stay preserved. An explicitly approved discard or independently verified
  squash merge requires a later reviewed workflow; this report cannot grant it.

Missing lifecycle records retain worktrees. A missing or prunable directory
requires manual reconciliation, not an automatic prune recommendation. Legacy,
stale or owner-exited leases also require reconciliation. Nothing releases them.
The report also probes the recorded positive owner PID with signal zero. A live
or unverifiable PID blocks readiness; PID reuse therefore blocks conservatively.
The host must bind that PID to the correct owner when creating the record. An
agent-supplied unrelated PID is not ownership evidence. The free state of a
supplied lease does not prove that every possible worker or other lease has been
found. `owner_state` remains a claim, not an OS attestation.

## Audit and trust

`--audit-context /trusted/host/context` explicitly records start and completion or
failure through existing `history.log_session_event`. A completion event contains
the report hash and target/counts, not file contents, command outputs, environment
values or lease tokens. Preserve the JSON report to verify that hash. If requested
audit persistence is unavailable, the command fails instead of claiming success.
Without this flag, the report writes no AFS state.
JSON output always contains the complete report and matches the completion audit
digest. `--limit` applies only to the human-readable display, including when its
value is lower than the inventory size.

`execution_authorized` is always false. A valid receipt hash proves the bytes
read, not who approved them or whether their claims are true. These reports,
mission records and cooperative leases are not tamper-proof against an agent
with the same host filesystem authority. Use a trusted controller to own policy,
audit and verification storage; give candidates separate writable directories.

Snapshots are point-in-time and cannot reserve the Git refs/worktree until a
later action. Revalidate all predicates at any actual merge/archive/delete
boundary. Concurrent hostile link replacement and process containment are not
solved here. This is visibility and conservative planning, not an OS sandbox.

Before adopting an eligible patch, review its exact base/head and evidence.
Before later cleanup, confirm the owner/worker exit, artifact retention, commit
preservation and absence of unique local content, then obtain action-specific
authorization. Never bulk-delete based on this report.
