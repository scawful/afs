# Policy-Checked Execution

AFS uses one policy-checked execution boundary for process launches that need a
portable, inspectable contract. The boundary is intentionally smaller than a
general-purpose shell: callers submit a typed request, trusted code supplies
policy, and the broker inspects the request again immediately before spawning.

## Public Contracts

The language-neutral JSON Schema contracts are packaged with AFS and available
through the schema registry and MCP resource surface:

- `afs://schemas/v1/execution/request`
- `afs://schemas/v1/execution/inspection`
- `afs://schemas/v1/execution/record`

The request binds the command, resolved working directory, caller and purpose,
selected environment, timeout and output limits, and requested isolation and
network modes. The inspection explains whether trusted policy permits that
request. The record captures the bounded result and audit metadata.

Python callers use the immutable `ExecutionRequest`, `ExecutionPolicy`,
`ExecutionInspection`, and `ExecutionRecord` types with
`inspect_execution(...)` and `execute_checked(...)`. Inspection never executes
the request. There is deliberately no generic `afs execution run` command.

```python
import sys
from pathlib import Path

from afs.execution import (
    ArgvCommand,
    ExecutionPolicy,
    ExecutionRequest,
    execute_checked,
    inspect_execution,
)

root = Path.cwd().resolve()
request = ExecutionRequest(
    command=ArgvCommand((sys.executable, "-m", "pytest", "-q")),
    caller="repo-verifier",
    purpose="run focused tests",
    cwd=root,
)
policy = ExecutionPolicy(
    allowed_cwd_roots=(root,),
    allowed_executables=frozenset({sys.executable}),
)

inspection = inspect_execution(request, policy)
record = execute_checked(request, policy) if inspection.allowed else None
```

The request cannot grant its own permissions; trusted caller code owns the
policy.

To inspect a request without launching it:

```bash
./scripts/afs execution inspect \
  --request request.json \
  --allowed-root "$PWD" \
  --allowed-executable python3 \
  --json
```

Exit codes are `0` when policy allows the request, `2` for invalid input, and
`3` when policy blocks it. Omitting `--allowed-executable` fails closed rather
than deriving executable permission from request data. Repeat
`--allowed-executable` as needed, and use `--allowed-env NAME` for each
non-baseline environment key the trusted inspection policy should permit.

## Current Backend

The first backend is a portable process boundary, not a security sandbox:

- structured argument vectors run without a shell
- working directories are resolved and must remain below a trusted allowed root
- the child environment is rebuilt from a small allowlist instead of inheriting
  the parent environment wholesale
- output is drained concurrently, capped per stream, and marked when truncated
- timeouts terminate the child process group or Windows process tree
- audit records include the request hash, resolved executable, redacted argv,
  environment key names and an environment hash, timing, return code, and
  truncation state

When present, the baseline environment keys are `PATH`, `HOME`, `TMPDIR`,
`LANG`, and `LC_ALL`. Additional inherited keys require trusted policy, as does
every explicit override (including overrides of baseline keys). `PYTHONPATH` is
not inherited by default.

The default timeout is 300 seconds and the v1 contract caps it at 3,600 seconds.
Each output stream defaults to a 1 MiB retained cap with a 10 MiB contract
maximum; reader threads continue draining beyond the retained prefix so a noisy
child cannot deadlock on a full pipe.

Records use explicit outcomes: `completed`, `failed`, `timed_out`, `blocked`,
or `spawn_error`.

This backend supports only `isolation=process` and `network=inherit`. Requests
for a sandbox, container, or network denial fail closed. A worktree or a clean
environment is useful operational isolation, but neither is a security boundary.

Audit fields never serialize raw environment values. Callers can redact
selected argv positions, and persisted session events contain audit metadata
rather than raw stdout, stderr, or environment values. Capped process output
can still contain anything the child prints and must be handled as sensitive
caller-visible data.

## Verification Configuration

Verification profiles should use structured executions:

```toml
[verification]
allow_legacy_shell = false

[[verification.profiles.repo.checks]]
name = "python-tests"
required = true

[[verification.profiles.repo.checks.executions]]
argv = [".venv/bin/python", "-m", "pytest", "-q"]
timeout_seconds = 600
inherit_env = ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"]
redact_argv_indices = []
```

Set `redact_argv_indices` when an argument must be replaced with `<redacted>` in
persisted verification audit metadata. `afs verify plan` reports each
structured execution before it can run. Existing
string entries in `commands` are legacy shell commands: they are deprecated and
blocked by default. A migration window permits an explicit configuration opt-in
or `afs verify run --allow-legacy-shell`; enabled legacy commands still pass
through the broker as resolved `bash -lc` requests and emit a warning on
stderr. Legacy verification shell commands are scheduled for removal in
AFS `0.4.0`.

When a required selected check is blocked, verification exits `2`. An optional
blocked check is skipped with a warning. Normal verification outcomes remain
`0` for success and `1` for a completed check failure.

## Trust Boundary

Request data is untrusted and cannot grant itself executable, environment,
filesystem, isolation, or network permission. `ExecutionPolicy` must come from
a trusted caller. A request hash provides integrity for exact policy review; it
does not establish provenance or approval by itself.

The process backend is the foundation for later sandboxed candidate trials. It
does not yet authorize the background job worker, generic agent shell tools, or
autonomous candidate execution.
