# Engineering Practices

Code standards for the AFS core. Each rule here exists because its
violation caused (or nearly caused) a real defect in this codebase; the
evidence is noted so the rule can be re-evaluated rather than cargo-culted.

Two ratchets enforce part of this document mechanically:

- **ruff** enforces the exception rules (`BLE001`, `S110`, `S112`) on every
  file not grandfathered in `pyproject.toml`'s per-file-ignores baseline.
- **mypy** checks the whole `afs` package (`make type-check`); modules not
  grandfathered in the `ignore_errors` baseline must stay clean.

**Ratchet contract:** never add an entry to either baseline. When you touch
a listed file or module, fix its violations and delete its entry. Baselines
were generated with ruff 0.15.* and mypy 1.11.* (pinned in CI); bumping a
tool pin requires regenerating the baseline in the same PR.

## Exceptions

Catch the narrowest exception the handler can actually do something about.

- Never `except Exception:` around state mutations. A handler that cannot
  distinguish "file absent" from "disk failing" cannot pick a correct
  recovery. *Evidence: a handoff event loader swallowed every `OSError`,
  so a transient open failure silently erased a stream's closed/ack state
  and let new revisions append to a closed stream.*
- "Optional file" means exactly `FileNotFoundError`. Swallow that; let
  other `OSError`s propagate.
- A silent `except ...: pass` needs to be provably safe. If suppression is
  the correct behavior, `contextlib.suppress(SpecificError)` states that
  intent; if you find yourself suppressing `Exception`, the design is
  wrong, not the linter.
- Legitimate broad catches exist — top-level daemon loops, plugin/extension
  boundaries, "must never crash the caller" telemetry. There: log the
  exception with context and add `# noqa: BLE001` plus one line saying why
  the boundary must not propagate.
- Best-effort side effects (history appends, notifications) still get a
  logged failure. An unrecorded skip is indistinguishable from success.
  *Evidence: hivemind's history append is `except Exception: pass`, so
  "sends append history events" is unverifiable when it matters.*

## Filesystem state

All AFS state files go through `afs.atomic_io`; plain `Path.write_text`
is for content whose torn or lost state is genuinely harmless.

- `atomic_write_text(path, text)` — mutable state files (JSON stores,
  manifests, queues). Readers must never observe a partial file.
  *Evidence: hivemind subscription files were written with `write_text`;
  a torn write silently reset the subscription.*
- `atomic_write_text(..., durable=True)` — state that must survive a
  crash immediately after the call returns (acks, watermarks, claims).
- `exclusive_create_text(path, text)` — artifacts that must never be
  overwritten (immutable revisions, one-shot claims). O_EXCL makes
  create-or-fail atomic; O_NOFOLLOW refuses planted symlinks.
- `secure_mkdir(path, mode=0o700)` — any directory holding private state.
  `Path.mkdir(mode=..., parents=True)` applies the mode **only to the
  leaf**; intermediates get umask defaults. *Evidence: artifact and
  handoff directories shipped with umask-mode intermediate directories.*
- Private files are 0600, private directories 0700, and permissions are
  applied **before** content becomes visible at the final path (both
  helpers do this for you).
- One exception: `agent_jobs.py` keeps a local, stricter publish helper
  whose error type distinguishes pre- from post-publish failure. Don't
  "simplify" it onto the shared helper.

## SQLite

`with sqlite3.connect(...)` commits or rolls back — it does **not** close
the connection. Use `contextlib.closing(...)` (or an explicit
`try/finally: connection.close()`) or every call leaks a file descriptor.
*Evidence: found leaking in `antigravity_status.py` and again in new
search code the same week — this misconception recurs; assume readers
have it.*

## Enforce at the resource, not the wrapper

Authorization, deny-lists, and validation belong to the layer that owns
the resource (queue, file store, embedding provider), not to the newest
API in front of it. Any older surface that reaches the resource directly
must inherit the same guard — especially during compatibility windows,
when legacy paths stay alive by design.

*Evidence (both from one review round): message-scope authorization lived
only in the new `MessageBus` view while legacy hivemind reads returned
any project's scoped messages; embedding deny-lists lived in the new
hybrid engine while the legacy index path happily embedded `.pem` keys
and symlink-escaped content.*

Corollary: security filename patterns are contracts — implement the
contract, not an approximation. `.env*` means `.envrc` too.

## Typing

- New modules must pass `mypy -p afs` with zero errors; they are enforced
  automatically (only baseline-listed modules are exempt).
- Narrow types per field, not in bulk: mypy cannot narrow through
  `all(isinstance(x, str) for x in (a, b, c))`. Write one `isinstance`
  check per variable and raise on failure. *Evidence: a bulk-`all()`
  guard added three baseline errors that a per-field guard avoids.*
- `# type: ignore` takes an error code and a reason, or it doesn't merge.

## Tests

- A test's name is a claim; the test must exercise what it names.
  *Evidence: a test named `..._audit_and_plan_...` never invoked the
  plan CLI path it named.*
- State-machine and containment code gets adversarial cases, not just
  happy paths: path traversal (`../`, absolute), symlinks (dangling,
  escaping), double-create, torn/interrupted writes, timezone-offset
  timestamps.
- Tests must not depend on ambient environment. `AFS_CONTEXT_ROOT`
  leaking into a test run fails ~10 unrelated tests; use `tmp_path` and
  `monkeypatch.delenv`/`setenv` explicitly.
- Failure-injection beats sleep-based racing: monkeypatch the failing
  syscall (`os.replace`, `os.fsync`) instead of racing timeouts.

## Compatibility surfaces

- CLI/MCP JSON output changes are additive-only during a compatibility
  window: never rename or remove fields; emit new optional fields.
- A deprecated-but-supported surface that silently diverges from the new
  surface's behavior is a delivery bug, not a compat feature. If the old
  path can't be routed through the new enforcement, it should warn loudly
  or refuse.

## Shrinking the baselines

```sh
# see remaining exception-hygiene debt
ruff check src/ tests/ --select BLE001,S110,S112 --statistics

# see remaining typing debt for a module you're touching
# (temporarily delete its line from the mypy overrides, then)
mypy -p afs
```

Fix, delete the baseline entry, and include both in the same commit as
your change.
