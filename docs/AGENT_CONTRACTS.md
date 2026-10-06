# Compact startup and extension contracts

AFS supplies project facts, recoverable state, and enforced boundaries. The host
owns its reasoning loop, skill loading, and opaque provider conversation state.
This revision reduces generic prompt coaching and adds small extension APIs.
It does not install or update host configurations.

## Startup and prompt settings

```sh
afs session bootstrap --short --json
afs session bootstrap --native-skills --token-budget 2000 --no-write-artifacts --json
```

`--short` reads only `state.md` and `deferred.md` from the authorized scratchpad
scopes, at most 1,200 characters per note, and returns configured skill roots.
Truncated notes carry a flag and a source path. It does not scan indexes, match
skills, register an agent, or write bootstrap artifacts. It is a brief, not a
complete health report. Omit `--short` for the existing full packet. The full
packet's token budget is an estimate, not a provider tokenizer limit.

The `afs.session.bootstrap` MCP prompt also accepts `short=true` and
`native_skills=true`. In a v2 context, supply the registered `project_path` to
select that project; omitted project scope follows the existing common-scope
rules. `afs fs ... messages` is a compatibility alias for the legacy `hivemind`
mount type; it does not move files or change serialized mount names.

Generated model prompts now default to `AFS_PROMPT_SCAFFOLDING=minimal`. This
omits the generic execution-profile block and prescribed repair loop. It keeps
base instructions, task facts, output schemas, repository policy, verification
requirements, and communication approval guidance. Set
`AFS_PROMPT_SCAFFOLDING=full` to restore the previous coaching for comparison.
Python callers can pass `scaffolding="minimal"` or `scaffolding="full"` directly.

Set `AFS_NATIVE_SKILLS=1` when assembling model prompts for a host that loads
skills itself. AFS supplies roots without repeating matched bodies. Use
`--native-skills` when building the full startup packet to also avoid the match
operation. Existing lifecycle-hook integrations retain their current behavior;
switch their startup command deliberately rather than running both loaders.

## Conditional file writes

```sh
afs fs read scratchpad notes/status.md --json
afs fs write scratchpad notes/status.md --input updated.md --if-match <sha256> --json
afs fs write scratchpad notes/new.md --input new.md --if-match missing --mkdirs --json
```

JSON reads return `sha256` for the exact raw bytes read, before newline or
decoding normalization. A write accepts that digest, or `missing` to require
creation. A mismatch returns a failure without changing the destination file.
Read again and reconcile the change; do not automatically retry without the
precondition. Appends also accept `--if-match` and publish a complete replacement
atomically. Existing ordinary writes participate in the same writer lock.

MCP `context.write` and compatibility `fs.write` accept `if_match`; both use the
same implementation as the filesystem API. `context.read` and `fs.read` return
the raw-byte digest. Python extensions can use
`ContextFileSystem.write_text(..., if_match=digest)` after normal scope resolution.

The lock coordinates these AFS file writers. Direct editor writes, file moves,
deletes, or other processes that do not use the contract are outside the
compare-and-write guarantee. Publication is atomic but does not promise
power-loss durability. POSIX writers lock the parent directory, open its
components without following links, and publish relative to that directory
descriptor. Windows uses a persistent sibling lock file and requires separate
platform acceptance testing.

## One-file delivery

```sh
afs fs push scratchpad notes/status.md \
  --host secondary-host \
  --remote-context-root /path/to/context \
  --remote-project /path/to/project \
  --if-match <destination-sha256> --json
```

This sends one UTF-8 text file over SSH to `afs fs write` on the destination.
The destination must run an AFS version supporting `--if-match`. Obtain the
destination hash with a destination read. Use `missing` only to create a new
file. `--destination`, `--destination-mount`, `--remote-afs`, and `--mkdirs` are
available when paths differ. SSH host configuration supplies authentication.

Content travels on stdin; command arguments are quoted separately. The
destination performs the conditional write under its own lock. The command
does not delete files, synchronize directories, or retry conflicts. After a
timeout or lost SSH connection, read the destination before retrying: the
remote write may have completed without delivering its receipt.

## Bind approval to final content

`WorkAssistantStore.create_approval` records `content_sha256` over a canonical
envelope containing `target_system`, `target_id`, `action`, and `preview`.
The preview must contain the final outgoing text, attachment content digests,
and delivery options. JSON object-key order is ignored. Text whitespace,
Unicode, array order, destinations, and message splitting change the digest.
Reusing a deduplication key with different content raises an error.

Existing approved records without a content digest return to pending when the
store opens. A new human decision binds their content. Approval decisions also
include the content digest in the human-authorization scope. Execution validates
the stored envelope both before and after claiming the approval.

Extensions own the actual send and must validate after their final transformation:

```python
from afs.approval_content import validate_approved_content

validate_approved_content(
    approval,
    target_system=final_target_system,
    target_id=final_target_id,
    action=final_action,
    preview=final_dispatch_payload,
)
send(final_dispatch_payload)
```

Use the approval supplied by the claimed executor payload. This helper validates
the content and human-confirmed status; the existing store owns claims, retries,
and execution results. It does not make arbitrary extension code safe if that
code omits the check or changes the payload afterward. Generic supervisor
`ApprovalGate` requests retain their existing agent/action/detail contract;
use the work approval store for content-bound external dispatch.

## Events, hooks, and hidden tools

```sh
afs events emit task.completed --source private-agent \
  --data '{"task_id":"example","result":"checked"}' --json
```

Python callers use `afs.external_events.emit_event`. Core assigns the event ID,
timestamp, type `external`, and source `afs.events.emit`. The reporting source
is metadata; caller data stays inside the payload. Data must be a JSON object
of at most 32 KiB. Core history redaction and logging settings apply. Disabled
logging produces a failure instead of a success receipt. These events never
create work-assistant approval state.

`afs doctor` warns about hook names core never emits and AFS package entry-point
groups it never loads. Custom hooks remain possible when an extension explicitly
dispatches them. Use `extension.toml` module declarations for discovery.

Extension tools omitted from the active catalog cannot be called just by knowing
their name. To allow deferred calls, an extension must set
`allow_hidden_call=True` on its `MCPToolDefinition`, or
`"allow_hidden_call": True` in a returned Python tool dictionary. This is per-tool
authorization. Agent `AFS_ALLOWED_TOOLS` restrictions still apply. Visible tools
remain callable under the existing agent restrictions; core compatibility tools
retain their existing behavior. Audit extensions before upgrading because
previously hidden extension tools may now be denied.

## Evidence and remaining evaluation

Provider guidance supports reducing generic coaching while retaining concrete
requirements. Anthropic discusses progressive disclosure and prompt simplification
in its [Claude 5 context engineering guidance](https://claude.dev/blog/the-new-rules-of-context-engineering-for-claude-5-generation-models/).
Google recommends concise prompts with native thinking in the
[Gemini 3 developer guide](https://ai.google.dev/gemini-api/docs/gemini-3).
These are design inputs, not measurements of AFS performance.

Keep model settings separate: consult the
[Opus 5.5 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5),
[Gemini Flash migration guide](https://ai.google.dev/gemini-api/docs/latest-model),
and [Gemini thinking guide](https://ai.google.dev/gemini-api/docs/thinking)
for the selected API and model. Sampling parameters and provider conversation
state are adapter concerns; this patch does not change them.

Compare minimal and full prompts on the same real tasks with the same model,
effort, tools, and acceptance criteria. Measure completed artifacts, permission
violations, tokens, elapsed time, and recovery after interruption. Unit tests
establish the local contracts; they do not establish a model-quality improvement,
remote-host acceptance, or the memory savings from retiring helper processes.
