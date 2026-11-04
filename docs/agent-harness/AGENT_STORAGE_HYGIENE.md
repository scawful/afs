# Agent Storage Hygiene

User-authorized policy, effective 2026-10-17. Applies to agent-generated local
build output, temporary artifacts, application backups, and nightly copies.

## Authorization and scope

Routine cleanup that meets this policy is pre-authorized. Do not ask the user
to approve each deletion or archive. This is the explicit exception to blanket
"ask before rm" instructions in the shared harness and workspace guidance.
Higher-priority tool restrictions still apply.

Clean up resources created or used by the task before declaring it complete.
Review older agent-generated output in that same project when building or
deploying again. Do not turn an ordinary task into a whole-home cleanup.

Source code, Git metadata/worktrees, uncommitted changes, models, datasets,
ROMs, saves, recordings, sample libraries, credentials, user backups, and agent
session history are outside this automatic deletion policy. Preserve those
even when an agent created them or they are inside an ignored directory.
Archiving unique data needs its own task authorization and verified copies.

## Before archiving a chat or worktree

Chat archiving can remove attached managed worktrees. Git recovery snapshots
omit ignored files. Before any chat/worktree archive or managed-checkout removal:

1. Run `python3 ~/src/scripts/worktree-preserve inventory --thread <exact-chat-id>`
   and save its complete output outside every checkout. It must cover every
   attached worktree and the chat's secondary primary checkout. Missing or
   unreadable attachment inventory blocks the archive.
2. Review every ignored/untracked entry and the total byte count. Give capture
   an explicit reviewed byte budget. ROMs, saves, states, profiles, helpers and
   runtime evidence remain protected regardless of ignore status. Do not exclude
   unknown contents or copy whole build trees without review; retain the
   checkout if full preservation cannot proceed.
3. Run `python3 ~/src/scripts/worktree-preserve capture --inventory <file> --max-bytes <budget>`.
   Its default store is `~/.local/share/worktree-preservation`, outside worktrees.
   Run `python3 ~/src/scripts/worktree-preserve restore --receipt <file> --destination <absent-folder>`
   to verify the actual captured bytes and metadata in a disposable destination.
4. Run `python3 ~/src/scripts/worktree-preserve check --receipt <file>` immediately before the
   separately authorized archive. A PASS is valid for 60 seconds and only for
   that exact chat and inventory. Any HOLD, changed/missing/new file, changed
   attachment, active process or pending manual test blocks the archive.
5. Record the receipt, restore result and durable store path in AFS. Keep the
   store and any pending-test artifacts; this policy does not authorize pruning
   preservation objects or stopping a process to make the gate pass.

The tool never archives or deletes originals. It fails closed on incomplete inspection,
insufficient capacity, unstable reads or failed verification.

The Codex automatic-cleanup toggle controls age/count cleanup only. Explicit
chat archives still remove worktrees, and direct UI/API archives can bypass this
guard. Do not use those archive paths before this preservation gate passes.
This is a required agent procedure, not an interception installed in the app.

## Retention defaults

| Artifact | Keep | Cleanup point |
| --- | --- | --- |
| Reusable build output | One directory per active checkout and required toolchain/configuration; reuse it instead of timestamping copies | Remove superseded configurations when no task needs them |
| Disposable task/worktree builds | Only output needed for an active task, pending acceptance, or a specific reproduction | Remove bulk output when the task completes; review inactive output older than 7 days on the next project task |
| Application rollback copies | The 2 newest verified working rollback copies per app, plus any explicitly required recovery copy | Prune older generated backups only after the replacement passes launch and task-relevant checks |
| Local nightly copies | The current target and the 2 newest verified working previous versions, plus any active test candidate | Prune older copies after verifying the current version and resolving current symlinks |
| Temporary staging/extractions | Only those still needed for the current task | Remove on success or handled failure after preserving useful diagnostics |

Age is a review trigger, not proof that another task has finished. Do not delete
an inactive build until its task is complete or confirmed abandoned. Pending
user acceptance protects its candidate and recovery copy. Record retained
exceptions with the task, path, size, reason, and a review date within 7 days;
the review date does not authorize deletion by itself.

Keep the compiler cache at the user-approved **10 GiB** limit. Use its supported
cleanup command; do not raise the limit to solve a build problem. Do not copy
regenerable caches or whole build trees to external storage merely to keep them.

## Before removing a directory

1. Identify the exact canonical path, artifact type, and task ownership. Reject
   symlinks, unexpected mounts, and unknown contents. A name such as `build`,
   `.backup`, or `predeploy`, or Git-ignore status alone, is insufficient proof.
2. Check running builds, app executable paths, open files, process working
   directories, and task ownership. Protect active or shared output and targets
   of current-version symlinks. Never stop another task to make cleanup possible.
3. Inspect for protected data. Preserve small logs, build configuration, exact
   revision, reproduction commands, and required evidence outside the disposable
   tree. If unique data cannot be separated safely, retain the directory.
4. Recheck the selected paths before deletion. Use explicit paths and existing
   retention tools where available; never use broad `build*`, app, or home globs
   directly as deletion arguments. Do not use sudo or change ownership to clean up.
5. Verify removal, retained current/rollback paths, and source/Git state. Record
   removed paths and measured sizes in a concise task receipt. Report any retained
   exception once; do not repeatedly request permission for eligible cleanup.

## Storage checks during work

Check free space before a large build, model download, or app staging operation.
If the internal disk has less than **30 GiB free**, clean eligible generated
output in scope before allocating more. If agent build output for one project
across its worktrees exceeds **50 GiB**, review obsolete copies before adding
another build. Protect active work even when these thresholds are exceeded;
report unresolved space pressure instead of deleting protected data.

Prefer targeted builds and tests. Do not rebuild every worktree to verify a
documentation change or recreate output just deleted by cleanup.

## Local application examples

For Yaze, reuse the documented preset/build directory in the active checkout.
Apply task-build cleanup under `~/src/hobby/yaze-worktrees` without deleting the
worktrees themselves. Apply the application retention rule to known generated
`/Applications/.yaze-before-*` and `yaze.app.predeploy-*` backups, including case
variants. Apply nightly retention to `~/.local/yaze/nightly/releases` while
protecting the resolved `~/.local/yaze/nightly/current` target.

For Mesen, apply the same application retention rule to known generated copies
under `/Applications/Mesen2-Backups`. Keep installed app bundles and all user
ROMs, save states, and profiles outside cleanup. These examples are candidate
locations, not blanket deletion commands.

## Completion report

Close build/deployment work with one storage status: what was removed, approximate
space reclaimed, and which rollback copies or pending-task artifacts remain.
If no generated output was created or made obsolete, no storage report is needed.
This policy is an instruction for agents; it does not install a scheduled cleaner.
