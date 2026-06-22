# Training And Feedback RFC

## Status

- Draft
- Audience: core `afs`, `afs-scawful`, and downstream agent/training workspaces

## Why This Exists

AFS already has generic context, event, session, and extension primitives. In
practice, training and feedback workflows still fall back to ad hoc shell loops,
one-off run wrappers, and repo-specific scripts.

That creates four problems:

- agents guess with `bash` instead of using stable tools
- run state lives in logs and command history instead of readable artifacts
- dataset pruning, eval failures, and feedback promotion are hard to compare
- downstream repos cannot reuse the same operator surface without copying glue

The goal of this RFC is to standardize the reusable infrastructure in core AFS
while keeping model-family, persona, and personal deployment policy in
extensions and downstream repos.

## Design Principle

Train agents to use explicit AFS tools for dataset, run, eval, and feedback
workflows. Treat `bash` as the escape hatch, not the default control surface.

This improves:

- determinism
- auditability
- reusable automation
- metric collection
- cross-repo portability

## Scope Split

### Core `afs`

Core AFS should own generic, reusable primitives:

- lifecycle orchestration for datasets, runs, evals, and feedback
- stable CLI and MCP command surfaces
- human-readable artifacts under `.context`
- event and metric schemas
- extension hooks for custom backends and policies

### `afs-scawful`

`afs-scawful` should own reusable but opinionated extensions:

- local log ingestors for Claude, Gemini, Codex, and similar tools
- backend adapters for MLX, Unsloth, Vast, LM Studio, Ollama, and local model managers
- scoring heuristics for tool use, reasoning discipline, and voice policy
- domain-specific dataset builders and feedback synthesizers

### Downstream Repos

Repos like `scawfulbot`, `halext-code`, and future training projects should own:

- concrete job specs
- project-specific datasets and eval packs
- persona policy
- promotion thresholds
- model selection

## Non-Goals

This RFC does not move the following into core `afs`:

- persona training recipes
- domain corpora
- local machine deployment paths
- model-specific prompt strategy
- subjective promotion policy for a specific assistant family

## Core Concepts

### Run Spec

A declarative file describing a training or evaluation job.

Example fields:

- `kind`: `train`, `eval`, `dataset_build`, `feedback_synthesize`
- `backend`: `mlx`, `unsloth`, `custom`
- `dataset`: logical dataset name or path
- `preset`: logical preset name
- `hyperparameters`
- `filters`
- `eval_packs`
- `artifacts`

Run specs should be portable and readable without knowing shell flags.

### Dataset Artifact

A dataset is not just a JSONL file. It is a tracked artifact bundle with:

- source provenance
- row counts
- outlier stats
- schema validation results
- quality filter reasons
- hashes for train/valid/test or pair splits

### Eval Pack

An eval pack is a named, versioned set of checks with:

- task cases
- expected tool behavior
- forbidden tool behavior
- pass/fail rules
- metric extraction rules

### Feedback Record

A feedback record is the durable unit for turning observed failures into future
training data.

Example sources:

- failed eval cases
- human review notes
- production traces
- tool-routing regressions
- unnecessary reasoning or tool spam

### Run Event Stream

Every dataset build, eval, or training job should emit structured events that
can be queried by both humans and agents.

## Proposed CLI Surface

Core AFS should standardize a command surface like this:

```bash
afs training dataset build <spec-or-name>
afs training dataset stats <dataset>
afs training dataset outliers <dataset>
afs training dataset validate <dataset>
afs training dataset prune <dataset>

afs training run start <run-spec>
afs training run status <run-id>
afs training run stop <run-id>
afs training run logs <run-id>
afs training run artifacts <run-id>

afs training eval run <eval-spec>
afs training eval status <eval-id>
afs training eval report <eval-id>

afs training feedback record <input>
afs training feedback list
afs training feedback synthesize <dataset-or-pack>
afs training feedback promote <feedback-id>
```

Existing generic commands like `training freshness-gate`, `training extract-sessions`,
and `training generate-router-data` should remain usable, but the long-term goal
is to fold them into a coherent lifecycle instead of a bag of unrelated subcommands.

## Proposed MCP Surface

Core AFS should expose matching MCP tools so agents can operate without shell
guesswork.

Suggested families:

- `training.dataset.build`
- `training.dataset.stats`
- `training.dataset.outliers`
- `training.dataset.validate`
- `training.dataset.prune`
- `training.run.start`
- `training.run.status`
- `training.run.stop`
- `training.run.logs`
- `training.eval.run`
- `training.eval.report`
- `training.feedback.record`
- `training.feedback.synthesize`
- `training.feedback.promote`

These tools should return structured payloads plus paths to readable artifacts.

## Artifact Layout In AFS

Default writable state should live under scratchpad.

Suggested layout:

```text
.context/
  scratchpad/
    training/
      datasets/
        <dataset-id>/
          manifest.json
          stats.json
          outliers.json
          filters.json
          notes.md
      runs/
        <run-id>/
          status.json
          status.md
          events.jsonl
          metrics.json
          artifacts.json
          logs/
      evals/
        <eval-id>/
          report.json
          report.md
          failures.jsonl
      feedback/
        inbox/
        promoted/
        rejected/
```

This keeps the current session state writable and reviewable. Durable summaries
can later be promoted into `memory` or `knowledge` deliberately.

## Shared Schemas

Core AFS should publish schemas for:

- dataset manifest
- run manifest
- run status snapshot
- run event
- eval case
- eval report
- feedback record
- feedback promotion decision

These should also be exposed as MCP schema resources, following the same style
as the existing `afs://schemas/...` resources.

## Metrics That Become Easy Once Tools Are Standardized

### Tooling Metrics

- first-tool correctness
- unnecessary reasoning-tool usage rate
- unnecessary `bash` fallback rate
- tool spam rate
- stop-after-evidence rate

### Dataset Metrics

- accepted vs rejected rows
- rejection reasons by source
- outlier counts by token/char band
- duplicate rate
- schema violation rate

### Run Metrics

- time to first progress report
- time to first checkpoint
- completion vs interruption rate
- failure causes by backend
- memory/latency summary by preset

### Feedback Metrics

- failures converted into feedback
- promoted vs rejected feedback
- repeat regression rate after promotion

## Extension Model

Core should define interfaces. Extensions should register implementations.

Examples:

- `dataset_builder` plugins
- `run_backend` plugins
- `eval_runner` plugins
- `feedback_synthesizer` plugins
- `metric_extractor` plugins

This lets core `afs` stay generic while `afs-scawful` and others contribute
useful adapters without forking the operator surface.

## Relationship To Existing AFS Surfaces

This RFC builds on existing AFS strengths rather than replacing them:

- `events.*` already provides a durable event model
- `session.*` already provides agent bootstrap and workflow hints
- `context.*` already provides deterministic file access
- extensions already provide the right boundary for domain logic

The main change is to stop treating training and feedback as shell-only
operator workflows and start treating them as first-class AFS objects.

## Rollout Plan

### Phase 1: Normalize Artifacts

- define manifests and event schemas
- standardize run directories in scratchpad
- wrap current ad hoc scripts with consistent status/report outputs

### Phase 2: Stabilize CLI

- add `training dataset ...`, `training run ...`, `training eval ...`, and
  `training feedback ...` lifecycle groups
- keep legacy commands as compatibility aliases where reasonable

### Phase 3: Add MCP Surface

- expose the same lifecycle through MCP tools
- add schema resources and status resources

### Phase 4: Move Agent Workflows Off Bash

- teach agents to prefer training tools over shell scripts
- use shell only for custom backend escape hatches

### Phase 5: Promote Extension Adapters

- move log-ingest, local backend, and scoring adapters into `afs-scawful`
- let downstream repos focus on specs and policy, not orchestration glue

## Recommended First Implementation Slice

The smallest useful slice is:

1. `training.run.start`
2. `training.run.status`
3. `training.run.stop`
4. `training.dataset.stats`
5. `training.dataset.outliers`
6. `training.eval.report`
7. `.context/scratchpad/training/runs/<run-id>/status.{json,md}`

That is enough to replace most ad hoc launch/inspect/stop shell loops and gives
agents a deterministic surface immediately.

## Boundary Rule

If a feature is about orchestrating datasets, runs, evals, feedback, artifacts,
or metrics in a generic way, it belongs in core `afs`.

If a feature is about a specific model family, persona, corpus, or personal
deployment environment, it belongs in an extension or downstream repo.
