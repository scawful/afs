# AFS Executive Technical Summary

AFS (Agentic File System) is a filesystem-native context platform for AI-agent
workflows. It turns the parts of agent work that are usually scattered across
prompts, chat history, tool glue, local scripts, and ad hoc notes into a durable,
inspectable `.context/` workspace that any compatible agent or MCP client can
use.

In product terms: AFS is an operating layer for context. It helps agents start
with the right project memory, act through governed tools, leave useful records
behind, and hand work to the next agent or human without losing the thread.

## The problem it solves

Modern agent workflows are powerful but brittle:

- context is fragmented across chats, docs, terminals, tools, and ticket systems
- handoffs are lossy because the next session cannot reliably see what mattered
- tool access is hard to govern consistently across CLI, MCP, and editor clients
- model upgrades and vendor swaps often force teams to rebuild workflow glue
- evaluation, approvals, and lineage are frequently bolted on after the fact

AFS makes those concerns explicit by storing context, memory, scratch work,
tool surfaces, approvals, events, and handoffs as structured filesystem-backed
artifacts.

## What AFS provides

### 1. A durable context workspace

Each project can have a `.context/` root with typed mounts such as:

- `scratchpad` for live working notes and session state
- `memory` for durable consolidated knowledge
- `knowledge` for reference material
- `history` for events and run records
- `items` for tasks and queued work
- `hivemind` for optional cross-agent messages
- `global` for indexes and shared metadata

This gives agents a stable substrate that persists beyond any one chat window.

### 2. Session bootstrap and handoff

Agents begin with an AFS bootstrap instead of a blank prompt. The bootstrap
summarizes repo health, scratchpad state, recent runs, handoffs, active tasks,
work-assistant state, and recommended next actions. At the end of work, agents
can write structured handoffs that the next session can ingest directly.

### 3. Context construction under a budget

AFS indexes project context and builds token-budgeted packs for different model
or client surfaces. This keeps context selection transparent: what was loaded,
why it was loaded, and where it came from can be inspected.

### 4. Governed tool access

The MCP server exposes a small default tool catalog for agent clients:

- `context.status`
- `context.query`
- `context.read`
- `context.write`
- `context.list`

Larger operational surfaces are available when explicitly enabled, but the
default posture is intentionally narrow. AFS also supports Claude-safe
underscore aliases (`context_status`, `context_query`, and so on) for clients
that reject dotted MCP tool names.

### 5. Human-centered approvals and work records

AFS includes work-assistant primitives for people, relationships, review routes,
approvals, and activity. External writes can be routed through approval records
and human terminal confirmation instead of being executed implicitly by an
agent.

### 6. Extension boundary

Core AFS stays focused on reusable context infrastructure. Domain-specific
connectors, private workflows, model-training stacks, and specialized skills
belong in companion extension repos. That keeps the public core understandable,
portable, and suitable for product or platform review.

## Why it is different

- **Harness-neutral.** AFS state can serve Codex, Claude, Gemini, editor
  extensions, and MCP clients without binding the context model to one vendor.
- **Filesystem-native.** The durable interface is files and JSON artifacts, not
  an opaque SaaS backend.
- **Traceable.** Runs, events, prompts, handoffs, approvals, and context packs
  can be inspected after the fact.
- **Governed by default.** Tool listing, sensitivity checks, approval records,
  and narrow default MCP exposure reduce accidental overreach.
- **Extensible without polluting core.** Specialized domains can plug in through
  extension manifests while core remains a general context platform.

## Research lineage

AFS is an independent implementation and extension of the agentic file system
abstraction described in **“Everything is Context: Agentic File System
Abstraction for Context Engineering”** (arXiv:2512.05470). The paper argues that
context engineering needs a persistent, governed filesystem abstraction for
memory, tools, knowledge, and human input. AFS maps that idea into a practical
Python repository with CLI workflows, MCP surfaces, context indexing, session
bootstrap, handoffs, approval gates, and extension packaging.

AFS is not affiliated with the paper authors, CSIRO Data61, ArcBlock, or the
AIGNE reference implementation.

## Demo path for a product conversation

A concise demo can show the platform without domain-specific material:

1. `./scripts/afs session bootstrap --json` — show how an agent starts with
   repo state, scratchpad, tasks, and handoff context.
2. `./scripts/afs context query "MCP tool surface" --path .` — show indexed
   retrieval over project context.
3. `docs/MCP_SERVER.md` — show the narrow default MCP tool surface and the
   compatibility aliases for clients that need underscore-only tool names.
4. `./scripts/afs work` — show people/review/approval records for governed
   external collaboration.
5. `docs/LINEAGE.md` and `docs/EXTENSION_MIGRATION.md` — show research lineage
   and the deliberate separation between core platform and extensions.

## Current maturity

AFS is an alpha-stage platform repo, but the core concepts are implemented and
usable locally: context roots, typed mounts, index/query, session bootstrap,
MCP tools, agent run records, handoffs, work approvals, profiles, and extension
loading. The repo is being cleaned so the public surface represents the core
platform clearly, without private or domain-specific implementation detail.

## One-sentence version

AFS is a durable context operating layer for AI agents: it gives agents a
shared filesystem-backed memory, governed tools, traceable handoffs, and an
extension model so teams can make agent workflows repeatable instead of
rebuilding context from scratch every session.
