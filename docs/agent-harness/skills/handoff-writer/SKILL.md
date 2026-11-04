---
name: handoff-writer
description: Write durable, task-specific handoff notes for another agent or future session. Use when asked for a handoff, continuation prompt, next-agent prompt, AFS handoff, breakpoint note, or when work must pause cleanly.
---

# Handoff Writer

## Goal
Leave enough verified state for the next agent to continue without reconstructing the conversation.

Follow the [shared transfer and usage rules](../../AGENT_HARNESS_CONTRACT.md#handoffs-and-ownership) and [compact payload guidance](../../AGENT_COMMUNICATION_STANDARD.md#agent-to-agent-handoffs).

## Workflow
1. Find the canonical location.
   - AFS task: discover context and write under `scratchpad/handoffs/`.
   - Repo task: use an existing `docs/`, `scratchpad/`, or handoff file pattern.
   - Chat-only request: return a concise copy-paste prompt.
2. Capture only useful facts:
   - goal and done criteria
   - files changed or intentionally left alone
   - commands run and outcomes
   - current blockers and exact next step
   - important paths, commit hashes, ports, device names, model IDs, socket paths
3. Mark evidence as verified, inferred, or unverified.
4. Avoid dumping chat history. Prefer concise routing and proof handles.
5. If the user asked for an AFS handoff, write the actual file before final response.

## Output Shape
Use the shared compact payload headings unless the task has an established handoff format.

## Guardrails
- Do not update `memory` or `knowledge` mounts unless explicitly requested.
- Do not invent completed verification.
