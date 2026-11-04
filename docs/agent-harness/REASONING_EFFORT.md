# Choose reasoning effort by task

Start bounded work at `medium` when selection is available and authorized. This is a house recommendation, not a change to model defaults or settings. Honor explicit user choices and the selected host's supported controls. Choose higher effort from the start when complexity warrants it; optimize for the accepted outcome and total work, not the lowest setting.

| Effort | Use it for this work. |
|---|---|
| `low` | Find symbols, format text, update ID-alias docs, or make a narrow edit whose solution is already known. |
| `medium` | Implement a bounded feature, adjust scene configuration, add focused tests, or review a contained change. |
| `high` | Debug uncertain mapper/camera state across subsystems, reason about 65816 timing/register invariants, or weigh coupled design choices. |
| `xhigh` | Resolve deep ambiguity left after a focused evidence-based attempt, or a consequential architecture decision such as AFS ownership/persistence boundaries. |
| `max` | Use for demanding problems when substantial reasoning serves the outcome; no arbitrary failed-attempt count is required. |
| `ultracode` | Include this additional effort exposed in Justin's UI when the task or his preference calls for it. Use the actual available selector; its API identifier remains unverified. |

Use these as flexible starting points, not rigid ceilings or guarantees. Higher effort can be useful; keep context focused and preserve required verification at every level.

Escalate for a specific unresolved question, conflicting evidence, or interacting invariants. Pause for missing data instead of retrying with more reasoning. Isolate the hard decision, then downshift mechanical follow-through. Preserve decisions, evidence, tests, and remaining uncertainty in the existing handoff so the next worker doesn't redo discovery. Explain an escalation briefly when useful; don't announce a ritual on every call.

## Distinguish the controls

Confirm the actual choices exposed by the selected host. Justin reports `max` and `ultracode` as additional UI efforts; include them even when a cached API catalog differs. Model choice, effort, service/fast mode, and plan quota remain separate. Don't silently map `ultracode` to `ultra` or send an unverified API identifier. Catalog capabilities don't grant delegation or spending authority.

Observed `~/.codex/models_cache.json` on 2026-10-30 with `fetched_at=2026-10-30T23:16:30.511430Z`: it lists `low`, `medium`, `high`, `xhigh`, and `max`; some models also list `ultra`. It has no `ultracode` entry, which differs from Justin's reported UI; `ultra` remains a distinct catalog identifier. Service tiers are separate fields. Recheck support when changing models or hosts. This snapshot doesn't establish billing, remaining quota, or a linear usage/latency benefit from lowering effort.

Justin reports fast mode as roughly 1.5× speed with more usage. Treat that as his observed tradeoff, not a constant speed guarantee or billing multiplier. Prefer the actual exposed control.

Justin prefers GPT-6-Astra for big problems and finds GPT-6.1-Sol useful for everyday work, with a tendency toward overengineering to watch for. Keep scope bounded on either model. Model distillation lineage and “loop” counts remain speculation, not routing facts.

## Compare only when useful

An optional matched comparison can hold the task, inputs, acceptance criteria, and verification constant while comparing medium with a chosen higher effort. Record accepted outcomes, rework, elapsed time, and actual usage with source/time/units when exposed; mark missing usage unknown. Avoid concurrent duplicate writers. This proposes a comparison, not a benchmark run or new spending authority.
