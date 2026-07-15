# Autonomous Optimization Protocol

## Status

- Protocol: `v1`
- Decision algorithm: `pareto-gate-1.0`
- Execution authority: **none**

AFS v1 defines a small language-neutral contract for comparing one immutable
candidate with one immutable baseline. It is the evidence and decision layer of
a future hill-climbing system, not a candidate generator, trial runner, approval
system, or deployment controller.

The command is deliberately pure:

```bash
./scripts/afs optimize decide \
  --baseline examples/optimization_gate/baseline.json \
  --candidate examples/optimization_gate/candidate.json \
  --policy examples/optimization_gate/policy.json \
  --json
```

It reads three bounded JSON files, writes only to stdout/stderr, and never starts
a subprocess, calls a model, edits an artifact, or changes active AFS state.

## Why Protocol First

Useful self-improvement needs separate trust boundaries:

```text
untrusted proposer -> isolated trial runner -> deterministic evidence
                   -> trusted decision gate -> human review
                   -> separately authorized activation -> canary + rollback
```

LLMs may eventually propose prompt, policy, skill, code, or tool-routing
variants. They must not grade their own work or authorize promotion. AFS core
owns stable evidence and decision contracts; mutation strategies and domain
evaluators remain extension-owned.

This matches current agent-evaluation practice: define tasks, trials, graders,
outcomes, and stable environments; run capability and regression suites
together; and prefer deterministic code-based graders where possible. See
[Anthropic's agent eval guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## Versioned Schemas

The canonical JSON Schema 2020-12 files ship as package data:

```text
src/afs/protocols/optimization/v1/
  evaluation.schema.json
  policy.schema.json
  decision.schema.json
```

They are also available through the existing CLI and MCP schema-resource
surface:

```text
afs://schemas/v1/optimization/evaluation
afs://schemas/v1/optimization/policy
afs://schemas/v1/optimization/decision
```

```bash
./scripts/afs schema show v1/optimization/evaluation
./scripts/afs schema validate \
  --schema v1/optimization/evaluation \
  --file examples/optimization_gate/candidate.json
```

### Evaluation record

An evaluation binds metrics and constraints to:

- candidate and parent IDs
- immutable artifact SHA-256 and optional artifact reference
- evaluation-suite name, version, and case-set SHA-256
- evaluator name/version, run ID, seed, and environment SHA-256
- metric values, units, sample counts, and optional standard errors
- named boolean constraints

`parent_id: "root"` is the v1 sentinel for a lineage root. A candidate compared
against a baseline must name the baseline `candidate_id` as its `parent_id`.

### Policy record

A policy declares:

- objective metrics that must improve by a positive `min_delta`
- guardrail metrics and their maximum allowed regression
- `min_delta` is objective-only: an objective must declare `min_delta > 0`, and
  a policy that declares `min_delta` on a guardrail is rejected as invalid
- explicit units and maximize/minimize direction
- minimum samples per metric
- required hard constraints
- a conservative confidence multiplier (`confidence_z`)
- whether environment fingerprints must match

V1 requires human review. `require_human_approval` is fixed to `true`; it is
policy metadata, not an authorization credential.

### Decision record

The decision contains:

- `rejected`, `inconclusive`, or `eligible_for_human_review`
- stable reason codes and per-metric gate results
- canonical hashes of all three inputs
- a deterministic decision hash
- no timestamp, random ID, approval, activation, or deployment state

An `eligible_for_human_review` result means only that the supplied evidence
passed this comparator. It does not prove the evidence is authentic and it does
not authorize a write. A later approval record must bind reviewer identity and a
one-shot nonce to the candidate, baseline, policy, suite, environment, and
decision hashes.

## Normative Decision Algorithm

`pareto-gate-1.0` performs these steps:

1. Strictly validate all schemas, reject unknown fields, duplicate metric names,
   non-standard JSON numbers, NaN, and infinity.
2. Verify parent lineage, evaluation-suite evidence, evaluator version, seed,
   and (when required) environment fingerprints.
3. Fail a candidate when any required constraint is explicitly false. Missing
   constraints are inconclusive.
4. For every policy metric, convert finite JSON numbers to base-10 decimals,
   calculate with 50 significant digits, and normalize direction so positive
   means better:

   ```text
   adjusted_delta = candidate - baseline          # maximize
   adjusted_delta = baseline - candidate          # minimize
   conservative_delta = adjusted_delta
                        - confidence_z * hypot(baseline_se, candidate_se)
   ```

5. Reject when a conservative delta exceeds a metric's allowed regression.
6. Return inconclusive for mismatched evidence, missing metrics/uncertainty,
   insufficient samples, or no objective meeting its positive `min_delta`.
7. Return `eligible_for_human_review` only when at least one objective improves
   and every constraint and guardrail passes.

This is a Pareto-style gate, not an opaque weighted score. It avoids hiding a
safety, latency, or cost regression inside an aggregate reward.

The v1 standard-error bound is intentionally small and conservative; it is not a
claim of statistical significance. A later protocol can add paired case/trial
records and bootstrap confidence intervals without changing v1 artifacts.

## Determinism And Integrity

- JSON is UTF-8 with sorted keys and no NaN/infinity.
- Threshold arithmetic uses 50-digit base-10 decimal math; derived values must
  remain representable as finite JSON/binary64 numbers.
- Hash input uses AFS v1 numeric canonicalization: plain decimal notation,
  trailing fractional zero removal, no exponent, and negative-zero collapse to
  `0`. Therefore `500`, `500.0`, and `5e2` hash identically, as do `1.2300` and
  `1.23e0`.
- Metric and constraint order is normalized before hashing.
- Identical semantic inputs produce byte-stable `--json` output.
- The decision hash covers the decision body but not the hash field itself.
- Content hashes provide integrity, not provenance, authenticity, or approval.
- Inputs must be regular files and are read through a 2 MiB bounded stream at
  the CLI boundary.

## Exit Codes

| Code | Meaning |
|---:|---|
| `0` | `eligible_for_human_review` |
| `1` | `rejected` |
| `2` | invalid input or schema |
| `3` | `inconclusive` |
| `4` | internal gate error — not an evidence verdict |

Codes `0`, `1`, and `3` are evidence verdicts. Code `4` means the gate itself
failed after inputs were accepted; treat it like `2` (do not promote, fix and
re-run), never like `rejected`.

## Path To Safe Hill Climbing

### P0 — execution boundary

AFS now has a typed, fail-closed execution broker and routes verification
through its portable process backend. Its request hash binds the resolved
executable, exact argv or explicitly enabled legacy shell input, cwd, selected
environment, timeout and output bounds, caller/purpose, and requested backend
policy. Structured argv is the default, and worktrees remain isolation aids
rather than security sandboxes.

The current backend supports only process isolation with inherited networking;
sandbox/container and network-deny requests remain blocked. Background jobs,
hooks, training, and generic agent tools are not migrated by this milestone.
See `docs/EXECUTION_BROKER.md`.

### P1 — immutable trials

Add candidate, eval-pack, trial, and search-state artifacts under
`.context/scratchpad/experiments/<id>/`. Claim work with transactional leases or
compare-and-swap semantics, record full traces, and keep holdout cases hidden
from proposers.

### P2 — bounded search

Add pluggable proposers:

- simple local hill climbing for one controlled change at a time
- reflective prompt evolution and Pareto-frontier search inspired by
  [GEPA](https://arxiv.org/abs/2507.19457)
- population-based code search inspired by
  [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- bandit or Bayesian selection where evaluation cost warrants it

Proposal mechanisms remain untrusted. Automated evaluators, regression gates,
budget limits, human review, canaries, and rollback remain authoritative.

### P3 — activation

Add exact-hash approval records, an atomic champion pointer, canary monitoring,
and one-command rollback. Only this layer may mutate active state.

## C++ And Other Language Clients

The protocol is the ABI. A Python rewrite or direct SQLite access is not
required. A C++ evaluator can emit the v1 JSON records, validate them against the
packaged schemas, and call the CLI today.

After the protocol and executor stabilize, a professional C++20 SDK should:

- expose transport-independent value types and strong IDs/enums
- use RAII for processes, files, locks, and cancellation
- use `std::filesystem`, `std::chrono`, and `std::stop_token`
- return typed results rather than exceptions crossing process boundaries
- communicate through CLI JSON or standards-conformant MCP JSON-RPC, never by
  embedding Python internals or reading AFS SQLite files directly
- ship relocatable CMake install/export targets and contract tests against the
  same golden JSON fixtures used by Python

Follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines),
[CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html),
and the [CMake import/export guide](https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html).
MCP interoperability should track the current
[MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/basic)
and preserve schemas as standalone artifacts rather than inventing a C ABI.

## Related AFS Surfaces

- `docs/TRAINING_FEEDBACK_RFC.md` — generic run/eval/feedback lifecycle
- `docs/LINEAGE.md` — filesystem contracts and cross-language consumers
- `src/afs/skill_mining.py` — future candidate proposer, not a trusted grader
- `src/afs/verification.py` — structured checks routed through the execution broker
- `src/afs/work_execution.py` — strongest current approval-oriented execution precedent
