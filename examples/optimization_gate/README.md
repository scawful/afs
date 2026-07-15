# Optimization Gate Example

This example exercises AFS's pure, language-neutral optimization decision
contract. It does not run, write, activate, or promote either candidate.

```bash
./scripts/afs optimize decide \
  --baseline examples/optimization_gate/baseline.json \
  --candidate examples/optimization_gate/candidate.json \
  --policy examples/optimization_gate/policy.json \
  --json
```

Exit codes are stable for automation:

- `0`: `eligible_for_human_review`
- `1`: `rejected`
- `2`: invalid input
- `3`: `inconclusive`
- `4`: internal gate error (not an evidence verdict)

The baseline uses `parent_id: "root"` as the v1 lineage sentinel. Every candidate
compared to it must name the baseline's `candidate_id` as its own `parent_id`.

Python, C++, Rust, Go, and TypeScript evaluators can emit the same JSON records.
Fetch the exact packaged contracts with:

```bash
./scripts/afs schema show v1/optimization/evaluation
./scripts/afs schema show v1/optimization/policy
./scripts/afs schema show v1/optimization/decision
```
