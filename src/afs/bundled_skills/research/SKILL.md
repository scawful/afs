---
name: research
triggers:
  - research
  - investigate
  - investigation
  - evidence
  - root cause
  - prior art
  - compare options
  - find out why
profiles:
  - general
requires:
  - afs
enforcement:
  - Every claim carries its evidence — path:line, commit, URL, or command output; label anything else as unverified.
  - Distinguish confirmed (you reproduced or read the primary source) from plausible (inferred) from unknown.
  - State what evidence is missing rather than papering over it; a labeled gap beats a guess.
verification:
  - Re-check that each conclusion in the final report links to at least one cited source.
---

# Research

Evidence-first investigation for engineering questions: root causes,
prior art, design comparisons, "why is it like this".

## Method

1. **Write the question down**, plus what evidence would settle it. If
   you cannot name the settling evidence, the question is too vague.
2. **Search what the team already knows** before the world:
   `afs search` / `context.query` (memory, scratchpad, handoffs),
   `git log -S`/`git blame` for why code changed, tickets and PR
   discussions, then external docs/web.
3. **Sweep from multiple angles** — by symptom, by component, by author,
   by time window. One angle misses; note which angles you used.
4. **Read primary sources.** A changelog claim is plausible; the diff is
   confirmed. A docstring is plausible; the code path is confirmed.
5. **Timebox.** Report at the box even if unresolved: findings so far,
   confidence, next cheapest probe.

## Report Shape

- Answer first, one paragraph, with confidence (confirmed / likely / open).
- Evidence list: claim → source (path:line, commit, URL, command + output).
- Explicit gaps: what was not checked and what it would cost to check.
- Staleness notes: anything cited that may have changed since written.

## Red Flags

- Conclusions that cite "the docs" without a location.
- A single search angle treated as exhaustive.
- Prior context ignored — the answer already sat in a handoff or memory.
- Investigation that quietly became implementation without deciding to.
