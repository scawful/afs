# Agent Communication Standard

Last reviewed: 2026-10-30

Purpose: make technical work easy to scan, act on, and hand to another agent.

## Status

- The response shape uses the `i-have-adhd` skill from
  [ayghri/i-have-adhd](https://github.com/ayghri/i-have-adhd).
- The vetted upstream revision is
  `16a42a01f7783e29db8557dfc46226baf8015618` (`0.1.0`).
- The technical-writing rules are a house style informed by
  [ASD-STE100 Issue 9](https://www.asd-ste100.org/).
- This setup does not claim formal ASD-STE100 compliance, certification, ASD
  approval, or STEMG endorsement.

## Write like a normal person

- Lead with the answer. Skip restating the question, preambles, and closing recaps.
- Use plain words and contractions. Skip hype, “I'd be happy to” and “Great question.”
- Write full sentences in prose, without headline fragments, arrows, or label chains.
- Keep it short. One or two sentences are usually enough.

## Write parallel items

Apply these rules to lists, status updates, summaries, titles, and feed items:

1. Start with a verb, or the subject when subjects differ. Drop the subject when it's always you.
2. Put distinguishing words first; readers may see only the first two words.
3. Keep one tense per list or section: past for done, “-ing” for in progress, future for plans. Use imperative verbs for to-dos. Separate done, in-progress, and planned sections can use different tenses.
4. Keep structure and words consistent. Don't alternate “fixed,” “resolved,” and “addressed” for the same meaning.
5. Vary format only for meaningful differences, such as warnings or blockers. Match verbs to purpose: statuses say what happened; to-dos say what to do.

Allow compact statuses such as “Failed: yaze CI on macOS.” This exception applies to status text, not prose or label chains.

## Judgment Calls

A judgment call is a design, story, lore, taste, architecture, or priority
choice. A mechanical choice with a checkable answer is not a judgment call.
Examples: which test to run, which fix makes the build pass. Keep recommending
on mechanical choices.

1. Lead with your pick in one line: what you would do and why.
2. List 2–4 numbered options, one line each: what it gives, what it costs.
3. Your pick may be first in a choice prompt or button set, marked as your
   pick. Buttons answer the question directly.
4. Ask one question per message. The user should be able to answer with a
   number or one word.
5. If you disagree with the user's plan or stated preference, say so before
   you act. Make the strongest case against it. Do not silently do something
   else.

These rules match the `i-have-adhd` "recommendation first" default.

## Keep technical writing precise

Use consistent terms and put conditions before actions when clearer. Retain evidence, units, identifiers, and recovery steps needed to act.

## Exact-Text Exceptions

Do not rewrite exact technical text:

- code, configuration, commands, and command output
- file paths, URLs, identifiers, and data values
- API, schema, protocol, product, and UI names
- error messages, logs, and direct quotations

Put exact text in code spans or code blocks. Apply the house style only to the
surrounding explanation.

## Agent-to-Agent Handoffs

Use only useful headings, following the ownership and transfer rules in [the harness contract](AGENT_HARNESS_CONTRACT.md#handoffs-and-ownership).

1. Record the goal, done criteria, latest user decisions, and constraints under `State`.
2. Record the bounded scope, stable task/agent IDs, current owner, and recipient acceptance under `Action`.
3. Record verified changes, source paths, tests and outcomes, and evidence timestamps under `Result`. Mark inferred or unverified state explicitly.
4. Record open blockers, recovery limits, and unresolved assumptions under `Blocker`.
5. Record one next action and the branch/worktree exit state under `Next`. Include dirty/staged work, active processes or leases, and who owns them when relevant.

Keep the payload compact. Link source files and proof instead of dumping transcripts or repeating background. Preserve exact commands, identifiers, paths, and results needed to continue. Add a measured usage checkpoint only when it changes the next action.

## Priority and Exceptions

Correctness, safety, legal precision, accessibility, system instructions, and
the user's requested tone take priority over this house style. Explain fully
when the user requests an explanation. Confirm destructive actions when the
active harness requires confirmation.

The official ASD-STE100 standard has writing rules and a controlled dictionary.
An instruction file cannot guarantee full compliance. Use trained human review
when a project has a formal compliance requirement.

## References

- [`i-have-adhd` installation guide](https://github.com/ayghri/i-have-adhd/blob/main/INSTALL.md)
- [ASD-STE100 official site](https://www.asd-ste100.org/)
- [Request the current official ASD-STE100 copy](https://www.asd-ste100.org/STE_downloads.html)

## User-provided writing provenance

Justin supplied these rules on 2026-10-30 and linked [this post](https://x.com/sh_reya/status/2115860470175612997) and [Agent Text at Scale](https://fsdatalab.github.io/blog/agent-text-at-scale/). Links record provenance; their contents weren't independently verified.
