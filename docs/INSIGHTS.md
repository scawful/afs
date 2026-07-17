# Insights

AFS Insights turns scoped local evidence into **reviewable learning
candidates**. It has two complementary commands:

- `research` retrieves relevant project and shared context for a question.
- `reflect` deterministically finds repeated failures in attributed history.

Neither command changes durable memory. Reflection creates pending candidates;
a person must inspect and explicitly accept or reject each one.

> **About “dreaming”**
>
> AFS sometimes uses dreaming as an explanatory metaphor for offline replay
> and reflection. It is not a CLI, API, or configuration term, and it does not
> mean stochastic model imagination. The public workflow is **Insights**:
> `research`, `reflect`, and human review.

## Quick start

Run local research from a registered project:

```bash
afs insights research "where is retry policy defined?" --path "$PWD"
```

By default, research refreshes the local text/symbol index before searching.
Use `--reuse-index` only when the existing index is known to be current.

Create and review deterministic reflection candidates:

```bash
afs insights reflect --path "$PWD"
afs insights list --path "$PWD"
afs insights show <candidate-id> --path "$PWD"
afs insights accept <candidate-id> --path "$PWD" \
  --because "This pattern is supported by the linked runs."
```

`accept` and `reject` require a rationale and an interactive human
confirmation. Accepted candidates become durable notes with provenance;
rejected candidates are archived. No agent or scheduled job can promote a
candidate automatically.

## Scope and storage

In the central v2 layout, research is limited to the **current registered
project plus `common`**. It has no all-projects mode. Knowing the central
context path does not grant access to another project.

Reflection and candidate review operate on one exact scope at a time:

- the current project by default; or
- shared context with `--common`.

For these lifecycle commands, `--common` selects common instead of combining
it with the current project. Project candidates are readable Markdown under
`scratchpad/projects/<project-id>/insights/candidates/`; common candidates are
under `scratchpad/common/insights/candidates/`. Decisions and accepted or
rejected archives remain in the same scope.

Only explicitly attributed events are eligible for reflection. Project
evidence must carry matching registry attribution; common evidence must be
explicitly attributed to common. Ambiguous or unattributed events are
ignored.

## Local and semantic research

Local retrieval is the default. Choose `--mode text` (the default) or
`--mode symbol`:

```bash
afs insights research "parser entry point" --path "$PWD" --mode symbol
```

Semantic retrieval is opt-in:

```bash
# Local Ollama embeddings
afs insights research "similar shutdown failures" --path "$PWD" \
  --semantic --provider ollama

# Gemini embeddings
afs insights research "similar shutdown failures" --path "$PWD" \
  --semantic --provider gemini
```

The provider is used only with `--semantic`. **Ollama keeps embedding input
local. Gemini transmits indexed content and the query to Gemini** to create
embeddings. Use `--model` to select a supported provider model; Gemini
otherwise uses the stable `gemini-embedding-2` default at 768 dimensions.

Semantic retrieval and internet research are separate permissions. Enabling
one does not enable the other.

## Optional internet research

Internet research requires both an enabled extension provider selected by
name and at least one allowed domain:

```bash
afs insights research "current protocol guidance" --path "$PWD" \
  --internet-provider <provider> \
  --allow-domain docs.example.com
```

Repeat `--allow-domain` for additional domains. The selected provider is
extension code and is part of the trusted computing base. It is expected to
cooperate with the contract by enforcing the allowlist during DNS resolution,
redirect handling, DNS-rebinding checks, private-IP rejection, and transport
timeouts.

AFS core does **not** independently mediate the provider's sockets, DNS, or
redirects. Core runs the selected provider out of process with bounded time
and output, then validates the returned evidence: result count, total bytes,
and final URLs (HTTPS, no credentials, default/443 port, and an allowed host
or subdomain). Internet results are returned to the caller; they are not
automatically ingested into context or promoted to memory.

The subprocess receives a scrubbed environment: API-key environment variables
are not inherited. For now, a provider that needs credentials should read a
credential file whose path is explicitly named in trusted AFS/extension
configuration. Do not scan default credential locations or inherit the full
parent environment. A future contract may add an explicit secret-name
allowlist; there is no implicit secret forwarding today.

Use `--internet-limit`, `--internet-timeout`, and `--internet-max-bytes` to
tighten the default bounds.

## Deterministic reflection

`afs insights reflect` uses no model and no network. It builds a bounded,
canonical evidence packet from attributed history, removes event payloads and
payload references, retains only allowlisted scalar metadata, and hashes the
packet. The same packet produces the same candidate identity.

Reflection highlights repeated failures only. Successful completions and
general activity are deliberately ignored so a rolling schedule does not
produce routine candidate spam. Failure frequency is not proof of causation.
Each candidate links to its evidence digest so a human can inspect the claim
before deciding. Narrow evidence with repeatable `--event-type` filters and
the scalar `--limit`; `--agent-name` labels the candidate's provenance rather
than filtering events.

## Optional scheduled agents

Neither Insights agent is part of the default supervisor set. Both write only
to the current scope's scratchpad and have no memory-promotion authority.

### Weekly reflection

To opt a profile into weekly project reflection:

```toml
[[profiles.work.agent_configs]]
name = "insights-reflect"
role = "learning"
description = "Create reviewable insight candidates from attributed local history."
module = "afs.agents.insights"
schedule = "weekly"
project_path = "/absolute/path/to/project"
limit = 100
```

Use `common = true` instead of `project_path` for shared-context reflection.
The scheduled agent inspects only the newest 1,000 physical history records
before scope filtering and the evidence `limit`, so its recurring work stays
bounded as the ledger grows. The interactive `afs insights reflect` command
retains complete-history behavior. The agent only creates pending candidates;
it cannot accept them.

### Weekly research

`insights-research` requires one explicit project and one research question.
It refreshes local code/context before searching unless `reuse_index = true`:

```toml
[[profiles.work.agent_configs]]
name = "insights-research"
role = "learning"
description = "Write a scoped scratchpad report for one recurring question."
module = "afs.agents.research"
schedule = "weekly"
project_path = "/absolute/path/to/project"
query = "Where are retry failures recurring?"
limit = 10
```

The report is readable Markdown in the project's scratchpad. It is not a
durable note and is never promoted automatically. Optional semantic settings
are `semantic`, `provider`, and `model`; the same Ollama-local and
Gemini-transmission boundary described above applies.

Scheduled internet research is fail-closed and requires all three of these
settings:

```toml
network_allowed = true
internet_provider = "enabled-provider-name"
allowed_domains = ["docs.example.com"]
```

The boolean must be the literal value `true`. Optional bounds are
`internet_limit`, `internet_timeout`, and `internet_max_bytes`. Setting a
provider without that consent, or consenting without both a provider and
domains, is an error. `reuse_index = true` is also available but trades away
the default freshness check.

Defining any custom `agent_configs` makes that profile's agent list explicit,
so copy in any shipped defaults you still want. Start the supervisor with
`afs services start agent-supervisor` after reviewing the profile.

## Machine-readable output

All Insights subcommands support `--json`. Research JSON also reports whether
semantic or network access was requested and whether content could have been
transmitted remotely. Treat that data-movement summary as part of the audit
record.
