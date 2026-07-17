# Extension Authoring Guide

Extensions let teams add domain behavior without forking core AFS. Use them for private connectors, model families, personal workflows, company policy, domain agents, and MCP tools that should not ship in the generic core package.

## Contract status

AFS treats the manifest format as a **versioned pre-1.0 contract**. Add
`api_version = 1` to new manifests. Existing `schema_version = "0.1"`
manifests remain compatible, but new manifests should use the integer API
version.

Core promises for `0.2.x`:

- `extension.toml` discovery from configured extension dirs and sibling repos
- automatic `src/` import path support for companion repos
- `knowledge_mounts`, `skill_roots`, `cli_modules`, `agent_modules`, `[mcp_tools]`, `[mcp_server]`, and `[[context_sources]]`
- first-found-wins resolution when multiple roots expose the same extension name

Core does **not** promise that private connector code, local model names, or domain datasets belong in this repo.

## Minimal extension

```text
afs_example/
  extension.toml
  src/afs_example/__init__.py
  src/afs_example/cli.py
  skills/example/SKILL.md
  knowledge/README.md
```

`extension.toml`:

```toml
api_version = 1
name = "afs_example"
description = "Example AFS extension"

knowledge_mounts = ["knowledge"]
skill_roots = ["skills"]
python_paths = ["src"]
cli_modules = ["afs_example.cli"]
```

Enable it from an AFS workspace:

```toml
[extensions]
enabled_extensions = ["afs_example"]
extension_repo_roots = ["~/src/lab"]
```

Or with environment variables:

```bash
export AFS_EXTENSION_REPO_ROOTS="$HOME/src/lab"
export AFS_ENABLED_EXTENSIONS="afs_example"
./scripts/afs extensions list --details
```

## CLI modules

A CLI module should expose a registration hook that accepts the AFS CLI parser surface. Keep commands explicit and dry-run first when they touch external systems.

Recommended command behavior:

- provide `--dry-run` for writes
- never read credentials from logs or prompts that echo secrets
- return non-zero on validation failures
- keep domain-specific output under the extension's namespace

## MCP tools

Use MCP extensions when a client needs tool access to extension behavior.

Tool-only manifest:

```toml
[mcp_tools]
module = "afs_example.mcp_tools"
factory = "register_mcp_tools"
# Optional; omitted tools inherit this value. The default is "full".
catalog = "slim"
```

Broader MCP manifest:

```toml
[mcp_server]
module = "afs_example.mcp_surface"
factory = "register_mcp_server"
```

A tool factory returns dictionaries with:

- `name`
- `description`
- `inputSchema` or `input_schema`
- `handler`
- optional `catalog` (`"slim"` or `"full"`)

Catalog exposure is separate from call-time permission. By default extension
tools are full-catalog-only. `[mcp_tools].catalog = "slim"` opts that factory's
tools into the default `tools/list`, while a per-tool `catalog = "full"` can opt
one tool back out. The manifest default does not apply to `[mcp_server]` or
profile-contributed tools; those must opt in per tool. Unknown catalog values
reject the manifest or MCP contribution rather than falling back to `"full"`
or `"slim"` silently.

Avoid overriding core names. Prefer extension-prefixed names such as `example.lookup` or `company.ticket.search`.

## Context source providers

Use `[[context_sources]]` for provider-neutral records. Core materialization is
currently v1-only and writes `.context/items/sources/`; v2 `sources sync`
fails closed before provider invocation until scoped ingestion exists.

```toml
[[context_sources]]
name = "docs-research"
module = "afs_example.sources"
factory = "register_docs_research"
kinds = ["doc"]
```

A minimal opt-in research provider implements `research(request)`. Importing
the module and calling its factory must only define/construct the provider:
do not connect to the network, read credentials, or perform other side effects
until AFS invokes `research` with an explicitly authorized request.

```python
from afs.sources import ContextSourceRecord, ResearchRequest

from .transport import search_docs_with_policy


class DocsResearchProvider:
    name = "docs-research"

    def research(self, request: ResearchRequest) -> list[ContextSourceRecord]:
        if not request.network_allowed:
            raise PermissionError("network research was not authorized")

        # This extension-owned transport must enforce allowed_domains on the
        # initial URL and every redirect, resolve DNS itself, reject private or
        # rebound addresses, and honor request.timeout_seconds and size limits.
        pages = search_docs_with_policy(
            request.query,
            allowed_domains=request.allowed_domains,
            timeout_seconds=request.timeout_seconds,
            limit=request.max_results,
            max_bytes=request.max_bytes,
        )
        return [
            ContextSourceRecord(
                id=page.id,
                kind="doc",
                title=page.title,
                body=page.excerpt,
                source="example-docs",
                uri=page.final_url,
            )
            for page in pages
        ]


def register_docs_research() -> DocsResearchProvider:
    return DocsResearchProvider()
```

The CLI still requires an enabled selected provider and an explicit domain
allowlist. The provider is trusted/cooperative extension code and owns DNS,
redirect, DNS-rebinding, private-IP, and transport-timeout enforcement. Core
AFS runs only the selected provider in a time/output-bounded subprocess, then
validates returned HTTPS URIs, record count, and total bytes. Core does not
mediate the provider's sockets.

The broker does not inherit API-key environment variables into this
subprocess. For now, put a credential-file path in trusted configuration and
have `research()` read that explicit file; never copy or inherit the full
parent environment. A future contract may add a narrow secret-name allowlist.

Providers should normalize remote records before writing into AFS. Keep the
original external system as provenance, not as a required runtime dependency
for every user. Future v2 ingestion will route project records to
`knowledge/projects/<project-id>/` and require an explicit choice for shared
`knowledge/common/` records; extensions must not write unscoped v2 `items`
records themselves.

## Manager actions

Extensions can expose human-triggered actions in the AFS manager:

```toml
[manager]
actions = [
  "afs example status",
  "afs example sync --dry-run",
]
```

Actions must be safe to preview and should not mutate external systems unless the user explicitly runs an apply/execute command.

## What to keep out of core

Move these to extensions:

- private workspace paths
- private/domain-specific corpora
- local model names and weights
- company-specific ticket/doc/chat connectors
- scripts that require private credentials
- examples that cannot run from a fresh clone of core AFS

## Example

See `examples/extension_hello_world/` for a tiny manifest and Python package that can be enabled from a checkout.
