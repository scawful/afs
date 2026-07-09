# Security Policy

AFS is an agent-facing context and workflow layer. Security-sensitive surfaces include MCP tools, filesystem access, shell execution, work-assistant external writes, credential handling, and extension loading.

## Supported versions

| Version | Supported |
| --- | --- |
| `0.2.x` | Yes |
| `0.1.x` | Best effort |

## Reporting a vulnerability

Please do not publish exploit details before maintainers have a chance to respond.

Report privately by opening a GitHub security advisory if available, or by contacting the repository owner directly through the GitHub profile at https://github.com/scawful.

Include:

- affected version or commit
- reproduction steps
- expected vs actual behavior
- whether credentials, external writes, or data deletion are involved
- suggested fix, if known

## Security principles

AFS changes should preserve these defaults:

- least-privilege MCP surfaces by default
- explicit opt-in for broader tool catalogs and extension loading
- dry-run first for setup, repair, and external operations
- approval gates for external writes
- no credential logging
- no destructive actions without explicit user intent
- clear boundary between core AFS and extension-owned domains

## Extension risk model

Extensions are code. Users should only enable extension repos they trust. Extension manifests should be reviewed before adding CLI modules, MCP tools, hooks, or Python import paths.
