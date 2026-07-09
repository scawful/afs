# AFS Examples

These examples are intentionally generic and runnable from a fresh clone of core AFS.

## Core examples

- `context_quickstart.py` — create a temporary context root and inspect AFS status.
- `plugin_skeleton/` — minimal Python plugin shape.
- `extension_hello_world/` — minimal manifest-based extension shape.

## Moved examples

Examples that depended on removed core modules or private/domain-specific models now belong in companion extensions:

- continuous learning loops -> extension-owned training/feedback packages
- cost optimization -> extension-owned infrastructure packages
- quality gates/deployment controllers -> extension-owned release packages
- private/domain/local-model workflows -> domain extension repos

Core AFS should stay useful without private data, local model weights, or workstation-specific services.
