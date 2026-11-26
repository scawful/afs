---
name: config-change-safety
description: Safely inspect and change local config, dotfiles, launchers, editor settings, window manager rules, shell startup, service config, or machine-specific harness files. Use when behavior depends on user-local configuration.
---

# Config Change Safety

## Goal
Fix local behavior without destabilizing the machine or hiding the root cause.

## Workflow
1. Diagnose live state before editing when behavior is currently observable.
2. Identify the active config file and precedence order. Do not edit a plausible config until it is confirmed active.
3. Prefer the smallest reversible change. Avoid adding toggles or workflows for restoring default behavior.
4. Preserve user-local unrelated settings.
5. For reloadable services or UI agents, run the supported reload/restart command and verify one real behavior after reload.
6. Report exact file paths changed and how to revert.

## High-Risk Areas
- Shell startup files, launch agents / systemd units, editor init files, window manager rules, SSH/tunnel scripts, MCP configs, model routing, device signing.
- Before broad deletion, force push, history rewrite, or killing user-owned processes, ask the user.

## Verification
- Parser check for structured config (`json`, `toml`, `yaml`).
- `bash -n` for shell launchers.
- Live process/query check for service config.
- One newly created window/session/request when stale state may persist.
