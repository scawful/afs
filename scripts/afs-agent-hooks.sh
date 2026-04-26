#!/usr/bin/env bash
# Source this after afs-shell-init.sh to route common AI harness commands
# through AFS session wrappers automatically.

if [ "${AFS_AGENT_HOOKS_DISABLE:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

if [ -n "${BASH_SOURCE[0]-}" ]; then
  AFS_AGENT_HOOKS_SOURCE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION-}" ]; then
  AFS_AGENT_HOOKS_SOURCE="${(%):-%N}"
else
  AFS_AGENT_HOOKS_SOURCE="$0"
fi

AFS_AGENT_HOOKS_DIR="$(cd "$(dirname "${AFS_AGENT_HOOKS_SOURCE}")" && pwd)"
AFS_ROOT="${AFS_ROOT:-$(cd "${AFS_AGENT_HOOKS_DIR}/.." && pwd)}"
export AFS_ROOT

unalias codex claude gemini hcode z3cli 2>/dev/null || true

codex() { "${AFS_ROOT}/scripts/afs-codex" "$@"; }
claude() { "${AFS_ROOT}/scripts/afs-claude" "$@"; }
gemini() { "${AFS_ROOT}/scripts/afs-gemini" "$@"; }
hcode() { "${AFS_ROOT}/scripts/afs-hcode" "$@"; }
z3cli() { "${AFS_ROOT}/scripts/afs-z3cli" "$@"; }

afs-agent-hooks-status() {
  printf '%s\n' "AFS agent hooks: enabled"
  printf '  codex -> %s\n' "${AFS_ROOT}/scripts/afs-codex"
  printf '  claude -> %s\n' "${AFS_ROOT}/scripts/afs-claude"
  printf '  gemini -> %s\n' "${AFS_ROOT}/scripts/afs-gemini"
  printf '  hcode -> %s\n' "${AFS_ROOT}/scripts/afs-hcode"
  printf '  z3cli -> %s\n' "${AFS_ROOT}/scripts/afs-z3cli"
}

afs-agent-hooks-off() {
  unset -f codex claude gemini hcode z3cli afs-agent-hooks-status afs-agent-hooks-off 2>/dev/null || true
  printf '%s\n' "AFS agent hooks disabled for this shell"
}
