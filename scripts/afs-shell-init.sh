#!/usr/bin/env bash
# Source this in bash/zsh to expose the local AFS CLI with aliases and helpers.
#
#   source /path/to/afs/scripts/afs-shell-init.sh
#
# Or add to .zshrc / .bashrc:
#   [ -f ~/src/lab/afs/scripts/afs-shell-init.sh ] && source ~/src/lab/afs/scripts/afs-shell-init.sh

# --- Resolve AFS_ROOT ---
if [ -n "${BASH_SOURCE[0]-}" ]; then
  AFS_SHELL_INIT_SOURCE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION-}" ]; then
  AFS_SHELL_INIT_SOURCE="${(%):-%N}"
else
  AFS_SHELL_INIT_SOURCE="$0"
fi

AFS_SHELL_INIT_DIR="$(cd "$(dirname "${AFS_SHELL_INIT_SOURCE}")" && pwd)"
AFS_ROOT="${AFS_ROOT:-$(cd "${AFS_SHELL_INIT_DIR}/.." && pwd)}"
export AFS_ROOT
export AFS_CLI="${AFS_ROOT}/scripts/afs"

if [ -z "${AFS_VENV:-}" ] && [ -d "${AFS_ROOT}/.venv" ]; then
  export AFS_VENV="${AFS_ROOT}/.venv"
fi

case ":${PATH}:" in
  *":${AFS_ROOT}/scripts:"*) ;;
  *) export PATH="${AFS_ROOT}/scripts:${PATH}" ;;
esac

# --- Aliases ---
# Keep `afs` on the user's installed/default command. Use `afs-dev` when the
# current repo wrapper should win explicitly.
alias afs-dev="${AFS_CLI}"
alias a='afs'
alias as='afs status'
alias asj='afs status --json'
alias ap='afs agents ps'
alias apa='afs agents ps --all'
alias aw='afs agents watch'
alias al='afs agents list'
alias ab='afs session bootstrap'
alias tl='afs tasks list'
alias hm='afs hivemind list'
alias sk='afs skills list'
alias pc='afs profile current'
alias ps_='afs profile switch'

# --- Helper functions ---

_afs_python_bin() {
  local candidate="${AFS_VENV:-${AFS_ROOT}/.venv}/bin/python"
  if [ -x "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  echo "python3 not found" >&2
  return 1
}

_afs_context_path() {
  local context_path="${AFS_CONTEXT_PATH:-.context}"
  if [ ! -d "$context_path" ] && [ ! -L "$context_path" ]; then
    echo "No AFS context at $context_path" >&2
    return 1
  fi
  printf '%s\n' "$context_path"
}

# Quick context status for current directory
afs-here() {
  if [ -d ".context" ] || [ -L ".context" ]; then
    afs status
  else
    echo "No .context in $(pwd)"
    echo "  Run: afs init --link-context --workspace-path . --workspace-name $(basename "$(pwd)")"
  fi
}

# Initialize context for the current project
afs-bootstrap() {
  local name="${1:-$(basename "$(pwd)")}"
  afs init --link-context --workspace-path . --workspace-name "$name"
  echo ""
  afs session bootstrap || true
  echo ""
  afs status
}

# Watch an agent's progress events (defaults to last 20)
afs-watch() {
  local agent="${1:?usage: afs-watch <agent-name> [limit]}"
  local limit="${2:-20}"
  afs agents watch "$agent" --limit "$limit"
}

# Spawn an agent in the background via the local supervisor
afs-spawn() {
  local name="${1:?usage: afs-spawn <agent-name> [module] [-- agent-args...]}"
  shift
  local module=""
  if [ $# -gt 0 ] && [ "$1" != "--" ] && [[ "$1" != -* ]]; then
    module="$1"
    shift
  fi
  if [ $# -gt 0 ] && [ "$1" = "--" ]; then
    shift
  fi
  local python_bin
  python_bin="$(_afs_python_bin)" || return 1
  "$python_bin" - "$name" "$module" "$@" <<'PY'
from __future__ import annotations

import sys

from afs.agents import get_agent
from afs.agents.supervisor import AgentSupervisor
from afs.config import load_config_model
from afs.profiles import resolve_active_profile

name = sys.argv[1]
requested_module = sys.argv[2].strip()
agent_args = sys.argv[3:]

config = load_config_model(merge_user=True)
profile = resolve_active_profile(config)
agent_config = next((item for item in profile.agent_configs if item.name == name), None)

module = requested_module
if not module and agent_config and agent_config.module:
    module = agent_config.module
if not module:
    spec = get_agent(name)
    if spec is not None:
        module = spec.entrypoint.__module__

if not module:
    raise SystemExit(
        f"Unable to resolve module for agent '{name}'. Pass a module explicitly or add it to your profile."
    )

supervisor = AgentSupervisor(config=config)
agent = supervisor.spawn(
    name,
    module,
    args=agent_args,
    reason="shell_helper",
    agent_config=agent_config,
)
print(f"  name: {agent.name}")
print(f"  pid: {agent.pid}")
print(f"  module: {agent.module}")
print(f"  state: {agent.state}")
PY
}

# Quick search across all context mounts
afs-find() {
  local pattern="${1:?usage: afs-find <pattern>}"
  for mount in knowledge memory scratchpad tools; do
    local mount_dir=".context/$mount"
    if [ -d "$mount_dir" ]; then
      local results
      results=$(find -L "$mount_dir" -type f -name "*${pattern}*" 2>/dev/null)
      if [ -n "$results" ]; then
        echo "[$mount]"
        echo "$results" | sed 's/^/  /'
      fi
    fi
  done
}

# Show what skills match a prompt
afs-match() {
  local prompt="${1:?usage: afs-match <prompt>}"
  afs skills match "$prompt"
}

# Quick task creation
afs-task() {
  local title="${1:?usage: afs-task <title> [priority]}"
  local priority="${2:-5}"
  local python_bin
  local context_path
  python_bin="$(_afs_python_bin)" || return 1
  context_path="$(_afs_context_path)" || return 1
  echo "Creating task: $title (priority=$priority)"
  "$python_bin" - "$context_path" "$title" "$priority" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from afs.tasks import TaskQueue

context_path = Path(sys.argv[1]).expanduser().resolve()
title = sys.argv[2]
try:
    priority = int(sys.argv[3])
except ValueError as exc:
    raise SystemExit("priority must be an integer") from exc

queue = TaskQueue(context_path)
task = queue.create(title, priority=priority)
print(f"  id: {task.id}")
print(f"  status: {task.status}")
PY
}

# Send a hivemind message
afs-say() {
  local from_agent="${1:?usage: afs-say <from-agent> <type> <key=value...>}"
  local msg_type="${2:?usage: afs-say <from-agent> <type> <key=value...>}"
  shift 2
  local python_bin
  local context_path
  python_bin="$(_afs_python_bin)" || return 1
  context_path="$(_afs_context_path)" || return 1
  "$python_bin" - "$context_path" "$from_agent" "$msg_type" "$@" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from afs.hivemind import HivemindBus

context_path = Path(sys.argv[1]).expanduser().resolve()
from_agent = sys.argv[2]
msg_type = sys.argv[3]
payload: dict[str, str] = {}

for raw in sys.argv[4:]:
    if "=" not in raw:
        raise SystemExit(f"payload items must use key=value: {raw}")
    key, value = raw.split("=", 1)
    if not key:
        raise SystemExit(f"payload key missing in: {raw}")
    payload[key] = value

bus = HivemindBus(context_path)
msg = bus.send(from_agent, msg_type, payload=payload)
print(f"  sent: {msg.id}")
PY
}

# Run a verification command and record the result against the active session.
afs-verify() {
  "${AFS_ROOT}/scripts/afs-session-verify" "$@"
}

# --- Completions (zsh) ---
if [ -n "${ZSH_VERSION-}" ]; then
  _afs_commands() {
    local commands=(
      'status:Show AFS status'
      'init:Initialize context root'
      'agents:Agent operations'
      'tasks:Task queue'
      'hivemind:Message bus'
      'skills:Skill metadata'
      'profile:Profile management'
      'context:Context operations'
      'mcp:MCP server'
      'plugins:Plugin management'
      'services:Service management'
      'bundle:Bundle operations'
      'health:Health checks'
      'fs:Filesystem operations'
      'help:Show help'
    )
    _describe 'command' commands
  }
  compdef _afs_commands afs
fi
