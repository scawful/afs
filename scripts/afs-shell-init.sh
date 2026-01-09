#!/usr/bin/env bash
# Source this in bash/zsh to expose the local AFS CLI.

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
