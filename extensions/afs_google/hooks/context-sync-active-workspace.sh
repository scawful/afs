#!/usr/bin/env bash
set -euo pipefail

# Template hook: call this from your workspace switch flow.
# Usage: context-sync-active-workspace.sh /abs/path/to/workspace

workspace_path="${1:-}"
if [[ -z "${workspace_path}" ]]; then
  echo "usage: $0 <workspace-path>" >&2
  exit 1
fi

if [[ -d "${workspace_path}" ]]; then
  workspace_path="$(cd "${workspace_path}" && pwd)"
fi

context_root="${AFS_CONTEXT_ROOT:-$HOME/.context}"
profile="${AFS_PROFILE:-default}"
out_dir="${context_root}/monorepo"
out_file="${out_dir}/active_workspace.toml"

mkdir -p "${out_dir}"

cat > "${out_file}" <<EOF
active_workspace = "${workspace_path}"
profile = "${profile}"
source = "context-sync-active-workspace"
updated_at = "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
EOF

echo "updated ${out_file}"
