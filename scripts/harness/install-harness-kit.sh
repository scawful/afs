#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: install-harness-kit.sh [options]

Install or update portable AFS agent rules, skills, instruction templates, and
rendered client MCP/hook configs on a local or work machine.

Default mode is dry-run: no files are modified until --apply is passed.
Existing instruction files and non-matching skill directories are preserved
unless --force is also passed.

Options:
  --home PATH            Target user home directory (default: $HOME)
  --afs-root PATH        AFS checkout path (default: auto-detected from script)
  --workspace-root PATH  Allowed workspace root for MCP config (default: <HOME>/src)
  --guides-dir PATH      Location for ~/src/docs/guides/ai-coder symlink/link
                         (default: <HOME>/src/docs/guides/ai-coder)
  --apply                Perform file copies, symlinks, and config rendering
  --force                Overwrite existing instruction files or skill targets
  --skills               Install portable skills into <HOME>/.agents/skills and link to hosts
  --instructions         Install instruction templates (CLAUDE.md, CODEX AGENTS.md, GEMINI.md, Cursor rules)
  --render-configs DIR   Render host hook/MCP snippets into DIR with paths substituted
  -h, --help             Show this help

Defaults when --apply is given without --skills or --instructions:
  Installs both the shared guides symlink and the portable skills pack.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_AFS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGET_HOME="${HOME}"
AFS_ROOT="${DEFAULT_AFS_ROOT}"
WORKSPACE_ROOT=""
GUIDES_DIR=""
APPLY=0
FORCE=0
DO_SKILLS=0
DO_INSTRUCTIONS=0
RENDER_DIR=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --home)
      TARGET_HOME="${2:-}"
      shift 2
      ;;
    --afs-root)
      AFS_ROOT="${2:-}"
      shift 2
      ;;
    --workspace-root)
      WORKSPACE_ROOT="${2:-}"
      shift 2
      ;;
    --guides-dir)
      GUIDES_DIR="${2:-}"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --skills)
      DO_SKILLS=1
      shift
      ;;
    --instructions)
      DO_INSTRUCTIONS=1
      shift
      ;;
    --render-configs)
      RENDER_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

TARGET_HOME="${TARGET_HOME%/}"
AFS_ROOT="${AFS_ROOT%/}"
if [ -z "${WORKSPACE_ROOT}" ]; then
  WORKSPACE_ROOT="${TARGET_HOME}/src"
else
  WORKSPACE_ROOT="${WORKSPACE_ROOT%/}"
fi
if [ -z "${GUIDES_DIR}" ]; then
  GUIDES_DIR="${TARGET_HOME}/src/docs/guides/ai-coder"
else
  GUIDES_DIR="${GUIDES_DIR%/}"
fi

KIT_DIR="${AFS_ROOT}/docs/agent-harness"
if [ ! -d "${KIT_DIR}" ]; then
  echo "Agent harness kit not found at: ${KIT_DIR}" >&2
  exit 1
fi

if [ "${DO_SKILLS}" -eq 0 ] && [ "${DO_INSTRUCTIONS}" -eq 0 ]; then
  DO_SKILLS=1
fi

log_action() {
  if [ "${APPLY}" -eq 1 ]; then
    echo "[apply] $*"
  else
    echo "[dry-run] $*"
  fi
}

link_guides() {
  local parent
  parent="$(dirname "${GUIDES_DIR}")"
  if [ -e "${GUIDES_DIR}" ] || [ -L "${GUIDES_DIR}" ]; then
    if [ "${FORCE}" -eq 1 ] && [ -L "${GUIDES_DIR}" ]; then
      log_action "ln -sfn ${KIT_DIR} ${GUIDES_DIR}"
      [ "${APPLY}" -eq 1 ] && ln -sfn "${KIT_DIR}" "${GUIDES_DIR}"
    else
      echo "[keep] guides path already exists: ${GUIDES_DIR}"
    fi
  else
    log_action "mkdir -p ${parent} && ln -s ${KIT_DIR} ${GUIDES_DIR}"
    if [ "${APPLY}" -eq 1 ]; then
      mkdir -p "${parent}"
      ln -s "${KIT_DIR}" "${GUIDES_DIR}"
    fi
  fi
}

copy_skill() {
  local skill_dir="$1"
  local skill_name
  skill_name="$(basename "${skill_dir}")"
  local canonical_root="${TARGET_HOME}/.agents/skills"
  local canonical_dest="${canonical_root}/${skill_name}"

  if [ -e "${canonical_dest}" ] && [ "${FORCE}" -eq 0 ]; then
    echo "[keep] skill exists: ${canonical_dest}"
  else
    log_action "install canonical skill ${canonical_dest}"
    if [ "${APPLY}" -eq 1 ]; then
      mkdir -p "${canonical_dest}"
      cp -R "${skill_dir}/." "${canonical_dest}/"
    fi
  fi

  local host_roots=(
    "${TARGET_HOME}/.codex/skills"
    "${TARGET_HOME}/.claude/skills"
    "${TARGET_HOME}/.gemini/config/skills"
  )
  local host_root host_link
  for host_root in "${host_roots[@]}"; do
    host_link="${host_root}/${skill_name}"
    if [ -L "${host_link}" ]; then
      if [ "${FORCE}" -eq 1 ]; then
        log_action "ln -sfn ${canonical_dest} ${host_link}"
        [ "${APPLY}" -eq 1 ] && ln -sfn "${canonical_dest}" "${host_link}"
      else
        echo "[keep] skill link exists: ${host_link}"
      fi
    elif [ -e "${host_link}" ]; then
      echo "[skip] non-symlink skill dir exists: ${host_link} (pass --force or link manually)"
    else
      log_action "ln -s ${canonical_dest} ${host_link}"
      if [ "${APPLY}" -eq 1 ]; then
        mkdir -p "${host_root}"
        ln -s "${canonical_dest}" "${host_link}"
      fi
    fi
  done
}

install_instruction() {
  local src="$1"
  local dst="$2"
  local dst_dir
  dst_dir="$(dirname "${dst}")"
  if [ -e "${dst}" ] && [ "${FORCE}" -eq 0 ]; then
    echo "[keep] instruction file exists: ${dst} (use --force to overwrite)"
    return 0
  fi
  log_action "install instruction ${src} -> ${dst}"
  if [ "${APPLY}" -eq 1 ]; then
    mkdir -p "${dst_dir}"
    cp "${src}" "${dst}"
  fi
}

render_templates() {
  local out_dir="${RENDER_DIR%/}"
  log_action "render templates into ${out_dir} (AFS_ROOT=${AFS_ROOT}, HOME=${TARGET_HOME}, WORKSPACE_ROOT=${WORKSPACE_ROOT})"
  if [ "${APPLY}" -eq 0 ]; then
    return 0
  fi
  mkdir -p "${out_dir}"
  python3 - "${KIT_DIR}/hooks" "${out_dir}" "${AFS_ROOT}" "${TARGET_HOME}" "${WORKSPACE_ROOT}" <<'PY'
import pathlib, sys

src_dir = pathlib.Path(sys.argv[1])
dst_dir = pathlib.Path(sys.argv[2])
afs_root, home_dir, workspace_root = sys.argv[3], sys.argv[4], sys.argv[5]

for src in sorted(src_dir.iterdir()):
    if not src.is_file():
        continue
    rendered = (
        src.read_text(encoding="utf-8")
        .replace("<AFS_ROOT>", afs_root)
        .replace("<HOME>", home_dir)
        .replace("<WORKSPACE_ROOT>", workspace_root)
    )
    out = dst_dir / src.name
    out.write_text(rendered, encoding="utf-8")
    if src.suffix == ".sh":
        out.chmod(0o755)
    print(f"  rendered: {out}")
PY
}

link_guides

if [ "${DO_SKILLS}" -eq 1 ]; then
  for skill in "${KIT_DIR}/skills"/*; do
    [ -d "${skill}" ] && copy_skill "${skill}"
  done
fi

if [ "${DO_INSTRUCTIONS}" -eq 1 ]; then
  install_instruction "${KIT_DIR}/instructions/CLAUDE.md" "${TARGET_HOME}/.claude/CLAUDE.md"
  install_instruction "${KIT_DIR}/instructions/CODEX_AGENTS.md" "${TARGET_HOME}/.codex/AGENTS.md"
  install_instruction "${KIT_DIR}/instructions/GEMINI.md" "${TARGET_HOME}/.gemini/GEMINI.md"
  install_instruction "${KIT_DIR}/instructions/CURSOR_RULES.mdc" "${TARGET_HOME}/.cursor/rules/afs-agent.mdc"
  install_instruction "${KIT_DIR}/instructions/HOME_AGENTS.md" "${TARGET_HOME}/AGENTS.md"
fi

if [ -n "${RENDER_DIR}" ]; then
  render_templates
fi

if [ "${APPLY}" -eq 0 ]; then
  echo "Dry-run complete. Re-run with --apply to write changes."
fi
