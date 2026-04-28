#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: setup_gws.sh [options]

Install/check the Google Workspace CLI and authenticate for AFS helpers.

Options:
  --credentials PATH    OAuth client_secret JSON to copy into the gws config dir
  --config-dir PATH     gws config directory (default: ~/.config/gws)
  --scopes LIST         Comma-separated auth scopes (default: gmail,calendar,drive)
  --no-install          Do not install gws; only check for it
  --no-auth             Do not run browser OAuth login
  --dry-run             Print planned actions without writing or authenticating
  --yes                 Do not pause before browser OAuth
  -h, --help            Show this help

Examples:
  scripts/setup_gws.sh --dry-run
  scripts/setup_gws.sh --credentials ~/Downloads/client_secret.json
  scripts/setup_gws.sh --scopes gmail,calendar
EOF
}

CONFIG_DIR="${GWS_CONFIG_DIR:-$HOME/.config/gws}"
SCOPES="gmail,calendar,drive"
CREDENTIALS=""
INSTALL=1
AUTH=1
DRY_RUN=0
YES=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --credentials)
      CREDENTIALS="${2:-}"
      shift 2
      ;;
    --config-dir)
      CONFIG_DIR="${2:-}"
      shift 2
      ;;
    --scopes)
      SCOPES="${2:-}"
      shift 2
      ;;
    --no-install)
      INSTALL=0
      shift
      ;;
    --no-auth)
      AUTH=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes|-y)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$CONFIG_DIR" ]; then
  echo "--config-dir requires a path" >&2
  exit 2
fi
if [ -z "$SCOPES" ]; then
  echo "--scopes requires a non-empty comma-separated list" >&2
  exit 2
fi

CLIENT_SECRET="${CONFIG_DIR%/}/client_secret.json"

info() { printf '[ok] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
step() { printf '\n== %s ==\n' "$*"; }
run_or_print() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'next:'
    printf ' %q' "$@"
    printf '\n'
  else
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  fi
}

find_credentials() {
  if [ -n "$CREDENTIALS" ]; then
    printf '%s\n' "$CREDENTIALS"
    return 0
  fi
  find "$HOME/Downloads" -maxdepth 1 \( -name 'client_secret*.json' -o -name '*oauth*.json' \) -type f 2>/dev/null | head -1
}

echo "AFS Google Workspace setup"
printf 'config_dir: %s\n' "$CONFIG_DIR"
printf 'scopes: %s\n' "$SCOPES"
printf 'mode: %s\n' "$([ "$DRY_RUN" -eq 1 ] && echo dry-run || echo apply)"

step "Check gws CLI"
if command -v gws >/dev/null 2>&1; then
  info "gws is installed: $(gws --version 2>/dev/null || echo unknown-version)"
else
  if [ "$INSTALL" -eq 0 ]; then
    warn "gws is not installed"
    echo "Install it with Homebrew or npm, then rerun this script:"
    echo "  brew install googleworkspace-cli"
    echo "  npm install -g @googleworkspace/cli"
    if [ "$DRY_RUN" -eq 0 ]; then
      exit 1
    fi
  fi
  if command -v brew >/dev/null 2>&1; then
    run_or_print brew install googleworkspace-cli
  elif command -v npm >/dev/null 2>&1; then
    run_or_print npm install -g @googleworkspace/cli
  else
    echo "Neither brew nor npm was found. Install gws with an approved package path first." >&2
    exit 1
  fi
fi

step "OAuth credentials"
credential_source="$(find_credentials || true)"
if [ -f "$CLIENT_SECRET" ]; then
  info "credentials already exist: $CLIENT_SECRET"
elif [ -n "$credential_source" ] && [ -f "$credential_source" ]; then
  run_or_print mkdir -p "$CONFIG_DIR"
  run_or_print cp "$credential_source" "$CLIENT_SECRET"
else
  warn "no client_secret JSON found"
  echo "Place a work-approved OAuth client_secret JSON at:"
  echo "  $CLIENT_SECRET"
  echo "or rerun with:"
  echo "  scripts/setup_gws.sh --credentials /path/to/client_secret.json"
  if [ "$DRY_RUN" -eq 0 ]; then
    exit 1
  fi
fi

if [ "$AUTH" -eq 1 ]; then
  step "Authenticate"
  if [ "$DRY_RUN" -eq 1 ]; then
    run_or_print gws auth login -s "$SCOPES"
  else
    if [ "$YES" -ne 1 ]; then
      printf 'This opens a browser OAuth flow for scopes: %s\n' "$SCOPES"
      printf 'Press Enter to continue, or Ctrl-C to stop. '
      read -r _
    fi
    run_or_print gws auth login -s "$SCOPES"
  fi
else
  warn "auth step skipped"
fi

step "Verify"
if command -v gws >/dev/null 2>&1; then
  if [ "$DRY_RUN" -eq 1 ]; then
    run_or_print gws auth status
    run_or_print gws gmail users.messages.list --params '{"userId":"me","maxResults":1}'
    run_or_print gws calendar events.list --params '{"calendarId":"primary","maxResults":1}'
    run_or_print gws drive files.list --params '{"pageSize":1}'
  else
    gws auth status 2>/dev/null || warn "gws auth status failed"
    gws gmail users.messages.list --params '{"userId":"me","maxResults":1}' >/dev/null 2>&1 \
      && info "gmail check ok" || warn "gmail check failed or scope unavailable"
    gws calendar events.list --params '{"calendarId":"primary","maxResults":1}' >/dev/null 2>&1 \
      && info "calendar check ok" || warn "calendar check failed or scope unavailable"
    gws drive files.list --params '{"pageSize":1}' >/dev/null 2>&1 \
      && info "drive check ok" || warn "drive check failed or scope unavailable"
  fi
else
  warn "gws is not available yet; rerun verification after install"
fi

cat <<EOF

Useful AFS commands:
  afs gws status
  afs gws agenda
  afs gws unread
  afs guide google-workspace
EOF
