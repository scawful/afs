#!/bin/bash
# chat-service.sh - Control script for AFS Chat infrastructure
# Can be called from CLI, sketchybar, or cortex hub
#
# Usage:
#   chat-service.sh start [simple|full]   - Start chat services
#   chat-service.sh stop                  - Stop all chat services
#   chat-service.sh status [--json]       - Show service status
#   chat-service.sh restart               - Restart services
#   chat-service.sh open                  - Open chat in browser
#   chat-service.sh logs                  - Tail service logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AFS_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$AFS_ROOT/docker"
STATE_DIR="${HOME}/.config/afs/services/state"
CACHE_FILE="${HOME}/.cache/afs/chat_status.cache"
SECRETS_ENV_FILE="${HOME}/.config/afs/openwebui.secrets.env"
LITELLM_ENV_FILE="${HOME}/.config/afs/litellm.env"
LITELLM_CONFIG_FILE="${HOME}/.config/afs/litellm.yaml"
AFS_CHAT_URL="${AFS_CHAT_URL:-http://localhost:3000}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

normalize_semicolon_list() {
  local input="${1:-}"
  local -a raw=()
  local -a cleaned=()
  IFS=';' read -r -a raw <<< "$input" || true
  for item in "${raw[@]-}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "$item" ]] && cleaned+=("$item")
  done
  printf '%s\n' "${cleaned[@]-}"
}

list_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

load_secrets() {
  local -a candidate_files=(
    "${AFS_SECRETS_FILE:-}"
    "${HOME}/.config/afs/secrets.env"
    "${HOME}/.secrets"
  )

  AFS_OPENAI_API_KEYS="${OPENAI_API_KEYS:-}"
  AFS_OPENAI_API_BASE_URLS="${OPENAI_API_BASE_URLS:-}"
  AFS_OPENAI_API_KEY="${OPENAI_API_KEY:-}"
  AFS_OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
  AFS_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
  AFS_CLAUDE_API_KEY="${CLAUDE_API_KEY:-}"
  AFS_GEMINI_API_KEY="${GEMINI_API_KEY:-}"
  AFS_GEMINI_API_BASE_URL="${GEMINI_API_BASE_URL:-}"
  AFS_LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-}"
  AFS_LITELLM_API_KEY="${LITELLM_API_KEY:-}"
  AFS_LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://litellm:4000/v1}"

  for file in "${candidate_files[@]}"; do
    [[ -n "$file" && -f "$file" ]] || continue
    while IFS= read -r line || [[ -n "$line" ]]; do
      line="${line%%$'\r'}"
      [[ -z "${line// }" ]] && continue
      [[ "$line" =~ ^[[:space:]]*# ]] && continue
      if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
        local key="${BASH_REMATCH[2]}"
        local value="${BASH_REMATCH[3]}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        if [[ "$value" =~ ^\"(.*)\"$ ]]; then
          value="${BASH_REMATCH[1]}"
        elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
          value="${BASH_REMATCH[1]}"
        fi
        case "$key" in
          OPENAI_API_KEYS) [[ -z "$AFS_OPENAI_API_KEYS" ]] && AFS_OPENAI_API_KEYS="$value" ;;
          OPENAI_API_BASE_URLS) [[ -z "$AFS_OPENAI_API_BASE_URLS" ]] && AFS_OPENAI_API_BASE_URLS="$value" ;;
          OPENAI_API_KEY) [[ -z "$AFS_OPENAI_API_KEY" ]] && AFS_OPENAI_API_KEY="$value" ;;
          OPENROUTER_API_KEY) [[ -z "$AFS_OPENROUTER_API_KEY" ]] && AFS_OPENROUTER_API_KEY="$value" ;;
          ANTHROPIC_API_KEY) [[ -z "$AFS_ANTHROPIC_API_KEY" ]] && AFS_ANTHROPIC_API_KEY="$value" ;;
          CLAUDE_API_KEY) [[ -z "$AFS_ANTHROPIC_API_KEY" ]] && AFS_ANTHROPIC_API_KEY="$value" ;;
          GEMINI_API_KEY) [[ -z "$AFS_GEMINI_API_KEY" ]] && AFS_GEMINI_API_KEY="$value" ;;
          GEMINI_API_BASE_URL) [[ -z "$AFS_GEMINI_API_BASE_URL" ]] && AFS_GEMINI_API_BASE_URL="$value" ;;
          LITELLM_MASTER_KEY) [[ -z "$AFS_LITELLM_MASTER_KEY" ]] && AFS_LITELLM_MASTER_KEY="$value" ;;
          LITELLM_API_KEY) [[ -z "$AFS_LITELLM_API_KEY" ]] && AFS_LITELLM_API_KEY="$value" ;;
          LITELLM_BASE_URL) [[ -z "$AFS_LITELLM_BASE_URL" ]] && AFS_LITELLM_BASE_URL="$value" ;;
        esac
      fi
  done < "$file"
  done

  if [[ -z "$AFS_ANTHROPIC_API_KEY" && -n "$AFS_CLAUDE_API_KEY" ]]; then
    AFS_ANTHROPIC_API_KEY="$AFS_CLAUDE_API_KEY"
  fi
}

sync_openwebui_secrets() {
  load_secrets
  local openai_api_keys="$AFS_OPENAI_API_KEYS"
  local openai_api_base_urls="$AFS_OPENAI_API_BASE_URLS"
  local openai_api_key="$AFS_OPENAI_API_KEY"
  local openrouter_api_key="$AFS_OPENROUTER_API_KEY"
  local anthropic_api_key="$AFS_ANTHROPIC_API_KEY"
  local gemini_api_key="$AFS_GEMINI_API_KEY"
  local gemini_api_base_url="$AFS_GEMINI_API_BASE_URL"
  local litellm_master_key="$AFS_LITELLM_MASTER_KEY"
  local litellm_api_key="$AFS_LITELLM_API_KEY"
  local litellm_base_url="$AFS_LITELLM_BASE_URL"

  local -a base_raw=()
  local -a key_raw=()
  local -a openai_base_list=()
  local -a openai_keys_list=()
  local openai_default_url="https://api.openai.com/v1"
  local openrouter_url="https://openrouter.ai/api/v1"
  local litellm_enabled=0
  local litellm_key=""

  while IFS= read -r item; do
    base_raw+=("$item")
  done < <(normalize_semicolon_list "$openai_api_base_urls")

  while IFS= read -r item; do
    key_raw+=("$item")
  done < <(normalize_semicolon_list "$openai_api_keys")

  local count=${#base_raw[@]}
  if (( ${#key_raw[@]} > count )); then
    count=${#key_raw[@]}
  fi

  for (( i=0; i<count; i++ )); do
    local base="${base_raw[i]-}"
    local key="${key_raw[i]:-}"
    if [[ -z "$base" ]]; then
      base="$openai_default_url"
    fi
    openai_base_list+=("$base")
    openai_keys_list+=("$key")
  done

  ensure_openai_pair() {
    local base="$1"
    local key="$2"
    local position="${3:-append}"
    local i
    for i in "${!openai_base_list[@]}"; do
      if [[ "${openai_base_list[$i]}" == "$base" ]]; then
        [[ -n "$key" ]] && openai_keys_list[$i]="$key"
        return 0
      fi
    done
    if [[ "$position" == "prepend" ]]; then
      openai_base_list=("$base" "${openai_base_list[@]-}")
      openai_keys_list=("$key" "${openai_keys_list[@]-}")
    else
      openai_base_list+=("$base")
      openai_keys_list+=("$key")
    fi
  }

  if [[ -n "$openai_api_key" ]]; then
    ensure_openai_pair "$openai_default_url" "$openai_api_key" "prepend"
  fi

  if [[ -n "$openrouter_api_key" ]]; then
    ensure_openai_pair "$openrouter_url" "$openrouter_api_key" "append"
  fi

  if [[ -n "$anthropic_api_key" || -n "$gemini_api_key" || -n "$litellm_master_key" || -n "$litellm_api_key" ]]; then
    litellm_enabled=1
  fi

  if [[ -n "$litellm_master_key" ]]; then
    litellm_key="$litellm_master_key"
  elif [[ -n "$litellm_api_key" ]]; then
    litellm_key="$litellm_api_key"
  fi

  if (( litellm_enabled )); then
    ensure_openai_pair "$litellm_base_url" "$litellm_key" "append"
  fi

  if [[ -z "$openrouter_api_key" ]]; then
    local -a filtered_bases=()
    local -a filtered_keys=()
    local i
    for i in "${!openai_base_list[@]}"; do
      if [[ "${openai_base_list[$i]}" == "$openrouter_url" ]]; then
        continue
      fi
      filtered_bases+=("${openai_base_list[$i]}")
      filtered_keys+=("${openai_keys_list[$i]}")
    done
    openai_base_list=("${filtered_bases[@]}")
    openai_keys_list=("${filtered_keys[@]}")
  fi

  if (( litellm_enabled == 0 )); then
    local -a filtered_bases=()
    local -a filtered_keys=()
    local i
    for i in "${!openai_base_list[@]}"; do
      if [[ "${openai_base_list[$i]}" == "$litellm_base_url" ]]; then
        continue
      fi
      filtered_bases+=("${openai_base_list[$i]}")
      filtered_keys+=("${openai_keys_list[$i]}")
    done
    openai_base_list=("${filtered_bases[@]}")
    openai_keys_list=("${filtered_keys[@]}")
  fi

  if (( ${#openai_base_list[@]} > 0 )); then
    openai_api_base_urls="$(IFS=';'; echo "${openai_base_list[*]}")"
  else
    openai_api_base_urls=""
  fi

  if (( ${#openai_keys_list[@]} > 0 )); then
    openai_api_keys="$(IFS=';'; echo "${openai_keys_list[*]}")"
  else
    openai_api_keys=""
  fi

  mkdir -p "$(dirname "$SECRETS_ENV_FILE")"
  : > "$SECRETS_ENV_FILE"
  chmod 600 "$SECRETS_ENV_FILE"

  if [[ -n "$openai_api_keys" ]]; then
    echo "OPENAI_API_KEYS=$openai_api_keys" >> "$SECRETS_ENV_FILE"
  fi
  if [[ -n "$openai_api_base_urls" ]]; then
    echo "OPENAI_API_BASE_URLS=$openai_api_base_urls" >> "$SECRETS_ENV_FILE"
  fi
  if [[ -n "$gemini_api_key" ]]; then
    echo "GEMINI_API_KEY=$gemini_api_key" >> "$SECRETS_ENV_FILE"
  fi
  if [[ -n "$gemini_api_base_url" ]]; then
    echo "GEMINI_API_BASE_URL=$gemini_api_base_url" >> "$SECRETS_ENV_FILE"
  fi
}

sync_litellm_secrets() {
  load_secrets

  mkdir -p "$(dirname "$LITELLM_ENV_FILE")"
  : > "$LITELLM_ENV_FILE"
  chmod 600 "$LITELLM_ENV_FILE"

  if [[ -n "$AFS_LITELLM_MASTER_KEY" ]]; then
    echo "LITELLM_MASTER_KEY=$AFS_LITELLM_MASTER_KEY" >> "$LITELLM_ENV_FILE"
  fi
  if [[ -n "$AFS_LITELLM_API_KEY" ]]; then
    echo "LITELLM_API_KEY=$AFS_LITELLM_API_KEY" >> "$LITELLM_ENV_FILE"
  fi
  if [[ -n "$AFS_OPENAI_API_KEY" ]]; then
    echo "OPENAI_API_KEY=$AFS_OPENAI_API_KEY" >> "$LITELLM_ENV_FILE"
  fi
  if [[ -n "$AFS_OPENROUTER_API_KEY" ]]; then
    echo "OPENROUTER_API_KEY=$AFS_OPENROUTER_API_KEY" >> "$LITELLM_ENV_FILE"
  fi
  if [[ -n "$AFS_ANTHROPIC_API_KEY" ]]; then
    echo "ANTHROPIC_API_KEY=$AFS_ANTHROPIC_API_KEY" >> "$LITELLM_ENV_FILE"
  fi
  if [[ -n "$AFS_GEMINI_API_KEY" ]]; then
    echo "GEMINI_API_KEY=$AFS_GEMINI_API_KEY" >> "$LITELLM_ENV_FILE"
  fi
  if [[ -n "$AFS_GEMINI_API_BASE_URL" ]]; then
    echo "GEMINI_API_BASE_URL=$AFS_GEMINI_API_BASE_URL" >> "$LITELLM_ENV_FILE"
  fi
}

ensure_litellm_config() {
  if [[ -f "$LITELLM_CONFIG_FILE" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$LITELLM_CONFIG_FILE")"
  cat <<'EOF' > "$LITELLM_CONFIG_FILE"
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20240620
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: claude-3-5-haiku
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: gemini-1.5-pro
    litellm_params:
      model: gemini/gemini-1.5-pro-latest
      api_key: os.environ/GEMINI_API_KEY
  - model_name: gemini-1.5-flash
    litellm_params:
      model: gemini/gemini-1.5-flash-latest
      api_key: os.environ/GEMINI_API_KEY
EOF
}

sync_openwebui_ollama_labels() {
  local container=""
  if check_container "afs-chat-simple"; then
    container="afs-chat-simple"
  elif check_container "afs-chat"; then
    container="afs-chat"
  else
    return 0
  fi

  local image
  image=$(docker inspect -f '{{.Config.Image}}' "$container" 2>/dev/null || true)
  local data_source
  data_source=$(docker inspect -f '{{ range .Mounts }}{{ if eq .Destination "/app/backend/data" }}{{ .Source }}{{ end }}{{ end }}' "$container" 2>/dev/null || true)

  if [[ -z "$image" || -z "$data_source" ]]; then
    return 0
  fi

  local output=""
  local status=0

  set +e
  output=$(docker run --rm --entrypoint python -v "$data_source":/data "$image" -c "
import json
import sqlite3
from pathlib import Path

db = Path('/data/webui.db')
if not db.exists():
    print('missing')
    raise SystemExit(0)

conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute('select id, data from config order by id desc limit 1')
row = cur.fetchone()

if row:
    config_id, data = row
    config = json.loads(data)
else:
    config_id = None
    config = {'version': 0, 'ui': {}}

desired = {
    '0': {
        'prefix_id': 'win',
        'tags': ['windows', 'gpu'],
        'connection_type': 'remote',
    },
    '1': {
        'prefix_id': 'mac',
        'tags': ['mac', 'local'],
        'connection_type': 'local',
    },
}

current = config.get('ollama', {}).get('api_configs', {})
if not isinstance(current, dict):
    current = {}

updated = dict(current)
changed = False

for idx, entry in desired.items():
    existing = updated.get(idx, {})
    if not isinstance(existing, dict):
        existing = {}
    merged = dict(existing)
    merged.update(entry)
    if merged != existing:
        changed = True
    updated[idx] = merged

if not changed and updated == current:
    print('unchanged')
    raise SystemExit(0)

config.setdefault('ollama', {})['api_configs'] = updated
payload = json.dumps(config)

if config_id is None:
    cur.execute('insert into config (data, version) values (?, ?)', (payload, config.get('version', 0)))
else:
    cur.execute('update config set data=?, updated_at=CURRENT_TIMESTAMP where id=?', (payload, config_id))

conn.commit()
print('updated')
")
  status=$?
  set -e

  if (( status != 0 )); then
    return 0
  fi

  if [[ "$output" == "updated" ]]; then
    docker restart "$container" >/dev/null 2>&1 || true
  fi
}

# Check if docker container is running
check_container() {
  docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q 'true'
}

# Check if port is listening
check_port() {
  lsof -i ":$1" -sTCP:LISTEN >/dev/null 2>&1
}

# Get full status
get_status() {
  local openwebui_status="stopped"
  local gateway_status="stopped"
  local ollama_status="stopped"
  local gateway_pid=""

  # Check OpenWebUI
  if check_container "afs-chat-simple" || check_container "afs-chat"; then
    openwebui_status="running"
  fi

  # Check Gateway
  if check_port 8000; then
    gateway_status="running"
    gateway_pid=$(lsof -ti :8000 2>/dev/null | head -1)
  fi

  # Check Ollama
  if check_port 11434; then
    ollama_status="running (local)"
  elif check_port 11435; then
    ollama_status="running (windows tunnel)"
  fi

  echo "$openwebui_status $gateway_status $ollama_status $gateway_pid"
}

cmd_start() {
  local mode="${1:-simple}"
  local litellm_enabled=0

  echo "Starting AFS Chat infrastructure..."
  load_secrets

  if [[ -n "$AFS_ANTHROPIC_API_KEY" || -n "$AFS_GEMINI_API_KEY" || -n "$AFS_LITELLM_MASTER_KEY" || -n "$AFS_LITELLM_API_KEY" ]]; then
    litellm_enabled=1
  fi

  if (( litellm_enabled )); then
    sync_litellm_secrets
    ensure_litellm_config
  fi

  sync_openwebui_secrets

  case "$mode" in
    simple)
      echo "  Mode: Simple (OpenWebUI → Ollama directly)"
      cd "$DOCKER_DIR"
      local compose_file="${AFS_CHAT_COMPOSE_FILE:-docker-compose.simple.yml}"
      if [[ ! -f "$compose_file" ]]; then
        echo "  Warning: compose file not found: $compose_file (falling back to docker-compose.simple.yml)"
        compose_file="docker-compose.simple.yml"
      fi
      local -a compose_args=("-f" "$compose_file")
      if [[ -n "${AFS_CHAT_COMPOSE_OVERRIDE:-}" ]]; then
        local override
        IFS=':' read -r -a overrides <<< "$AFS_CHAT_COMPOSE_OVERRIDE"
        for override in "${overrides[@]-}"; do
          [[ -z "$override" ]] && continue
          if [[ -f "$override" ]]; then
            compose_args+=("-f" "$override")
          else
            echo "  Warning: compose override not found: $override"
          fi
        done
      fi
      if (( litellm_enabled )); then
        docker compose "${compose_args[@]}" --profile litellm up -d
      else
        docker compose "${compose_args[@]}" up -d
      fi
      sync_openwebui_ollama_labels || true
      echo -e "  ${GREEN}✓${NC} OpenWebUI started at $AFS_CHAT_URL"
      ;;
    full)
      echo "  Mode: Full (OpenWebUI → Gateway → Ollama)"

      # Start gateway first
      echo "  Starting Gateway..."
      cd "$AFS_ROOT"
      source .venv/bin/activate 2>/dev/null || true
      mkdir -p "$STATE_DIR"

      PYTHONPATH=src nohup python -m uvicorn afs.gateway.server:app \
        --host 0.0.0.0 --port 8000 \
        > /tmp/afs-gateway.log 2>&1 &

      local pid=$!
      echo "{\"pid\": $pid, \"started_at\": \"$(date -Iseconds)\"}" > "$STATE_DIR/gateway.json"
      echo -e "  ${GREEN}✓${NC} Gateway started at http://localhost:8000 (PID: $pid)"

      # Wait for gateway to be ready
      sleep 2

      # Start OpenWebUI
      echo "  Starting OpenWebUI..."
      cd "$DOCKER_DIR"
      docker compose -f docker-compose.yml up -d
      echo -e "  ${GREEN}✓${NC} OpenWebUI started at $AFS_CHAT_URL"
      ;;
    gateway)
      echo "  Mode: Gateway only"
      cd "$AFS_ROOT"
      source .venv/bin/activate 2>/dev/null || true
      mkdir -p "$STATE_DIR"

      PYTHONPATH=src nohup python -m uvicorn afs.gateway.server:app \
        --host 0.0.0.0 --port 8000 \
        > /tmp/afs-gateway.log 2>&1 &

      local pid=$!
      echo "{\"pid\": $pid, \"started_at\": \"$(date -Iseconds)\"}" > "$STATE_DIR/gateway.json"
      echo -e "  ${GREEN}✓${NC} Gateway started at http://localhost:8000 (PID: $pid)"
      ;;
    *)
      echo "Unknown mode: $mode"
      echo "Valid modes: simple, full, gateway"
      return 1
      ;;
  esac

  # Clear cache
  rm -f "$CACHE_FILE"

  echo ""
  echo "Chat ready! Open $AFS_CHAT_URL"
}

cmd_stop() {
  echo "Stopping AFS Chat infrastructure..."

  # Stop OpenWebUI containers
  cd "$DOCKER_DIR"
  local compose_file="${AFS_CHAT_COMPOSE_FILE:-docker-compose.simple.yml}"
  if [[ ! -f "$compose_file" ]]; then
    echo "  Warning: compose file not found: $compose_file (falling back to docker-compose.simple.yml)"
    compose_file="docker-compose.simple.yml"
  fi
  local -a compose_args=("-f" "$compose_file")
  if [[ -n "${AFS_CHAT_COMPOSE_OVERRIDE:-}" ]]; then
    local override
    IFS=':' read -r -a overrides <<< "$AFS_CHAT_COMPOSE_OVERRIDE"
    for override in "${overrides[@]-}"; do
      [[ -z "$override" ]] && continue
      if [[ -f "$override" ]]; then
        compose_args+=("-f" "$override")
      else
        echo "  Warning: compose override not found: $override"
      fi
    done
  fi
  docker compose "${compose_args[@]}" down 2>/dev/null || true
  docker compose -f docker-compose.yml down 2>/dev/null || true
  echo -e "  ${GREEN}✓${NC} OpenWebUI stopped"

  # Stop Gateway
  if check_port 8000; then
    pkill -f "uvicorn afs.gateway.server:app" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Gateway stopped"
  fi

  rm -f "$STATE_DIR/gateway.json"
  rm -f "$CACHE_FILE"

  echo "All services stopped."
}

cmd_status() {
  local json_mode=0
  [[ "${1:-}" == "--json" ]] && json_mode=1

  read -r openwebui gateway ollama pid <<<"$(get_status)"

  if [[ $json_mode -eq 1 ]]; then
    cat <<EOF
{
  "openwebui": {"status": "$openwebui"},
  "gateway": {"status": "$gateway", "pid": ${pid:-null}},
  "ollama": {"status": "$ollama"},
  "url": "$AFS_CHAT_URL"
}
EOF
  else
    echo "AFS Chat Status"
    echo "---------------"

    if [[ "$openwebui" == "running" ]]; then
      echo -e "  OpenWebUI: ${GREEN}●${NC} running ($AFS_CHAT_URL)"
    else
      echo -e "  OpenWebUI: ${RED}○${NC} stopped"
    fi

    if [[ "$gateway" == "running" ]]; then
      echo -e "  Gateway:   ${GREEN}●${NC} running (http://localhost:8000) [PID: $pid]"
    else
      echo -e "  Gateway:   ${RED}○${NC} stopped"
    fi

    if [[ "$ollama" == "stopped" ]]; then
      echo -e "  Ollama:    ${RED}○${NC} stopped"
    else
      echo -e "  Ollama:    ${GREEN}●${NC} $ollama"
    fi
  fi
}

cmd_restart() {
  cmd_stop
  sleep 1
  cmd_start "${1:-simple}"
}

cmd_open() {
  open "$AFS_CHAT_URL"
}

cmd_logs() {
  echo "=== Gateway logs ==="
  tail -f /tmp/afs-gateway.log 2>/dev/null || echo "No gateway logs available"
}

cmd_help() {
  cat <<EOF
AFS Chat Service Control

Usage: $(basename "$0") <command> [options]

Commands:
  start [mode]     Start chat services
                   Modes: simple (default), full, gateway
  stop             Stop all chat services
  status [--json]  Show service status
  restart [mode]   Restart services
  open             Open chat in browser
  logs             Tail service logs
  help             Show this help

Examples:
  $(basename "$0") start              # Start OpenWebUI (simple mode)
  $(basename "$0") start full         # Start with Gateway
  $(basename "$0") status --json      # Get status as JSON
  $(basename "$0") restart full       # Restart in full mode
EOF
}

# Main
case "${1:-help}" in
  start)   shift; cmd_start "$@" ;;
  stop)    cmd_stop ;;
  status)  shift; cmd_status "$@" ;;
  restart) shift; cmd_restart "$@" ;;
  open)    cmd_open ;;
  logs)    cmd_logs ;;
  help|--help|-h) cmd_help ;;
  *)
    echo "Unknown command: $1"
    cmd_help
    exit 1
    ;;
esac
