#!/bin/bash
# Quick start script for AFS Chat infrastructure
# Usage: ./chat-start.sh [simple|full|gateway]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AFS_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$AFS_ROOT/docker"

case "${1:-simple}" in
    simple)
        "$SCRIPT_DIR/chat-service.sh" start simple
        ;;
    full)
        "$SCRIPT_DIR/chat-service.sh" start full
        ;;
    gateway)
        echo "Starting AFS Gateway only (development mode)..."
        cd "$AFS_ROOT"
        python -m afs gateway serve --reload
        ;;
    stop)
        "$SCRIPT_DIR/chat-service.sh" stop
        ;;
    *)
        echo "Usage: $0 [simple|full|gateway|stop]"
        echo ""
        echo "Modes:"
        echo "  simple   - Open WebUI only, connects to local Ollama (default)"
        echo "  full     - Open WebUI + AFS Gateway (Docker)"
        echo "  gateway  - Gateway API only, development mode with reload"
        echo "  stop     - Stop all containers"
        exit 1
        ;;
esac
