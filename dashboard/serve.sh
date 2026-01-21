#!/bin/bash

# AFS Training Dashboard - Launch Script
# Starts the Flask backend and opens the dashboard in a browser

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PORT=5000
HOST="localhost"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  AFS Training Dashboard Server                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found$(NC)"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}→ Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}→ Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo -e "${YELLOW}→ Installing dependencies...${NC}"
pip install -q flask flask-cors requests 2>/dev/null || {
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
}
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Starting Flask server on http://${HOST}:${PORT}${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Change to dashboard directory
cd "$SCRIPT_DIR"

# Start Flask app in background
python3 api.py &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Check if server started successfully
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Failed to start server${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
echo ""

# Open browser
if command -v open &> /dev/null; then
    # macOS
    echo -e "${YELLOW}→ Opening dashboard in default browser...${NC}"
    open "http://${HOST}:${PORT}"
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open "http://${HOST}:${PORT}"
elif command -v start &> /dev/null; then
    # Windows
    start "http://${HOST}:${PORT}"
else
    echo -e "${YELLOW}→ Please open your browser and navigate to:${NC}"
    echo -e "${BLUE}  http://${HOST}:${PORT}${NC}"
fi

echo ""
echo -e "${GREEN}Dashboard ready!${NC}"
echo ""
echo -e "${YELLOW}Controls:${NC}"
echo "  • Press Ctrl+C to stop the server"
echo "  • Auto-refresh every 30 seconds"
echo "  • Dark mode toggle in header"
echo "  • Export data as CSV or JSON"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Keep the script running and handle cleanup
trap "echo ''; echo -e '${YELLOW}Shutting down server...${NC}'; kill $SERVER_PID 2>/dev/null || true; echo -e '${GREEN}Server stopped${NC}'; exit 0" INT TERM

wait $SERVER_PID
