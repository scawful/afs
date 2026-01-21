#!/bin/bash
# Examples and usage patterns for training_monitor_tui.py
#
# This file contains common use cases and command patterns for the
# VAST.AI Training Monitor TUI.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/training_monitor_tui.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== VAST.AI Training Monitor - Usage Examples ===${NC}\n"

# ==============================================================================
# Example 1: Basic Monitoring
# ==============================================================================
echo -e "${GREEN}Example 1: Monitor specific instances${NC}"
echo "Command:"
echo "  python3 $SCRIPT --instances 30007012,30007267,30007268,30007269,30007270"
echo ""
echo "Description:"
echo "  Monitors 5 specific vast.ai instances with default 10-second refresh interval"
echo ""
echo "Use case: Watch your current training jobs"
echo ""

# ==============================================================================
# Example 2: Monitor All Instances
# ==============================================================================
echo -e "${GREEN}Example 2: Monitor all running instances${NC}"
echo "Command:"
echo "  python3 $SCRIPT --all"
echo ""
echo "Description:"
echo "  Automatically discovers and monitors all running vast.ai instances"
echo ""
echo "Use case: Unattended monitoring of entire training swarm"
echo ""

# ==============================================================================
# Example 3: Custom Refresh Rate
# ==============================================================================
echo -e "${GREEN}Example 3: High-frequency monitoring${NC}"
echo "Command:"
echo "  python3 $SCRIPT --instances 30007012,30007267 --interval 5"
echo ""
echo "Description:"
echo "  Updates every 5 seconds instead of default 10 seconds"
echo ""
echo "Use case: Close monitoring during critical training phases"
echo "Note: Higher frequency = more API calls, potential rate limiting"
echo ""

# ==============================================================================
# Example 4: Test Mode
# ==============================================================================
echo -e "${GREEN}Example 4: Test the UI without API (mock mode)${NC}"
echo "Command:"
echo "  python3 $SCRIPT --instances 30007012,30007267,30007268,30007269,30007270 --mock"
echo ""
echo "Description:"
echo "  Uses simulated data instead of connecting to vast.ai API"
echo ""
echo "Use case: Test display, verify functionality without API credentials"
echo ""

# ==============================================================================
# Example 5: Background Monitoring with Logging
# ==============================================================================
echo -e "${GREEN}Example 5: Run in background with output logging${NC}"
echo "Command:"
echo "  nohup python3 $SCRIPT --instances 30007012,30007267 > monitor.log 2>&1 &"
echo ""
echo "Description:"
echo "  Runs monitor in background and logs output to file"
echo ""
echo "Use case: Long-running monitoring that survives SSH disconnect"
echo "Note: Will not display interactive UI, but logs metrics to file"
echo ""

# ==============================================================================
# Example 6: Monitor with Slowdown for Low Power Machine
# ==============================================================================
echo -e "${GREEN}Example 6: Gentle monitoring (15-second intervals)${NC}"
echo "Command:"
echo "  python3 $SCRIPT --instances 30007012,30007267,30007268 --interval 15"
echo ""
echo "Description:"
echo "  Updates less frequently to reduce API load and CPU usage"
echo ""
echo "Use case: Low-power laptops, machines with slow network"
echo ""

# ==============================================================================
# Example 7: Capture Session for Recording/Sharing
# ==============================================================================
echo -e "${GREEN}Example 7: Record session to file (screen/asciinema)${NC}"
echo "Commands:"
echo "  # Using asciinema (requires: pip install asciinema)"
echo "  asciinema rec training-session.cast"
echo "  python3 $SCRIPT --instances 30007012,30007267"
echo ""
echo "  # Using screen"
echo "  script training-session.log"
echo "  python3 $SCRIPT --instances 30007012,30007267"
echo ""
echo "Description:"
echo "  Records the entire monitor session for review or sharing"
echo ""
echo "Use case: Create training documentation, share progress with team"
echo ""

# ==============================================================================
# Example 8: Real-time SSH Tunnel to Remote Machine
# ==============================================================================
echo -e "${GREEN}Example 8: Monitor remote vast.ai instances from local machine${NC}"
echo "Commands:"
echo "  # Via SSH with terminal multiplexing"
echo "  ssh -t user@remote-server \\\"cd ~/src/lab/afs && python3 scripts/training_monitor_tui.py --instances 30007012,30007267\\\""
echo ""
echo "Description:"
echo "  Connect to a remote server that has vast.ai credentials configured"
echo ""
echo "Use case: Central monitoring server monitoring training swarm"
echo ""

# ==============================================================================
# Example 9: Automated Monitoring with Systemd Service
# ==============================================================================
echo -e "${GREEN}Example 9: Set up as system service${NC}"
echo "Create file: ~/.config/systemd/user/training-monitor.service"
echo ""
cat << 'EOF'
[Unit]
Description=VAST.AI Training Monitor
After=network-online.target

[Service]
Type=simple
User=scawful
WorkingDirectory=/Users/scawful/src/lab/afs
ExecStart=/usr/bin/python3 scripts/training_monitor_tui.py --instances 30007012,30007267,30007268,30007269,30007270 --interval 10
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
echo ""
echo "Then enable and start:"
echo "  systemctl --user enable training-monitor"
echo "  systemctl --user start training-monitor"
echo ""

# ==============================================================================
# Example 10: Debug Mode with Detailed Logging
# ==============================================================================
echo -e "${GREEN}Example 10: Debug mode for troubleshooting${NC}"
echo "Command:"
echo "  python3 -u $SCRIPT --instances 30007012,30007267 2> debug.log"
echo ""
echo "Description:"
echo "  Writes debug logs to file for troubleshooting API issues"
echo ""
echo "Use case: Diagnosing connection problems, API errors"
echo "Then: tail -f debug.log"
echo ""

# ==============================================================================
# Example 11: Monitor During Training Kick-off
# ==============================================================================
echo -e "${GREEN}Example 11: Full training workflow with monitoring${NC}"
echo "Commands:"
echo "  # 1. Launch training instances"
echo "  python3 scripts/vastai_setup.py --all-models --budget 100"
echo ""
echo "  # 2. Wait a moment for instances to start"
echo "  sleep 30"
echo ""
echo "  # 3. Start monitoring"
echo "  python3 $SCRIPT --all"
echo ""
echo "Description:"
echo "  Typical workflow for launching and monitoring training swarm"
echo ""

# ==============================================================================
# Example 12: Cost-Conscious Monitoring
# ==============================================================================
echo -e "${GREEN}Example 12: Minimize API calls for cost-conscious users${NC}"
echo "Command:"
echo "  python3 $SCRIPT --instances 30007012,30007267 --interval 60"
echo ""
echo "Description:"
echo "  Updates once per minute instead of every 10 seconds"
echo ""
echo "Use case: Reduce vast.ai API call rates (1 call per 60s instead of per 10s)"
echo "Note: Reduces responsiveness but significantly lower API overhead"
echo ""

# ==============================================================================
# Keyboard Controls Reference
# ==============================================================================
echo ""
echo -e "${YELLOW}=== Interactive Controls (while monitoring) ===${NC}\n"
echo "q             Quit the monitor"
echo "r             Force refresh metrics immediately"
echo "p             Pause/Resume automatic updates"
echo "c             Clear all training logs"
echo "Ctrl+C        Force quit (same as 'q')"
echo ""

# ==============================================================================
# Tips and Best Practices
# ==============================================================================
echo -e "${YELLOW}=== Tips and Best Practices ===${NC}\n"

echo "1. Choose appropriate refresh interval:"
echo "   - Fast training (10-100 epochs): 5-10 seconds"
echo "   - Standard training (3-10 epochs): 10-30 seconds"
echo "   - Long training (1-2 epochs): 30-60 seconds"
echo ""

echo "2. Monitor cost in real-time:"
echo "   - Check 'Cost/hr' column for per-instance rates"
echo "   - Use 'p' key to pause and calculate totals"
echo "   - Watch for GPU utilization dips (may indicate issues)"
echo ""

echo "3. Detect training problems early:"
echo "   - Look for yellow/red status indicators"
echo "   - Check loss values (should decrease gradually)"
echo "   - Watch GPU% (should be >50% for healthy training)"
echo "   - Monitor 'ETA' divergence (should decrease over time)"
echo ""

echo "4. Efficient log monitoring:"
echo "   - Recent logs show last 10 training outputs"
echo "   - Use 'c' key to clear logs when reviewing"
echo "   - Logs auto-update, can see training progress in real-time"
echo ""

echo "5. Remote monitoring setup:"
echo "   - Configure SSH keys for passwordless access"
echo "   - Use tmux/screen for persistent sessions"
echo "   - Consider running on low-power server for 24/7 monitoring"
echo ""

# ==============================================================================
# Troubleshooting Quick Reference
# ==============================================================================
echo -e "${YELLOW}=== Troubleshooting Quick Reference ===${NC}\n"

echo "Issue: 'No instances found'"
echo "Fix:   1. Verify vast.ai CLI: which vastai"
echo "       2. Check API key: vastai show user"
echo "       3. Ensure instances running: vastai show instances"
echo ""

echo "Issue: 'Logs not updating'"
echo "Fix:   1. Check SSH access: ssh root@<instance-ip> ls /workspace/output/"
echo "       2. Verify log format in training script"
echo "       3. Use --mock to test UI independently"
echo ""

echo "Issue: 'High CPU usage'"
echo "Fix:   1. Increase --interval (e.g., --interval 30)"
echo "       2. Check if vast.ai API is slow"
echo "       3. Reduce number of monitored instances"
echo ""

echo "Issue: 'Terminal display issues'"
echo "Fix:   1. Set TERM: export TERM=xterm-256color"
echo "       2. Ensure terminal is wide enough (80+ cols)"
echo "       3. Try: clear && python3 $SCRIPT --instances <IDS>"
echo ""

echo -e "${BLUE}=== End of Examples ===${NC}\n"
echo "For more information, see: $SCRIPT_DIR/TRAINING_MONITOR_README.md"
