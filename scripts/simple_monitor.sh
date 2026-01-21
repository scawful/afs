#!/bin/bash
# Simple vast.ai training monitor - no SSH needed
# Just checks instance status via vastai CLI

INSTANCES="30007012 30007267 30007268 30007269 30007270"
LOG_FILE="/Users/scawful/.context/training/monitor.log"

echo "=== Training Monitor Started: $(date) ===" >> "$LOG_FILE"

while true; do
    echo "" >> "$LOG_FILE"
    echo "--- Status Check: $(date) ---" >> "$LOG_FILE"

    # Check each instance
    for instance in $INSTANCES; do
        status=$(vastai show instances | grep "^$instance" | awk '{print $3, $4, $14}')
        echo "Instance $instance: $status" >> "$LOG_FILE"
    done

    # Check if any are still running
    running=$(vastai show instances | grep -E "$(echo $INSTANCES | tr ' ' '|')" | grep -c "running")

    if [ "$running" -eq 0 ]; then
        echo "" >> "$LOG_FILE"
        echo "=== ALL TRAINING COMPLETE: $(date) ===" >> "$LOG_FILE"

        # Create completion marker
        echo "$(date)" > /Users/scawful/.context/training/training_complete.marker

        break
    fi

    echo "Still running: $running/5 instances" >> "$LOG_FILE"

    # Wait 5 minutes
    sleep 300
done

echo "Monitor exiting. Check $LOG_FILE for details."
