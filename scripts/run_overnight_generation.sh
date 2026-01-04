#!/bin/bash
# Overnight data generation script for AFS expert models
# Generates training data for all 4 domains in parallel

set -e

cd /Users/scawful/src/lab/afs
source .venv/bin/activate

# Output directory
OUTPUT_DIR="generated_data/$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/generation.log"
echo "Starting overnight generation at $(date)" | tee "$LOG_FILE"

# Generate 2500 samples per domain (10K total)
# Using only Gemini to stay within rate limits
# Rate: ~8 req/min = ~5 hours per domain

echo "Starting parallel generation for all domains..." | tee -a "$LOG_FILE"

# Run all 4 domains in parallel, redirecting output to separate logs
PYTHONPATH=src python -m afs.generators.curriculum_generator \
    --domain din --count 2500 --output "$OUTPUT_DIR/din_train.jsonl" \
    > "$OUTPUT_DIR/din.log" 2>&1 &
DIN_PID=$!

PYTHONPATH=src python -m afs.generators.curriculum_generator \
    --domain nayru --count 2500 --output "$OUTPUT_DIR/nayru_train.jsonl" \
    > "$OUTPUT_DIR/nayru.log" 2>&1 &
NAYRU_PID=$!

PYTHONPATH=src python -m afs.generators.curriculum_generator \
    --domain farore --count 2500 --output "$OUTPUT_DIR/farore_train.jsonl" \
    > "$OUTPUT_DIR/farore.log" 2>&1 &
FARORE_PID=$!

PYTHONPATH=src python -m afs.generators.curriculum_generator \
    --domain veran --count 2500 --output "$OUTPUT_DIR/veran_train.jsonl" \
    > "$OUTPUT_DIR/veran.log" 2>&1 &
VERAN_PID=$!

echo "Started generation processes:" | tee -a "$LOG_FILE"
echo "  Din: PID $DIN_PID" | tee -a "$LOG_FILE"
echo "  Nayru: PID $NAYRU_PID" | tee -a "$LOG_FILE"
echo "  Farore: PID $FARORE_PID" | tee -a "$LOG_FILE"
echo "  Veran: PID $VERAN_PID" | tee -a "$LOG_FILE"

# Wait for all to complete
wait $DIN_PID && echo "Din complete" | tee -a "$LOG_FILE" || echo "Din failed" | tee -a "$LOG_FILE"
wait $NAYRU_PID && echo "Nayru complete" | tee -a "$LOG_FILE" || echo "Nayru failed" | tee -a "$LOG_FILE"
wait $FARORE_PID && echo "Farore complete" | tee -a "$LOG_FILE" || echo "Farore failed" | tee -a "$LOG_FILE"
wait $VERAN_PID && echo "Veran complete" | tee -a "$LOG_FILE" || echo "Veran failed" | tee -a "$LOG_FILE"

echo "Generation complete at $(date)" | tee -a "$LOG_FILE"

# Summary
echo "" | tee -a "$LOG_FILE"
echo "Summary:" | tee -a "$LOG_FILE"
for f in "$OUTPUT_DIR"/*.jsonl; do
    if [ -f "$f" ]; then
        count=$(wc -l < "$f")
        echo "  $(basename $f): $count samples" | tee -a "$LOG_FILE"
    fi
done
