#!/bin/bash
# Setup script for Agahnim v2 training on vast.ai
# Run this after uploading training data

set -e

echo "=== Setting up Agahnim v2 Training ==="

# Install dependencies
pip install --quiet unsloth "transformers>=4.46" datasets trl peft accelerate bitsandbytes

# Create directories
mkdir -p /workspace/data /workspace/output

# Check for training data
if [ ! -f "/workspace/data/agahnim_v2_training.jsonl" ]; then
    echo "ERROR: Training data not found!"
    echo "Upload agahnim_v2_training.jsonl to /workspace/data/"
    exit 1
fi

echo "Training data found: $(wc -l /workspace/data/agahnim_v2_training.jsonl) samples"

# Download training script if not present
if [ ! -f "/workspace/train_agahnim.py" ]; then
    echo "Training script not found, please upload train_agahnim.py"
    exit 1
fi

echo "=== Starting Training ==="
echo "Estimated time: ~1.5 hours for 518 samples (3 epochs)"

# Run training
nohup python /workspace/train_agahnim.py \
    --data /workspace/data/agahnim_v2_training.jsonl \
    --output /workspace/output/agahnim-v2-lora \
    --epochs 3 \
    --batch-size 4 \
    > /workspace/train.log 2>&1 &

echo "Training started in background. Monitor with:"
echo "  tail -f /workspace/train.log"
