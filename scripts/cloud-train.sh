#!/bin/bash
# Quick cloud training script for Vast.ai
# Usage: ./scripts/cloud-train.sh <training_data.jsonl> [gpu_type]

set -e

DATA_FILE="${1:?Usage: $0 <training_data.jsonl> [gpu_type]}"
GPU_TYPE="${2:-RTX_5090}"

# Check prerequisites
command -v vastai >/dev/null || { echo "Install: pip install vastai"; exit 1; }
[[ -f "$DATA_FILE" ]] || { echo "Data file not found: $DATA_FILE"; exit 1; }

echo "=== Vast.ai Cloud Training ==="
echo "Data: $DATA_FILE"
echo "GPU: $GPU_TYPE"

# Find best offer
echo -e "\n[1/7] Searching for $GPU_TYPE..."
OFFER=$(vastai search offers "gpu_name=$GPU_TYPE rentable=true disk_space>=50" --order 'dph_total' --raw | head -1 | jq -r '.id')
[[ -z "$OFFER" || "$OFFER" == "null" ]] && { echo "No $GPU_TYPE available"; exit 1; }
PRICE=$(vastai search offers "gpu_name=$GPU_TYPE rentable=true" --order 'dph_total' --raw | head -1 | jq -r '.dph_total')
echo "Found offer $OFFER at \$$PRICE/hr"

# Create instance
echo -e "\n[2/7] Creating instance..."
RESULT=$(vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 50 --ssh --direct)
INSTANCE=$(echo "$RESULT" | grep -oP "new_contract': \K\d+" || echo "$RESULT" | jq -r '.new_contract')
echo "Instance: $INSTANCE"

# Wait for startup
echo -e "\n[3/7] Waiting for instance..."
for i in {1..30}; do
    STATUS=$(vastai show instance $INSTANCE 2>/dev/null | awk 'NR==2 {print $3}')
    echo -n "."
    [[ "$STATUS" == "running" ]] && break
    sleep 10
done
echo " Ready!"

# Get SSH details
SSH_URL=$(vastai ssh-url $INSTANCE)
SSH_HOST=$(echo $SSH_URL | sed 's|ssh://root@||' | cut -d: -f1)
SSH_PORT=$(echo $SSH_URL | sed 's|ssh://root@||' | cut -d: -f2)

# Attach SSH key
vastai attach ssh $INSTANCE "$(cat ~/.ssh/id_rsa.pub)" >/dev/null 2>&1
sleep 5

# Upload data
echo -e "\n[4/7] Uploading training data..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "mkdir -p /workspace/training/data" 2>/dev/null
scp -o StrictHostKeyChecking=no -P $SSH_PORT "$DATA_FILE" root@$SSH_HOST:/workspace/training/data/training.jsonl

# Install packages
echo -e "\n[5/7] Installing packages..."
ssh -p $SSH_PORT root@$SSH_HOST "pip install -q peft transformers datasets accelerate bitsandbytes" 2>/dev/null

# Run training
echo -e "\n[6/7] Training..."
ssh -p $SSH_PORT root@$SSH_HOST "cd /workspace/training && python train.py" 2>&1 | tail -20

# Download results
echo -e "\n[7/7] Downloading adapters..."
mkdir -p models/cloud-adapters
scp -P $SSH_PORT root@$SSH_HOST:/workspace/training/output/adapter_config.json models/cloud-adapters/
scp -P $SSH_PORT root@$SSH_HOST:/workspace/training/output/adapter_model.safetensors models/cloud-adapters/

# Cleanup
echo -e "\n=== Cleanup ==="
vastai destroy instance $INSTANCE
echo "Instance destroyed"

echo -e "\n=== Complete ==="
echo "Adapters saved to: models/cloud-adapters/"
ls -la models/cloud-adapters/
