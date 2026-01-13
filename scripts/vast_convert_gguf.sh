#!/bin/bash
# Convert trained model to GGUF format for LMStudio
# Run this after training completes

set -e

MODEL_NAME="${1:-agahnim-v2}"
INPUT_DIR="/workspace/output/${MODEL_NAME}-lora-merged"
OUTPUT_DIR="/workspace/output/gguf"

echo "=== Converting ${MODEL_NAME} to GGUF ==="

# Install llama.cpp if not present
if [ ! -d "/workspace/llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    cd /workspace
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    pip install -r requirements.txt
fi

mkdir -p ${OUTPUT_DIR}

# Convert to GGUF (F16)
echo "Converting to GGUF..."
python /workspace/llama.cpp/convert_hf_to_gguf.py \
    ${INPUT_DIR} \
    --outfile ${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf \
    --outtype f16

# Quantize to Q8_0 (best quality, reasonable size)
echo "Quantizing to Q8_0..."
/workspace/llama.cpp/llama-quantize \
    ${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf \
    ${OUTPUT_DIR}/${MODEL_NAME}-q8_0.gguf \
    Q8_0

echo "=== Conversion Complete ==="
echo "F16 model: ${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf"
echo "Q8 model:  ${OUTPUT_DIR}/${MODEL_NAME}-q8_0.gguf"

# Show file sizes
ls -lh ${OUTPUT_DIR}/*.gguf

echo ""
echo "Download with:"
echo "  scp -P PORT root@ssh.vast.ai:${OUTPUT_DIR}/${MODEL_NAME}-q8_0.gguf ."
