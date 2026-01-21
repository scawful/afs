# Production Dockerfile for AFS training
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml setup.py* ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev,test]"

# Install Unsloth
RUN pip install --no-cache-dir "unsloth[cu121-torch210] @ git+https://github.com/unslothai/unsloth.git"

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY evaluations/ ./evaluations/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["/bin/bash"]
