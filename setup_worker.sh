#!/usr/bin/env bash
# Set up a worker venv on a remote machine.
# Detects the platform and installs the appropriate torch build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    uv venv .venv
fi

# Detect platform
UNAME="$(uname -s)"
ARCH="$(uname -m)"

if [ "$UNAME" = "Darwin" ]; then
    echo "Detected macOS — installing MPS torch..."
    uv pip install --python .venv torch pt-kmeans safetensors huggingface_hub pydantic pyyaml transformers accelerate
elif command -v nvidia-smi &>/dev/null; then
    echo "Detected NVIDIA GPU — installing CUDA torch..."
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "NVIDIA driver: $DRIVER_VERSION"
    uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu132
    uv pip install --python .venv pt-kmeans safetensors huggingface_hub pydantic pyyaml transformers accelerate
else
    echo "No NVIDIA GPU detected — installing CPU torch..."
    uv pip install --python .venv torch pt-kmeans safetensors huggingface_hub pydantic pyyaml transformers accelerate
fi

echo "Worker venv ready: .venv/bin/python"
echo "Test: .venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
