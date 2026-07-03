#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] TARGET

Download artifacts (models, eval datasets) to local cache.

Targets:
    model MODEL_ID       Download a HuggingFace model (default: Qwen/Qwen3.6-35B-A3B)
    dataset DATASET_ID   Download a HuggingFace dataset (default: allenai/c4)

Options:
    --eval-samples N     Number of C4 validation samples to cache (default: 1000)
    --help               Show this help message

Examples:
    $(basename "$0") model Qwen/Qwen3.6-35B-A3B
    $(basename "$0") dataset allenai/c4
    $(basename "$0") --eval-samples 500 dataset allenai/c4
EOF
}

MODEL_ID="Qwen/Qwen3.6-35B-A3B"
DATASET_ID="allenai/c4"
EVAL_SAMPLES=1000

download_model() {
    local model_id="$1"
    echo "Downloading model: $model_id"
    uv run hf download "$model_id"
    echo "Model cached: $model_id"
}

download_dataset() {
    local dataset_id="$1"
    local split="${2:-validation}"
    echo "Pre-caching dataset: $dataset_id (split=$split, samples=$EVAL_SAMPLES)"
    uv run python -c "
from datasets import load_dataset
ds = load_dataset('$dataset_id', 'en', split='$split', streaming=True)
texts = []
for i, sample in enumerate(ds):
    if i >= $EVAL_SAMPLES:
        break
    texts.append(sample['text'])
print(f'Cached {len(texts)} samples ({sum(len(t) for t in texts)} chars)')
"
    echo "Dataset cached: $dataset_id"
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-samples)
            EVAL_SAMPLES="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        model)
            download_model "$2"
            shift 2
            ;;
        dataset)
            download_dataset "$2"
            shift 2
            ;;
        *)
            echo "Unknown command: $1" >&2
            usage
            exit 1
            ;;
    esac
done
