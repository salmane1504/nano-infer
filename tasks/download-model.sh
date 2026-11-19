#!/usr/bin/env bash
# Download a Hugging Face model into a local directory.
# Usage: tasks/download-model.sh [REPO_ID] [LOCAL_DIR]
set -euo pipefail

cd "$(dirname "$0")/.."
repo="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
local_dir="${2:-model_files/qwen2.5_0.5B-instruct}"

uv run hf download "$repo" --local-dir "$local_dir"
