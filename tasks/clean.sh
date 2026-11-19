#!/usr/bin/env bash
# Remove Python bytecode and pytest caches. Leaves .venv and model_files alone.
set -euo pipefail

cd "$(dirname "$0")/.."
find . -path ./.venv -prune -o \( -name __pycache__ -o -name .pytest_cache \) -print -exec rm -rf {} +
