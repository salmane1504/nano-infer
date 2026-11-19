#!/usr/bin/env bash
# Install runtime and development dependencies from uv.lock.
set -euo pipefail

cd "$(dirname "$0")/.."
uv sync --locked --extra dev
