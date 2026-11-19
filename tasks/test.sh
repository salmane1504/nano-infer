#!/usr/bin/env bash
# Run the test suite. Extra arguments are passed to pytest, e.g. tasks/test.sh tests/test_sampler.py
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --locked --extra dev python -m pytest -q "$@"
