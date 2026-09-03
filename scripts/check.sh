#!/usr/bin/env bash
# Purpose: Sync dependencies and run the repository's static quality checks.
# Usage: Run from anywhere inside the repository: ./scripts/check.sh

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"

uv sync --locked --all-groups
uv run --locked --no-sync ruff check .
uv run --locked --no-sync ruff format --check .
uv run --locked --no-sync ty check src tests
