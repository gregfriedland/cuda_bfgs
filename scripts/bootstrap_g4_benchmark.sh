#!/usr/bin/env bash
# Purpose: Install the project environment for manually launched benchmarks.
# Usage: Run as root on the G4 VM: sudo ./scripts/bootstrap_g4_benchmark.sh

set -euo pipefail

cd /opt/batched-bfgs
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes python3.12-dev
if [[ ! -x /usr/local/bin/uv ]]; then
    curl -fLsS https://astral.sh/uv/install.sh --output /tmp/install-uv.sh
    UV_INSTALL_DIR=/usr/local/bin sh /tmp/install-uv.sh
fi

UV_LINK_MODE=copy /usr/local/bin/uv sync --locked --no-dev
