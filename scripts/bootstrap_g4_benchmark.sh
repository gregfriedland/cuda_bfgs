#!/usr/bin/env bash

set -euo pipefail

cd /opt/batched-bfgs
if [[ ! -x /usr/local/bin/uv ]]; then
    curl -fLsS https://astral.sh/uv/install.sh --output /tmp/install-uv.sh
    UV_INSTALL_DIR=/usr/local/bin sh /tmp/install-uv.sh
fi

UV_LINK_MODE=copy /usr/local/bin/uv sync --locked --no-dev
install -m 0644 scripts/bfgs-benchmark.service \
    /etc/systemd/system/bfgs-benchmark.service
systemctl daemon-reload
systemctl enable --now bfgs-benchmark.service
