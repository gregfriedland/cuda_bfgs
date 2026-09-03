#!/usr/bin/env bash

set -euo pipefail

exec > >(tee -a /var/log/g4-driver-install.log) 2>&1

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes ca-certificates curl python3

installer_dir=/opt/google/cuda-installer
install -d -m 0755 "$installer_dir"
curl -fSsL \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
    --output "$installer_dir/cuda_installer.pyz"

python3 "$installer_dir/cuda_installer.pyz" install_driver \
    --installation-mode=binary \
    --installation-branch=prod

nvidia-smi
