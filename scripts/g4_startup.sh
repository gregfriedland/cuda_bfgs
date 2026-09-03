#!/usr/bin/env bash

set -euo pipefail

exec > >(tee -a /var/log/g4-setup.log) 2>&1

ready_file=/var/lib/bfgs-g4-ready
if [[ -f "$ready_file" ]]; then
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes build-essential ca-certificates curl python3

installer_dir=/opt/google/cuda-installer
install -d -m 0755 "$installer_dir"
curl -fSsL \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
    --output "$installer_dir/cuda_installer.pyz"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    python3 "$installer_dir/cuda_installer.pyz" install_driver \
        --installation-mode=binary \
        --installation-branch=prod
fi

if [[ ! -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    curl -fSsL \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        --output "$installer_dir/cuda-keyring.deb"
    dpkg -i "$installer_dir/cuda-keyring.deb"
    apt-get update
    apt-get install --yes cuda-toolkit-12-8
fi

nvidia-smi
/usr/local/cuda-12.8/bin/nvcc --version
touch "$ready_file"
rm -rf /var/lib/apt/lists/*
