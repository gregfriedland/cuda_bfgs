#!/usr/bin/env bash
# Purpose: Record an interrupted benchmark before a G4 Spot VM stops.
# Usage: Installed as the GCE shutdown-script metadata hook by the VM manager.

set -euo pipefail

state_dir=/var/lib/bfgs-benchmark
if [[ -f "$state_dir/RUNNING.json" && ! -f "$state_dir/DONE.json" ]]; then
    install -d -m 0755 "$state_dir"
    temporary="$state_dir/.PREEMPTED.json.tmp"
    printf '{"boot_id":"%s","stopped_at":"%s"}\n' \
        "$(</proc/sys/kernel/random/boot_id)" \
        "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" >"$temporary"
    mv "$temporary" "$state_dir/PREEMPTED.json"
    sync
fi
