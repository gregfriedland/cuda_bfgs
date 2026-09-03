#!/usr/bin/env bash
# Purpose: Profile high-batch fused CUDA BFGS with Nsight Systems and cuobjdump.
# Usage: On the G4 VM, run `sudo /opt/batched-bfgs/scripts/run_cuda_profile_remote.sh`.

set -euo pipefail

state_dir=/var/lib/bfgs-benchmark
profile_dir="$state_dir/profile-cuda-65536"
test -s "$state_dir/COMPILED_DONE.json"
if [[ -s "$state_dir/PROFILE_DONE.json" && -d "$profile_dir" ]]; then
    exit 0
fi
rm -f "$state_dir/PROFILE_FAILED.json"
install -d -m 0755 "$profile_dir"

commit="$(</opt/batched-bfgs/SOURCE_COMMIT)"
printf '{"commit":"%s","pid":%d,"started_at":"%s"}\n' \
    "$commit" "$$" "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
    >"$state_dir/.PROFILE_RUNNING.json.tmp"
mv "$state_dir/.PROFILE_RUNNING.json.tmp" "$state_dir/PROFILE_RUNNING.json"

record_failure() {
    local exit_code=$?
    printf '{"exit_code":%d,"failed_at":"%s"}\n' \
        "$exit_code" "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
        >"$state_dir/.PROFILE_FAILED.json.tmp"
    mv "$state_dir/.PROFILE_FAILED.json.tmp" "$state_dir/PROFILE_FAILED.json"
    rm -f "$state_dir/PROFILE_RUNNING.json"
    exit "$exit_code"
}
trap record_failure ERR

command -v nsys >/dev/null
profile_cases=(
    "extended_rosenbrock|16"
)
for profile_case in "${profile_cases[@]}"; do
    IFS='|' read -r objective dimension <<<"$profile_case"
    prefix="$profile_dir/${objective}-${dimension}d"
    PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
        BFGS_CUDA_RESOURCE_USAGE=1 \
        TORCH_EXTENSIONS_DIR="$profile_dir/torch-extensions" \
        nsys profile \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --sample=none \
        --cpuctxsw=none \
        --output="$prefix" \
        .venv/bin/python -m batched_bfgs profile-cuda \
        --objective "$objective" \
        --dimension "$dimension" \
        --batch-size 65536 \
        >"$prefix-metrics.json" 2>"$prefix-profile.log"
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum \
        "$prefix.nsys-rep" >"$prefix-stats.txt"
done

grep -hE 'ptxas info|Used [0-9]+ registers|stack frame|spill' \
    "$profile_dir"/*-profile.log >"$profile_dir/resource-usage.txt"
test -s "$profile_dir/resource-usage.txt"

printf '{"commit":"%s","finished_at":"%s","profile_dir":"%s"}\n' \
    "$commit" "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$profile_dir" \
    >"$state_dir/.PROFILE_DONE.json.tmp"
mv "$state_dir/.PROFILE_DONE.json.tmp" "$state_dir/PROFILE_DONE.json"
rm -f "$state_dir/PROFILE_RUNNING.json"
