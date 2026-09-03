#!/usr/bin/env bash
# Purpose: Run the GPU benchmark with durable attempt and terminal-state markers.
# Usage: On the G4 VM, run `sudo /opt/batched-bfgs/scripts/run_benchmark_remote.sh`.

set -euo pipefail

state_dir=/var/lib/bfgs-benchmark
install -d -m 0755 "$state_dir"
if [[ -f "$state_dir/DONE.json" && -s "$state_dir/report.json" ]]; then
    exit 0
fi

attempt_file="$state_dir/attempt"
attempt=1
if [[ -f "$attempt_file" ]]; then
    attempt=$(( $(<"$attempt_file") + 1 ))
fi
printf '%d\n' "$attempt" >"$attempt_file"

boot_id="$(</proc/sys/kernel/random/boot_id)"
commit="$(</opt/batched-bfgs/SOURCE_COMMIT)"
log_path="$state_dir/benchmark-attempt-${attempt}.log"
failed_tmp="$state_dir/.FAILED.json.tmp"

record_failure() {
    local exit_code=$?
    printf '{"attempt":%d,"boot_id":"%s","commit":"%s","exit_code":%d,"failed_at":"%s","log":"%s"}\n' \
        "$attempt" "$boot_id" "$commit" "$exit_code" \
        "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$log_path" >"$failed_tmp"
    mv "$failed_tmp" "$state_dir/FAILED.json"
    rm -f "$state_dir/RUNNING.json"
    exit "$exit_code"
}
trap record_failure ERR

rm -f "$state_dir/FAILED.json" "$state_dir/PREEMPTED.json"
running_tmp="$state_dir/.RUNNING.json.tmp"
printf '{"attempt":%d,"boot_id":"%s","commit":"%s","pid":%d,"started_at":"%s"}\n' \
    "$attempt" "$boot_id" "$commit" "$$" \
    "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" >"$running_tmp"
mv "$running_tmp" "$state_dir/RUNNING.json"

report_tmp="$state_dir/.report.json.tmp"
timing_state="$state_dir/timing-state.json"
benchmark_cases=(
    "extended_rosenbrock|16"
)
printf '{\n  "benchmarks": [\n' >"$report_tmp"
separator=""
for benchmark_case in "${benchmark_cases[@]}"; do
    IFS='|' read -r objective dimension <<<"$benchmark_case"
    case_report="$state_dir/.${objective}-${dimension}d.json.tmp"
    PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
        .venv/bin/python -m batched_bfgs benchmark \
        --batch-sizes 64 256 4096 65536 \
        --repeats 5 \
        --objective "$objective" \
        --dimension "$dimension" \
        --state-file "$timing_state" \
        --device cuda >"$case_report" 2>>"$log_path"
    printf '%s' "$separator" >>"$report_tmp"
    cat "$case_report" >>"$report_tmp"
    rm "$case_report"
    separator=$',\n'
done
printf '\n  ]\n}\n' >>"$report_tmp"
.venv/bin/python -m json.tool "$report_tmp" >/dev/null
mv "$report_tmp" "$state_dir/report.json"

done_tmp="$state_dir/.DONE.json.tmp"
printf '{"attempt":%d,"boot_id":"%s","commit":"%s","finished_at":"%s","report":"%s"}\n' \
    "$attempt" "$boot_id" "$commit" \
    "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
    "$state_dir/report.json" >"$done_tmp"
mv "$done_tmp" "$state_dir/DONE.json"
rm -f "$state_dir/RUNNING.json"
