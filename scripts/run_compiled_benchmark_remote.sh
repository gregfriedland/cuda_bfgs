#!/usr/bin/env bash
# Purpose: Run the torch.compile chunked benchmark after the standard GPU suite.
# Usage: On the G4 VM, run `sudo /opt/batched-bfgs/scripts/run_compiled_benchmark_remote.sh`.

set -euo pipefail

state_dir=/var/lib/bfgs-benchmark
standard_report="$state_dir/report.json"
compiled_report="$state_dir/compiled-report.json"
combined_report="$state_dir/report-with-compiled.json"
timing_state="$state_dir/timing-state.json"

test -s "$state_dir/DONE.json"
test -s "$standard_report"
if [[ -s "$state_dir/COMPILED_DONE.json" && -s "$combined_report" ]]; then
    exit 0
fi
rm -f "$state_dir/COMPILED_FAILED.json"

commit="$(</opt/batched-bfgs/SOURCE_COMMIT)"
printf '{"commit":"%s","pid":%d,"started_at":"%s"}\n' \
    "$commit" "$$" "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
    >"$state_dir/.COMPILED_RUNNING.json.tmp"
mv "$state_dir/.COMPILED_RUNNING.json.tmp" \
    "$state_dir/COMPILED_RUNNING.json"

record_failure() {
    local exit_code=$?
    printf '{"exit_code":%d,"failed_at":"%s"}\n' \
        "$exit_code" "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
        >"$state_dir/.COMPILED_FAILED.json.tmp"
    mv "$state_dir/.COMPILED_FAILED.json.tmp" \
        "$state_dir/COMPILED_FAILED.json"
    rm -f "$state_dir/COMPILED_RUNNING.json"
    exit "$exit_code"
}
trap record_failure ERR

benchmark_cases=(
    "extended_rosenbrock|16"
)
printf '{\n  "benchmarks": [\n' >"$state_dir/.compiled-report.json.tmp"
separator=""
for benchmark_case in "${benchmark_cases[@]}"; do
    IFS='|' read -r objective dimension <<<"$benchmark_case"
    case_report="$state_dir/.compiled-${objective}-${dimension}d.json.tmp"
    PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
        .venv/bin/python -m batched_bfgs.compiled_benchmark \
        --batch-sizes 64 256 4096 65536 \
        --repeats 5 \
        --objective "$objective" \
        --dimension "$dimension" \
        --state-file "$timing_state" \
        --device cuda >"$case_report"
    printf '%s' "$separator" >>"$state_dir/.compiled-report.json.tmp"
    cat "$case_report" >>"$state_dir/.compiled-report.json.tmp"
    rm "$case_report"
    separator=$',\n'
done
printf '\n  ]\n}\n' >>"$state_dir/.compiled-report.json.tmp"
.venv/bin/python -m json.tool "$state_dir/.compiled-report.json.tmp" >/dev/null
mv "$state_dir/.compiled-report.json.tmp" "$compiled_report"

printf '{\n  "standard": ' >"$state_dir/.report-with-compiled.json.tmp"
cat "$standard_report" >>"$state_dir/.report-with-compiled.json.tmp"
printf ',\n  "compiled_chunked": ' >>"$state_dir/.report-with-compiled.json.tmp"
cat "$compiled_report" >>"$state_dir/.report-with-compiled.json.tmp"
printf '}\n' >>"$state_dir/.report-with-compiled.json.tmp"
.venv/bin/python -m json.tool \
    "$state_dir/.report-with-compiled.json.tmp" >/dev/null
mv "$state_dir/.report-with-compiled.json.tmp" "$combined_report"

printf '{"finished_at":"%s","report":"%s"}\n' \
    "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$combined_report" \
    >"$state_dir/.COMPILED_DONE.json.tmp"
mv "$state_dir/.COMPILED_DONE.json.tmp" "$state_dir/COMPILED_DONE.json"
rm -f "$state_dir/COMPILED_RUNNING.json"
