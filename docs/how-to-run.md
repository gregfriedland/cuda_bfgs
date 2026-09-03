# How to run

## Local setup

Install the locked runtime and development dependencies:

```bash
/opt/homebrew/bin/uv sync --locked --all-groups
```

## Standard benchmark

Run the Python loop and eager PyTorch variants on CPU:

```bash
/opt/homebrew/bin/uv run --locked --no-sync batched-bfgs benchmark \
  --objective extended_rosenbrock \
  --dimension 16 \
  --device cpu
```

Use `--device cuda` to move the tensor implementations to the GPU and include
the fused CUDA kernel for a supported dimension. The default batch sizes are
64, 256, 4096, and 65536; the Python loop is omitted above its configured
benchmark limit.

Persist completed timing cases so an interrupted benchmark can reuse them:

```bash
/opt/homebrew/bin/uv run --locked --no-sync batched-bfgs benchmark \
  --objective extended_rosenbrock \
  --dimension 16 \
  --device cuda \
  --state-file timing-state.json
```

## Compiled benchmark

Use the same `benchmark` command with `--compiled`. A state file is required:

```bash
/opt/homebrew/bin/uv run --locked --no-sync batched-bfgs benchmark \
  --compiled \
  --batch-sizes 64 256 4096 65536 \
  --objective extended_rosenbrock \
  --dimension 16 \
  --device cuda \
  --state-file timing-state.json
```

The first run includes TorchInductor compilation. Reports separate that first
run from warmed steady-state timing and assert that timed repeats do not create
new graphs or graph breaks.

## CUDA profiling workload

Run one warmed fused-kernel workload with an NVTX range:

```bash
/opt/homebrew/bin/uv run --locked --no-sync batched-bfgs profile-cuda \
  --objective extended_rosenbrock \
  --dimension 16 \
  --batch-size 65536
```

For full Nsight Systems capture and compiler resource-usage output, use the
remote profiling stage described below.

## GCP Spot VM infrastructure

The repository includes a lifecycle manager and startup/shutdown scripts for a
persistent `g4-standard-48` Spot VM. The manager supports exactly three actions:
`create`, `stop`, and `start`.

Authenticate the gcloud account and ensure it can access the target project:

```bash
gcloud auth login ACCOUNT
```

The manager always passes both `--account` and `--project` to gcloud. It defaults
to region `us-east5`, discovers a compatible zone when one is not specified,
and uses the instance name `bfgs-g4-spot` by default.

### Create the VM

Preview the exact provisioning command first:

```bash
.venv/bin/python -m scripts.manage_g4_spot_vm create \
  --project PROJECT_ID \
  --account ACCOUNT \
  --zone us-east5-a \
  --dry-run
```

Create the instance:

```bash
.venv/bin/python -m scripts.manage_g4_spot_vm create \
  --project PROJECT_ID \
  --account ACCOUNT \
  --region us-east5
```

The create command configures Ubuntu 24.04, a 50 GB Hyperdisk Balanced boot
disk, Spot provisioning, and a stop-on-preemption policy. Automatic boot-disk
deletion is disabled, so the source checkout, compiled extensions, and benchmark
artifacts survive a stop or preemption. The preserved disk continues to incur
storage charges.

### Stop and start the VM

Stop the instance without deleting its disk:

```bash
.venv/bin/python -m scripts.manage_g4_spot_vm stop \
  --project PROJECT_ID \
  --account ACCOUNT
```

Start the preserved instance:

```bash
.venv/bin/python -m scripts.manage_g4_spot_vm start \
  --project PROJECT_ID \
  --account ACCOUNT
```

When no zone is supplied for `stop` or `start`, the manager locates the named
instance within the requested region and verifies that its current state allows
the operation.

### Bootstrap behavior

The startup script installs the NVIDIA driver, CUDA toolkit, project source, and
locked Python environment. Benchmark stages are started explicitly with the
commands below.

The shutdown script records preemption state under `/var/lib/bfgs-benchmark`.
Benchmark scripts also write atomic running, failure, and completion markers in
that directory so interrupted work can be diagnosed and resumed.

### Run the remote stages

Run the standard implementation matrix first:

```bash
sudo /opt/batched-bfgs/scripts/run_benchmark_remote.sh
```

Run the fixed-shape compiled chunked benchmark after the standard report exists:

```bash
sudo /opt/batched-bfgs/scripts/run_compiled_benchmark_remote.sh
```

Run the Nsight Systems CUDA profile after compiled validation completes:

```bash
sudo /opt/batched-bfgs/scripts/run_cuda_profile_remote.sh
```

The current remote scripts run 16D extended Rosenbrock. Standard and compiled
timings share a durable timing cache, while the profiling stage writes Nsight,
CUDA API, kernel-summary, resource-usage, timing, convergence, and peak-memory
artifacts under `/var/lib/bfgs-benchmark`.
