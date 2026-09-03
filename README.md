# Batched strong-Wolfe BFGS

This repository implements full BFGS with a strong-Wolfe line search using:

1. A Python loop over batch members.
2. Masked batched PyTorch tensor operations.
3. One fused fixed-dimensional optimization per CUDA thread.

The only benchmark objectives are extended Rosenbrock, defined as independent
two-variable blocks, and extended Powell singular, defined as independent
four-variable blocks. The Python and PyTorch implementations accept arbitrary
valid dimensions. The fused CUDA extension supports 2D Rosenbrock plus 16D
extended Rosenbrock and extended Powell through compiled objective kernels.

## Local setup

```bash
/opt/homebrew/bin/uv sync --locked --all-groups
```

Run an extended Rosenbrock benchmark in 16 dimensions:

```bash
/opt/homebrew/bin/uv run --locked --no-sync bfgs-benchmark \
  --objective extended_rosenbrock \
  --dimension 16 \
  --device cpu
```

Switch to the extended Powell singular strong-Wolfe stress case:

```bash
/opt/homebrew/bin/uv run --locked --no-sync bfgs-benchmark \
  --objective extended_powell \
  --dimension 16 \
  --device cpu
```

Use `--device cuda` with either 16D objective to include its fused CUDA kernel.

## Local and remote CI checks

### Local checks

The shared check script updates `.venv` from `uv.lock`, then runs Ruff linting,
Ruff's formatting check, and ty type checking:

```bash
./scripts/check.sh
```

Run the unit tests separately:

```bash
/opt/homebrew/bin/uv run --locked --no-sync pytest -q
```

Enable the tracked pre-push hook once per clone:

```bash
git config core.hooksPath .githooks
```

### Remote checks

GitHub Actions runs the same `./scripts/check.sh` command for every push and
pull request. View results on the
[Static checks Actions page](https://github.com/gregfriedland/cuda_bfgs/actions/workflows/static-checks.yml).

```bash
gh run list --workflow static-checks.yml --limit 5
```

## GCP Spot VM

Create a full-GPU `g4-standard-48` Spot VM. The manager requires a project and
authenticated gcloud account, and defaults to region `us-east5`.

```bash
.venv/bin/python -m scripts.manage_g4_spot_vm create \
  --project PROJECT_ID \
  --account ACCOUNT \
  --region us-east5
```

The VM uses a 50 GB Hyperdisk Balanced boot disk and Ubuntu 24.04. Spot
preemption stops the VM, and the boot disk remains available with auto-delete
disabled. The preserved disk continues to incur storage charges.

Preview create arguments without provisioning a VM:

```bash
.venv/bin/python -m scripts.manage_g4_spot_vm create \
  --project PROJECT_ID \
  --account ACCOUNT \
  --zone us-east5-a \
  --dry-run
```

Stop the VM while preserving its boot disk, then start it again:

```bash
.venv/bin/python -m scripts.manage_g4_spot_vm stop --project PROJECT_ID --account ACCOUNT
.venv/bin/python -m scripts.manage_g4_spot_vm start --project PROJECT_ID --account ACCOUNT
```

VM bootstrap installs but disables `bfgs-benchmark.service`. Run the benchmark
manually on the VM with:

```bash
sudo /opt/batched-bfgs/scripts/run_benchmark_remote.sh
```
