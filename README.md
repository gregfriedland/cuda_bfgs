# Batched strong-Wolfe BFGS

This repository compares three implementations of complete BFGS optimization
for independent 2D Rosenbrock problems:

1. A Python loop over batch members.
2. Masked batched PyTorch tensor operations.
3. One complete optimization per CUDA thread in a custom extension.

The CUDA kernel duplicates the Rosenbrock value and gradient because device
code cannot call a Python objective callback.

## Local setup and CPU checks

```bash
/opt/homebrew/bin/uv sync --all-groups
.venv/bin/python -m pytest -q tests/test_cpu_equivalence.py
```

## Local and remote CI checks

### Local checks

The shared check script creates or updates `.venv` from `uv.lock`, then runs
Ruff linting, Ruff's formatting check, and ty type checking:

```bash
./scripts/check.sh
```

Enable the tracked pre-push hook once per clone:

```bash
git config core.hooksPath .githooks
```

### Remote checks

GitHub Actions runs the same `./scripts/check.sh` command for every push and
pull request. View the workflow and its results on the
[Static checks Actions page](https://github.com/gregfriedland/cuda_bfgs/actions/workflows/static-checks.yml).

To inspect recent runs from the command line:

```bash
gh run list --workflow static-checks.yml --limit 5
```

The local machine does not need CUDA for the CPU equivalence test. The complete
three-way benchmark requires a CUDA development environment:

```bash
.venv/bin/python -m batched_bfgs.benchmark \
  --batch_sizes 64 256 4096 65536 \
  --repeats 5
```

JIT extension compilation and the warmup call are excluded from measured CUDA
event timings. The Python-loop baseline is capped at batch size 256 because it
intentionally measures per-member Python and kernel-launch overhead.

## GCP Spot VM

Create a full-GPU `g4-standard-48` Spot VM. The launcher defaults to project
`<gcp-project>`, region `us-east5`, and the authenticated gcloud account
`<user>`. It discovers a zone in the requested region where
the machine type is advertised; pass `--zone` to choose one explicitly.

```bash
./scripts/create_g4_spot_vm.sh \
  --project "<gcp-project>" \
  --region us-east5
```

The VM uses a 50 GB Hyperdisk Balanced boot disk and Ubuntu 24.04. A startup
script installs the current production NVIDIA driver. Spot preemption stops
the VM, and the boot disk remains available with auto-delete disabled. The
preserved disk continues to incur storage charges.

Preview all create arguments without provisioning a VM:

```bash
./scripts/create_g4_spot_vm.sh --zone us-east5-a --dry-run
```

The launcher requires an existing gcloud credential for the selected account
and access to the selected project. Authenticate with
`gcloud auth login "<user>"` if the credential is absent.
