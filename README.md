# Batched strong-Wolfe BFGS

This repository compares three implementations of complete BFGS optimization
for independent 2D Rosenbrock problems:

1. A Python loop over batch members.
2. Masked batched PyTorch tensor operations.
3. One complete optimization per CUDA thread in a custom extension.

The numerical contract is pinned in [ALGORITHM.md](ALGORITHM.md). The CUDA
kernel duplicates the Rosenbrock value and gradient because device code cannot
call a Python objective callback.

## Local setup and CPU checks

```bash
/opt/homebrew/bin/uv sync --all-groups
.venv/bin/python -m pytest -q tests/test_cpu_equivalence.py
```

## Static checks

The shared check script creates or updates `.venv` from `uv.lock`, then runs
Ruff linting, Ruff's formatting check, and ty type checking:

```bash
./scripts/check.sh
```

Enable the tracked pre-push hook once per clone:

```bash
git config core.hooksPath .githooks
```

The same script runs in GitHub Actions for pushes and pull requests.

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

## Flyte submission

The standalone task uses public `flyte`, `torch`, and Kubernetes APIs. It does
not import from the Rezo monorepo. Its pod template pins one full
`g4-standard-48` node in `atlas-east5`.

```bash
PYTHONPATH=src .venv/bin/python -m batched_bfgs.submit \
  --run_dir "$PWD/run_260902_g4_benchmark"
```

The submission command records each run before returning and reuses a recorded
active run. After diagnosing a failed run, pass `--resume` to submit a linked
replacement within the three-attempt retry budget.

The standalone bounded waiter accepts the execution ID returned by submission:

```bash
PYTHONPATH=src .venv/bin/python -m batched_bfgs.wait RUN_ID \
  --status_file "$PWD/run_260902_g4_benchmark/status/RUN_ID.json"
```
