# Static checks and tests

## Shared static-check command

Run the repository check script from anywhere inside the checkout:

```bash
./scripts/check.sh
```

The script synchronizes `.venv` from `uv.lock`, then runs Ruff linting, Ruff's
format check, and ty type checking against `src` and `tests`.

## Unit tests

Run the unit-test suite separately:

```bash
/opt/homebrew/bin/uv run --locked --no-sync pytest -q
```

The static-check script intentionally does not run pytest.

## Individual checks

Use these commands when working on one class of failure:

```bash
/opt/homebrew/bin/uv run --locked --no-sync ruff check .
/opt/homebrew/bin/uv run --locked --no-sync ruff format --check .
/opt/homebrew/bin/uv run --locked --no-sync ty check src tests
```

## Pre-push hook

Enable the tracked hook once per clone:

```bash
git config core.hooksPath .githooks
```

## GitHub Actions

GitHub Actions runs `./scripts/check.sh` for every push and pull request. View
results on the [Static checks Actions page](https://github.com/gregfriedland/cuda_bfgs/actions/workflows/static-checks.yml)
or list recent runs from the command line:

```bash
gh run list --workflow static-checks.yml --limit 5
```
