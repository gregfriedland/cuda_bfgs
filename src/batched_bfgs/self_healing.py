"""Fail-closed capability preflight for the Flyte monitor heartbeat."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import secrets
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import flyte
from flyte.remote import Run

from batched_bfgs.models import BaseModelNoExtra


class SelfHealingCheck(BaseModelNoExtra):
    """Probe and attest the heartbeat's self-healing capabilities."""

    requirements: dict[str, Any]
    wake_id: str
    ttl_seconds: int = 120

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _run(
        command: list[str], cwd: Path, timeout_seconds: int
    ) -> dict[str, Any]:
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise RuntimeError(
                f"command exited {result.returncode}: {command!r}; "
                f"output tail={output[-1000:].strip()!r}"
            )
        encoded = output.encode(errors="replace")
        return {
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "output_bytes": len(encoded),
            "output_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    @staticmethod
    def _write_probe(directory: Path, nonce: str, label: str) -> dict[str, Any]:
        if not directory.is_dir():
            raise RuntimeError(f"{label} is not a directory: {directory}")
        path = directory / f".self-healing-{nonce}-{label}.probe"
        try:
            path.write_text(nonce, encoding="utf-8")
            if path.read_text(encoding="utf-8") != nonce:
                raise RuntimeError(f"{label} probe readback mismatch")
        finally:
            path.unlink(missing_ok=True)
        return {"directory": str(directory), "create_read_remove": True}

    def _flyte_query(self) -> dict[str, Any]:
        config = self.requirements["flyte"]
        flyte.init(
            endpoint=config["endpoint"],
            project=config["project"],
            domain=config["domain"],
        )
        run = Run.get(name=config["known_run_id"])
        phase = str(run.phase)
        if not phase:
            raise RuntimeError("known Flyte run returned an empty phase")
        return {"known_run_id": config["known_run_id"], "phase": phase}

    def _credentials(self) -> dict[str, Any]:
        env_names = list(self.requirements.get("credential_env", []))
        file_names = list(self.requirements.get("credential_files", []))
        missing_env = [name for name in env_names if not os.environ.get(name)]
        missing_files = [
            name for name in file_names if not Path(name).is_file()
        ]
        if missing_env or missing_files:
            raise RuntimeError(
                "required credential evidence is missing "
                f"(environment={len(missing_env)}, files={len(missing_files)})"
            )
        return {
            "environment_entries_present": len(env_names),
            "credential_files_present": len(file_names),
            "authenticated_query_required": True,
        }

    def _submission_symbols(self) -> dict[str, Any]:
        symbols = list(self.requirements["submission_symbols"])
        for value in symbols:
            module_name, separator, attribute_name = value.partition(":")
            if not separator:
                raise RuntimeError(f"expected module:attribute: {value}")
            attribute = getattr(
                importlib.import_module(module_name), attribute_name
            )
            if not callable(attribute):
                raise RuntimeError(
                    f"submission symbol is not callable: {value}"
                )
        return {"callable_symbols": symbols}

    def _automation_config(self) -> dict[str, Any]:
        path = Path(self.requirements["automation_config"]).resolve()
        with path.open("rb") as handle:
            config = tomllib.load(handle)
        expected = self.requirements
        if config.get("id") != expected["automation_id"]:
            raise RuntimeError(
                "automation config id does not match requirements"
            )
        if (
            config.get("kind") != "heartbeat"
            or config.get("status") != "ACTIVE"
        ):
            raise RuntimeError("automation is not an active heartbeat")
        if config.get("target_thread_id") != expected["origin_thread_id"]:
            raise RuntimeError("automation target does not match origin thread")
        if not config.get("rrule"):
            raise RuntimeError("automation config has no schedule")
        updated_at = config.get("updated_at")
        if not isinstance(updated_at, int):
            raise RuntimeError("automation config has no numeric updated_at")
        prompt = str(config.get("prompt", "")).encode()
        return {
            "path": str(path),
            "updated_at": updated_at,
            "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        }

    def _git_commit_readiness(
        self, worktree: Path, nonce: str
    ) -> dict[str, Any]:
        probe = worktree / f".self-healing-{nonce}-git.probe"
        before = self._run(
            ["git", "diff", "--cached", "--name-only", "-z"], worktree, 15
        )
        try:
            probe.write_text(nonce, encoding="utf-8")
            self._run(["git", "add", "--", probe.name], worktree, 15)
            dry_run = self._run(
                ["git", "commit", "--dry-run", "--allow-empty"], worktree, 15
            )
        finally:
            subprocess.run(
                ["git", "restore", "--staged", "--", probe.name],
                cwd=worktree,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            probe.unlink(missing_ok=True)
        after = self._run(
            ["git", "diff", "--cached", "--name-only", "-z"], worktree, 15
        )
        if before["output_sha256"] != after["output_sha256"]:
            raise RuntimeError(
                "git index changed during commit-readiness probe"
            )
        return {"dry_run": dry_run, "index_restored": True}

    def _command_check(self, key: str, worktree: Path) -> dict[str, Any]:
        command = self.requirements[key]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise RuntimeError(f"{key} must be a non-empty string array")
        timeout = int(self.requirements.get("command_timeout_seconds", 300))
        return self._run(command, worktree, timeout)

    def _checks(self, nonce: str) -> dict[str, Callable[[], dict[str, Any]]]:
        worktree = Path(self.requirements["worktree"]).resolve()
        run_dir = Path(self.requirements["run_dir"]).resolve()
        source_dir = Path(self.requirements["source_edit_dir"]).resolve()
        return {
            "automation_config_read": self._automation_config,
            "credentials": self._credentials,
            "flyte_query": self._flyte_query,
            "git_commit": lambda: self._git_commit_readiness(worktree, nonce),
            "resume_command": lambda: self._command_check(
                "resume_check_command", worktree
            ),
            "run_directory_write": lambda: self._write_probe(
                run_dir, nonce, "run-dir"
            ),
            "source_edit": lambda: self._write_probe(
                source_dir, nonce, "source-edit"
            ),
            "submission_imports": self._submission_symbols,
            "targeted_tests": lambda: self._command_check(
                "targeted_test_command", worktree
            ),
            "worktree_write": lambda: self._write_probe(
                worktree, nonce, "worktree"
            ),
        }

    def probe(self, pending_path: Path) -> int:
        """Run all capability probes and write the pending attestation."""
        started = time.monotonic()
        nonce = secrets.token_hex(16)
        if self.requirements.get("required_monitor_class") != "self_healing":
            raise RuntimeError("required monitor class must be self_healing")
        if (
            self.requirements["origin_thread_id"]
            != self.requirements["delivery_thread_id"]
        ):
            raise RuntimeError("origin and delivery thread IDs must match")
        checks = self._checks(nonce)
        results = self._execute_checks(checks)
        failures = sorted(
            name
            for name, result in results.items()
            if result["status"] != "pass"
        )
        pending = self._pending_record(
            nonce, results, failures, started, len(checks)
        )
        self._atomic_json(pending_path, pending)
        print(
            json.dumps(
                {
                    "elapsed_ms": pending["probe_elapsed_ms"],
                    "failures": failures,
                    "nonce_marker": (
                        pending["nonce_marker"] if not failures else None
                    ),
                    "pending_path": str(pending_path),
                },
                sort_keys=True,
            )
        )
        return 0 if not failures else 1

    @staticmethod
    def _execute_checks(
        checks: dict[str, Callable[[], dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=len(checks),
            thread_name_prefix="self-healing-capability",
        ) as executor:
            futures = {
                executor.submit(check): name for name, check in checks.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = {
                        "status": "pass",
                        "evidence": future.result(),
                    }
                except Exception as error:
                    results[name] = {
                        "status": "fail",
                        "error": f"{type(error).__name__}: {error}",
                    }
        return results

    def _pending_record(
        self,
        nonce: str,
        results: dict[str, dict[str, Any]],
        failures: list[str],
        started: float,
        check_count: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "required_monitor_class": "self_healing",
            "verified_monitor_class": "pending"
            if not failures
            else "unverified",
            "automation_id": self.requirements["automation_id"],
            "origin_thread_id": self.requirements["origin_thread_id"],
            "delivery_thread_id": self.requirements["delivery_thread_id"],
            "wake_id": self.wake_id,
            "nonce": nonce,
            "nonce_marker": f"SELF_HEALING_PROBE_NONCE={nonce}",
            "probe_started_at": self._now(),
            "probe_elapsed_ms": round((time.monotonic() - started) * 1000),
            "ttl_seconds": self.ttl_seconds,
            "automation_config": str(
                Path(self.requirements["automation_config"]).resolve()
            ),
            "capabilities": dict(sorted(results.items())),
            "probe_execution": {
                "strategy": "thread_pool",
                "check_count": check_count,
                "max_workers": check_count,
            },
            "failures": failures,
            "remote_submission_performed": False,
        }

    @classmethod
    def finalize(cls, pending_path: Path, profile_path: Path) -> int:
        """Attest a same-wake heartbeat update and write the final profile."""
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        failures = cls._finalization_failures(pending, pending_path)
        profile = {
            "schema_version": 1,
            "required_monitor_class": "self_healing",
            "verified_monitor_class": "self_healing"
            if not failures
            else "unverified",
            "automation_id": pending.get("automation_id"),
            "origin_thread_id": pending.get("origin_thread_id"),
            "delivery_thread_id": pending.get("delivery_thread_id"),
            "verified_at": cls._now(),
            "expires_at_epoch": (
                time.time() + int(pending.get("ttl_seconds", 0))
                if not failures
                else None
            ),
            "capabilities": pending.get("capabilities", {}),
            "heartbeat_self_update": {
                "status": "pass" if not failures else "fail",
                "evidence": "same-wake nonce persisted by the product update",
            },
            "failures": failures,
            "remote_submission_performed": False,
        }
        cls._atomic_json(profile_path, profile)
        pending_path.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "failures": failures,
                    "profile_path": str(profile_path),
                    "verified_monitor_class": profile["verified_monitor_class"],
                },
                sort_keys=True,
            )
        )
        return 0 if not failures else 1

    @classmethod
    def _finalization_failures(
        cls, pending: dict[str, Any], pending_path: Path
    ) -> list[str]:
        failures: list[str] = []
        if pending.get("failures"):
            failures.append("one or more capability probes failed")
        if pending.get("verified_monitor_class") != "pending":
            failures.append("profile is not awaiting heartbeat attestation")
        age = time.time() - pending_path.stat().st_mtime
        if age > int(pending.get("ttl_seconds", 0)):
            failures.append("pending profile expired")
        with Path(pending["automation_config"]).open("rb") as handle:
            view = tomllib.load(handle)
        cls._check_updated_heartbeat(pending, view, failures)
        return failures

    @staticmethod
    def _check_updated_heartbeat(
        pending: dict[str, Any], view: dict[str, Any], failures: list[str]
    ) -> None:
        if view.get("id") != pending.get("automation_id"):
            failures.append("automation ID mismatch")
        if view.get("kind") != "heartbeat" or view.get("status") != "ACTIVE":
            failures.append("automation is not an active heartbeat")
        if view.get("target_thread_id") != pending.get("origin_thread_id"):
            failures.append("heartbeat target does not match origin thread")
        if pending.get("origin_thread_id") != pending.get("delivery_thread_id"):
            failures.append("origin and delivery thread IDs do not match")
        marker = str(pending.get("nonce_marker", ""))
        if not marker or marker not in str(view.get("prompt", "")):
            failures.append("fresh nonce is absent from heartbeat prompt")
        prior = pending["capabilities"]["automation_config_read"]["evidence"]
        before = prior.get("updated_at")
        after = view.get("updated_at")
        if not isinstance(before, int) or not isinstance(after, int):
            failures.append("numeric heartbeat timestamps are required")
        elif after <= before:
            failures.append("heartbeat config did not advance after update")

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> Self:
        """Build a checker from parsed probe arguments."""
        requirements = json.loads(
            arguments.requirements.read_text(encoding="utf-8")
        )
        return cls(
            requirements=requirements,
            wake_id=arguments.wake_id,
            ttl_seconds=arguments.ttl_seconds,
        )


class SelfHealingCli:
    """Parse and execute self-healing preflight commands."""

    @staticmethod
    def run() -> int:
        """Execute the requested capability check stage."""
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        probe = commands.add_parser("probe")
        probe.add_argument("--requirements", type=Path, required=True)
        probe.add_argument("--wake-id", required=True)
        probe.add_argument("--pending", type=Path, required=True)
        probe.add_argument("--ttl-seconds", type=int, default=120)
        finalize = commands.add_parser("finalize")
        finalize.add_argument("--pending", type=Path, required=True)
        finalize.add_argument("--profile", type=Path, required=True)
        arguments = parser.parse_args()
        if arguments.command == "probe":
            return SelfHealingCheck.from_arguments(arguments).probe(
                arguments.pending
            )
        return SelfHealingCheck.finalize(arguments.pending, arguments.profile)


if __name__ == "__main__":
    raise SystemExit(SelfHealingCli.run())
