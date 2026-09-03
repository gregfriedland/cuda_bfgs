"""Idempotent Flyte submission entry point for the BFGS benchmark."""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import flyte
from flyte.remote import Run

from batched_bfgs.flyte_app import test
from batched_bfgs.flyte_constants import (
    FLYTE_DOMAIN,
    FLYTE_ENDPOINT,
    FLYTE_PROJECT,
    TERMINAL_PHASES,
)


class FlyteCampaign:
    """Submit or resume one durable benchmark campaign."""

    def __init__(
        self,
        run_dir: Path,
        batch_sizes: list[int],
        repeats: int,
        allow_retry: bool,
    ) -> None:
        """Initialize the campaign.

        Args:
            run_dir: Durable state and output directory.
            batch_sizes: Remote benchmark batch sizes.
            repeats: Timing repetitions.
            allow_retry: Whether a failed recorded run may be replaced.

        """
        self._run_dir = run_dir.resolve()
        self._state_path = self._run_dir / "campaign_state.json"
        self._batch_sizes = batch_sizes
        self._repeats = repeats
        self._allow_retry = allow_retry
        self._root = Path(__file__).resolve().parents[2]

    def run(self) -> dict[str, Any]:
        """Reuse an active run or submit the next permitted attempt.

        Returns:
            Updated campaign state.

        """
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "status").mkdir(exist_ok=True)
        (self._run_dir / "logs").mkdir(exist_ok=True)
        self._initialize_flyte()
        state = self._load_state()
        active_run_id = state.get("active_run_id")
        if active_run_id is not None:
            current = Run.get(name=active_run_id)
            phase = str(current.phase).rsplit(".", maxsplit=1)[-1]
            state["phase"] = phase
            state["last_checked_at"] = self._now()
            self._write_state(state)
            if phase not in TERMINAL_PHASES:
                return state
            if phase == "SUCCEEDED":
                return self._collect(current, state)
            if not self._allow_retry:
                raise RuntimeError(
                    "recorded run failed; pass --resume after diagnosis",
                )
        return self._submit(state)

    def _submit(self, state: dict[str, Any]) -> dict[str, Any]:
        self._assert_submission_ready(state)
        attempts = int(state.get("attempts", 0)) + 1
        if attempts > 3:
            raise RuntimeError("campaign retry budget of three is exhausted")
        execution = flyte.with_runcontext(
            copy_style="all",
            interruptible=False,
        ).run(
            test,
            batch_sizes=self._batch_sizes,
            repeats=self._repeats,
        )
        previous_id = state.get("active_run_id")
        run_record = {
            "run_id": execution.name,
            "url": execution.url,
            "submitted_at": self._now(),
            "replaces": previous_id,
        }
        run_ids = list(state.get("runs", []))
        run_ids.append(run_record)
        state.update(
            {
                "phase": "SUBMITTED",
                "attempts": attempts,
                "active_run_id": execution.name,
                "active_run_url": execution.url,
                "runs": run_ids,
                "flyte_project": FLYTE_PROJECT,
                "flyte_domain": FLYTE_DOMAIN,
            },
        )
        self._write_state(state)
        return state

    def _assert_submission_ready(self, state: dict[str, Any]) -> None:
        requirements = self._read_json("capability_requirements.json")
        profile = self._read_json("heartbeat_profile.json")
        readiness = self._read_json("submission_readiness.json")
        required_class = requirements["required_monitor_class"]
        verified_class = profile["verified_monitor_class"]
        if required_class != "self_healing" or verified_class != required_class:
            raise RuntimeError("a verified self-healing monitor is required")
        if profile["automation_id"] != requirements["automation_id"]:
            raise RuntimeError("heartbeat identity does not match requirements")
        if (
            requirements["origin_thread_id"]
            != requirements["delivery_thread_id"]
        ):
            raise RuntimeError("origin and delivery thread IDs must match")
        capabilities = profile["capabilities"]
        if not capabilities or not all(
            item["verified"] for item in capabilities.values()
        ):
            raise RuntimeError("not every heartbeat capability was verified")
        commit = self._git_commit()
        if readiness["commit"] != commit:
            raise RuntimeError("submission readiness does not match HEAD")
        submitter = self._root / "src/batched_bfgs/submit.py"
        digest = hashlib.sha256(submitter.read_bytes()).hexdigest()
        if readiness["submitter_sha256"] != digest:
            raise RuntimeError(
                "submission entry point changed after validation"
            )
        state["required_monitor_class"] = required_class
        state["verified_monitor_class"] = verified_class
        state["monitor_verified_at"] = profile["verified_at"]

    def _collect(self, execution: Run, state: dict[str, Any]) -> dict[str, Any]:
        outputs = execution.outputs()
        if len(outputs) != 1:
            raise RuntimeError("benchmark task must return exactly one output")
        encoded_report = outputs[0]
        if not isinstance(encoded_report, str):
            raise TypeError("benchmark output must be a JSON string")
        report = json.loads(encoded_report)
        self._validate_report(report)
        report_path = self._run_dir / "benchmark_report.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(report_path)
        state["phase"] = "COMPLETE"
        state["terminal_verified_at"] = self._now()
        state["validated_artifacts"] = [str(report_path)]
        self._write_state(state)
        return state

    @staticmethod
    def _validate_report(report: dict[str, Any]) -> None:
        correctness = report["correctness"]
        if not correctness["all_converged"]:
            raise RuntimeError("correctness batch did not converge")
        if not correctness["all_steps_satisfied_strong_wolfe"]:
            raise RuntimeError("a correctness step violated strong Wolfe")
        timings = report["timings"]
        if not timings:
            raise RuntimeError("benchmark report has no timings")
        for timing in timings:
            elapsed = float(timing["median_ms"])
            fraction = float(timing["converged_fraction"])
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise RuntimeError("timing is not finite and positive")
            if not -1e-12 <= fraction <= 1.0 + 1e-12:
                raise RuntimeError("converged fraction is outside [0, 1]")

    def _read_json(self, name: str) -> dict[str, Any]:
        path = self._run_dir / name
        if not path.is_file():
            raise RuntimeError(f"required file is missing: {path}")
        return json.loads(path.read_text())

    def _git_commit(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self._root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _initialize_flyte(self) -> None:
        flyte.init(
            endpoint=FLYTE_ENDPOINT,
            project=FLYTE_PROJECT,
            domain=FLYTE_DOMAIN,
            root_dir=self._root,
        )

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {
                "phase": "READY",
                "attempts": 0,
                "runs": [],
                "validated_artifacts": [],
                "required_monitor_class": "self_healing",
                "verified_monitor_class": "pending",
                "origin_thread_id": "01a02020-8997-71a2-a7a9-8232782e7922",
                "delivery_thread_id": "01a02020-8997-71a2-a7a9-8232782e7922",
            }
        return json.loads(self._state_path.read_text())

    def _write_state(self, state: dict[str, Any]) -> None:
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary.replace(self._state_path)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()


class SubmitCli:
    """Parse campaign arguments and print its durable state."""

    @staticmethod
    def run() -> None:
        """Execute the submission command."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--run_dir", type=Path, required=True)
        parser.add_argument(
            "--batch_sizes",
            nargs="+",
            type=int,
            default=[64, 256, 4096, 65536],
        )
        parser.add_argument("--repeats", type=int, default=5)
        parser.add_argument("--resume", action="store_true")
        arguments = parser.parse_args()
        state = FlyteCampaign(
            run_dir=arguments.run_dir,
            batch_sizes=arguments.batch_sizes,
            repeats=arguments.repeats,
            allow_retry=arguments.resume,
        ).run()
        print(json.dumps(state, indent=2, sort_keys=True))


def main() -> None:
    """Run the submission CLI."""
    SubmitCli.run()


if __name__ == "__main__":
    main()
