"""Idempotent Flyte submission entry point for the BFGS benchmark."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import flyte
from flyte.remote import Run

from batched_bfgs.flyte_app import run_benchmark
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
            state["phase"] = str(current.phase)
            state["last_checked_at"] = self._now()
            self._write_state(state)
            if str(current.phase) not in TERMINAL_PHASES:
                return state
            if str(current.phase) == "SUCCEEDED":
                return state
            if not self._allow_retry:
                raise RuntimeError(
                    "recorded run failed; pass --resume after diagnosis",
                )
        return self._submit(state)

    def _submit(self, state: dict[str, Any]) -> dict[str, Any]:
        attempts = int(state.get("attempts", 0)) + 1
        if attempts > 3:
            raise RuntimeError("campaign retry budget of three is exhausted")
        execution = flyte.with_runcontext(
            copy_style="all",
            interruptible=False,
        ).run(
            run_benchmark,
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
