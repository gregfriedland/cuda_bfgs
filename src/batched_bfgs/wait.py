"""Bounded standalone waiter for one Flyte benchmark run."""

import argparse
import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import flyte
from flyte.remote import Action, Run

from batched_bfgs.flyte_constants import (
    FLYTE_DOMAIN,
    FLYTE_ENDPOINT,
    FLYTE_PROJECT,
    TERMINAL_PHASES,
)


class FlyteWaiter:
    """Poll Flyte authoritatively and persist bounded status snapshots."""

    def __init__(
        self,
        run_id: str,
        status_file: Path,
        poll_seconds: float,
        max_wait_seconds: float,
    ) -> None:
        """Initialize the waiter.

        Args:
            run_id: Flyte execution identifier.
            status_file: Durable latest-status JSON path.
            poll_seconds: Delay between authoritative queries.
            max_wait_seconds: Maximum wall-clock wait.

        """
        self._run_id = run_id
        self._status_file = status_file.resolve()
        self._poll_seconds = poll_seconds
        self._max_wait_seconds = max_wait_seconds

    def run(self) -> int:
        """Wait for terminal state or the bounded deadline.

        Returns:
            Zero on success, one on remote failure, or two on waiter timeout.

        """
        return asyncio.run(self._run_async())

    async def _run_async(self) -> int:
        flyte.init(
            endpoint=FLYTE_ENDPOINT,
            project=FLYTE_PROJECT,
            domain=FLYTE_DOMAIN,
        )
        started = time.monotonic()
        while True:
            execution = Run.get(name=self._run_id)
            phase = str(execution.phase)
            snapshot = await self._snapshot(execution, phase)
            self._write_snapshot(snapshot)
            if phase in TERMINAL_PHASES:
                return 0 if phase == "SUCCEEDED" else 1
            if time.monotonic() - started >= self._max_wait_seconds:
                snapshot["waiter_timed_out"] = True
                self._write_snapshot(snapshot)
                return 2
            await asyncio.sleep(self._poll_seconds)

    async def _snapshot(self, execution: Run, phase: str) -> dict[str, Any]:
        actions = []
        action_iterator = cast(
            AsyncIterator[Action],
            Action.listall(for_run_name=self._run_id),
        )
        async for action in action_iterator:
            actions.append(
                {
                    "name": str(getattr(action, "name", "")),
                    "task_name": str(getattr(action, "task_name", "")),
                    "phase": str(action.phase),
                },
            )
        return {
            "run_id": self._run_id,
            "phase": phase,
            "url": execution.url,
            "checked_at": datetime.now(UTC).isoformat(),
            "actions": actions,
            "error_metadata_location": (
                f"gs://uc-us-east5-rezotx/metadata/v2/{FLYTE_PROJECT}/"
                f"{FLYTE_DOMAIN}/{self._run_id}/"
            ),
        }

    def _write_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._status_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._status_file.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        )
        temporary.replace(self._status_file)


class WaitCli:
    """Parse waiter arguments."""

    @staticmethod
    def run() -> int:
        """Execute the waiter command."""
        parser = argparse.ArgumentParser()
        parser.add_argument("run_id")
        parser.add_argument("--status_file", type=Path, required=True)
        parser.add_argument("--poll_seconds", type=float, default=30.0)
        parser.add_argument("--max_wait_seconds", type=float, default=7200.0)
        arguments = parser.parse_args()
        return FlyteWaiter(
            run_id=arguments.run_id,
            status_file=arguments.status_file,
            poll_seconds=arguments.poll_seconds,
            max_wait_seconds=arguments.max_wait_seconds,
        ).run()


def main() -> None:
    """Run the bounded Flyte waiter."""
    raise SystemExit(WaitCli.run())


if __name__ == "__main__":
    main()
