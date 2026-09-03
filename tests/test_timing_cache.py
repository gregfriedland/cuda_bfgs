"""Tests for durable benchmark timing state."""

import json
from pathlib import Path

from batched_bfgs.timing_cache import TimingCache, TimingConfiguration


class TestTimingCache:
    """Check desired, skipped, and completed timing entries."""

    def test_persists_timing_and_not_desired_sentinel(
        self,
        tmp_path: Path,
    ) -> None:
        """Every planned configuration receives an explicit state entry."""
        state_path = tmp_path / "timing-state.json"
        desired = self._configuration("cuda_kernel", 65536)
        skipped = self._configuration("python_loop", 65536)
        cache = TimingCache(state_path)
        cache.initialize([(desired, True), (skipped, False)])

        assert cache.timing(desired) is None
        state = json.loads(state_path.read_text())
        assert state["configurations"][skipped.key]["timing"] == "not_desired"

        timing = {
            "implementation": "cuda_kernel",
            "batch_size": 65536,
            "median_ms": 4.0,
            "members_per_second": 16_384_000.0,
            "median_iterations": 100.0,
            "converged_fraction": 1.0,
            "repeats": 5,
        }
        cache.record(desired, timing)

        assert TimingCache(state_path).timing(desired) == timing
        assert not state_path.with_suffix(".json.tmp").exists()

    def test_compiled_configuration_has_distinct_environment_key(self) -> None:
        """Compiled timings cannot reuse another compiler environment."""
        base = self._configuration("pytorch (compiled chunked)", 256)
        first = base.model_copy(
            update={
                "torch_version": "2.8.0",
                "compile_mode": "inductor-fullgraph-static",
                "chunk_size": 16,
            }
        )
        second = first.model_copy(update={"torch_version": "2.9.0"})

        assert first.key != second.key

    @staticmethod
    def _configuration(
        implementation: str,
        batch_size: int,
    ) -> TimingConfiguration:
        return TimingConfiguration(
            objective="extended_rosenbrock",
            dimension=16,
            implementation=implementation,
            batch_size=batch_size,
            device="test-gpu",
            dtype="float32",
            tolerance=1e-4,
            repeats=5,
        )
