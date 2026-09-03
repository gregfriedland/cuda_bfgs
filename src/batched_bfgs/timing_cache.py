"""Durable per-configuration timing cache."""

import math
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from batched_bfgs.models import BaseModelNoExtra
from batched_bfgs.objective import ObjectiveType


class TimingConfiguration(BaseModelNoExtra):
    """Fields that determine whether a timing result can be reused."""

    objective: ObjectiveType
    dimension: int
    implementation: str
    batch_size: int
    device: str
    dtype: str
    tolerance: float
    repeats: int
    torch_version: str | None = None
    compile_mode: str | None = None
    chunk_size: int | None = None

    @property
    def key(self) -> str:
        """Stable, human-readable state key."""
        fields = (
            self.objective,
            str(self.dimension),
            self.implementation,
            str(self.batch_size),
            self.device,
            self.dtype,
            f"{self.tolerance:.17g}",
            str(self.repeats),
        )
        if self.compile_mode is not None:
            fields += (
                self.torch_version or "unknown-torch",
                self.compile_mode,
                str(self.chunk_size),
            )
        return "|".join(fields)


class TimingRecord(BaseModelNoExtra):
    """Validated benchmark timing stored in the cache."""

    implementation: str
    batch_size: int
    median_ms: float
    members_per_second: float
    median_iterations: float
    converged_fraction: float
    repeats: int
    first_run_ms: float | None = None
    estimated_compile_overhead_ms: float | None = None
    compiled_graphs: int | None = None
    steady_state_new_graphs: int | None = None
    graph_breaks: int | None = None
    peak_memory_mb: float | None = None

    @model_validator(mode="after")
    def _validate_metrics(self) -> Self:
        """Validate all recorded timing metrics."""
        values = (
            self.median_ms,
            self.members_per_second,
            self.median_iterations,
            self.converged_fraction,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("timing metrics must be finite")
        if self.median_ms <= 0.0:
            raise ValueError("median_ms must be positive")
        if not -1e-12 <= self.converged_fraction <= 1.0 + 1e-12:
            raise ValueError("converged_fraction is outside tolerance")
        optional_values = (
            self.first_run_ms,
            self.estimated_compile_overhead_ms,
            self.peak_memory_mb,
        )
        if not all(
            value is None or math.isfinite(value) for value in optional_values
        ):
            raise ValueError("optional timing metrics must be finite")
        return self


class TimingCacheEntry(BaseModelNoExtra):
    """One desired, completed, or intentionally skipped configuration."""

    configuration: TimingConfiguration
    timing: TimingRecord | Literal["not_desired"] | None


class TimingCacheState(BaseModelNoExtra):
    """On-disk cache schema."""

    schema_version: Literal[1] = 1
    configurations: dict[str, TimingCacheEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_keys(self) -> Self:
        """Validate configuration keys and timing identities."""
        for key, entry in self.configurations.items():
            if key != entry.configuration.key:
                raise ValueError(
                    "timing cache key does not match configuration"
                )
            if isinstance(entry.timing, TimingRecord):
                if (
                    entry.timing.implementation
                    != entry.configuration.implementation
                    or entry.timing.batch_size != entry.configuration.batch_size
                    or entry.timing.repeats != entry.configuration.repeats
                ):
                    raise ValueError(
                        "timing record does not match configuration"
                    )
        return self


class TimingCache:
    """Persist timing results atomically after every configuration."""

    def __init__(self, path: Path) -> None:
        """Load an existing cache or initialize an empty state."""
        self._path = path
        self._state = (
            TimingCacheState.model_validate_json(path.read_text())
            if path.exists()
            else TimingCacheState()
        )

    def initialize(
        self,
        configurations: list[tuple[TimingConfiguration, bool]],
    ) -> None:
        """Record the complete desired and skipped configuration matrix."""
        entries = dict(self._state.configurations)
        for configuration, desired in configurations:
            existing = entries.get(configuration.key)
            if existing is not None and desired:
                timing = (
                    None
                    if existing.timing == "not_desired"
                    else existing.timing
                )
            else:
                timing = None if desired else "not_desired"
            entries[configuration.key] = TimingCacheEntry(
                configuration=configuration,
                timing=timing,
            )
        self._state = TimingCacheState(configurations=entries)
        self._write()

    def timing(
        self, configuration: TimingConfiguration
    ) -> dict[str, Any] | None:
        """Return a cached desired timing, or ``None`` when it must run."""
        entry = self._state.configurations[configuration.key]
        if entry.timing == "not_desired":
            raise ValueError("not-desired configurations cannot be timed")
        if entry.timing is None:
            return None
        return entry.timing.model_dump(exclude_none=True)

    def record(
        self,
        configuration: TimingConfiguration,
        timing: dict[str, Any],
    ) -> None:
        """Persist one completed timing before the next configuration runs."""
        entries = dict(self._state.configurations)
        existing = entries.get(configuration.key)
        if existing is None or existing.timing == "not_desired":
            raise ValueError("configuration is not pending a timing")
        entries[configuration.key] = TimingCacheEntry(
            configuration=configuration,
            timing=TimingRecord.model_validate(timing),
        )
        self._state = TimingCacheState(configurations=entries)
        self._write()

    def _write(self) -> None:
        """Atomically persist the current cache state."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        temporary.write_text(self._state.model_dump_json(indent=2) + "\n")
        temporary.replace(self._path)
