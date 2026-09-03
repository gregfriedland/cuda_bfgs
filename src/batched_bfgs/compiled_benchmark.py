"""GPU benchmark for the compiled chunked PyTorch BFGS implementation."""

import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch._dynamo.utils import counters

from batched_bfgs.benchmark import BenchmarkRunner
from batched_bfgs.chunked import (
    ChunkedVectorizedBfgs,
    CompiledChunkedVectorizedBfgs,
)
from batched_bfgs.models import BfgsConfig, OptimizationResult
from batched_bfgs.objective import ObjectiveType
from batched_bfgs.timing_cache import TimingCache, TimingConfiguration


class CompiledBenchmarkRunner:
    """Validate and time fixed-shape Inductor-compiled BFGS runs."""

    implementation = "pytorch (compiled chunked)"
    compile_mode = "inductor-fullgraph-static"

    def __init__(
        self,
        batch_sizes: list[int],
        repeats: int,
        objective_name: ObjectiveType,
        dimension: int,
        chunk_size: int = 16,
    ) -> None:
        """Initialize one compiled benchmark case."""
        # Validate the requested fixed-shape timing matrix.
        if not batch_sizes or min(batch_sizes) <= 0:
            raise ValueError("batch_sizes must contain positive values")
        if repeats < 3:
            raise ValueError("repeats must be at least three")

        # Store benchmark settings and construct the analytic objective.
        self._batch_sizes = batch_sizes
        self._repeats = repeats
        self._objective_name = objective_name
        self._dimension = dimension
        self._chunk_size = chunk_size
        self._config = BfgsConfig(tolerance=1e-4, max_iterations=300)
        self._objective = objective_name.create(dimension)

    def run(self, device: torch.device, state_path: Path) -> dict[str, Any]:
        """Run GPU parity checks and steady-state timings."""
        # Require an available CUDA device for Inductor GPU compilation.
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError(
                "compiled benchmark requires an available CUDA GPU"
            )

        # Validate parity and initialize the persistent timing matrix.
        correctness = self._check_correctness(device)
        cache = TimingCache(state_path)
        configurations = [
            self._configuration(device, batch_size)
            for batch_size in self._batch_sizes
        ]
        cache.initialize(
            [(configuration, True) for configuration in configurations]
        )

        # Reuse cached cases and time each remaining configuration.
        timings = []
        for configuration in configurations:
            cached = cache.timing(configuration)
            timing = cached or self._time_configuration(
                device,
                configuration.batch_size,
            )
            if cached is None:
                cache.record(configuration, timing)
            timings.append(timing)

        # Return correctness, timing, and compilation metadata.
        return {
            "device": torch.cuda.get_device_properties(device).name,
            "torch_version": torch.__version__,
            "objective": self._objective_name.value,
            "dimension": self._dimension,
            "timing_dtype": "float32",
            "timing_tolerance": self._config.tolerance,
            "compile_mode": self.compile_mode,
            "chunk_size": self._chunk_size,
            "correctness": correctness,
            "timings": timings,
        }

    def _configuration(
        self,
        device: torch.device,
        batch_size: int,
    ) -> TimingConfiguration:
        """Build the cache identity for one compiled timing."""
        # Include every input that can alter compilation or runtime.
        return TimingConfiguration(
            objective=self._objective_name,
            dimension=self._dimension,
            implementation=self.implementation,
            batch_size=batch_size,
            device=torch.cuda.get_device_properties(device).name,
            dtype="float32",
            tolerance=self._config.tolerance,
            repeats=self._repeats,
            torch_version=torch.__version__,
            compile_mode=self.compile_mode,
            chunk_size=self._chunk_size,
        )

    def _check_correctness(self, device: torch.device) -> dict[str, Any]:
        """Validate compiled results against the eager chunked optimizer."""
        # Build one deterministic batch for eager-versus-compiled parity.
        starts = self._objective.make_starts(
            16,
            device,
            torch.float32,
        )
        eager = ChunkedVectorizedBfgs(
            self._config,
            self._objective,
            self._chunk_size,
        ).run(starts)
        compiled = CompiledChunkedVectorizedBfgs(
            self._config,
            self._objective,
            self._chunk_size,
        ).run(starts)

        # Compare all public numerical and status outputs.
        parity_tolerance = 5.0 * self._config.tolerance
        torch.testing.assert_close(
            compiled.x,
            eager.x,
            atol=parity_tolerance,
            rtol=parity_tolerance,
        )
        torch.testing.assert_close(
            compiled.objective,
            eager.objective,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            compiled.gradient,
            eager.gradient,
            atol=parity_tolerance,
            rtol=parity_tolerance,
        )
        torch.testing.assert_close(compiled.converged, eager.converged)
        torch.testing.assert_close(
            compiled.wolfe_satisfied,
            eager.wolfe_satisfied,
        )

        # Validate the compiled result against the analytic minimizer.
        target = self._objective.minimizer(starts)
        BenchmarkRunner._assert_result(
            self.implementation,
            compiled,
            self._objective.value_and_gradient(starts)[0],
            target,
            parity_tolerance
            if self._objective_name is ObjectiveType.EXTENDED_ROSENBROCK
            else 1e-2,
        )

        # Return detailed parity and convergence diagnostics.
        return {
            "batch_size": 16,
            "dtype": "float32",
            "maximum_position_difference_from_eager": float(
                (compiled.x - eager.x).abs().amax()
            ),
            "maximum_iteration_difference_from_eager": int(
                (compiled.iterations - eager.iterations).abs().amax()
            ),
            "maximum_position_error": float((compiled.x - target).abs().amax()),
            "maximum_gradient": float(compiled.gradient.abs().amax()),
            "parity_tolerance": parity_tolerance,
            "all_converged": True,
            "all_steps_satisfied_strong_wolfe": True,
        }

    def _time_configuration(
        self,
        device: torch.device,
        batch_size: int,
    ) -> dict[str, Any]:
        """Measure compilation overhead and steady-state GPU timing."""
        # Reset compiler and memory counters for this fixed shape.
        torch.compiler.reset()
        counters.clear()
        torch.cuda.reset_peak_memory_stats(device)

        # Construct inputs and a fresh compiled optimizer instance.
        starts = self._objective.make_starts(
            batch_size,
            device,
            torch.float32,
        )
        optimizer = CompiledChunkedVectorizedBfgs(
            self._config,
            self._objective,
            self._chunk_size,
        )

        # Time the first call, which includes compilation overhead.
        torch.cuda.synchronize(device)
        first_begin = time.perf_counter()
        optimizer.run(starts)
        torch.cuda.synchronize(device)
        first_run_ms = (time.perf_counter() - first_begin) * 1000.0

        # Warm the compiled graph before recording steady-state counters.
        optimizer.run(starts)
        optimizer.run(starts)
        torch.cuda.synchronize(device)
        graphs_before = self._counter("stats", "unique_graphs")
        breaks_before = self._graph_breaks()

        # Measure repeated execution with CUDA device events.
        elapsed_ms = []
        result: OptimizationResult | None = None
        for _repeat in range(self._repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = optimizer.run(starts)
            end.record()
            end.synchronize()
            elapsed_ms.append(begin.elapsed_time(end))

        # Reject recompilation or graph breaks inside the timed region.
        new_graphs = self._counter("stats", "unique_graphs") - graphs_before
        graph_breaks = self._graph_breaks() - breaks_before
        if new_graphs != 0 or graph_breaks != 0:
            raise RuntimeError(
                "compiled timing triggered new graphs or graph breaks: "
                f"new_graphs={new_graphs}, graph_breaks={graph_breaks}"
            )

        # Validate execution and bounded convergence statistics.
        if result is None:
            raise RuntimeError("compiled benchmark did not execute")
        median_ms = statistics.median(elapsed_ms)
        converged_fraction = float(result.converged.float().mean())
        if not -1e-12 <= converged_fraction <= 1.0 + 1e-12:
            raise RuntimeError("converged fraction is outside [0, 1]")

        # Return compilation cost, runtime, throughput, and memory metrics.
        return {
            "implementation": self.implementation,
            "batch_size": batch_size,
            "median_ms": median_ms,
            "members_per_second": batch_size * 1000.0 / median_ms,
            "median_iterations": float(result.iterations.float().median()),
            "converged_fraction": converged_fraction,
            "repeats": self._repeats,
            "first_run_ms": first_run_ms,
            "estimated_compile_overhead_ms": first_run_ms - median_ms,
            "compiled_graphs": self._counter("stats", "unique_graphs"),
            "steady_state_new_graphs": new_graphs,
            "graph_breaks": graph_breaks,
            "peak_memory_mb": torch.cuda.max_memory_allocated(device)
            / (1024.0 * 1024.0),
        }

    @staticmethod
    def _counter(group: str, name: str) -> int:
        """Read one TorchDynamo counter."""
        return int(counters[group][name])

    @staticmethod
    def _graph_breaks() -> int:
        """Return the total recorded TorchDynamo graph breaks."""
        return sum(int(value) for value in counters["graph_break"].values())
