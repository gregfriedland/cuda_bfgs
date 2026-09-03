"""Nsight workload for the fused CUDA BFGS kernel."""

from typing import Any

import torch

from batched_bfgs.cuda import CudaBfgs
from batched_bfgs.models import BfgsConfig
from batched_bfgs.objective import ObjectiveType


class CudaProfileWorkload:
    """Run one warmed high-batch CUDA kernel inside an NVTX range."""

    def __init__(
        self,
        objective_name: ObjectiveType,
        dimension: int,
        batch_size: int,
    ) -> None:
        """Store the fixed profiling configuration."""
        # Validate and retain the requested profiling case.
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._objective_name = objective_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._config = BfgsConfig(tolerance=1e-4, max_iterations=300)
        self._objective = objective_name.create(dimension)

    def run(self) -> dict[str, Any]:
        """Compile, warm up, and measure one fused CUDA optimizer launch."""
        # Require a CUDA device before allocating profile inputs.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiling requires an available GPU")

        # Build the selected starts and compile the fused extension.
        device = torch.device("cuda")
        starts = self._objective.make_starts(
            self._batch_size,
            device,
            torch.float32,
        )
        optimizer = CudaBfgs(self._config, self._objective_name)
        optimizer.compile(verbose=True)

        # Warm the kernel and reset peak-memory accounting.
        optimizer.run(starts)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Give the profiled launch a descriptive NVTX range label.
        label = (
            f"cuda_bfgs:{self._objective_name.value}:"
            f"{self._dimension}d:batch={self._batch_size}"
        )

        # Capture exactly one fused launch inside the profiler interval.
        torch.cuda.cudart().cudaProfilerStart()
        begin.record()
        with torch.cuda.nvtx.range(label):
            result = optimizer.run(starts)
        end.record()
        end.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        # Validate correctness indicators before reporting performance.
        converged_fraction = float(result.converged.float().mean())
        if not -1e-12 <= converged_fraction <= 1.0 + 1e-12:
            raise RuntimeError("converged fraction is outside [0, 1]")
        wolfe_satisfied_fraction = float(result.wolfe_satisfied.float().mean())
        if not -1e-12 <= wolfe_satisfied_fraction <= 1.0 + 1e-12:
            raise RuntimeError("strong-Wolfe fraction is outside [0, 1]")

        # Return machine-readable timing, convergence, and device metadata.
        return {
            "objective": self._objective_name.value,
            "dimension": self._dimension,
            "batch_size": self._batch_size,
            "dtype": "float32",
            "tolerance": self._config.tolerance,
            "elapsed_ms": begin.elapsed_time(end),
            "converged_fraction": converged_fraction,
            "wolfe_satisfied_fraction": wolfe_satisfied_fraction,
            "median_iterations": float(result.iterations.float().median()),
            "median_line_search_evaluations": float(
                result.line_search_evaluations.float().median()
            ),
            "peak_memory_mb": torch.cuda.max_memory_allocated(device)
            / (1024.0 * 1024.0),
            "gpu": torch.cuda.get_device_properties(device).name,
            "torch_version": torch.__version__,
        }
