"""Nsight workload for the fused CUDA BFGS kernel."""

import argparse
import json
from typing import Any

import torch

from batched_bfgs.benchmark import ObjectiveFactory, ObjectiveName
from batched_bfgs.cuda import CudaBfgs
from batched_bfgs.models import BfgsConfig


class CudaProfileWorkload:
    """Run one warmed high-batch CUDA kernel inside an NVTX range."""

    def __init__(
        self,
        objective_name: ObjectiveName,
        dimension: int,
        batch_size: int,
    ) -> None:
        """Store the fixed profiling configuration."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._objective_name = objective_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._config = BfgsConfig(tolerance=1e-4, max_iterations=300)

    def run(self) -> dict[str, Any]:
        """Compile, warm up, and measure one fused CUDA optimizer launch."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiling requires an available GPU")
        device = torch.device("cuda")
        starts = ObjectiveFactory.make_starts(
            self._objective_name,
            self._batch_size,
            self._dimension,
            device,
            torch.float32,
        )
        optimizer = CudaBfgs(self._config, self._objective_name.value)
        optimizer.compile(verbose=True)
        optimizer.run(starts)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        label = (
            f"cuda_bfgs:{self._objective_name.value}:"
            f"{self._dimension}d:batch={self._batch_size}"
        )
        torch.cuda.cudart().cudaProfilerStart()
        begin.record()
        with torch.cuda.nvtx.range(label):
            result = optimizer.run(starts)
        end.record()
        end.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        converged_fraction = float(result.converged.float().mean())
        if not -1e-12 <= converged_fraction <= 1.0 + 1e-12:
            raise RuntimeError("converged fraction is outside [0, 1]")
        if not bool(result.wolfe_satisfied.all()):
            raise RuntimeError("profiled run failed a strong-Wolfe search")
        return {
            "objective": self._objective_name.value,
            "dimension": self._dimension,
            "batch_size": self._batch_size,
            "dtype": "float32",
            "tolerance": self._config.tolerance,
            "elapsed_ms": begin.elapsed_time(end),
            "converged_fraction": converged_fraction,
            "median_iterations": float(result.iterations.float().median()),
            "median_line_search_evaluations": float(
                result.line_search_evaluations.float().median()
            ),
            "peak_memory_mb": torch.cuda.max_memory_allocated(device)
            / (1024.0 * 1024.0),
            "gpu": torch.cuda.get_device_properties(device).name,
            "torch_version": torch.__version__,
        }


class CudaProfileCli:
    """Parse arguments and run the CUDA profiling workload."""

    @staticmethod
    def run() -> None:
        """Execute one profiler-visible high-batch workload."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--objective",
            type=ObjectiveName,
            choices=list(ObjectiveName),
            required=True,
        )
        parser.add_argument("--dimension", type=int, required=True)
        parser.add_argument("--batch-size", type=int, default=65536)
        arguments = parser.parse_args()
        report = CudaProfileWorkload(
            objective_name=arguments.objective,
            dimension=arguments.dimension,
            batch_size=arguments.batch_size,
        ).run()
        print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    """Run the CUDA profile CLI."""
    CudaProfileCli.run()


if __name__ == "__main__":
    main()
