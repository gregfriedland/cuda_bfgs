"""Correctness and timing harness for the three BFGS implementations."""

import argparse
import json
import statistics
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch

from batched_bfgs.cuda import CudaBfgs
from batched_bfgs.loop import LoopBfgs
from batched_bfgs.models import BfgsConfig, OptimizationResult
from batched_bfgs.objective import RosenbrockObjective
from batched_bfgs.vectorized import VectorizedBfgs


class Implementation(Enum):
    """Available BFGS execution strategies."""

    PYTHON_LOOP = "python_loop"
    PYTORCH_BATCHED = "pytorch_batched"
    CUDA_KERNEL = "cuda_kernel"


class BenchmarkRunner:
    """Validate and time the three implementations on fixed inputs."""

    def __init__(
        self,
        batch_sizes: list[int],
        repeats: int,
        loop_max_batch: int = 256,
    ) -> None:
        """Initialize the benchmark.

        Args:
            batch_sizes: Batch sizes timed for vectorized and CUDA paths.
            repeats: Number of measured repetitions per case.
            loop_max_batch: Largest batch timed through the Python loop.

        """
        if not batch_sizes or min(batch_sizes) <= 0:
            raise ValueError("batch_sizes must contain positive values")
        if repeats < 3:
            raise ValueError("repeats must be at least three")
        self._batch_sizes = batch_sizes
        self._repeats = repeats
        self._loop_max_batch = loop_max_batch
        self._config = BfgsConfig()

    def run(self, device: torch.device) -> dict[str, Any]:
        """Run correctness checks followed by timing.

        Args:
            device: Device used by every implementation.

        Returns:
            JSON-serializable benchmark report.

        """
        if device.type != "cuda":
            raise ValueError("the three-way benchmark requires CUDA")
        cuda = CudaBfgs(self._config)
        cuda.compile()
        correctness = self._check_correctness(device, cuda)
        records: list[dict[str, Any]] = []
        for batch_size in self._batch_sizes:
            starts = self.make_starts(batch_size, device, torch.float32)
            implementations = self._implementations(cuda, batch_size)
            for name, implementation in implementations.items():
                records.append(
                    self._time_implementation(name, implementation, starts),
                )
        properties = torch.cuda.get_device_properties(device)
        return {
            "device": properties.name,
            "compute_capability": list(
                torch.cuda.get_device_capability(device)
            ),
            "torch_version": torch.__version__,
            "correctness": correctness,
            "timings": records,
        }

    def _check_correctness(
        self,
        device: torch.device,
        cuda: CudaBfgs,
    ) -> dict[str, Any]:
        starts = self.make_starts(32, device, torch.float64)
        initial, _gradient = RosenbrockObjective.value_and_gradient(starts)
        loop = LoopBfgs(self._config).run(starts)
        vectorized = VectorizedBfgs(self._config).run(starts)
        kernel = cuda.run(starts)
        for name, result in (
            (Implementation.PYTHON_LOOP.value, loop),
            (Implementation.PYTORCH_BATCHED.value, vectorized),
            (Implementation.CUDA_KERNEL.value, kernel),
        ):
            self._assert_result(name, result, initial)
        torch.testing.assert_close(vectorized.x, loop.x, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(kernel.x, loop.x, atol=1e-6, rtol=1e-6)
        return {
            "batch_size": 32,
            "dtype": "float64",
            "maximum_position_error": float(
                torch.maximum(
                    (vectorized.x - loop.x).abs().amax(),
                    (kernel.x - loop.x).abs().amax(),
                ),
            ),
            "all_converged": True,
            "all_steps_satisfied_strong_wolfe": True,
        }

    def _implementations(
        self,
        cuda: CudaBfgs,
        batch_size: int,
    ) -> dict[str, Callable[[torch.Tensor], OptimizationResult]]:
        implementations = {
            Implementation.PYTORCH_BATCHED.value: VectorizedBfgs(
                self._config
            ).run,
            Implementation.CUDA_KERNEL.value: cuda.run,
        }
        if batch_size <= self._loop_max_batch:
            implementations[Implementation.PYTHON_LOOP.value] = LoopBfgs(
                self._config
            ).run
        return implementations

    def _time_implementation(
        self,
        name: str,
        implementation: Callable[[torch.Tensor], OptimizationResult],
        starts: torch.Tensor,
    ) -> dict[str, Any]:
        implementation(starts)
        torch.cuda.synchronize(starts.device)
        elapsed_ms: list[float] = []
        result: OptimizationResult | None = None
        for _repeat in range(self._repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = implementation(starts)
            end.record()
            end.synchronize()
            elapsed_ms.append(begin.elapsed_time(end))
        if result is None:
            raise RuntimeError("benchmark did not execute")
        median_ms = statistics.median(elapsed_ms)
        converged_fraction = float(result.converged.float().mean())
        if not -1e-12 <= converged_fraction <= 1.0 + 1e-12:
            raise RuntimeError("converged fraction is outside [0, 1]")
        return {
            "implementation": name,
            "batch_size": starts.shape[0],
            "median_ms": median_ms,
            "members_per_second": starts.shape[0] * 1000.0 / median_ms,
            "median_iterations": float(result.iterations.float().median()),
            "converged_fraction": converged_fraction,
            "repeats": self._repeats,
        }

    def _assert_result(
        self,
        name: str,
        result: OptimizationResult,
        initial_objective: torch.Tensor,
    ) -> None:
        if not bool(torch.isfinite(result.x).all()):
            raise AssertionError(f"{name} produced non-finite coordinates")
        if not bool(torch.isfinite(result.objective).all()):
            raise AssertionError(f"{name} produced non-finite objectives")
        if not bool((result.objective <= initial_objective + 1e-12).all()):
            raise AssertionError(f"{name} increased an objective")
        if not bool(result.wolfe_satisfied.all()):
            raise AssertionError(f"{name} failed a strong-Wolfe line search")
        if not bool(result.converged.all()):
            maximum_gradient = float(result.gradient.abs().amax())
            raise AssertionError(
                f"{name} did not converge; max gradient={maximum_gradient}",
            )
        target = torch.ones_like(result.x)
        torch.testing.assert_close(result.x, target, atol=1e-6, rtol=1e-6)

    @staticmethod
    def make_starts(
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create deterministic, non-identical Rosenbrock starting points.

        Args:
            batch_size: Number of independent optimization problems.
            device: Destination device.
            dtype: Floating-point dtype.

        Returns:
            Starting coordinates with shape ``[batch_size, 2]``.

        """
        index = torch.arange(batch_size, dtype=dtype, device=device)
        x0 = -1.2 + 0.25 * torch.sin(index * 0.37)
        x1 = 1.0 + 0.25 * torch.cos(index * 0.53)
        return torch.stack((x0, x1), dim=-1)


class BenchmarkCli:
    """Parse command-line arguments and run the benchmark."""

    @staticmethod
    def run() -> None:
        """Execute the command-line benchmark."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--batch_sizes",
            nargs="+",
            type=int,
            default=[64, 256, 4096, 65536],
        )
        parser.add_argument("--repeats", type=int, default=5)
        arguments = parser.parse_args()
        report = BenchmarkRunner(
            batch_sizes=arguments.batch_sizes,
            repeats=arguments.repeats,
        ).run(torch.device("cuda"))
        print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    """Run the benchmark CLI."""
    BenchmarkCli.run()


if __name__ == "__main__":
    main()
