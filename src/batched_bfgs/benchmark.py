"""Correctness and timing harness for the BFGS implementations."""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from enum import Enum, StrEnum
from typing import Any

import torch

from batched_bfgs.cuda import CudaBfgs
from batched_bfgs.loop import LoopBfgs
from batched_bfgs.models import BfgsConfig, OptimizationResult
from batched_bfgs.objective import (
    ExtendedPowellSingularObjective,
    ExtendedRosenbrockObjective,
    TensorObjective,
)
from batched_bfgs.vectorized import VectorizedBfgs


class Implementation(Enum):
    """Available BFGS execution strategies."""

    PYTHON_LOOP = "python_loop"
    PYTORCH_BATCHED = "pytorch_batched"
    CUDA_KERNEL = "cuda_kernel"


class ObjectiveName(StrEnum):
    """Available analytic benchmark objectives."""

    EXTENDED_ROSENBROCK = "extended_rosenbrock"
    EXTENDED_POWELL = "extended_powell"


class ObjectiveFactory:
    """Construct objectives, starts, and known minimizers by name."""

    @staticmethod
    def create(name: ObjectiveName) -> TensorObjective:
        """Create the requested analytic objective."""
        if name is ObjectiveName.EXTENDED_ROSENBROCK:
            return ExtendedRosenbrockObjective()
        return ExtendedPowellSingularObjective()

    @staticmethod
    def make_starts(
        name: ObjectiveName,
        batch_size: int,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create deterministic standard starts for an objective."""
        if name is ObjectiveName.EXTENDED_ROSENBROCK:
            return ExtendedRosenbrockObjective.make_starts(
                batch_size,
                dimension,
                device,
                dtype,
            )
        return ExtendedPowellSingularObjective.make_starts(
            batch_size,
            dimension,
            device,
            dtype,
        )

    @staticmethod
    def minimizer(name: ObjectiveName, starts: torch.Tensor) -> torch.Tensor:
        """Return the known global minimizer for an objective."""
        if name is ObjectiveName.EXTENDED_ROSENBROCK:
            return torch.ones_like(starts)
        return torch.zeros_like(starts)


class BenchmarkRunner:
    """Validate and time implementations on one analytic objective."""

    def __init__(
        self,
        batch_sizes: list[int],
        repeats: int,
        objective_name: ObjectiveName = ObjectiveName.EXTENDED_ROSENBROCK,
        dimension: int = 16,
        loop_max_batch: int = 256,
    ) -> None:
        """Initialize the benchmark."""
        if not batch_sizes or min(batch_sizes) <= 0:
            raise ValueError("batch_sizes must contain positive values")
        if repeats < 3:
            raise ValueError("repeats must be at least three")
        self._batch_sizes = batch_sizes
        self._repeats = repeats
        self._objective_name = objective_name
        self._dimension = dimension
        self._loop_max_batch = loop_max_batch
        self._config = BfgsConfig(max_iterations=300)
        self._objective = ObjectiveFactory.create(objective_name)
        ObjectiveFactory.make_starts(
            objective_name,
            1,
            dimension,
            torch.device("cpu"),
            torch.float64,
        )

    def run(self, device: torch.device) -> dict[str, Any]:
        """Run correctness checks followed by timing."""
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        cuda = self._cuda_implementation(device)
        correctness = self._check_correctness(device, cuda)
        records: list[dict[str, Any]] = []
        for batch_size in self._batch_sizes:
            starts = self.make_starts(batch_size, device, torch.float32)
            implementations = self._implementations(cuda, batch_size)
            for name, implementation in implementations.items():
                records.append(
                    self._time_implementation(name, implementation, starts),
                )
        return {
            "device": self._device_name(device),
            "torch_version": torch.__version__,
            "objective": self._objective_name.value,
            "dimension": self._dimension,
            "cuda_kernel_included": cuda is not None,
            "cuda_kernel_constraint": None
            if cuda is not None
            else (
                "the fused CUDA kernels support 2D Rosenbrock and 16D "
                "extended Rosenbrock/Powell"
            ),
            "correctness": correctness,
            "timings": records,
        }

    def make_starts(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create starts for this runner's objective and dimension."""
        return ObjectiveFactory.make_starts(
            self._objective_name,
            batch_size,
            self._dimension,
            device,
            dtype,
        )

    def _cuda_implementation(self, device: torch.device) -> CudaBfgs | None:
        compatible = device.type == "cuda" and (
            self._dimension == 16
            or (
                self._objective_name is ObjectiveName.EXTENDED_ROSENBROCK
                and self._dimension == 2
            )
        )
        if not compatible:
            return None
        cuda = CudaBfgs(self._config, self._objective_name.value)
        cuda.compile()
        return cuda

    def _check_correctness(
        self,
        device: torch.device,
        cuda: CudaBfgs | None,
    ) -> dict[str, Any]:
        starts = self.make_starts(16, device, torch.float64)
        initial, _gradient = self._objective.value_and_gradient(starts)
        loop = LoopBfgs(self._config, self._objective).run(starts)
        vectorized = VectorizedBfgs(self._config, self._objective).run(starts)
        results = [
            (Implementation.PYTHON_LOOP.value, loop),
            (Implementation.PYTORCH_BATCHED.value, vectorized),
        ]
        if cuda is not None:
            results.append((Implementation.CUDA_KERNEL.value, cuda.run(starts)))
        target = ObjectiveFactory.minimizer(self._objective_name, starts)
        target_tolerance = (
            1e-4
            if self._objective_name is ObjectiveName.EXTENDED_ROSENBROCK
            else 1e-2
        )
        for name, result in results:
            self._assert_result(
                name,
                result,
                initial,
                target,
                target_tolerance,
            )
        equivalence_tolerance = target_tolerance
        if (
            self._objective_name is ObjectiveName.EXTENDED_ROSENBROCK
            and self._dimension == 2
        ):
            equivalence_tolerance = 1e-6
        torch.testing.assert_close(
            vectorized.x,
            loop.x,
            atol=equivalence_tolerance,
            rtol=equivalence_tolerance,
        )
        maximum_error = float((vectorized.x - loop.x).abs().amax())
        if cuda is not None:
            kernel = results[-1][1]
            torch.testing.assert_close(
                kernel.x,
                loop.x,
                atol=equivalence_tolerance,
                rtol=equivalence_tolerance,
            )
            maximum_error = max(
                maximum_error,
                float((kernel.x - loop.x).abs().amax()),
            )
        return {
            "batch_size": 16,
            "dtype": "float64",
            "maximum_position_error": maximum_error,
            "all_converged": True,
            "all_steps_satisfied_strong_wolfe": True,
        }

    def _implementations(
        self,
        cuda: CudaBfgs | None,
        batch_size: int,
    ) -> dict[str, Callable[[torch.Tensor], OptimizationResult]]:
        implementations = {
            Implementation.PYTORCH_BATCHED.value: VectorizedBfgs(
                self._config,
                self._objective,
            ).run,
        }
        if cuda is not None:
            implementations[Implementation.CUDA_KERNEL.value] = cuda.run
        if batch_size <= self._loop_max_batch:
            implementations[Implementation.PYTHON_LOOP.value] = LoopBfgs(
                self._config,
                self._objective,
            ).run
        return implementations

    def _time_implementation(
        self,
        name: str,
        implementation: Callable[[torch.Tensor], OptimizationResult],
        starts: torch.Tensor,
    ) -> dict[str, Any]:
        implementation(starts)
        elapsed_ms: list[float] = []
        result: OptimizationResult | None = None
        for _repeat in range(self._repeats):
            if starts.device.type == "cuda":
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                result = implementation(starts)
                end.record()
                end.synchronize()
                elapsed_ms.append(begin.elapsed_time(end))
            else:
                begin_time = time.perf_counter()
                result = implementation(starts)
                elapsed_ms.append((time.perf_counter() - begin_time) * 1000.0)
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

    @staticmethod
    def _assert_result(
        name: str,
        result: OptimizationResult,
        initial_objective: torch.Tensor,
        target: torch.Tensor,
        target_tolerance: float,
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
        torch.testing.assert_close(
            result.x,
            target,
            atol=target_tolerance,
            rtol=target_tolerance,
        )

    @staticmethod
    def _device_name(device: torch.device) -> str:
        if device.type == "cuda":
            return torch.cuda.get_device_properties(device).name
        return str(device)


class BenchmarkCli:
    """Parse command-line arguments and run an analytic benchmark."""

    @staticmethod
    def run() -> None:
        """Execute the command-line benchmark."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--batch-sizes",
            nargs="+",
            type=int,
            default=[64, 256, 4096, 65536],
        )
        parser.add_argument("--repeats", type=int, default=5)
        parser.add_argument(
            "--objective",
            type=ObjectiveName,
            choices=list(ObjectiveName),
            default=ObjectiveName.EXTENDED_ROSENBROCK,
        )
        parser.add_argument("--dimension", type=int, default=16)
        parser.add_argument("--device", default="cuda")
        arguments = parser.parse_args()
        report = BenchmarkRunner(
            batch_sizes=arguments.batch_sizes,
            repeats=arguments.repeats,
            objective_name=arguments.objective,
            dimension=arguments.dimension,
        ).run(torch.device(arguments.device))
        print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    """Run the benchmark CLI."""
    BenchmarkCli.run()


if __name__ == "__main__":
    main()
