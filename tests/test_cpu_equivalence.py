"""CPU checks for analytic objectives and generic-dimensional BFGS."""

import torch

from batched_bfgs.loop import LoopBfgs
from batched_bfgs.models import BfgsConfig, OptimizationResult
from batched_bfgs.objective import (
    ExtendedPowellSingularObjective,
    ExtendedRosenbrockObjective,
    TensorObjective,
)
from batched_bfgs.vectorized import VectorizedBfgs


class TestObjectiveGradients:
    """Compare each analytic gradient with central finite differences."""

    def test_extended_rosenbrock_gradient(self) -> None:
        """Extended Rosenbrock reports the derivative of every block."""
        objective = ExtendedRosenbrockObjective()
        point = torch.tensor(
            (-1.2, 1.0, 0.4, -0.3, 1.1, 0.8, -0.8, 1.4),
            dtype=torch.float64,
        )
        self._assert_gradient(objective, point)

    def test_extended_powell_gradient(self) -> None:
        """Extended Powell reports the derivative of every block."""
        objective = ExtendedPowellSingularObjective()
        point = torch.tensor(
            (3.0, -1.0, 0.2, 1.0, 1.4, -0.3, 0.7, -0.5),
            dtype=torch.float64,
        )
        self._assert_gradient(objective, point)

    @staticmethod
    def _assert_gradient(
        objective: TensorObjective,
        point: torch.Tensor,
    ) -> None:
        _value, gradient = objective.value_and_gradient(point)
        finite_difference = torch.empty_like(point)
        epsilon = 1e-6
        for index in range(point.shape[0]):
            offset = torch.zeros_like(point)
            offset[index] = epsilon
            upper, _gradient = objective.value_and_gradient(point + offset)
            lower, _gradient = objective.value_and_gradient(point - offset)
            finite_difference[index] = (upper - lower) / (2.0 * epsilon)
        torch.testing.assert_close(
            gradient,
            finite_difference,
            atol=1e-6,
            rtol=1e-7,
        )


class TestCpuEquivalence:
    """Check numerical properties shared by CPU-capable implementations."""

    def test_loop_and_vectorized_extended_rosenbrock(self) -> None:
        """Both implementations converge in sixteen dimensions."""
        device = torch.device("cpu")
        objective = ExtendedRosenbrockObjective()
        starts = objective.make_starts(16, 16, device, torch.float64)
        initial, _gradient = objective.value_and_gradient(starts)
        config = BfgsConfig(tolerance=1e-7, max_iterations=200)
        loop = LoopBfgs(config, objective).run(starts)
        vectorized = VectorizedBfgs(config, objective).run(starts)
        self._assert_equivalent(
            loop,
            vectorized,
            initial,
            torch.ones_like(starts),
            position_tolerance=1e-4,
            equivalence_tolerance=1e-4,
        )

    def test_loop_and_vectorized_extended_powell(self) -> None:
        """Both implementations solve the singular strong-Wolfe stress case."""
        device = torch.device("cpu")
        objective = ExtendedPowellSingularObjective()
        starts = objective.make_starts(4, 16, device, torch.float64)
        initial, _gradient = objective.value_and_gradient(starts)
        config = BfgsConfig(tolerance=1e-6, max_iterations=300)
        loop = LoopBfgs(config, objective).run(starts)
        vectorized = VectorizedBfgs(config, objective).run(starts)
        self._assert_equivalent(
            loop,
            vectorized,
            initial,
            torch.zeros_like(starts),
            position_tolerance=1e-2,
            equivalence_tolerance=1e-2,
        )

    def test_rejects_empty_dimension(self) -> None:
        """The public APIs reject empty batches and empty dimensions."""
        config = BfgsConfig()
        for starts in (
            torch.zeros((0, 4), dtype=torch.float64),
            torch.zeros((4, 0), dtype=torch.float64),
        ):
            for implementation in (LoopBfgs(config), VectorizedBfgs(config)):
                try:
                    implementation.run(starts)
                except ValueError as error:
                    assert "[batch, dimension]" in str(error)
                else:
                    raise AssertionError("empty input shape was accepted")

    @staticmethod
    def _assert_equivalent(
        loop: OptimizationResult,
        vectorized: OptimizationResult,
        initial: torch.Tensor,
        target: torch.Tensor,
        position_tolerance: float = 1e-6,
        equivalence_tolerance: float = 1e-6,
    ) -> None:
        assert bool(loop.converged.all())
        assert bool(vectorized.converged.all())
        assert bool(loop.wolfe_satisfied.all())
        assert bool(vectorized.wolfe_satisfied.all())
        assert bool((loop.objective <= initial + 1e-12).all())
        assert bool((vectorized.objective <= initial + 1e-12).all())
        torch.testing.assert_close(
            loop.x,
            target,
            atol=position_tolerance,
            rtol=position_tolerance,
        )
        torch.testing.assert_close(
            vectorized.x,
            loop.x,
            atol=equivalence_tolerance,
            rtol=equivalence_tolerance,
        )
