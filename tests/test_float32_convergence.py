"""Regression checks for float32 convergence tolerances."""

import torch

from batched_bfgs.loop import LoopBfgs
from batched_bfgs.models import BfgsConfig
from batched_bfgs.objective import ExtendedRosenbrockObjective
from batched_bfgs.vectorized import VectorizedBfgs


class TestFloat32Convergence:
    """Check convergence above the float32 Rosenbrock gradient floor."""

    def test_extended_rosenbrock(self) -> None:
        """Both CPU implementations converge with tolerance 1e-4."""
        device = torch.device("cpu")
        objective = ExtendedRosenbrockObjective(dimension=16)
        starts = objective.make_starts(64, device, torch.float32)
        config = BfgsConfig(tolerance=1e-4, max_iterations=300)
        loop = LoopBfgs(config, objective).run(starts)
        vectorized = VectorizedBfgs(config, objective).run(starts)

        assert bool(loop.converged.all())
        assert bool(vectorized.converged.all())
        assert bool(loop.wolfe_satisfied.all())
        assert bool(vectorized.wolfe_satisfied.all())
