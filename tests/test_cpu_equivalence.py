"""CPU checks for the Python-loop and vectorized BFGS implementations."""

import torch

from batched_bfgs.benchmark import BenchmarkRunner
from batched_bfgs.loop import LoopBfgs
from batched_bfgs.models import BfgsConfig
from batched_bfgs.objective import RosenbrockObjective
from batched_bfgs.vectorized import VectorizedBfgs


class TestCpuEquivalence:
    """Check numerical properties shared by CPU-capable implementations."""

    def test_loop_and_vectorized_converge(self) -> None:
        """Both implementations converge to the Rosenbrock minimum."""
        device = torch.device("cpu")
        starts = BenchmarkRunner.make_starts(16, device, torch.float64)
        initial, _gradient = RosenbrockObjective.value_and_gradient(starts)
        config = BfgsConfig(tolerance=1e-7)
        loop = LoopBfgs(config).run(starts)
        vectorized = VectorizedBfgs(config).run(starts)
        assert bool(loop.converged.all())
        assert bool(vectorized.converged.all())
        assert bool(loop.wolfe_satisfied.all())
        assert bool(vectorized.wolfe_satisfied.all())
        assert bool((loop.objective <= initial + 1e-12).all())
        assert bool((vectorized.objective <= initial + 1e-12).all())
        torch.testing.assert_close(
            loop.x, torch.ones_like(loop.x), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(vectorized.x, loop.x, atol=1e-6, rtol=1e-6)

    def test_rejects_wrong_shape(self) -> None:
        """The public APIs reject non-two-dimensional problems."""
        starts = torch.zeros((4, 3), dtype=torch.float64)
        config = BfgsConfig()
        for implementation in (LoopBfgs(config), VectorizedBfgs(config)):
            try:
                implementation.run(starts)
            except ValueError as error:
                assert "[batch, 2]" in str(error)
            else:
                raise AssertionError("wrong input shape was accepted")
