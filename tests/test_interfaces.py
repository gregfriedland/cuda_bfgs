"""Contracts shared by objectives and BFGS implementations."""

import inspect
import subprocess
import sys
from typing import cast

import pytest
import torch
from pydantic import ValidationError

from batched_bfgs.__main__ import Cli
from batched_bfgs.base import Bfgs
from batched_bfgs.chunked import (
    ChunkedVectorizedBfgs,
    CompiledChunkedVectorizedBfgs,
)
from batched_bfgs.cuda import CudaBfgs
from batched_bfgs.loop import LoopBfgs
from batched_bfgs.models import BfgsConfig
from batched_bfgs.objective import (
    ExtendedRosenbrockObjective,
    ObjectiveType,
    TensorObjective,
)
from batched_bfgs.timing_cache import TimingConfiguration
from batched_bfgs.vectorized import VectorizedBfgs


class TestBfgsInterface:
    """Check that every optimizer implements the shared interface."""

    def test_base_is_abstract(self) -> None:
        """The base class cannot be instantiated directly."""
        assert inspect.isabstract(Bfgs)

    def test_all_implementations_are_subclasses(self) -> None:
        """Every public implementation satisfies the BFGS contract."""
        implementations = (
            LoopBfgs,
            VectorizedBfgs,
            ChunkedVectorizedBfgs,
            CompiledChunkedVectorizedBfgs,
            CudaBfgs,
        )

        assert all(
            issubclass(implementation, Bfgs)
            for implementation in implementations
        )

    def test_tensor_objective_is_an_abstract_base(self) -> None:
        """Every analytic objective satisfies the shared abstract contract."""
        assert inspect.isabstract(TensorObjective)
        assert issubclass(ExtendedRosenbrockObjective, TensorObjective)

    def test_objective_dimensions_use_pydantic_validation(self) -> None:
        """Invalid objective dimensions fail during model construction."""
        with pytest.raises(ValidationError, match="positive and even"):
            ExtendedRosenbrockObjective(dimension=3)


class TestObjectiveType:
    """Check enum use at API and persistence boundaries."""

    def test_cuda_dispatch_covers_every_objective(self) -> None:
        """Every objective enum member has an explicit CUDA code."""
        config = BfgsConfig()

        for objective_type in ObjectiveType:
            CudaBfgs(config, objective_type)

        with pytest.raises(TypeError, match="ObjectiveType"):
            CudaBfgs(
                config,
                cast(ObjectiveType, "extended_rosenbrock"),
            )

    def test_enum_constructs_objectives_with_owned_minimizers(self) -> None:
        """The enum selects a class whose minimizer behavior stays local."""
        rosenbrock = ObjectiveType.EXTENDED_ROSENBROCK.create(16)
        assert isinstance(rosenbrock, ExtendedRosenbrockObjective)
        assert torch.equal(
            rosenbrock.minimizer(torch.empty(2, 16)),
            torch.ones(2, 16),
        )

    def test_timing_cache_loads_legacy_string(self) -> None:
        """Old JSON strings become enums without changing serialized values."""
        configuration = TimingConfiguration(
            objective="extended_rosenbrock",
            dimension=16,
            implementation="cuda_kernel",
            batch_size=64,
            device="test-gpu",
            dtype="float32",
            tolerance=1e-4,
            repeats=5,
        )

        assert configuration.objective is ObjectiveType.EXTENDED_ROSENBROCK
        serialized = configuration.model_dump_json()
        assert '"objective":"extended_rosenbrock"' in serialized


class TestPackageCli:
    """Check the unified package command structure."""

    @pytest.mark.parametrize(
        "command",
        ("benchmark", "profile-cuda"),
    )
    def test_subcommand_help(self, command: str) -> None:
        """Every workload is exposed through the package entry point."""
        result = subprocess.run(
            [sys.executable, "-m", "batched_bfgs", command, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        assert f"usage: batched-bfgs {command}" in result.stdout

    def test_default_dimension_is_sixteen(self) -> None:
        """Benchmark and profile commands default to the active 16D scope."""
        benchmark = Cli._parser().parse_args(["benchmark"])
        profile = Cli._parser().parse_args(
            ["profile-cuda", "--objective", "extended_rosenbrock"]
        )

        assert benchmark.dimension == 16
        assert profile.dimension == 16
