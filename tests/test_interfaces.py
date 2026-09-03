"""Contracts shared by objectives and BFGS implementations."""

import inspect
from typing import cast

import pytest
import torch
from pydantic import ValidationError

from batched_bfgs.base import Bfgs
from batched_bfgs.chunked import (
    ChunkedVectorizedBfgs,
    CompiledChunkedVectorizedBfgs,
)
from batched_bfgs.cuda import CudaBfgs
from batched_bfgs.loop import LoopBfgs
from batched_bfgs.models import BfgsConfig
from batched_bfgs.objective import (
    ExtendedPowellSingularObjective,
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
        assert issubclass(ExtendedPowellSingularObjective, TensorObjective)

    def test_objective_dimensions_use_pydantic_validation(self) -> None:
        """Invalid objective dimensions fail during model construction."""
        with pytest.raises(ValidationError, match="positive and even"):
            ExtendedRosenbrockObjective(dimension=3)
        with pytest.raises(ValidationError, match="multiple of four"):
            ExtendedPowellSingularObjective(dimension=6)


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
        powell = ObjectiveType.EXTENDED_POWELL.create(16)

        assert isinstance(rosenbrock, ExtendedRosenbrockObjective)
        assert isinstance(powell, ExtendedPowellSingularObjective)
        assert torch.equal(
            rosenbrock.minimizer(torch.empty(2, 16)),
            torch.ones(2, 16),
        )
        assert torch.equal(
            powell.minimizer(torch.empty(2, 16)),
            torch.zeros(2, 16),
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
