"""Tensor objectives used by the Python and vectorized implementations."""

from abc import ABC, abstractmethod
from enum import StrEnum

import torch
from pydantic import Field, field_validator

from batched_bfgs.models import BaseModelNoExtra


class ObjectiveType(StrEnum):
    """Analytic objective available to the benchmark implementations."""

    EXTENDED_ROSENBROCK = "extended_rosenbrock"
    EXTENDED_POWELL = "extended_powell"

    def create(self, dimension: int) -> "TensorObjective":
        """Construct the objective selected by this enum value."""
        if self is ObjectiveType.EXTENDED_ROSENBROCK:
            return ExtendedRosenbrockObjective(dimension=dimension)
        return ExtendedPowellSingularObjective(dimension=dimension)


class TensorObjective(BaseModelNoExtra, ABC):
    """Contract for an objective with an analytic gradient."""

    dimension: int = Field(gt=0)

    @abstractmethod
    def value_and_gradient(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate objective values and gradients at ``x``."""

    @abstractmethod
    def make_starts(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create deterministic starting coordinates."""

    @abstractmethod
    def minimizer(self, like: torch.Tensor) -> torch.Tensor:
        """Return the known global minimizer shaped like ``like``."""

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate a runtime tensor against the configured dimension."""
        if x.ndim == 0 or x.shape[-1] != self.dimension:
            raise ValueError(
                f"expected inputs with final dimension {self.dimension}",
            )


class ExtendedRosenbrockObjective(TensorObjective):
    """Evaluate independent two-variable Rosenbrock blocks."""

    def value_and_gradient(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the extended Rosenbrock value and analytic gradient.

        Args:
            x: Tensor with a positive, even-sized final dimension.

        Returns:
            Objective values and gradients with matching leading dimensions.

        """
        self._validate_input(x)
        odd = x[..., 0::2]
        even = x[..., 1::2]
        residual = even - odd.square()
        objective = ((1.0 - odd).square() + 100.0 * residual.square()).sum(
            dim=-1
        )
        gradient = torch.empty_like(x)
        gradient[..., 0::2] = -2.0 * (1.0 - odd) - 400.0 * odd * residual
        gradient[..., 1::2] = 200.0 * residual
        return objective, gradient

    def make_starts(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create deterministic starts from repeated ``[-1.2, 1]`` blocks."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        block = torch.tensor((-1.2, 1.0), dtype=dtype, device=device)
        starts = (
            block.repeat(self.dimension // 2).expand(batch_size, -1).clone()
        )
        index = torch.arange(batch_size, dtype=dtype, device=device)
        starts[:, 0::2] += 0.05 * torch.sin(index[:, None] * 0.37)
        starts[:, 1::2] += 0.05 * torch.cos(index[:, None] * 0.53)
        return starts

    def minimizer(self, like: torch.Tensor) -> torch.Tensor:
        """Return the all-ones Rosenbrock minimizer."""
        self._validate_input(like)
        return torch.ones_like(like)

    @field_validator("dimension")
    @classmethod
    def _validate_dimension(cls, dimension: int) -> int:
        """Validate an extended Rosenbrock dimension."""
        if dimension < 2 or dimension % 2 != 0:
            raise ValueError(
                "extended Rosenbrock dimension must be positive and even",
            )
        return dimension


class ExtendedPowellSingularObjective(TensorObjective):
    """Evaluate independent four-variable Powell singular blocks."""

    def value_and_gradient(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the extended Powell singular value and gradient.

        Args:
            x: Tensor whose final dimension is a positive multiple of four.

        Returns:
            Objective values and gradients with matching leading dimensions.

        """
        self._validate_input(x)
        blocks = x.reshape(*x.shape[:-1], -1, 4)
        x1, x2, x3, x4 = blocks.unbind(dim=-1)
        first = x1 + 10.0 * x2
        second = x3 - x4
        third = x2 - 2.0 * x3
        fourth = x1 - x4
        objective = (
            first.square()
            + 5.0 * second.square()
            + third.pow(4)
            + 10.0 * fourth.pow(4)
        ).sum(dim=-1)
        gradient_blocks = torch.stack(
            (
                2.0 * first + 40.0 * fourth.pow(3),
                20.0 * first + 4.0 * third.pow(3),
                10.0 * second - 8.0 * third.pow(3),
                -10.0 * second - 40.0 * fourth.pow(3),
            ),
            dim=-1,
        )
        return objective, gradient_blocks.reshape_as(x)

    def make_starts(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create repeated standard Powell starts ``[3, -1, 0, 1]``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        block = torch.tensor((3.0, -1.0, 0.0, 1.0), dtype=dtype, device=device)
        return block.repeat(self.dimension // 4).expand(batch_size, -1).clone()

    def minimizer(self, like: torch.Tensor) -> torch.Tensor:
        """Return the all-zero Powell minimizer."""
        self._validate_input(like)
        return torch.zeros_like(like)

    @field_validator("dimension")
    @classmethod
    def _validate_dimension(cls, dimension: int) -> int:
        """Validate an extended Powell dimension."""
        if dimension < 4 or dimension % 4 != 0:
            raise ValueError(
                "extended Powell dimension must be a positive multiple of four",
            )
        return dimension
