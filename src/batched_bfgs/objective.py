"""Tensor objectives used by the Python and vectorized implementations."""

from abc import ABC, abstractmethod
from enum import StrEnum

import torch
from pydantic import Field, field_validator

from batched_bfgs.models import BaseModelNoExtra


class ObjectiveType(StrEnum):
    """Analytic objective available to the benchmark implementations."""

    EXTENDED_ROSENBROCK = "extended_rosenbrock"

    def create(self, dimension: int) -> "TensorObjective":
        """Construct the objective selected by this enum value."""
        # Resolve the enum to one concrete validated model.
        return ExtendedRosenbrockObjective(dimension=dimension)


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
        # Reject scalar or mismatched runtime coordinates immediately.
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
        # Validate the runtime shape before slicing coordinate pairs.
        self._validate_input(x)
        odd = x[..., 0::2]
        even = x[..., 1::2]
        residual = even - odd.square()

        # Accumulate the independent pairwise objective terms.
        objective = ((1.0 - odd).square() + 100.0 * residual.square()).sum(
            dim=-1
        )

        # Assemble analytic derivatives in the original coordinate order.
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
        # Validate the batch before constructing its coordinates.
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Repeat the canonical pair across the configured dimension.
        block = torch.tensor((-1.2, 1.0), dtype=dtype, device=device)
        starts = (
            block.repeat(self.dimension // 2).expand(batch_size, -1).clone()
        )

        # Add deterministic member-specific perturbations.
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
        # Rosenbrock coordinates require complete pairs.
        if dimension < 2 or dimension % 2 != 0:
            raise ValueError(
                "extended Rosenbrock dimension must be positive and even",
            )
        return dimension
