"""Tensor objectives used by the Python and vectorized implementations."""

from typing import Protocol

import torch


class TensorObjective(Protocol):
    """Contract for an objective with an analytic gradient."""

    def value_and_gradient(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate objective values and gradients at ``x``."""
        ...


class ExtendedRosenbrockObjective:
    """Evaluate independent two-variable Rosenbrock blocks."""

    @staticmethod
    def value_and_gradient(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the extended Rosenbrock value and analytic gradient.

        Args:
            x: Tensor with a positive, even-sized final dimension.

        Returns:
            Objective values and gradients with matching leading dimensions.

        """
        ExtendedRosenbrockObjective._validate(x)
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

    @staticmethod
    def make_starts(
        batch_size: int,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create deterministic starts from repeated ``[-1.2, 1]`` blocks."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        ExtendedRosenbrockObjective._validate_dimension(dimension)
        block = torch.tensor((-1.2, 1.0), dtype=dtype, device=device)
        starts = block.repeat(dimension // 2).expand(batch_size, -1).clone()
        index = torch.arange(batch_size, dtype=dtype, device=device)
        starts[:, 0::2] += 0.05 * torch.sin(index[:, None] * 0.37)
        starts[:, 1::2] += 0.05 * torch.cos(index[:, None] * 0.53)
        return starts

    @staticmethod
    def _validate(x: torch.Tensor) -> None:
        if x.ndim == 0:
            raise ValueError("extended Rosenbrock inputs must have a dimension")
        ExtendedRosenbrockObjective._validate_dimension(x.shape[-1])

    @staticmethod
    def _validate_dimension(dimension: int) -> None:
        if dimension < 2 or dimension % 2 != 0:
            raise ValueError(
                "extended Rosenbrock dimension must be positive and even",
            )


class ExtendedPowellSingularObjective:
    """Evaluate independent four-variable Powell singular blocks."""

    @staticmethod
    def value_and_gradient(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the extended Powell singular value and gradient.

        Args:
            x: Tensor whose final dimension is a positive multiple of four.

        Returns:
            Objective values and gradients with matching leading dimensions.

        """
        ExtendedPowellSingularObjective._validate(x)
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

    @staticmethod
    def make_starts(
        batch_size: int,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create repeated standard Powell starts ``[3, -1, 0, 1]``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        ExtendedPowellSingularObjective._validate_dimension(dimension)
        block = torch.tensor((3.0, -1.0, 0.0, 1.0), dtype=dtype, device=device)
        return block.repeat(dimension // 4).expand(batch_size, -1).clone()

    @staticmethod
    def _validate(x: torch.Tensor) -> None:
        if x.ndim == 0:
            raise ValueError("extended Powell inputs must have a dimension")
        ExtendedPowellSingularObjective._validate_dimension(x.shape[-1])

    @staticmethod
    def _validate_dimension(dimension: int) -> None:
        if dimension < 4 or dimension % 4 != 0:
            raise ValueError(
                "extended Powell dimension must be a positive multiple of four",
            )
