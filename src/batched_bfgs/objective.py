"""The objective shared by the Python and vectorized implementations."""

import torch


class RosenbrockObjective:
    """Evaluate the two-dimensional Rosenbrock function and gradient."""

    @staticmethod
    def value_and_gradient(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate values and analytic gradients over the leading dimensions.

        Args:
            x: Tensor whose final dimension has size two.

        Returns:
            Objective values and gradients with matching leading dimensions.

        """
        if x.shape[-1] != 2:
            raise ValueError("Rosenbrock inputs must have final dimension 2")
        x0 = x[..., 0]
        x1 = x[..., 1]
        residual = x1 - x0.square()
        objective = (1.0 - x0).square() + 100.0 * residual.square()
        gradient = torch.stack(
            (
                -2.0 * (1.0 - x0) - 400.0 * x0 * residual,
                200.0 * residual,
            ),
            dim=-1,
        )
        return objective, gradient
