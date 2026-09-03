"""Naive Python-loop implementation of batched strong-Wolfe BFGS."""

import math

import torch

from batched_bfgs.base import Bfgs
from batched_bfgs.models import (
    BfgsConfig,
    OptimizationResult,
    ScalarLineSearchResult,
)
from batched_bfgs.objective import ExtendedRosenbrockObjective, TensorObjective


class LoopBfgs(Bfgs):
    """Optimize each batch member through explicit Python control flow."""

    def __init__(
        self,
        config: BfgsConfig,
        objective: TensorObjective | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            config: Shared numerical configuration.
            objective: Objective and analytic gradient evaluator.

        """
        self._config = config
        self._objective = objective or ExtendedRosenbrockObjective(dimension=2)

    @torch.no_grad()
    def run(self, starts: torch.Tensor) -> OptimizationResult:
        """Optimize all starts sequentially.

        Args:
            starts: Initial coordinates with shape ``[batch, dimension]``.

        Returns:
            One optimization result per batch member.

        """
        if starts.ndim != 2 or starts.shape[0] == 0 or starts.shape[1] == 0:
            raise ValueError("starts must have shape [batch, dimension]")
        members = [self._optimize_one(start) for start in starts]
        fields = zip(*members, strict=True)
        stacked = [torch.stack(tuple(values)) for values in fields]
        return OptimizationResult(*stacked)

    def _optimize_one(self, start: torch.Tensor) -> OptimizationResult:
        """Optimize one starting point."""
        x = start.clone()
        dimension = x.shape[0]
        hessian = torch.eye(dimension, dtype=x.dtype, device=x.device)
        objective, gradient = self._objective.value_and_gradient(x)
        iterations = 0
        evaluations = 0
        wolfe_satisfied = True
        converged = self._gradient_norm(gradient) <= self._config.tolerance
        for _iteration in range(self._config.max_iterations):
            if converged:
                break
            direction = -(hessian @ gradient)
            if float(torch.dot(gradient, direction)) >= 0.0:
                hessian = torch.eye(
                    dimension,
                    dtype=x.dtype,
                    device=x.device,
                )
                direction = -gradient
            line = self._strong_wolfe(x, objective, gradient, direction)
            evaluations += line.evaluations
            if not line.accepted:
                wolfe_satisfied = False
                break
            step = line.step * direction
            change = line.gradient - gradient
            hessian = self._update_hessian(hessian, step, change)
            x, objective, gradient = x + step, line.objective, line.gradient
            iterations += 1
            converged = self._gradient_norm(gradient) <= self._config.tolerance
            if self._gradient_norm(step) <= self._config.step_tolerance:
                break
        return self._make_result(
            x,
            objective,
            gradient,
            iterations,
            evaluations,
            converged,
            wolfe_satisfied,
        )

    def _strong_wolfe(
        self,
        x: torch.Tensor,
        objective: torch.Tensor,
        gradient: torch.Tensor,
        direction: torch.Tensor,
    ) -> ScalarLineSearchResult:
        """Find a step satisfying the strong-Wolfe conditions."""
        derivative0 = float(torch.dot(gradient, direction))
        previous_step = 0.0
        previous_value = objective
        previous_gradient = gradient
        previous_derivative = derivative0
        step = self._config.initial_step
        for iteration in range(self._config.max_bracket_iterations):
            value, trial_gradient = self._evaluate(x, direction, step)
            derivative = float(torch.dot(trial_gradient, direction))
            armijo = float(objective) + self._config.c1 * step * derivative0
            too_high = not math.isfinite(float(value)) or float(value) > armijo
            nondecreasing = iteration > 0 and float(value) >= float(
                previous_value
            )
            if too_high or nondecreasing:
                return self._zoom(
                    x,
                    direction,
                    objective,
                    derivative0,
                    (
                        previous_step,
                        previous_value,
                        previous_gradient,
                        previous_derivative,
                    ),
                    (step, value, trial_gradient, derivative),
                    iteration + 1,
                )
            if abs(derivative) <= -self._config.c2 * derivative0:
                return ScalarLineSearchResult(
                    step,
                    value,
                    trial_gradient,
                    iteration + 1,
                    True,
                )
            if derivative >= 0.0:
                return self._zoom(
                    x,
                    direction,
                    objective,
                    derivative0,
                    (step, value, trial_gradient, derivative),
                    (
                        previous_step,
                        previous_value,
                        previous_gradient,
                        previous_derivative,
                    ),
                    iteration + 1,
                )
            previous_step, previous_value = step, value
            previous_gradient, previous_derivative = trial_gradient, derivative
            step = min(2.0 * step, self._config.maximum_step)
        return ScalarLineSearchResult(
            0.0,
            objective,
            gradient,
            self._config.max_bracket_iterations,
            False,
        )

    def _zoom(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        objective0: torch.Tensor,
        derivative0: float,
        low: tuple[float, torch.Tensor, torch.Tensor, float],
        high: tuple[float, torch.Tensor, torch.Tensor, float],
        evaluations: int,
    ) -> ScalarLineSearchResult:
        """Refine a bracketed strong-Wolfe step."""
        low_step, low_value, low_gradient, low_derivative = low
        high_step, high_value, _high_gradient, high_derivative = high
        for iteration in range(self._config.max_zoom_iterations):
            step = self._cubic_step(
                low_step,
                float(low_value),
                low_derivative,
                high_step,
                float(high_value),
                high_derivative,
            )
            value, gradient = self._evaluate(x, direction, step)
            derivative = float(torch.dot(gradient, direction))
            armijo = float(objective0) + self._config.c1 * step * derivative0
            if not math.isfinite(float(value)) or float(value) > armijo:
                high_step, high_value = step, value
                high_derivative = derivative
            elif float(value) >= float(low_value):
                high_step, high_value = step, value
                high_derivative = derivative
            else:
                if abs(derivative) <= -self._config.c2 * derivative0:
                    return ScalarLineSearchResult(
                        step,
                        value,
                        gradient,
                        evaluations + iteration + 1,
                        True,
                    )
                if derivative * (high_step - low_step) >= 0.0:
                    high_step, high_value = low_step, low_value
                    high_derivative = low_derivative
                low_step, low_value = step, value
                low_gradient, low_derivative = gradient, derivative
        return ScalarLineSearchResult(
            0.0,
            objective0,
            low_gradient,
            evaluations + self._config.max_zoom_iterations,
            False,
        )

    def _cubic_step(
        self,
        step1: float,
        value1: float,
        derivative1: float,
        step2: float,
        value2: float,
        derivative2: float,
    ) -> float:
        """Choose a safeguarded cubic-interpolation step."""
        lower, upper = sorted((step1, step2))
        midpoint = 0.5 * (lower + upper)
        if step1 == step2:
            return midpoint
        d1 = derivative1 + derivative2
        d1 -= 3.0 * (value1 - value2) / (step1 - step2)
        discriminant = d1 * d1 - derivative1 * derivative2
        if discriminant < 0.0 or not math.isfinite(discriminant):
            return midpoint
        d2 = math.sqrt(discriminant)
        if step1 <= step2:
            denominator = derivative2 - derivative1 + 2.0 * d2
            if abs(denominator) <= 1e-20:
                return midpoint
            candidate = step2 - (step2 - step1) * (
                (derivative2 + d2 - d1) / denominator
            )
        else:
            denominator = derivative1 - derivative2 + 2.0 * d2
            if abs(denominator) <= 1e-20:
                return midpoint
            candidate = step1 - (step1 - step2) * (
                (derivative1 + d2 - d1) / denominator
            )
        guard = 0.1 * (upper - lower)
        if (
            not math.isfinite(candidate)
            or not lower + guard < candidate < upper - guard
        ):
            return midpoint
        return candidate

    def _evaluate(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        step: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the objective along one search direction."""
        return self._objective.value_and_gradient(x + step * direction)

    def _update_hessian(
        self,
        hessian: torch.Tensor,
        step: torch.Tensor,
        change: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one inverse-Hessian BFGS update."""
        curvature = torch.dot(step, change)
        threshold = self._config.curvature_eps * torch.linalg.vector_norm(step)
        threshold *= torch.linalg.vector_norm(change)
        if float(curvature) <= float(threshold):
            return hessian
        rho = curvature.reciprocal()
        identity = torch.eye(
            step.shape[0],
            dtype=step.dtype,
            device=step.device,
        )
        left = identity - rho * torch.outer(step, change)
        return left @ hessian @ left.T + rho * torch.outer(step, step)

    @staticmethod
    def _gradient_norm(value: torch.Tensor) -> float:
        """Return the gradient infinity norm."""
        return float(value.abs().amax())

    @staticmethod
    def _make_result(
        x: torch.Tensor,
        objective: torch.Tensor,
        gradient: torch.Tensor,
        iterations: int,
        evaluations: int,
        converged: bool,
        wolfe_satisfied: bool,
    ) -> OptimizationResult:
        """Pack scalar optimizer state into tensor outputs."""
        device = x.device
        return OptimizationResult(
            x,
            objective,
            gradient,
            torch.tensor(iterations, dtype=torch.int32, device=device),
            torch.tensor(evaluations, dtype=torch.int32, device=device),
            torch.tensor(converged, dtype=torch.bool, device=device),
            torch.tensor(wolfe_satisfied, dtype=torch.bool, device=device),
        )
