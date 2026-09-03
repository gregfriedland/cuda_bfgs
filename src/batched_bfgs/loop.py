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
        # Store immutable optimizer inputs for repeated runs.
        self._config = config
        self._objective = objective or ExtendedRosenbrockObjective(dimension=16)

    @torch.no_grad()
    def run(self, starts: torch.Tensor) -> OptimizationResult:
        """Optimize all starts sequentially.

        Args:
            starts: Initial coordinates with shape ``[batch, dimension]``.

        Returns:
            One optimization result per batch member.

        """
        # Validate the common batched input contract.
        if starts.ndim != 2 or starts.shape[0] == 0 or starts.shape[1] == 0:
            raise ValueError("starts must have shape [batch, dimension]")

        # Optimize members independently and stack each result field.
        members = [self._optimize_one(start) for start in starts]
        fields = zip(*members, strict=True)
        stacked = [torch.stack(tuple(values)) for values in fields]
        return OptimizationResult(*stacked)

    def _optimize_one(self, start: torch.Tensor) -> OptimizationResult:
        """Optimize one starting point."""
        # Initialize coordinates, inverse Hessian, and derivatives.
        x = start.clone()
        dimension = x.shape[0]
        hessian = torch.eye(dimension, dtype=x.dtype, device=x.device)
        objective, gradient = self._objective.value_and_gradient(x)

        # Initialize scalar progress and termination state.
        iterations = 0
        evaluations = 0
        wolfe_satisfied = True
        converged = self._gradient_norm(gradient) <= self._config.tolerance

        # Advance BFGS until convergence or a terminal line-search result.
        for _iteration in range(self._config.max_iterations):
            if converged:
                break

            # Compute a descent direction and reset invalid Hessians.
            direction = -(hessian @ gradient)
            if float(torch.dot(gradient, direction)) >= 0.0:
                hessian = torch.eye(
                    dimension,
                    dtype=x.dtype,
                    device=x.device,
                )
                direction = -gradient

            # Find and account for a strong-Wolfe step.
            line = self._strong_wolfe(x, objective, gradient, direction)
            evaluations += line.evaluations
            if not line.accepted:
                wolfe_satisfied = False
                break

            # Update the inverse Hessian from the accepted displacement.
            step = line.step * direction
            change = line.gradient - gradient
            hessian = self._update_hessian(hessian, step, change)

            # Commit the accepted point and update termination state.
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
        # Initialize the directional derivative and prior endpoint.
        derivative0 = float(torch.dot(gradient, direction))
        previous_step = 0.0
        previous_value = objective
        previous_gradient = gradient
        previous_derivative = derivative0
        step = self._config.initial_step

        # Expand the trial step until acceptance or bracketing.
        for iteration in range(self._config.max_bracket_iterations):
            # Evaluate the new point and sufficient-decrease bound.
            value, trial_gradient = self._evaluate(x, direction, step)
            derivative = float(torch.dot(trial_gradient, direction))
            armijo = float(objective) + self._config.c1 * step * derivative0
            too_high = not math.isfinite(float(value)) or float(value) > armijo
            nondecreasing = iteration > 0 and float(value) >= float(
                previous_value
            )

            # Zoom when the trial overshoots or stops decreasing.
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

            # Accept a trial that satisfies strong curvature.
            if abs(derivative) <= -self._config.c2 * derivative0:
                return ScalarLineSearchResult(
                    step,
                    value,
                    trial_gradient,
                    iteration + 1,
                    True,
                )

            # Zoom across a derivative sign change.
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

            # Shift the lower endpoint and continue expanding.
            previous_step, previous_value = step, value
            previous_gradient, previous_derivative = trial_gradient, derivative
            step = min(2.0 * step, self._config.maximum_step)
        # Report exhaustion without changing the original point.
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
        # Unpack the current lower and upper bracket endpoints.
        low_step, low_value, low_gradient, low_derivative = low
        high_step, high_value, _high_gradient, high_derivative = high

        # Refine the bracket until a strong-Wolfe point is found.
        for iteration in range(self._config.max_zoom_iterations):
            # Interpolate and evaluate a safeguarded trial point.
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

            # Shrink the upper endpoint when decrease is insufficient.
            if not math.isfinite(float(value)) or float(value) > armijo:
                high_step, high_value = step, value
                high_derivative = derivative
            elif float(value) >= float(low_value):
                high_step, high_value = step, value
                high_derivative = derivative
            else:
                # Accept curvature or move the lower endpoint inward.
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

        # Report bracket exhaustion without changing the original point.
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
        # Establish a safe interval and handle collapsed brackets.
        lower, upper = sorted((step1, step2))
        midpoint = 0.5 * (lower + upper)
        if step1 == step2:
            return midpoint

        # Compute the cubic discriminant and reject invalid curvature.
        d1 = derivative1 + derivative2
        d1 -= 3.0 * (value1 - value2) / (step1 - step2)
        discriminant = d1 * d1 - derivative1 * derivative2
        if discriminant < 0.0 or not math.isfinite(discriminant):
            return midpoint
        d2 = math.sqrt(discriminant)

        # Interpolate from the endpoint ordering supplied by the bracket.
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

        # Fall back to bisection when interpolation leaves the safe interior.
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
        # Reject updates without sufficiently positive curvature.
        curvature = torch.dot(step, change)
        threshold = self._config.curvature_eps * torch.linalg.vector_norm(step)
        threshold *= torch.linalg.vector_norm(change)
        if float(curvature) <= float(threshold):
            return hessian

        # Apply the factored inverse-Hessian update.
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
        # Materialize scalar metadata on the result device.
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
