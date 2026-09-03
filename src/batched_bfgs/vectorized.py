"""Vectorized PyTorch implementation of batched strong-Wolfe BFGS."""

import torch

from batched_bfgs.models import (
    BatchedLineSearchResult,
    BfgsConfig,
    OptimizationResult,
)
from batched_bfgs.objective import ExtendedRosenbrockObjective, TensorObjective


class VectorizedBfgs:
    """Optimize a batch through masked tensor operations."""

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
        self._objective = objective or ExtendedRosenbrockObjective()

    @torch.no_grad()
    def run(self, starts: torch.Tensor) -> OptimizationResult:
        """Optimize all starts with batched tensor operations.

        Args:
            starts: Initial coordinates with shape ``[batch, dimension]``.

        Returns:
            One optimization result per batch member.

        """
        if starts.ndim != 2 or starts.shape[0] == 0 or starts.shape[1] == 0:
            raise ValueError("starts must have shape [batch, dimension]")
        x = starts.clone()
        batch, dimension = x.shape
        identity = torch.eye(dimension, dtype=x.dtype, device=x.device)
        hessian = identity.expand(batch, -1, -1).clone()
        objective, gradient = self._objective.value_and_gradient(x)
        iterations = torch.zeros(batch, dtype=torch.int32, device=x.device)
        evaluations = torch.zeros_like(iterations)
        converged = self._norm(gradient) <= self._config.tolerance
        wolfe = torch.ones(batch, dtype=torch.bool, device=x.device)
        active = ~converged
        for _iteration in range(self._config.max_iterations):
            if not bool(active.any()):
                break
            direction = -torch.bmm(hessian, gradient.unsqueeze(-1)).squeeze(-1)
            derivative = (gradient * direction).sum(dim=-1)
            reset = active & (derivative >= 0.0)
            hessian = torch.where(reset[:, None, None], identity, hessian)
            direction = torch.where(reset[:, None], -gradient, direction)
            line = self._strong_wolfe(x, objective, gradient, direction, active)
            evaluations += line.evaluations
            accepted = active & line.accepted
            wolfe &= ~active | line.accepted
            step = line.step[:, None] * direction
            change = line.gradient - gradient
            hessian = self._update_hessian(
                hessian,
                step,
                change,
                accepted,
                identity,
            )
            x = torch.where(accepted[:, None], x + step, x)
            objective = torch.where(accepted, line.objective, objective)
            gradient = torch.where(accepted[:, None], line.gradient, gradient)
            iterations += accepted.to(iterations.dtype)
            newly_converged = accepted & (
                self._norm(gradient) <= self._config.tolerance
            )
            converged |= newly_converged
            stagnant = accepted & (
                self._norm(step) <= self._config.step_tolerance
            )
            active = accepted & ~converged & ~stagnant
        return OptimizationResult(
            x,
            objective,
            gradient,
            iterations,
            evaluations,
            converged,
            wolfe,
        )

    def _strong_wolfe(
        self,
        x: torch.Tensor,
        objective: torch.Tensor,
        gradient: torch.Tensor,
        direction: torch.Tensor,
        active: torch.Tensor,
    ) -> BatchedLineSearchResult:
        batch = x.shape[0]
        derivative0 = (gradient * direction).sum(dim=-1)
        previous_step = torch.zeros_like(objective)
        previous_value = objective.clone()
        previous_gradient = gradient.clone()
        previous_derivative = derivative0.clone()
        step = torch.full_like(objective, self._config.initial_step)
        output_step = torch.zeros_like(objective)
        output_value = objective.clone()
        output_gradient = gradient.clone()
        evaluations = torch.zeros(batch, dtype=torch.int32, device=x.device)
        accepted = torch.zeros(batch, dtype=torch.bool, device=x.device)
        bracketed = torch.zeros_like(accepted)
        low = (
            previous_step,
            previous_value,
            previous_gradient,
            previous_derivative,
        )
        high = (
            step.clone(),
            previous_value.clone(),
            previous_gradient.clone(),
            previous_derivative.clone(),
        )
        pending = active.clone()
        for iteration in range(self._config.max_bracket_iterations):
            value, trial_gradient = self._evaluate(x, direction, step, pending)
            derivative = (trial_gradient * direction).sum(dim=-1)
            evaluations += pending.to(evaluations.dtype)
            armijo = objective + self._config.c1 * step * derivative0
            bad = pending & (~torch.isfinite(value) | (value > armijo))
            bad |= pending & (iteration > 0) & (value >= previous_value)
            curved = (
                pending
                & ~bad
                & (derivative.abs() <= -self._config.c2 * derivative0)
            )
            reverse = pending & ~bad & ~curved & (derivative >= 0.0)
            accepted |= curved
            output_step = torch.where(curved, step, output_step)
            output_value = torch.where(curved, value, output_value)
            output_gradient = torch.where(
                curved[:, None],
                trial_gradient,
                output_gradient,
            )
            new_bracket = bad | reverse
            low = self._set_bracket_low(
                low,
                bad,
                previous_step,
                previous_value,
                previous_gradient,
                previous_derivative,
                reverse,
                step,
                value,
                trial_gradient,
                derivative,
            )
            high = self._set_bracket_high(
                high,
                bad,
                step,
                value,
                trial_gradient,
                derivative,
                reverse,
                previous_step,
                previous_value,
                previous_gradient,
                previous_derivative,
            )
            bracketed |= new_bracket
            continuing = pending & ~new_bracket & ~curved
            previous_step = torch.where(continuing, step, previous_step)
            previous_value = torch.where(continuing, value, previous_value)
            previous_gradient = torch.where(
                continuing[:, None],
                trial_gradient,
                previous_gradient,
            )
            previous_derivative = torch.where(
                continuing,
                derivative,
                previous_derivative,
            )
            step = torch.where(
                continuing,
                torch.clamp(2.0 * step, max=self._config.maximum_step),
                step,
            )
            pending = continuing
        return self._zoom(
            x,
            direction,
            objective,
            derivative0,
            low,
            high,
            bracketed,
            output_step,
            output_value,
            output_gradient,
            evaluations,
            accepted,
        )

    def _zoom(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        objective0: torch.Tensor,
        derivative0: torch.Tensor,
        low: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        high: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        active: torch.Tensor,
        output_step: torch.Tensor,
        output_value: torch.Tensor,
        output_gradient: torch.Tensor,
        evaluations: torch.Tensor,
        accepted: torch.Tensor,
    ) -> BatchedLineSearchResult:
        low_step, low_value, low_gradient, low_derivative = low
        high_step, high_value, high_gradient, high_derivative = high
        for _iteration in range(self._config.max_zoom_iterations):
            step = self._cubic_step(
                low_step,
                low_value,
                low_derivative,
                high_step,
                high_value,
                high_derivative,
            )
            value, gradient = self._evaluate(x, direction, step, active)
            derivative = (gradient * direction).sum(dim=-1)
            evaluations += active.to(evaluations.dtype)
            armijo = objective0 + self._config.c1 * step * derivative0
            bad = active & (~torch.isfinite(value) | (value > armijo))
            bad |= active & (value >= low_value)
            curved = (
                active
                & ~bad
                & (derivative.abs() <= -self._config.c2 * derivative0)
            )
            accepted |= curved
            output_step = torch.where(curved, step, output_step)
            output_value = torch.where(curved, value, output_value)
            output_gradient = torch.where(
                curved[:, None],
                gradient,
                output_gradient,
            )
            good = active & ~bad & ~curved
            flip = good & (derivative * (high_step - low_step) >= 0.0)
            high_step = torch.where(flip, low_step, high_step)
            high_value = torch.where(flip, low_value, high_value)
            high_gradient = torch.where(
                flip[:, None],
                low_gradient,
                high_gradient,
            )
            high_derivative = torch.where(flip, low_derivative, high_derivative)
            high_step = torch.where(bad, step, high_step)
            high_value = torch.where(bad, value, high_value)
            high_gradient = torch.where(bad[:, None], gradient, high_gradient)
            high_derivative = torch.where(bad, derivative, high_derivative)
            low_step = torch.where(good, step, low_step)
            low_value = torch.where(good, value, low_value)
            low_gradient = torch.where(good[:, None], gradient, low_gradient)
            low_derivative = torch.where(good, derivative, low_derivative)
            active &= ~curved
        return BatchedLineSearchResult(
            output_step,
            output_value,
            output_gradient,
            evaluations,
            accepted,
        )

    @staticmethod
    def _set_bracket_low(
        current: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        first_mask: torch.Tensor,
        first_step: torch.Tensor,
        first_value: torch.Tensor,
        first_gradient: torch.Tensor,
        first_derivative: torch.Tensor,
        second_mask: torch.Tensor,
        second_step: torch.Tensor,
        second_value: torch.Tensor,
        second_gradient: torch.Tensor,
        second_derivative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step, value, gradient, derivative = current
        step = torch.where(first_mask, first_step, step)
        value = torch.where(first_mask, first_value, value)
        gradient = torch.where(first_mask[:, None], first_gradient, gradient)
        derivative = torch.where(first_mask, first_derivative, derivative)
        step = torch.where(second_mask, second_step, step)
        value = torch.where(second_mask, second_value, value)
        gradient = torch.where(second_mask[:, None], second_gradient, gradient)
        derivative = torch.where(second_mask, second_derivative, derivative)
        return step, value, gradient, derivative

    @staticmethod
    def _set_bracket_high(
        current: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        first_mask: torch.Tensor,
        first_step: torch.Tensor,
        first_value: torch.Tensor,
        first_gradient: torch.Tensor,
        first_derivative: torch.Tensor,
        second_mask: torch.Tensor,
        second_step: torch.Tensor,
        second_value: torch.Tensor,
        second_gradient: torch.Tensor,
        second_derivative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return VectorizedBfgs._set_bracket_low(
            current,
            first_mask,
            first_step,
            first_value,
            first_gradient,
            first_derivative,
            second_mask,
            second_step,
            second_value,
            second_gradient,
            second_derivative,
        )

    def _cubic_step(
        self,
        step1: torch.Tensor,
        value1: torch.Tensor,
        derivative1: torch.Tensor,
        step2: torch.Tensor,
        value2: torch.Tensor,
        derivative2: torch.Tensor,
    ) -> torch.Tensor:
        lower = torch.minimum(step1, step2)
        upper = torch.maximum(step1, step2)
        midpoint = 0.5 * (lower + upper)
        separation = step1 - step2
        safe_separation = torch.where(
            separation.abs() > 1e-20,
            separation,
            torch.ones_like(separation),
        )
        d1 = derivative1 + derivative2
        d1 -= 3.0 * (value1 - value2) / safe_separation
        discriminant = d1.square() - derivative1 * derivative2
        d2 = torch.sqrt(torch.clamp(discriminant, min=0.0))
        forward_denominator = derivative2 - derivative1 + 2.0 * d2
        reverse_denominator = derivative1 - derivative2 + 2.0 * d2
        denominator = torch.where(
            step1 <= step2,
            forward_denominator,
            reverse_denominator,
        )
        safe_denominator = torch.where(
            denominator.abs() > 1e-20,
            denominator,
            torch.ones_like(denominator),
        )
        forward = step2 - (step2 - step1) * (
            (derivative2 + d2 - d1) / safe_denominator
        )
        reverse = step1 - (step1 - step2) * (
            (derivative1 + d2 - d1) / safe_denominator
        )
        candidate = torch.where(step1 <= step2, forward, reverse)
        guard = 0.1 * (upper - lower)
        usable = torch.isfinite(candidate) & (discriminant >= 0.0)
        usable &= denominator.abs() > 1e-20
        usable &= candidate > lower + guard
        usable &= candidate < upper - guard
        return torch.where(usable, candidate, midpoint)

    def _evaluate(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        step: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_step = torch.where(active, step, torch.zeros_like(step))
        return self._objective.value_and_gradient(
            x + safe_step[:, None] * direction,
        )

    def _update_hessian(
        self,
        hessian: torch.Tensor,
        step: torch.Tensor,
        change: torch.Tensor,
        accepted: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        curvature = (step * change).sum(dim=-1)
        threshold = self._config.curvature_eps
        threshold *= torch.linalg.vector_norm(step, dim=-1)
        threshold *= torch.linalg.vector_norm(change, dim=-1)
        usable = accepted & torch.isfinite(curvature) & (curvature > threshold)
        safe_curvature = torch.where(
            usable,
            curvature,
            torch.ones_like(curvature),
        )
        rho = safe_curvature.reciprocal()
        left = identity - rho[:, None, None] * (
            step[:, :, None] * change[:, None, :]
        )
        updated = torch.bmm(torch.bmm(left, hessian), left.transpose(1, 2))
        updated += rho[:, None, None] * (step[:, :, None] * step[:, None, :])
        return torch.where(usable[:, None, None], updated, hessian)

    @staticmethod
    def _norm(value: torch.Tensor) -> torch.Tensor:
        return value.abs().amax(dim=-1)
