"""Configuration and result types shared by the implementations."""

from typing import NamedTuple, Self

import torch
from pydantic import BaseModel, ConfigDict, model_validator


class BaseModelNoExtra(BaseModel):
    """Pydantic model that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BfgsConfig(BaseModelNoExtra):
    """Numerical contract shared by every BFGS implementation."""

    c1: float = 1e-4
    c2: float = 0.9
    tolerance: float = 1e-5
    step_tolerance: float = 1e-12
    curvature_eps: float = 1e-10
    initial_step: float = 1.0
    maximum_step: float = 64.0
    max_iterations: int = 100
    max_bracket_iterations: int = 20
    max_zoom_iterations: int = 25

    @model_validator(mode="after")
    def _validate_values(self) -> Self:
        if not 0.0 < self.c1 < self.c2 < 1.0:
            raise ValueError("Require 0 < c1 < c2 < 1")
        if self.initial_step <= 0.0:
            raise ValueError("initial_step must be positive")
        if self.maximum_step < self.initial_step:
            raise ValueError("maximum_step must be at least initial_step")
        if min(self.max_iterations, self.max_zoom_iterations) <= 0:
            raise ValueError("iteration limits must be positive")
        if self.max_bracket_iterations <= 0:
            raise ValueError("iteration limits must be positive")
        return self


class OptimizationResult(NamedTuple):
    """Per-member BFGS outputs."""

    x: torch.Tensor
    objective: torch.Tensor
    gradient: torch.Tensor
    iterations: torch.Tensor
    line_search_evaluations: torch.Tensor
    converged: torch.Tensor
    wolfe_satisfied: torch.Tensor


class ScalarLineSearchResult(NamedTuple):
    """Line-search result for one batch member."""

    step: float
    objective: torch.Tensor
    gradient: torch.Tensor
    evaluations: int
    accepted: bool


class BatchedLineSearchResult(NamedTuple):
    """Line-search result for a whole tensor batch."""

    step: torch.Tensor
    objective: torch.Tensor
    gradient: torch.Tensor
    evaluations: torch.Tensor
    accepted: torch.Tensor
