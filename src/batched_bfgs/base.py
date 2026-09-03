"""Shared interface for BFGS implementations."""

from abc import ABC, abstractmethod

import torch

from batched_bfgs.models import OptimizationResult


class Bfgs(ABC):
    """Interface implemented by every BFGS optimizer."""

    @abstractmethod
    def run(self, starts: torch.Tensor) -> OptimizationResult:
        """Optimize a batch of starting coordinates."""
