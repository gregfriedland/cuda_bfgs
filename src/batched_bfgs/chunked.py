"""Chunked PyTorch implementation of batched strong-Wolfe BFGS."""

from collections import deque
from typing import NamedTuple

import torch

from batched_bfgs.models import BfgsConfig, OptimizationResult
from batched_bfgs.objective import TensorObjective
from batched_bfgs.vectorized import VectorizedBfgs


class _ChunkState(NamedTuple):
    """Tensor state carried between fixed-size iteration chunks."""

    x: torch.Tensor
    objective: torch.Tensor
    gradient: torch.Tensor
    hessian: torch.Tensor
    iterations: torch.Tensor
    evaluations: torch.Tensor
    converged: torch.Tensor
    wolfe: torch.Tensor
    active: torch.Tensor


class _PendingCheck(NamedTuple):
    """One asynchronous device-to-host convergence check."""

    ready: torch.cuda.Event
    host_active: torch.Tensor
    device_active: torch.Tensor


class ChunkedVectorizedBfgs(VectorizedBfgs):
    """Run iteration chunks with nonblocking CUDA convergence polls."""

    def __init__(
        self,
        config: BfgsConfig,
        objective: TensorObjective | None = None,
        chunk_size: int = 16,
    ) -> None:
        """Initialize the optimizer with a fixed iteration chunk size."""
        super().__init__(config, objective)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = chunk_size

    @torch.no_grad()
    def run(self, starts: torch.Tensor) -> OptimizationResult:
        """Optimize a batch without synchronizing after every iteration."""
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
        state = _ChunkState(
            x,
            objective,
            gradient,
            hessian,
            iterations,
            evaluations,
            converged,
            wolfe,
            ~converged,
        )
        if starts.is_cuda:
            state = self._run_cuda_chunks(state, identity)
        else:
            state = self._run_cpu_chunks(state, identity)
        return OptimizationResult(
            state.x,
            state.objective,
            state.gradient,
            state.iterations,
            state.evaluations,
            state.converged,
            state.wolfe,
        )

    def _run_cpu_chunks(
        self,
        state: _ChunkState,
        identity: torch.Tensor,
    ) -> _ChunkState:
        completed = 0
        while completed < self._config.max_iterations:
            count = min(
                self._chunk_size,
                self._config.max_iterations - completed,
            )
            state = self._run_chunk(state, identity, count)
            completed += count
            if not bool(state.active.any()):
                break
        return state

    def _run_cuda_chunks(
        self,
        state: _ChunkState,
        identity: torch.Tensor,
    ) -> _ChunkState:
        check_stream = torch.cuda.Stream(device=state.x.device)
        pending: deque[_PendingCheck] = deque()
        completed = 0
        while completed < self._config.max_iterations:
            count = min(
                self._chunk_size,
                self._config.max_iterations - completed,
            )
            state = self._run_chunk(state, identity, count)
            completed += count
            pending.append(self._schedule_check(state.active, check_stream))
            if self._poll_converged(pending):
                break
        return state

    @staticmethod
    def _schedule_check(
        active: torch.Tensor,
        check_stream: torch.cuda.Stream,
    ) -> _PendingCheck:
        device_active = active.any()
        reduction_done = torch.cuda.Event()
        reduction_done.record(torch.cuda.current_stream(active.device))
        host_active = torch.empty((), dtype=torch.bool, pin_memory=True)
        with torch.cuda.stream(check_stream):
            check_stream.wait_event(reduction_done)
            host_active.copy_(device_active, non_blocking=True)
            device_active.record_stream(check_stream)
            ready = torch.cuda.Event()
            ready.record(check_stream)
        return _PendingCheck(ready, host_active, device_active)

    @staticmethod
    def _poll_converged(pending: deque[_PendingCheck]) -> bool:
        while pending and pending[0].ready.query():
            check = pending.popleft()
            if not bool(check.host_active):
                return True
        return False

    def _run_chunk(
        self,
        state: _ChunkState,
        identity: torch.Tensor,
        count: int,
    ) -> _ChunkState:
        (
            x,
            objective,
            gradient,
            hessian,
            iterations,
            evaluations,
            converged,
            wolfe,
            active,
        ) = state
        for _iteration in range(count):
            direction = -torch.bmm(hessian, gradient.unsqueeze(-1)).squeeze(-1)
            derivative = (gradient * direction).sum(dim=-1)
            reset = active & (derivative >= 0.0)
            hessian = torch.where(reset[:, None, None], identity, hessian)
            direction = torch.where(reset[:, None], -gradient, direction)
            line = self._strong_wolfe(
                x,
                objective,
                gradient,
                direction,
                active,
            )
            evaluations = evaluations + line.evaluations
            accepted = active & line.accepted
            wolfe = wolfe & (~active | line.accepted)
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
            iterations = iterations + accepted.to(iterations.dtype)
            newly_converged = accepted & (
                self._norm(gradient) <= self._config.tolerance
            )
            converged = converged | newly_converged
            stagnant = accepted & (
                self._norm(step) <= self._config.step_tolerance
            )
            active = accepted & ~converged & ~stagnant
        return _ChunkState(
            x,
            objective,
            gradient,
            hessian,
            iterations,
            evaluations,
            converged,
            wolfe,
            active,
        )
