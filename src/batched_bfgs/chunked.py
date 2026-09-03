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
        # Store the shared optimizer inputs before validating chunk control.
        super().__init__(config, objective)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = chunk_size

    @torch.no_grad()
    def run(self, starts: torch.Tensor) -> OptimizationResult:
        """Optimize a batch without synchronizing after every iteration."""
        # Validate and copy the common batched input contract.
        if starts.ndim != 2 or starts.shape[0] == 0 or starts.shape[1] == 0:
            raise ValueError("starts must have shape [batch, dimension]")
        x = starts.clone()

        # Initialize one inverse Hessian per batch member.
        batch, dimension = x.shape
        identity = torch.eye(dimension, dtype=x.dtype, device=x.device)
        hessian = identity.expand(batch, -1, -1).clone()

        # Initialize objective values and per-member progress state.
        objective, gradient = self._objective.value_and_gradient(x)
        iterations = torch.zeros(batch, dtype=torch.int32, device=x.device)
        evaluations = torch.zeros_like(iterations)
        converged = self._norm(gradient) <= self._config.tolerance
        wolfe = torch.ones(batch, dtype=torch.bool, device=x.device)

        # Package tensor state for repeated fixed-size chunks.
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

        # Select asynchronous CUDA checks or synchronous CPU checks.
        if starts.is_cuda:
            state = self._run_cuda_chunks(state, identity)
        else:
            state = self._run_cpu_chunks(state, identity)

        # Return the public subset of the internal chunk state.
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
        """Run chunks with synchronous CPU convergence checks."""
        # Execute bounded chunks until all members finish or the budget expires.
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
        """Run chunks with asynchronous CUDA convergence checks."""
        # Maintain a separate stream and queue for nonblocking activity checks.
        check_stream = torch.cuda.Stream(device=state.x.device)
        pending: deque[_PendingCheck] = deque()
        completed = 0

        # Keep launching work while completed checks report active members.
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
        """Schedule a nonblocking device-to-host activity check."""
        # Reduce activity on the compute stream and mark its completion.
        device_active = active.any()
        reduction_done = torch.cuda.Event()
        reduction_done.record(torch.cuda.current_stream(active.device))

        # Copy the scalar into pinned memory on the independent check stream.
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
        """Poll completed activity checks without blocking the host."""
        # Consume only ready checks and stop after the first all-done result.
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
        """Run a fixed number of eager BFGS iterations."""
        # Preserve fixed host control while applying tensor-only iterations.
        for _iteration in range(count):
            state = self._run_iteration(state, identity)
        return state

    def _run_iteration(
        self,
        state: _ChunkState,
        identity: torch.Tensor,
    ) -> _ChunkState:
        """Advance every active batch member by one BFGS iteration."""
        # Unpack immutable tuple state for local tensor updates.
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

        # Compute descent directions and reset invalid inverse Hessians.
        direction = -torch.bmm(hessian, gradient.unsqueeze(-1)).squeeze(-1)
        derivative = (gradient * direction).sum(dim=-1)
        reset = active & (derivative >= 0.0)
        hessian = torch.where(reset[:, None, None], identity, hessian)
        direction = torch.where(reset[:, None], -gradient, direction)

        # Find strong-Wolfe steps for every currently active member.
        line = self._strong_wolfe(
            x,
            objective,
            gradient,
            direction,
            active,
        )

        # Record accepted line searches and their evaluation counts.
        evaluations = evaluations + line.evaluations
        accepted = active & line.accepted
        wolfe = wolfe & (~active | line.accepted)
        step = line.step[:, None] * direction
        change = line.gradient - gradient

        # Update inverse Hessians using accepted step-gradient pairs.
        hessian = self._update_hessian(
            hessian,
            step,
            change,
            accepted,
            identity,
        )

        # Commit accepted coordinates, objectives, and gradients.
        x = torch.where(accepted[:, None], x + step, x)
        objective = torch.where(accepted, line.objective, objective)
        gradient = torch.where(accepted[:, None], line.gradient, gradient)
        iterations = iterations + accepted.to(iterations.dtype)

        # Retain only members that still require optimization.
        newly_converged = accepted & (
            self._norm(gradient) <= self._config.tolerance
        )
        converged = converged | newly_converged
        stagnant = accepted & (self._norm(step) <= self._config.step_tolerance)
        active = accepted & ~converged & ~stagnant

        # Repack the updated tensor state for the next iteration.
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


class CompiledChunkedVectorizedBfgs(ChunkedVectorizedBfgs):
    """Compile one tensor-only iteration while keeping chunk control eager."""

    def __init__(
        self,
        config: BfgsConfig,
        objective: TensorObjective | None = None,
        chunk_size: int = 16,
    ) -> None:
        """Initialize an Inductor-compiled fixed-shape optimizer."""
        # Compile only the tensor iteration while leaving chunk control eager.
        super().__init__(config, objective, chunk_size)
        self._compiled_iteration = torch.compile(
            self._run_iteration,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )

    def _run_chunk(
        self,
        state: _ChunkState,
        identity: torch.Tensor,
        count: int,
    ) -> _ChunkState:
        """Run a fixed number of compiled BFGS iterations."""
        # Reuse the fixed-shape graph for every iteration in this chunk.
        for _iteration in range(count):
            state = self._compiled_iteration(state, identity)
        return state
