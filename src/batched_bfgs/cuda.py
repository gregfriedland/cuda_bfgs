"""Python binding for the custom CUDA BFGS kernel."""

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

from batched_bfgs.base import Bfgs
from batched_bfgs.models import BfgsConfig, OptimizationResult
from batched_bfgs.objective import ObjectiveType


class CudaBfgs(Bfgs):
    """Run one fused fixed-dimensional optimization per CUDA thread."""

    def __init__(
        self,
        config: BfgsConfig,
        objective: ObjectiveType = ObjectiveType.EXTENDED_ROSENBROCK,
    ) -> None:
        """Initialize the optimizer without compiling the extension.

        Args:
            config: Shared numerical configuration.
            objective: Analytic objective implemented by the CUDA extension.

        """
        # Store the shared configuration for the single supported objective.
        self._config = config

        # Reject invalid public inputs before loading the extension.
        if not isinstance(objective, ObjectiveType):
            raise TypeError("objective must be an ObjectiveType")
        self._extension: ModuleType | None = None

    def compile(self, verbose: bool = True) -> None:
        """Compile and load the CUDA extension for the visible GPU.

        Args:
            verbose: Whether the extension builder should emit build output.

        """
        # Require a visible device before resolving its architecture.
        if not torch.cuda.is_available():
            raise RuntimeError("CudaBfgs requires a CUDA device")
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

        # Configure the C++ and CUDA extension sources.
        source_dir = Path(__file__).resolve().parent / "csrc"
        cuda_flags = ["-O3", "-lineinfo"]
        if os.environ.get("BFGS_CUDA_RESOURCE_USAGE") == "1":
            cuda_flags.append("--resource-usage")

        # Compile and load the extension into this process.
        self._extension = load(
            name="batched_bfgs_cuda_v3",
            sources=[
                str(source_dir / "bfgs.cpp"),
                str(source_dir / "bfgs_kernel.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_flags,
            with_cuda=True,
            verbose=verbose,
        )

    @torch.no_grad()
    def run(self, starts: torch.Tensor) -> OptimizationResult:
        """Optimize a contiguous CUDA batch.

        Args:
            starts: CUDA tensor with shape ``[batch, 2]`` or ``[batch, 16]``.

        Returns:
            One optimization result per batch member.

        """
        # Validate device, shape, batch size, and numeric type.
        if not starts.is_cuda:
            raise ValueError("starts must be on a CUDA device")
        if starts.ndim != 2 or starts.shape[1] not in (2, 16):
            raise ValueError("starts must have shape [batch, 2] or [batch, 16]")
        if starts.shape[0] == 0:
            raise ValueError("starts must contain at least one batch member")
        if starts.dtype not in (torch.float32, torch.float64):
            raise ValueError("starts must use float32 or float64")

        # Compile lazily and assert that loading succeeded.
        if self._extension is None:
            self.compile()
        if self._extension is None:
            raise RuntimeError("CUDA extension failed to load")

        # Invoke the fused kernel with the shared numerical configuration.
        values = self._extension.optimize(
            starts.contiguous(),
            self._config.c1,
            self._config.c2,
            self._config.tolerance,
            self._config.step_tolerance,
            self._config.curvature_eps,
            self._config.initial_step,
            self._config.maximum_step,
            self._config.max_iterations,
            self._config.max_bracket_iterations,
            self._config.max_zoom_iterations,
        )
        return OptimizationResult(*values)
