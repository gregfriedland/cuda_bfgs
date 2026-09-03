"""Unified command-line interface for batched BFGS workloads."""

import argparse
import json
from pathlib import Path

import torch

from batched_bfgs.benchmark import BenchmarkRunner
from batched_bfgs.compiled_benchmark import CompiledBenchmarkRunner
from batched_bfgs.objective import ObjectiveType
from batched_bfgs.profile_cuda import CudaProfileWorkload


class Cli:
    """Parse and execute package subcommands."""

    @staticmethod
    def run() -> None:
        """Execute the selected workload."""
        # Parse the shared command and dispatch one workload.
        arguments = Cli._parser().parse_args()
        if arguments.command == "benchmark":
            report = Cli._run_benchmark(arguments)
        else:
            report = Cli._run_cuda_profile(arguments)

        # Emit one machine-readable report for every subcommand.
        print(json.dumps(report, indent=2, sort_keys=True))

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        """Build the package command parser."""
        # Create the required top-level subcommand group.
        parser = argparse.ArgumentParser(prog="batched-bfgs")
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Configure both standard and compiled benchmark modes.
        benchmark = subparsers.add_parser("benchmark")
        benchmark.add_argument(
            "--batch-sizes",
            nargs="+",
            type=int,
            default=[64, 256, 4096, 65536],
        )
        benchmark.add_argument("--repeats", type=int, default=5)
        Cli._add_objective(benchmark, required=False)
        benchmark.add_argument("--dimension", type=int, default=16)
        benchmark.add_argument("--device", default="cuda")
        benchmark.add_argument("--state-file", type=Path)
        benchmark.add_argument(
            "--compiled",
            action="store_true",
            help="run only the fixed-shape compiled implementation",
        )

        # Configure the Nsight-visible CUDA profiling workload.
        profile = subparsers.add_parser("profile-cuda")
        Cli._add_objective(profile, required=True)
        profile.add_argument("--dimension", type=int, default=16)
        profile.add_argument("--batch-size", type=int, default=65536)
        return parser

    @staticmethod
    def _add_objective(
        parser: argparse.ArgumentParser,
        required: bool,
    ) -> None:
        """Add the shared objective argument to a subcommand."""
        # Preserve the benchmark default while requiring explicit profile cases.
        default = None if required else ObjectiveType.EXTENDED_ROSENBROCK
        parser.add_argument(
            "--objective",
            type=ObjectiveType,
            choices=list(ObjectiveType),
            default=default,
            required=required,
        )

    @staticmethod
    def _run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
        """Run the selected standard or compiled benchmark mode."""
        # Route compiled mode through its fixed-shape benchmark runner.
        if arguments.compiled:
            if arguments.state_file is None:
                raise ValueError("--state-file is required with --compiled")
            return Cli._run_compiled_benchmark(arguments)

        # Construct the standard multi-implementation benchmark.
        runner = BenchmarkRunner(
            batch_sizes=arguments.batch_sizes,
            repeats=arguments.repeats,
            objective_name=arguments.objective,
            dimension=arguments.dimension,
        )
        return runner.run(torch.device(arguments.device), arguments.state_file)

    @staticmethod
    def _run_compiled_benchmark(
        arguments: argparse.Namespace,
    ) -> dict[str, object]:
        """Run the fixed-shape compiled benchmark."""
        # Construct the requested compiled benchmark case.
        runner = CompiledBenchmarkRunner(
            batch_sizes=arguments.batch_sizes,
            repeats=arguments.repeats,
            objective_name=arguments.objective,
            dimension=arguments.dimension,
        )
        device = torch.device(arguments.device)
        return runner.run(device, arguments.state_file)

    @staticmethod
    def _run_cuda_profile(arguments: argparse.Namespace) -> dict[str, object]:
        """Run one CUDA profiling workload."""
        # Construct the fixed objective and batch profile.
        workload = CudaProfileWorkload(
            objective_name=arguments.objective,
            dimension=arguments.dimension,
            batch_size=arguments.batch_size,
        )
        return workload.run()


def main() -> None:
    """Run the package command-line interface."""
    Cli.run()


if __name__ == "__main__":
    main()
