"""CLI contract for the GCP Spot VM launcher."""

import subprocess
import sys
from pathlib import Path


class TestGcpLauncher:
    """Check the launcher's non-mutating interface."""

    def test_help_has_required_defaults(self) -> None:
        """Help exposes the requested project, region, account, and machine."""
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "scripts.manage_g4_spot_vm", "--help"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "g4-standard-48" in result.stdout
        assert "--project PROJECT" in result.stdout
        assert "default: us-east5" in result.stdout
        assert "--account ACCOUNT" in result.stdout
        assert "create" in result.stdout
        assert "stop" in result.stdout
        assert "start" in result.stdout

    def test_create_command_is_spot_g4(self) -> None:
        """The create command uses the required G4 Spot safety contract."""
        root = Path(__file__).resolve().parents[1]
        launcher_path = root / "scripts/manage_g4_spot_vm.py"
        launcher = launcher_path.read_text()
        assert launcher_path.stat().st_mode & 0o111
        assert not (root / "scripts/manage_g4_spot_vm.sh").exists()
        required_arguments = (
            'MACHINE_TYPE = "g4-standard-48"',
            "--boot-disk-type=hyperdisk-balanced",
            "--provisioning-model=SPOT",
            "--instance-termination-action=STOP",
            "--no-boot-disk-auto-delete",
            "--maintenance-policy=TERMINATE",
            "--no-restart-on-failure",
        )
        for argument in required_arguments:
            assert argument in launcher
        assert "default=50" in launcher
        assert '("create", "stop", "start")' in launcher
        assert "shutdown-script={shutdown_script}" in launcher

    def test_remote_runner_has_durable_markers(self) -> None:
        """Startup and benchmark scripts persist explicit terminal state."""
        root = Path(__file__).resolve().parents[1]
        startup = (root / "scripts/g4_startup.sh").read_text()
        runner = (root / "scripts/run_benchmark_remote.sh").read_text()
        compiled_path = root / "scripts/run_compiled_benchmark_remote.sh"
        compiled_runner = compiled_path.read_text()
        profile_path = root / "scripts/run_cuda_profile_remote.sh"
        profile_runner = profile_path.read_text()
        service = (root / "scripts/bfgs-benchmark.service").read_text()
        assert "bfgs-g4-ready" in startup
        assert "bfgs-g4-failed" in startup
        assert "cuda-toolkit-12-8" in startup
        assert "RUNNING.json" in runner
        assert "DONE.json" in runner
        assert "FAILED.json" in runner
        assert '"extended_rosenbrock|2"' in runner
        assert '"extended_rosenbrock|16"' in runner
        assert '"extended_powell|16"' in runner
        assert "--objective" in runner
        assert "--dimension" in runner
        assert "--state_file" in runner
        assert "timing-state.json" in runner
        assert "WantedBy=multi-user.target" in service
        assert compiled_path.stat().st_mode & 0o111
        assert "COMPILED_DONE.json" in compiled_runner
        assert "COMPILED_RUNNING.json" in compiled_runner
        assert "report-with-compiled.json" in compiled_runner
        assert "batched_bfgs.compiled_benchmark" in compiled_runner
        assert profile_path.stat().st_mode & 0o111
        assert "PROFILE_RUNNING.json" in profile_runner
        assert "PROFILE_DONE.json" in profile_runner
        assert "nsys profile" in profile_runner
        assert "--capture-range=cudaProfilerApi" in profile_runner
        assert "BFGS_CUDA_RESOURCE_USAGE=1" in profile_runner

    def test_benchmark_labels_both_pytorch_implementations(self) -> None:
        """Reports distinguish naive and chunked PyTorch implementations."""
        root = Path(__file__).resolve().parents[1]
        benchmark = (root / "src/batched_bfgs/benchmark.py").read_text()

        assert 'PYTORCH_NAIVE = "pytorch (naive)"' in benchmark
        assert 'PYTORCH_CHUNKED = "pytorch (chunked)"' in benchmark
