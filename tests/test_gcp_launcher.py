"""CLI contract for the GCP Spot VM launcher."""

import subprocess
from pathlib import Path


class TestGcpLauncher:
    """Check the launcher's non-mutating interface."""

    def test_help_has_required_defaults(self) -> None:
        """Help exposes the requested project, region, account, and machine."""
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["bash", str(root / "scripts/manage_g4_spot_vm.sh"), "--help"],
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
        launcher = (root / "scripts/manage_g4_spot_vm.sh").read_text()
        required_arguments = (
            'machine_type="g4-standard-48"',
            "--boot-disk-type=hyperdisk-balanced",
            "--provisioning-model=SPOT",
            "--instance-termination-action=STOP",
            "--no-boot-disk-auto-delete",
            "--maintenance-policy=TERMINATE",
            "--no-restart-on-failure",
        )
        for argument in required_arguments:
            assert argument in launcher
        assert "boot_disk_size_gb=50" in launcher
        assert "compute instances stop" in launcher
        assert "compute instances start" in launcher
