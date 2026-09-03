#!/usr/bin/env python3
# Purpose: Create, stop, or start the persistent G4 Spot benchmark VM.
# Usage: .venv/bin/python -m scripts.manage_g4_spot_vm ACTION
#   --project PROJECT --account ACCOUNT
"""Manage the lifecycle of the GCP G4 Spot benchmark VM."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class ManagerError(RuntimeError):
    """Report an invalid configuration or unsupported VM state."""


class G4SpotVmManager:
    """Create, stop, or start one persistent G4 Spot VM."""

    MACHINE_TYPE = "g4-standard-48"

    def __init__(
        self,
        *,
        action: str,
        account: str,
        boot_disk_size_gb: int,
        dry_run: bool,
        instance_name: str,
        project: str,
        region: str,
        zone: str | None,
    ) -> None:
        """Store and validate the requested lifecycle operation."""
        if boot_disk_size_gb < 40:
            raise ManagerError("disk size must be at least 40 GB")
        if zone is not None and not zone.startswith(f"{region}-"):
            raise ManagerError(f"zone {zone} is not in region {region}")
        gcloud = shutil.which("gcloud")
        if gcloud is None:
            raise ManagerError("gcloud is not installed")
        self.action = action
        self.account = account
        self.boot_disk_size_gb = boot_disk_size_gb
        self.dry_run = dry_run
        self.instance_name = instance_name
        self.project = project
        self.region = region
        self.zone = zone
        self.gcloud = gcloud

    def run_command(
        self,
        command: list[str],
        *,
        capture_output: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run one gcloud command and surface a concise failure."""
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            output = (result.stdout or "") + (result.stderr or "")
            raise ManagerError(
                output.strip() or f"command exited {result.returncode}"
            )
        return result

    def gcloud_command(self, *arguments: str) -> list[str]:
        """Build a project-scoped gcloud command with explicit identity."""
        return [
            self.gcloud,
            f"--account={self.account}",
            f"--project={self.project}",
            *arguments,
        ]

    def verify_access(self) -> None:
        """Verify the requested credential and project before mutation."""
        credential = self.run_command(
            [
                self.gcloud,
                "auth",
                "list",
                f"--filter=account={self.account}",
                "--format=value(account)",
            ],
            capture_output=True,
        ).stdout.strip()
        if credential != self.account:
            raise ManagerError(
                f"authenticate first: gcloud auth login {self.account}"
            )
        self.run_command(
            [
                self.gcloud,
                f"--account={self.account}",
                "projects",
                "describe",
                self.project,
                "--format=value(projectId)",
            ],
            capture_output=True,
        )

    def discover_create_zone(self) -> str:
        """Find the first available regional zone advertising G4."""
        result = self.run_command(
            self.gcloud_command(
                "compute",
                "zones",
                "list",
                f"--filter=region.basename()={self.region} AND status=UP",
                "--format=value(name)",
            ),
            capture_output=True,
        )
        for candidate in sorted(result.stdout.splitlines()):
            described = self.run_command(
                self.gcloud_command(
                    "compute",
                    "machine-types",
                    "describe",
                    self.MACHINE_TYPE,
                    f"--zone={candidate}",
                ),
                capture_output=True,
                check=False,
            )
            if described.returncode == 0:
                return candidate
        raise ManagerError(
            f"{self.MACHINE_TYPE} is unavailable in {self.region}"
        )

    def discover_existing_zone(self) -> str:
        """Resolve an exact-name VM in the requested region."""
        result = self.run_command(
            self.gcloud_command(
                "compute", "instances", "list", "--format=json"
            ),
            capture_output=True,
        )
        instances: list[dict[str, Any]] = json.loads(result.stdout)
        zones = [
            str(instance["zone"]).rsplit("/", 1)[-1]
            for instance in instances
            if instance.get("name") == self.instance_name
            and str(instance.get("zone", ""))
            .rsplit("/", 1)[-1]
            .startswith(f"{self.region}-")
        ]
        if not zones:
            raise ManagerError(f"instance not found in region {self.region}")
        if len(zones) > 1:
            raise ManagerError("instance name exists in multiple zones")
        return zones[0]

    def resolve_zone(self) -> str:
        """Return the explicit zone or discover the required one."""
        if self.zone is not None:
            return self.zone
        if self.action == "create":
            return self.discover_create_zone()
        return self.discover_existing_zone()

    def create_command(self, zone: str) -> list[str]:
        """Build the safety-constrained G4 Spot create command."""
        script_directory = Path(__file__).resolve().parent
        startup_script = script_directory / "g4_startup.sh"
        shutdown_script = script_directory / "g4_shutdown.sh"
        if not startup_script.is_file():
            raise ManagerError(f"missing startup script: {startup_script}")
        if not shutdown_script.is_file():
            raise ManagerError(f"missing shutdown script: {shutdown_script}")
        self.run_command(
            self.gcloud_command(
                "compute",
                "machine-types",
                "describe",
                self.MACHINE_TYPE,
                f"--zone={zone}",
            ),
            capture_output=True,
        )
        return self.gcloud_command(
            "compute",
            "instances",
            "create",
            self.instance_name,
            f"--zone={zone}",
            f"--machine-type={self.MACHINE_TYPE}",
            "--image-family=ubuntu-2404-lts-amd64",
            "--image-project=ubuntu-os-cloud",
            "--boot-disk-type=hyperdisk-balanced",
            f"--boot-disk-size={self.boot_disk_size_gb}GB",
            "--boot-disk-provisioned-iops=3000",
            "--boot-disk-provisioned-throughput=140",
            "--no-boot-disk-auto-delete",
            "--provisioning-model=SPOT",
            "--instance-termination-action=STOP",
            "--maintenance-policy=TERMINATE",
            "--no-restart-on-failure",
            "--no-service-account",
            "--no-scopes",
            "--no-shielded-secure-boot",
            "--shielded-vtpm",
            "--shielded-integrity-monitoring",
            "--metadata-from-file="
            f"startup-script={startup_script},shutdown-script={shutdown_script}",
        )

    def existing_vm_command(self, zone: str) -> list[str] | None:
        """Build a stop/start command, handling an already-achieved state."""
        status = self.run_command(
            self.gcloud_command(
                "compute",
                "instances",
                "describe",
                self.instance_name,
                f"--zone={zone}",
                "--format=value(status)",
            ),
            capture_output=True,
        ).stdout.strip()
        if self.action == "stop":
            if status == "TERMINATED":
                print(f"{self.instance_name} is already stopped in {zone}")
                return None
            if status != "RUNNING":
                raise ManagerError(f"cannot stop VM in state {status}")
        else:
            if status == "RUNNING":
                print(f"{self.instance_name} is already running in {zone}")
                return None
            if status != "TERMINATED":
                raise ManagerError(f"cannot start VM in state {status}")
        return self.gcloud_command(
            "compute",
            "instances",
            self.action,
            self.instance_name,
            f"--zone={zone}",
        )

    def execute(self) -> None:
        """Validate, build, and execute the requested lifecycle action."""
        self.verify_access()
        zone = self.resolve_zone()
        command = (
            self.create_command(zone)
            if self.action == "create"
            else self.existing_vm_command(zone)
        )
        if command is None:
            return
        print(
            f"Action: {self.action}\nProject: {self.project}\n"
            f"Region: {self.region}\nZone: {zone}\n"
            f"Instance: {self.instance_name}"
        )
        if self.dry_run:
            print(shlex.join(command))
            return
        self.run_command(command)
        self.run_command(
            self.gcloud_command(
                "compute",
                "instances",
                "describe",
                self.instance_name,
                f"--zone={zone}",
                "--format=table(name,zone.basename(),status,"
                "machineType.basename(),scheduling.provisioningModel)",
            )
        )


class Cli:
    """Parse arguments for the G4 Spot VM manager."""

    @staticmethod
    def run() -> int:
        """Execute the requested VM lifecycle action."""
        parser = argparse.ArgumentParser(
            description="Create, stop, or start a GCP g4-standard-48 Spot VM.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "action",
            choices=("create", "stop", "start"),
            help="VM lifecycle action",
        )
        parser.add_argument("--project", required=True, help="GCP project ID")
        parser.add_argument(
            "--account", required=True, help="authenticated gcloud account"
        )
        parser.add_argument("--region", default="us-east5", help="GCP region")
        parser.add_argument("--zone", help="exact zone; otherwise discover it")
        parser.add_argument(
            "--name",
            default="bfgs-g4-spot",
            dest="instance_name",
            help="instance name",
        )
        parser.add_argument(
            "--boot-disk-size-gb",
            type=int,
            default=50,
            help="boot disk size in GB",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="print the mutating command without running it",
        )
        arguments = parser.parse_args()
        try:
            G4SpotVmManager(
                action=arguments.action,
                account=arguments.account,
                boot_disk_size_gb=arguments.boot_disk_size_gb,
                dry_run=arguments.dry_run,
                instance_name=arguments.instance_name,
                project=arguments.project,
                region=arguments.region,
                zone=arguments.zone,
            ).execute()
        except (ManagerError, json.JSONDecodeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(Cli.run())
