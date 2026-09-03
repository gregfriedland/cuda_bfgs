"""Standalone Flyte task for the G4 BFGS benchmark."""

import json

import flyte
import torch
from kubernetes.client import (
    V1Container,
    V1PodSpec,
    V1ResourceRequirements,
    V1SecurityContext,
    V1Toleration,
)

from batched_bfgs.benchmark import BenchmarkRunner

CUDA_BASE_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"

BFGS_IMAGE = (
    flyte.Image.from_base(CUDA_BASE_IMAGE)
    .clone(
        name="batched-bfgs",
        extendable=True,
    )
    .with_pip_packages(
        "flyte==2.5.20",
        "kubernetes>=32,<34",
        "ninja>=1.11,<2",
        "pydantic>=2.10,<3",
    )
)

G4_POD = flyte.PodTemplate(
    primary_container_name="primary",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                resources=V1ResourceRequirements(
                    requests={
                        "cpu": "47000m",
                        "memory": "169440M",
                        "ephemeral-storage": "129.69G",
                        "nvidia.com/gpu": "1",
                    },
                    limits={
                        "cpu": "48000m",
                        "memory": "169440M",
                        "ephemeral-storage": "129.69G",
                        "nvidia.com/gpu": "1",
                    },
                ),
                security_context=V1SecurityContext(privileged=True),
            ),
        ],
        node_selector={
            "cloud.google.com/gke-accelerator": "nvidia-rtx-pro-6000",
            "node.kubernetes.io/instance-type": "g4-standard-48",
        },
        tolerations=[
            V1Toleration(
                effect="NoSchedule",
                key="nvidia.com/gpu",
                operator="Equal",
                value="present",
            ),
        ],
    ),
)

BFGS_ENVIRONMENT = flyte.TaskEnvironment(
    name="run",
    image=BFGS_IMAGE,
    interruptible=False,
    pod_template=G4_POD,
)


@BFGS_ENVIRONMENT.task(retries=0)
async def test(batch_sizes: list[int], repeats: int) -> str:
    """Run the three-way BFGS benchmark on one G4 GPU.

    Args:
        batch_sizes: Batch sizes for the vectorized and CUDA implementations.
        repeats: Measured timing repetitions per implementation and batch.

    Returns:
        JSON benchmark report.

    """
    if not torch.cuda.is_available():
        raise RuntimeError("Flyte task was not assigned a CUDA device")
    report = BenchmarkRunner(batch_sizes=batch_sizes, repeats=repeats).run(
        torch.device("cuda"),
    )
    return json.dumps(report, indent=2, sort_keys=True)
