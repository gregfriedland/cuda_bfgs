"""Resource contract for the standalone G4 Flyte task."""

from batched_bfgs.flyte_app import G4_POD


class TestFlyteContract:
    """Check that the task requests the intended complete G4 node."""

    def test_g4_standard_48_resources(self) -> None:
        """The task requests one full RTX PRO 6000 G4 node."""
        pod_spec = G4_POD.pod_spec
        assert pod_spec is not None
        assert pod_spec.node_selector == {
            "cloud.google.com/gke-accelerator": "nvidia-rtx-pro-6000",
            "node.kubernetes.io/instance-type": "g4-standard-48",
        }
        assert pod_spec.containers is not None
        resources = pod_spec.containers[0].resources
        assert resources is not None
        assert resources.requests is not None
        assert resources.limits is not None
        assert resources.requests["nvidia.com/gpu"] == "1"
        assert resources.requests["cpu"] == "47000m"
        assert resources.limits["cpu"] == "48000m"
