"""Flyte connection constants shared by submission and monitoring."""

FLYTE_ENDPOINT = "https://rezotx.hosted.unionai.cloud"
FLYTE_PROJECT = "atlas-east5"
FLYTE_DOMAIN = "development"
TERMINAL_PHASES = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"}
