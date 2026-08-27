"""Rented GPUs behind one seam: `provider.py` is the contract and the
registry, `runpod.py` the first vendor. `tools/cloud-gpu.py` is the
runbook as a command (offers → launch → push → run → pull → terminate);
docs/34 is the runbook as prose."""

from rq_pipeline.cloud.provider import (
    DEFAULT_PROVIDER,
    Action,
    GpuOffer,
    GpuProvider,
    Machine,
    MachineSpec,
    MissingCredentialError,
    ProviderError,
    SshEndpoint,
    Tier,
    credential,
    provider,
    providers,
    resolve,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "Action",
    "GpuOffer",
    "GpuProvider",
    "Machine",
    "MachineSpec",
    "MissingCredentialError",
    "ProviderError",
    "SshEndpoint",
    "Tier",
    "credential",
    "provider",
    "providers",
    "resolve",
]
