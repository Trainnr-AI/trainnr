"""The GPU-provider seam: machines rented by name, the way engines are.

A training run at recipe scale rents a card (docs/31 §5). Which vendor
rents it is the same kind of decision as which physics engine steps
the scene — a backend behind a seam, never a foundation (docs/30 §3.5).
This module is the seam: what an offer looks like, what a machine
looks like, what a provider must do, and a registry that resolves
providers by name (`rq_pipeline.gpu_providers` entry points, the
engine registry's pattern). The built-in is "runpod"; a second vendor
is one module and one entry point. Standard library only; a provider
does its own HTTP.

Credentials never pass through this code as values: `credential()`
reads an environment variable and refuses, naming it, when it is
missing. The repo's `.env` holds them; tools load it, nothing prints it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from rq_pipeline.plugins import load_group

ENTRY_POINT_GROUP = "rq_pipeline.gpu_providers"
# The modules that register the built-in providers; imported by
# `providers()` so a checkout works before its entry points are installed.
BUILTIN_MODULES = ("rq_pipeline.cloud.runpod",)
DEFAULT_PROVIDER = "runpod"


class Tier(str, Enum):
    """Where the hardware sits: the vendor's own datacenter, or a host
    the vendor vets. The same names every vendor uses for the split."""

    SECURE = "SECURE"
    COMMUNITY = "COMMUNITY"


class Action(str, Enum):
    """The state transitions a machine accepts."""

    START = "start"
    STOP = "stop"
    RESTART = "restart"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class GpuOffer:
    """One rentable GPU type at one provider, priced per card-hour."""

    provider: str
    gpu: str  # the provider's identifier, what `MachineSpec.gpu` takes
    name: str
    memory_gb: int
    tier: Tier
    price_per_hour: float | None  # None when the tier does not carry it
    availability: str  # the provider's word: NONE / LOW / MEDIUM / HIGH
    cuda_versions: tuple[str, ...] = ()  # host driver versions on offer
    max_count: int | None = None


@dataclass(frozen=True)
class MachineSpec:
    """What to rent: one container image on N cards of one GPU type."""

    name: str
    gpu: str
    image: str
    count: int = 1
    disk_gb: int = 50
    tier: Tier = Tier.SECURE
    env: Mapping[str, str] = field(default_factory=dict)
    ssh: bool = True
    ports: Sequence[str] = ("22/tcp",)  # a published sshd: rsync needs it
    data_centers: Sequence[str] = ()


@dataclass(frozen=True)
class SshEndpoint:
    """One way onto a machine over SSH, as parts and as the command."""

    host: str
    port: int
    username: str
    command: str


@dataclass(frozen=True)
class Machine:
    """A rented machine as the provider reports it."""

    provider: str
    id: str
    name: str
    status: str  # the provider's lifecycle word
    gpu: str | None
    count: int
    tier: Tier
    cost_per_hour: float
    actions: tuple[Action, ...]
    data_center: str | None = None
    cuda_version: str | None = None
    # `direct` reaches the machine's own sshd (rsync, port forwards);
    # `proxy` is a shell through the vendor. Either is None until the
    # machine can take it.
    ssh_direct: SshEndpoint | None = None
    ssh_proxy: SshEndpoint | None = None

    @property
    def running(self) -> bool:
        return self.status == "RUNNING"


class ProviderError(RuntimeError):
    """The provider refused or failed a request; the message carries
    the provider's own words (status, title, detail)."""


class MissingCredentialError(RuntimeError):
    """An API credential's environment variable is unset."""


def credential(env_var: str, *, provider: str) -> str:
    """The credential in `env_var`, refused by name when it is unset.
    The VALUE is returned to the caller and to nobody else: never log
    it, never put it in a URL, never write it to a manifest."""
    value = os.environ.get(env_var, "").strip()
    if not value:
        raise MissingCredentialError(
            f"{provider} needs {env_var} in the environment — the repo's "
            ".env at the root holds it (tools load that file; nothing prints it)"
        )
    return value


@runtime_checkable
class GpuProvider(Protocol):
    """What a vendor must do to sit behind the seam."""

    name: str

    def offers(
        self, *, tier: Tier = Tier.SECURE, min_cuda: str | None = None
    ) -> list[GpuOffer]:
        """The catalog with live availability, priced for `tier`."""

    def launch(self, spec: MachineSpec) -> Machine:
        """Rent it. Billing starts here."""

    def machine(self, machine_id: str) -> Machine: ...

    def machines(self) -> list[Machine]: ...

    def act(self, machine_id: str, action: Action) -> None:
        """A state transition; TERMINATE is permanent."""

    def logs(self, machine_id: str, *, tail: int = 100) -> str:
        """The container's recent output, as text."""


@dataclass(frozen=True)
class ProviderEntry:
    """A registered provider: its name, its constructor, one line about it."""

    name: str
    build: Callable[..., Any]  # build(**kwargs) -> a GpuProvider
    doc: str


_REGISTRY: dict[str, ProviderEntry] = {}


def provider(name: str, *, doc: str = ""):
    """Decorate a provider class (or factory); refuses a duplicate name."""
    if not name or "/" in name or " " in name:
        raise ValueError(f"provider names are single words, got {name!r}")

    def decorate(build: Callable[..., Any]) -> Callable[..., Any]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing.build is not build:
            raise ValueError(f"provider {name!r} registered twice")
        line = doc or (build.__doc__ or "").strip().split("\n")[0]
        _REGISTRY[name] = ProviderEntry(name=name, build=build, doc=line)
        return build

    return decorate


def providers() -> Mapping[str, ProviderEntry]:
    """Every registered provider: the built-ins plus installed plugins."""
    load_group(ENTRY_POINT_GROUP, BUILTIN_MODULES)
    return dict(_REGISTRY)


def resolve(name: str) -> ProviderEntry:
    """The entry for `name`; an unknown name is refused with the known ones."""
    known = providers()
    if name not in known:
        raise KeyError(f"unknown GPU provider {name!r}; known: {sorted(known)}")
    return known[name]
