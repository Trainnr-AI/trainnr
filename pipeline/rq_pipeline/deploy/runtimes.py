"""The gate runtimes, by name: what can drive an exported policy through
its manifest and be judged the evaluation's way.

One table feeds the door's refusal, the tool's choices, the record each
runtime writes, the instrument a card names, and the opener the gate
calls — so a runtime is added here and nowhere else. Openers and stacks
are imported when asked for, because the DDS runtime's stack (uinput,
CycloneDDS, Unitree's SDK) exists on Linux only and must fail by name
there, not at import time everywhere.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

    from rq_pipeline.deploy.manifest import Manifest

LINUX = "linux"


class GateRuntime(Protocol):
    """What the gate's trial loop asks of a runtime (`deploy.gate.run_trial`):
    the seven calls, the held command it reads, the instrument string
    the record names, and the command envelope it can reach (None when
    the manifest's ranges are the only bound)."""

    command: np.ndarray

    @property
    def instrument(self) -> str: ...
    @property
    def command_limit(self) -> float | None: ...

    def reset(self) -> None: ...
    def observe(self) -> np.ndarray: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
    def apply(self, action: np.ndarray) -> None: ...
    def base_velocity_b(self) -> np.ndarray: ...
    def fell_over(self) -> bool: ...
    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(base position, base quaternion wxyz, joints in the policy
        order): what the Studio's mirror draws (`deploy/mirror.py`)."""
        ...


Opener = Callable[..., GateRuntime]
StackFactory = Callable[..., AbstractContextManager[Any]]


@dataclass(frozen=True)
class RuntimeSpec:
    """A gate runtime as the registry knows it."""

    name: str
    module: str
    opener: str
    description: str  # what drove the policy, as the record states it
    instrument: str  # the field's name for it, as a card shows it
    record_file: str  # the gate record this runtime writes beside the manifest
    platforms: tuple[str, ...] = ()  # empty: every platform
    needs_assets: bool = True  # the bundle's meshes for the scene
    stack: str | None = None  # "module:attr" of a context that surrounds the gate

    def open(self) -> Opener:
        require_platform(self)
        return getattr(import_module(self.module), self.opener)

    def stack_for(self, manifest: Manifest, **options: Any) -> AbstractContextManager:
        """The processes this runtime needs around the gate, started on
        enter and stopped on exit; nothing for a runtime that is a library."""
        if self.stack is None:
            return nullcontext()
        require_platform(self)
        module, attr = self.stack.split(":", 1)
        return getattr(import_module(module), attr)(manifest, **options)


def require_platform(spec: RuntimeSpec) -> None:
    """Refuse by name a runtime this platform cannot run."""
    if spec.platforms and sys.platform not in spec.platforms:
        raise RuntimeError(
            f"the {spec.name} gate runtime ({spec.description}) runs on "
            f"{', '.join(spec.platforms)} only; this is {sys.platform} "
            f"({platform.machine()})"
        )


RUNTIMES: dict[str, RuntimeSpec] = {
    "mujoco": RuntimeSpec(
        name="mujoco",
        module="rq_pipeline.deploy.runtime",
        opener="open_runtime",
        description="plain MuJoCo + onnxruntime, driven by the manifest alone",
        instrument="plain MuJoCo",
        record_file="gate.json",
    ),
    "dds": RuntimeSpec(
        name="dds",
        module="rq_pipeline.deploy.dds_runtime",
        opener="open_dds_runtime",
        description="Unitree's own simulator and controller over DDS, "
        "commanded from a virtual gamepad",
        instrument="Unitree's simulator and controller over DDS",
        record_file="gate-dds.json",
        platforms=(LINUX,),
        needs_assets=False,
        stack="rq_pipeline.deploy.unitree_stage:UnitreeStack",
    ),
}
DEFAULT_RUNTIME = "mujoco"


def runtime_names() -> tuple[str, ...]:
    return tuple(RUNTIMES)


def runtime_spec(name: str) -> RuntimeSpec:
    """The named runtime, or a refusal naming the legal set."""
    try:
        return RUNTIMES[name]
    except KeyError:
        raise ValueError(
            f"unknown gate runtime {name!r}; one of {', '.join(RUNTIMES)}"
        ) from None


def open_named(
    name: str, manifest: Manifest, *, assets_dir: Path | None
) -> GateRuntime:
    """Build the named runtime for a manifest."""
    return runtime_spec(name).open()(manifest, assets_dir=assets_dir)
