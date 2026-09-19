"""The walk families: locomotion environments a project declares by a
spec and rq_mjlab builds in its own venv (docs/77 §2).

A walk is not a `Task` (no scripted expert, no manipulation protocol);
it is the knobs of mjlab's velocity task the loop cares about, named by
their content: the actuator randomization span around the identified
or declared basis, the terrain, the episode length, the paired trials
an evaluation runs. `build_*` returns a `WalkTask` — the spec, its
stamp, the robot's bundle when the project holds one — and compiles the
bundle's model as the honesty check that the robot exists. The
learnability smoke (two environments, two iterations, in rq_mjlab) is
the walk's acceptance, run by `tools/accept-task.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

from rq_pipeline.bundles.bundle import model_file_of
from rq_pipeline.bundles.hashing import content_stamp
from rq_pipeline.bundles.locate import find_bundle
from rq_pipeline.tasks.registry import register, resolve, walk_entries

TERRAINS = ("flat",)  # rough terrain arrives with its own sensors (docs/77)
DEFAULT_SPAN = 0.10  # the walk study's middle arm
DEFAULT_EPISODE_S = 20.0  # mjlab's velocity task
DEFAULT_TRIALS = 40  # the walk verdict's paired trials


@dataclass(frozen=True)
class WalkSpec:
    """A walk's knobs. `dr_span` is the declared randomization span
    around the actuator basis (0 = the basis exactly); `terrain` is
    flat until the rough variant's sensors are wired; `episode_s` the
    episode length; `trials` the paired trials an evaluation runs."""

    dr_span: float = DEFAULT_SPAN
    terrain: str = "flat"
    episode_s: float = DEFAULT_EPISODE_S
    trials: int = DEFAULT_TRIALS

    def __post_init__(self) -> None:
        if self.terrain not in TERRAINS:
            raise ValueError(
                f"terrain is one of {', '.join(TERRAINS)}, got {self.terrain!r}"
            )
        if not 0.0 <= self.dr_span < 1.0:
            raise ValueError(f"dr_span is a fraction in [0, 1), got {self.dr_span}")
        if self.trials < 1 or self.episode_s <= 0:
            raise ValueError("trials must be >= 1 and episode_s > 0")


DEFAULT_WALK = WalkSpec()


@dataclass(frozen=True)
class WalkTask:
    """What a declared walk is, for the doors and the readers: its
    name, the robot it walks (the walk registry's name and the bundle
    when the project holds one), its spec and stamp, the compiled
    model's spec for the scene card."""

    name: str
    robot: str
    task_spec: WalkSpec
    bundle_dir: Path | None
    spec: Any  # an MjSpec of the bundle's model, or None for a simulator asset
    instruction: str = "track the commanded velocity without falling"
    # mjlab's velocity task: physics at 0.005 s, one action per 4 steps.
    control_hz: int = 50

    @property
    def stamp(self) -> str:
        return content_stamp(self.name, asdict(self.task_spec))


def _walk(name: str, robot: str, spec: WalkSpec, *, bundle: str | None) -> WalkTask:
    bundle_dir = find_bundle(bundle) if bundle else None
    mjspec = None
    if bundle:
        if bundle_dir is None:
            raise FileNotFoundError(
                f"no bundle {bundle!r} in the project or the library: onboard the "
                f"robot (onboard_robot) before declaring its walk"
            )
        model_file = model_file_of(bundle_dir)
        if model_file is None:
            raise FileNotFoundError(f"{bundle_dir}: no model file recorded to compile")
        mjspec = mujoco.MjSpec.from_file(str(model_file))
        mjspec.compile()  # the honesty check: the robot exists and builds
    return WalkTask(
        name=name, robot=robot, task_spec=spec, bundle_dir=bundle_dir, spec=mjspec
    )


@register("go2-walk", rig="go2", walk=True)
def build_go2_walk(*, spec: WalkSpec = DEFAULT_WALK) -> WalkTask:
    """The Unitree Go2 walk on the project's onboarded bundle."""
    return _walk("go2-walk", "go2", spec, bundle="go2")


@register("microduck-walk", rig="microduck", walk=True)
def build_microduck_walk(*, spec: WalkSpec = DEFAULT_WALK) -> WalkTask:
    """The microduck walk on its certified bundle."""
    return _walk("microduck-walk", "microduck", spec, bundle="microduck")


@register("go1-walk", rig="go1", walk=True)
def build_go1_walk(*, spec: WalkSpec = DEFAULT_WALK) -> WalkTask:
    """mjlab's own Go1 walk (the asset ships with the simulator)."""
    return _walk("go1-walk", "go1", spec, bundle=None)


def walk_robot(task_id: str) -> str | None:
    """The walk robot of a walk family (its registered rig), else None
    for a family that is not a walk."""
    entry = resolve(task_id)
    return entry.rig if entry.walk else None


def walk_robots() -> tuple[str, ...]:
    """Every robot a walk family is registered for, sorted: what the
    walk doors accept by name (one truth: the task registry)."""
    return tuple(sorted({e.rig for e in walk_entries().values()}))
