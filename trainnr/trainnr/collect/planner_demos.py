"""Demonstrations from the planner expert (docs/66 §3 source 2, D3).

One episode: seat the scene at its paired start, read the object's pose
and size off the live model (privileged - demo generation may), let
`PickPlacePlanner` write the beats for the task's goal, execute them
by chained IK, then hold the final pose until the protocol's horizon
so the referee's settle window sees it. The press is the scripted
press with this as its driver: the same DR draws, cameras, manifests
and referee, the expert stamp naming the planner's knobs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trainnr.collect.choreography import Episode, PickPlacePlanner, PlannerKnobs
from trainnr.collect.scripted_demos import Driver, DrRanges, generate_demos

if TYPE_CHECKING:
    from trainnr.collect.press import DemoBatch, PressFeed
    from trainnr.tasks.task import Task


@dataclass(frozen=True)
class PlannerRig:
    """What a rig tells the planner about one task: its gripper on a
    compiled model, the goal read off the live scene, the object body."""

    gripper: Callable[[Any], Any]
    goal: Callable[[Any, Any], Any]
    object_body: str


def object_pose(model: Any, data: Any, body: str) -> tuple[Any, Any]:
    """The body's centre and half-extents (its first geom's box size)."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    geom = int(model.body_geomadr[body_id])
    return np.array(data.xpos[body_id]), np.array(model.geom_size[geom])


def drive_planned(  # noqa: PLR0913 - one episode's facts, named
    model: Any,
    stepper: Any,
    *,
    gripper: Any,
    goal: Any,
    object_body: str,
    on_control: Any = None,
    knobs: PlannerKnobs | None = None,
) -> Any:
    """Run the planner on a seated stepper to its horizon; returns the
    commanded rows (ticks, nu)."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    mujoco.mj_forward(model, stepper.data)
    # The home ctrl: the seated pose's joint positions (position servos).
    ctrl = np.zeros(model.nu)
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        ctrl[actuator] = stepper.data.qpos[int(model.jnt_qposadr[joint])]
    ctrl[gripper.jaw_ctrl] = gripper.jaw(1.0)
    episode = Episode(stepper, mujoco.MjData(model), on_control, ctrl)

    planner = PickPlacePlanner(knobs)
    xyz, half = object_pose(model, stepper.data, object_body)
    beats = planner.plan(xyz, half, goal, gripper.closing_axis(stepper.data, xyz))
    planner.run(episode, gripper, beats)
    # Hold to the horizon: the referee judges the settle window.
    while not stepper.done:
        episode.advance(0.5, episode.ctrl)
    return np.asarray(episode.actions)


def planned_episode(  # noqa: PLR0913 - one episode's facts, named
    model: Any,
    initial: Any,
    *,
    task: Task,
    gripper: Any,
    goal: Any,
    object_body: str,
    on_control: Any = None,
    knobs: PlannerKnobs | None = None,
) -> tuple[Any, Any, Any]:
    """One planned episode from a seated start on a fresh backend:
    (states, sensors, actions) over the protocol's whole horizon."""
    from trainnr.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

    backend = MuJoCoBackend()
    backend.load_model(model)
    stepper = backend.stepper(initial, task.protocol.steps)
    actions = drive_planned(
        model,
        stepper,
        gripper=gripper,
        goal=goal,
        object_body=object_body,
        on_control=on_control,
        knobs=knobs,
    )
    return stepper.states, stepper.sensors, actions


def planner_driver(rig: PlannerRig, knobs: PlannerKnobs) -> Driver:
    """The press driver: the planner on every attempt's compiled model."""

    def drive(
        task: Task, model: Any, stepper: Any, capture: Callable[[int, Any], None]
    ) -> Any:
        import mujoco  # noqa: PLC0415 - sim extra

        mujoco.mj_forward(model, stepper.data)
        return drive_planned(
            model,
            stepper,
            gripper=rig.gripper(model),
            goal=rig.goal(model, stepper.data),
            object_body=rig.object_body,
            on_control=capture,
            knobs=knobs,
        )

    return drive


def generate_planned_demos(  # noqa: PLR0913 - every knob of the loop, named
    out: Path,
    *,
    task_factory: Callable[[], Task],
    rig: PlannerRig,
    dr: DrRanges,
    basis: str,
    episodes: int,
    seed: int,
    knobs: PlannerKnobs | None = None,
    frame_every: int = 5,
    first_episode: int = 0,
    feed: PressFeed | None = None,
    say: Callable[[str], None] = print,
) -> DemoBatch:
    """Press `episodes` kept planner demonstrations: the scripted
    press's loop with the planner driving, stamped `planner@<knobs>`."""
    knobs = knobs or PlannerKnobs()
    return generate_demos(
        out,
        task_factory=task_factory,
        driver=planner_driver(rig, knobs),
        expert=knobs.stamp,
        dr=dr,
        basis=basis,
        episodes=episodes,
        seed=seed,
        frame_every=frame_every,
        first_episode=first_episode,
        feed=feed,
        say=say,
    )
