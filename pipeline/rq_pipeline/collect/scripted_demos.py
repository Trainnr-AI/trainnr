"""The open-loop scripted-expert press adapter — C1's dataset generator.

The paired study (docs/e2e-research/62) needs the SAME expert pressed
under two different dynamics-draw conditions, everything else equal.
This adapter serves any task whose expert is a step-scheduled policy
`policy(step, sensordata) -> ctrl` (so101's lift/stack/insert family):
per attempt it rebuilds the scene, scales the servo dynamics by a draw
from the CONDITION'S declared ranges, starts from one of the
protocol's own paired trials, runs the expert on the CPU reference
with frames, and lets the task's referee keep or discard. The
condition is carried on every manifest as the dynamics dict + the
basis string — two datasets differing ONLY in those lines is the whole
experimental design.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rq_pipeline.collect.press import DemoBatch, EpisodeManifest, PressResult, press

if TYPE_CHECKING:
    from numpy.random import Generator

    from rq_pipeline.tasks.task import Task

# The servo-DR rule has ONE home (physics/servo_dr.py); this module's
# own copy grew a different joint filter and drifted from the rig's
# (review, 2026-09-01).
from rq_pipeline.physics.servo_dr import scale_servo_dynamics

# {"damping": (low, high), "gain": (low, high)} — the ONLY thing the
# study's two conditions are allowed to disagree on.
DrRanges = dict[str, tuple[float, float]]


def _attempt(  # noqa: PLR0913 - bound by the closure below, not callers
    task_factory: Callable[[], Task],
    policy: Callable[[int, Any], Any],
    rng: Generator,
    *,
    dr: DrRanges,
    basis: str,
    frame_every: int,
) -> PressResult:
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    from rq_pipeline.physics.mujoco_backend import (  # noqa: PLC0415
        MuJoCoBackend,
        keyframe_state,
    )

    task = task_factory()
    dynamics = {
        param: float(rng.uniform(low, high)) for param, (low, high) in dr.items()
    }
    scale_servo_dynamics(
        task.spec,
        damping_scale=dynamics["damping"],
        gain_scale=dynamics["gain"],
    )
    model = task.spec.compile()
    backend = MuJoCoBackend()
    backend.load_model(model)
    protocol = task.protocol
    home = (
        keyframe_state(model, protocol.home)
        if protocol.home
        else backend.default_initial_state()
    )
    trial = int(rng.integers(protocol.trials))
    initial = protocol.perturb(trial, home)
    draws: dict[str, Any] = {"trial": trial}

    # frame_every=0: no renderer at all — headless generation and the
    # unit tests (a GL context is a per-box concern, not the physics').
    camera = task.cameras[0] if frame_every else None
    renderer = (
        mujoco.Renderer(model, height=camera.height, width=camera.width)
        if camera
        else None
    )
    frames: list[tuple[int, Any]] = []
    interval = protocol.control_interval
    ticks = protocol.steps // interval
    actions = np.empty((ticks, model.nu))
    stepper = backend.stepper(initial, protocol.steps)
    try:
        for tick in range(ticks):
            control = np.asarray(
                policy(tick * interval, stepper.data.sensordata), dtype=float
            )
            actions[tick] = control
            stepper.advance(control, interval)
            if renderer is not None and tick % frame_every == 0:
                renderer.update_scene(stepper.data, camera=camera.camera_name)
                frames.append((tick * interval, renderer.render().copy()))
    finally:
        if renderer is not None:
            renderer.close()
    succeeded = bool(protocol.success(stepper.states, stepper.sensors))
    return PressResult(
        succeeded,
        dynamics,
        draws,
        [],
        basis,
        stepper.states,
        stepper.sensors,
        actions,
        frames,
    )


def generate_scripted_demos(  # noqa: PLR0913 - every knob of the loop, named
    out: Path,
    *,
    task_factory: Callable[[], Task],
    policy: Callable[[int, Any], Any],
    expert: str,
    dr: DrRanges,
    basis: str,
    episodes: int,
    seed: int,
    frame_every: int = 5,
    first_episode: int = 0,
    say: Callable[[str], None] = print,
) -> DemoBatch:
    """Press `episodes` kept demonstrations of the open-loop `policy`
    on `task_factory`'s task, dynamics drawn per episode from `dr`
    (the condition), keep/discard by the task's own referee."""
    from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

    task = task_factory()
    backend = MuJoCoBackend()
    backend.load_spec(task_factory().spec)
    protocol = task.protocol
    control_hz = round(1.0 / (backend.model.opt.timestep * protocol.control_interval))
    if sorted(dr) != ["damping", "gain"]:
        raise ValueError(
            f"dr must declare exactly damping and gain ranges, got {sorted(dr)}"
        )

    def attempt_fn(rng: Generator, *, frame_every: int) -> PressResult:
        return _attempt(
            task_factory, policy, rng, dr=dr, basis=basis, frame_every=frame_every
        )

    def manifest_fn(result: PressResult, attempt: int) -> EpisodeManifest:
        return EpisodeManifest(
            seed=seed,
            attempt=attempt,
            task=getattr(task, "stamp", None)
            or getattr(task, "name", type(task).__name__),
            expert=expert,
            instrument=backend.instrument,
            dynamics=result.dynamics,
            draws=result.draws,
            retries=result.retries,
            control_hz=control_hz,
            frame_every_control_ticks=frame_every,
            dynamics_basis=result.dynamics_basis,
        )

    return press(
        out,
        task=getattr(task, "stamp", None) or getattr(task, "name", type(task).__name__),
        expert=expert,
        instrument=backend.instrument,
        control_hz=control_hz,
        attempt_fn=attempt_fn,
        episodes=episodes,
        seed=seed,
        frame_every=frame_every,
        first_episode=first_episode,
        manifest_fn=manifest_fn,
        say=say,
    )
