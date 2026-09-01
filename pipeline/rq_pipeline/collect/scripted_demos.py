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

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rq_pipeline.collect.press import (
    DemoBatch,
    EpisodeManifest,
    PressFeed,
    PressResult,
    press,
)
from rq_pipeline.evaluate.variations import Variation, VariationKeys, draw_random

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


def _task_label(task: Any) -> str:
    """The task's stamp when it has one, its name otherwise — what every
    episode record carries as its `task` field."""
    return str(
        getattr(task, "stamp", None) or getattr(task, "name", type(task).__name__)
    )


def _apply_visuals(
    model: Any, visuals: Sequence[Variation], rng: Generator
) -> dict[str, Any]:
    """Draw every visual knob from `rng` and write it onto the compiled
    model through the engine's appliers — what a knob DOES has one home
    (physics/variations.py, the same appliers evaluation sweeps); the
    press only draws and records. Returns the drawn values, JSON-ready,
    for the manifest."""
    from rq_pipeline.physics.variations import (  # noqa: PLC0415 - sim extra
        apply_key,
        snapshot_all,
    )

    nominal = snapshot_all(model)
    drawn: dict[str, Any] = {}
    for variation in visuals:
        value = draw_random(variation, rng)
        apply_key(model, nominal, variation.key, value)
        drawn[variation.key] = list(value) if isinstance(value, tuple) else value
    return drawn


def _attempt(  # noqa: PLR0913 - bound by the closure below, not callers
    task_factory: Callable[[], Task],
    policy: Callable[[int, Any], Any],
    rng: Generator,
    *,
    dr: DrRanges,
    basis: str,
    frame_every: int,
    visuals: Sequence[Variation] = (),
    visual_basis: str = "",
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
    drawn = _apply_visuals(model, visuals, rng) if visuals else {}
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
    # Otherwise EVERY declared camera renders (docs/66 §4: the task
    # declares a rig, the dataset carries the rig), renderers shared
    # across cameras of the same resolution.
    cameras = task.cameras if frame_every else ()
    renderers: dict[tuple[int, int], Any] = {}
    for spec in cameras:
        size = (spec.height, spec.width)
        if size not in renderers:
            renderers[size] = mujoco.Renderer(
                model, height=spec.height, width=spec.width
            )
    camera_frames: dict[str, list[tuple[int, Any]]] = {c.key: [] for c in cameras}
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
            if cameras and tick % frame_every == 0:
                for spec in cameras:
                    renderer = renderers[(spec.height, spec.width)]
                    renderer.update_scene(stepper.data, camera=spec.camera_name)
                    camera_frames[spec.key].append(
                        (tick * interval, renderer.render().copy())
                    )
    finally:
        for renderer in renderers.values():
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
        visuals=drawn,
        visual_basis=visual_basis if drawn else "",
        camera_frames=camera_frames or None,
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
    visuals: Sequence[Variation] = (),
    visual_basis: str = "",
    feed: PressFeed | None = None,
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
    unknown = [v.key for v in visuals if v.name not in VariationKeys.VISUAL_NAMES]
    if unknown:
        raise ValueError(
            f"visual DR draws only {sorted(VariationKeys.VISUAL_NAMES)} knobs "
            f"(dynamics draws have their own contract, `dr`); got {unknown}"
        )
    if visuals and not visual_basis:
        raise ValueError(
            "visual DR needs a visual_basis naming where the ranges came from"
        )

    def attempt_fn(rng: Generator, *, frame_every: int) -> PressResult:
        return _attempt(
            task_factory,
            policy,
            rng,
            dr=dr,
            basis=basis,
            frame_every=frame_every,
            visuals=visuals,
            visual_basis=visual_basis,
        )

    def manifest_fn(result: PressResult, attempt: int) -> EpisodeManifest:
        return EpisodeManifest(
            seed=seed,
            attempt=attempt,
            task=_task_label(task),
            expert=expert,
            instrument=backend.instrument,
            dynamics=result.dynamics,
            draws=result.draws,
            retries=result.retries,
            control_hz=control_hz,
            frame_every_control_ticks=frame_every,
            dynamics_basis=result.dynamics_basis,
            visuals=result.visuals,
            visual_basis=result.visual_basis,
        )

    return press(
        out,
        task=_task_label(task),
        expert=expert,
        instrument=backend.instrument,
        control_hz=control_hz,
        attempt_fn=attempt_fn,
        episodes=episodes,
        seed=seed,
        frame_every=frame_every,
        first_episode=first_episode,
        manifest_fn=manifest_fn,
        feed=feed,
        say=say,
    )
