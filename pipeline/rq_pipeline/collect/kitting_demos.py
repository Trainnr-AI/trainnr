"""Scripted kitting demonstrations — T5's data source — as a function.

Each attempt: draw both parts uniformly over the task's DECLARED band
(`KittingSpec.part_spawn`, the band the harness evaluates on; until
2026-08-27 a tool drew the front half only, a workaround for corners
the expert could not grasp), draw the dynamics within ±`dr_span` of the
bundle's values (domain randomisation centred on the identified
parameters), run the scripted choreography, and keep the episode ONLY
if the task's own referee scores it a success — the same judge policies
will face.

The keep/discard LOOP lives in `press.press` now (docs/e2e-research/60
§2: the generalization of exactly this file); what remains here is the
kitting-shaped attempt — scene recompile under scaled dynamics, spawn
draws, the scripted expert, the referee — and the LEGACY `Manifest`
this task's committed batches and exporters read. New tasks write the
press's `EpisodeManifest` instead; kitting migrates when its exporters
do.

`first_episode` numbers a shard: N generators with disjoint ranges and
seeds fill ONE batch directory in parallel (the render is the cost — 95
s/episode on the WSL card, 45 on a rented one; six EGL contexts on WSL
livelocked, three did not — docs/07 2026-08-28). The sim extra is
imported when the function runs, so the package imports without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # library + task types as annotations only (sim extra)
    from numpy.random import Generator

    from rq_pipeline.tasks.task import Task

from rq_pipeline.collect.kitting_export import Manifest
from rq_pipeline.collect.press import (
    ATTEMPTS_PER_EPISODE,
    DemoBatch,
    PressResult,
    Say,
    press,
)
from rq_pipeline.tasks.aloha2 import (
    KITTING_SPEC,
    PART_ORDER,
    PART_STATE_SLICE,
    KittingSpec,
    KittingStats,
    build_kitting,
    expert_stamp,
    scale_dynamics,
    scripted_kitting_episode,
)

__all__ = ["ATTEMPTS_PER_EPISODE", "DR_SPAN", "DemoBatch", "generate_demos"]

# ±30 %: measured 2026-08-26 with the gain scaled correctly (BOTH kp
# terms) the expert keeps 8/10 at 0.10 and 10/10 at 0.30. An earlier
# one-sided gain scale moved the setpoints and read as "nominal only".
DR_SPAN = 0.30


def generate_demos(  # noqa: PLR0913 - every knob of the generator, named
    out: Path,
    *,
    episodes: int,
    seed: int,
    dr_span: float = DR_SPAN,
    frame_every: int = 1,
    first_episode: int = 0,
    max_attempts: int | None = None,
    spec: KittingSpec = KITTING_SPEC,
    say: Say = print,
) -> DemoBatch:
    """Keep `episodes` referee-passing demonstrations under `out`, numbered
    from `first_episode`; stop early after `max_attempts` draws."""
    import mujoco  # noqa: PLC0415 - sim extra

    from rq_pipeline.physics.backend import instrument_stamp  # noqa: PLC0415

    task = build_kitting(spec=spec)

    def attempt_fn(rng: Generator, *, frame_every: int) -> PressResult:
        return _attempt(task, spec, rng, dr_span=dr_span, frame_every=frame_every)

    def manifest_fn(result: PressResult, attempt: int) -> Manifest:
        return Manifest(
            seed=seed,
            attempt=attempt,
            draws=result.draws,
            dr_span=dr_span,
            damping_scale=result.dynamics["damping"],
            gain_scale=result.dynamics["gain"],
            retries=result.retries,
            control_hz=task.control_hz,
            frame_every_control_ticks=frame_every,
            expert=expert_stamp(),
        )

    return press(
        out,
        task=task.stamp,
        expert=expert_stamp(),
        instrument=instrument_stamp("mujoco", mujoco.__version__),
        control_hz=task.control_hz,
        attempt_fn=attempt_fn,
        manifest_fn=manifest_fn,
        episodes=episodes,
        seed=seed,
        frame_every=frame_every,
        first_episode=first_episode,
        max_attempts=max_attempts,
        say=say,
    )


def _attempt(
    task: Task,
    spec: KittingSpec,
    rng: Generator,
    *,
    dr_span: float,
    frame_every: int,
) -> PressResult:
    import mujoco  # noqa: PLC0415 - sim extra

    from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: PLC0415

    # Domain randomisation: recompile the scene with scaled dynamics
    # (`scale_dynamics` scales BOTH kp terms of a position servo).
    scene = build_kitting(spec=spec).spec
    dynamics = {
        "damping": float(1.0 + rng.uniform(-dr_span, dr_span)),
        "gain": float(1.0 + rng.uniform(-dr_span, dr_span)),
    }
    basis = f"caller-declared span ±{dr_span:g} (kitting DR, docs/31)"

    scale_dynamics(
        scene, damping_scale=dynamics["damping"], gain_scale=dynamics["gain"]
    )
    model = scene.compile()
    initial = keyframe_state(model, task.protocol.home)
    draws: dict[str, Any] = {}
    for arm in PART_ORDER:
        (x_low, x_high), (y_low, y_high) = spec.part_spawn[arm]
        x, y = float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))
        part = PART_STATE_SLICE[arm]
        initial[part.start], initial[part.start + 1] = x, y
        draws[arm] = (x, y)

    camera = task.cameras[0]
    renderer = mujoco.Renderer(model, height=camera.height, width=camera.width)
    frames: list[tuple[int, Any]] = []
    every = frame_every * task.protocol.control_interval

    # renderer/frames bound as defaults — a late-binding closure would see
    # only the LAST attempt's objects (ruff B023).
    def snap(tick: int, live: Any, renderer=renderer, frames=frames) -> None:
        if tick % every == 0:  # tick is a physics step here
            renderer.update_scene(live, camera=camera.camera_name)
            frames.append((tick, renderer.render().copy()))

    stats = KittingStats()
    try:
        states, sensors, actions = scripted_kitting_episode(
            model, initial, on_control=snap, stats=stats, spec=spec
        )
    except RuntimeError as error:
        # An unreachable draw under this DR sample is a discard, not a
        # crash — the library keeps its honesty (it raises), the
        # generator keeps its throughput (it filters).
        return PressResult(
            False, dynamics, draws, stats.retries, basis, note=f"IK: {error}"
        )
    finally:
        renderer.close()
    succeeded = bool(task.protocol.success(states, sensors))
    return PressResult(
        succeeded,
        dynamics,
        draws,
        stats.retries,
        basis,
        states,
        sensors,
        actions,
        frames,
    )
