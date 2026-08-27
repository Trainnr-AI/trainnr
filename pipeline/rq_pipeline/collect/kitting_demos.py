"""Scripted kitting demonstrations — T5's data source — as a function.

Each attempt: draw both parts uniformly over the task's DECLARED band
(`KittingSpec.part_spawn`, the band the harness evaluates on; until
2026-08-27 a tool drew the front half only, a workaround for corners
the expert could not grasp), draw the dynamics within ±`dr_span` of the
bundle's values (domain randomisation centred on the identified
parameters), run the scripted choreography, and keep the episode ONLY
if the task's own referee scores it a success — the same judge policies
will face. A kept episode is written by `kitting_export.write_episode`
with its manifest (seed, draws, DR scales, retries, the expert's stamp).

`first_episode` numbers a shard: N generators with disjoint ranges and
seeds fill ONE batch directory in parallel (the render is the cost — 95
s/episode on the WSL card, 45 on a rented one; six EGL contexts on WSL
livelocked, three did not — docs/07 2026-08-28). The sim extra is
imported when the function runs, so the package imports without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.collect.kitting_export import Manifest, write_episode
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

# The DR sampler can in principle draw only unreachable dynamics: give up
# after this many draws per wanted episode.
ATTEMPTS_PER_EPISODE = 20
# ±30 %: measured 2026-08-26 with the gain scaled correctly (BOTH kp
# terms) the expert keeps 8/10 at 0.10 and 10/10 at 0.30. An earlier
# one-sided gain scale moved the setpoints and read as "nominal only".
DR_SPAN = 0.30
Say = Callable[[str], None]


@dataclass(frozen=True)
class DemoBatch:
    """What a generator run produced: the accounting a chain log and a
    docs entry quote."""

    out: Path
    wanted: int
    kept: int
    attempts: int
    first_episode: int
    expert: str
    task: str

    @property
    def complete(self) -> bool:
        return self.kept >= self.wanted

    @property
    def last_episode(self) -> int:
        return self.first_episode + self.kept - 1


@dataclass(frozen=True)
class Attempt:
    succeeded: bool
    damping_scale: float
    gain_scale: float
    draws: dict[str, tuple[float, float]]
    stats: KittingStats
    states: Any = None
    sensors: Any = None
    actions: Any = None
    frames: list[tuple[int, Any]] | None = None


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
    import numpy as np  # noqa: PLC0415 - sim extra

    task = build_kitting(spec=spec)
    expert = expert_stamp()
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    say(f"expert {expert}, task {task.stamp}")
    limit = ATTEMPTS_PER_EPISODE * episodes if max_attempts is None else max_attempts
    kept = attempts = 0
    while kept < episodes and attempts < limit:
        attempts += 1
        result = _attempt(
            task,
            spec,
            rng,
            dr_span=dr_span,
            frame_every=frame_every,
            attempt=attempts,
            say=say,
        )
        say(
            f"attempt {attempts}: {'KEEP' if result.succeeded else 'discard'} "
            f"(damping x{result.damping_scale:.2f}, gain x{result.gain_scale:.2f}, "
            f"retries {len(result.stats.retries)})"
        )
        if not result.succeeded:
            continue
        write_episode(
            out,
            first_episode + kept,
            states=result.states,
            sensors=result.sensors,
            actions=result.actions,
            frames=result.frames or [],
            manifest=Manifest(
                seed=seed,
                attempt=attempts,
                draws=result.draws,
                dr_span=dr_span,
                damping_scale=result.damping_scale,
                gain_scale=result.gain_scale,
                retries=result.stats.retries,
                control_hz=task.control_hz,
                frame_every_control_ticks=frame_every,
                expert=expert,
            ),
        )
        kept += 1
    batch = DemoBatch(out, episodes, kept, attempts, first_episode, expert, task.stamp)
    say(
        f"kept {kept}/{attempts} episodes -> {out} "
        f"(episodes {first_episode}..{batch.last_episode})"
        if kept
        else f"gave up after {attempts} attempts: kept 0/{episodes} -> {out}"
    )
    return batch


def _attempt(  # noqa: PLR0913 - the loop's bindings, each named
    task: Any,
    spec: KittingSpec,
    rng: Any,
    *,
    dr_span: float,
    frame_every: int,
    attempt: int,
    say: Say,
) -> Attempt:
    import mujoco  # noqa: PLC0415 - sim extra

    from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: PLC0415

    # Domain randomisation: recompile the scene with scaled dynamics
    # (`scale_dynamics` scales BOTH kp terms of a position servo).
    scene = build_kitting(spec=spec).spec
    damping_scale = float(1.0 + rng.uniform(-dr_span, dr_span))
    gain_scale = float(1.0 + rng.uniform(-dr_span, dr_span))
    scale_dynamics(scene, damping_scale=damping_scale, gain_scale=gain_scale)
    model = scene.compile()
    initial = keyframe_state(model, task.protocol.home)
    draws: dict[str, tuple[float, float]] = {}
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
        say(f"attempt {attempt}: discard (IK: {error})")
        return Attempt(False, damping_scale, gain_scale, draws, stats)
    finally:
        renderer.close()
    succeeded = bool(task.protocol.success(states, sensors))
    return Attempt(
        succeeded,
        damping_scale,
        gain_scale,
        draws,
        stats,
        states,
        sensors,
        actions,
        frames,
    )
