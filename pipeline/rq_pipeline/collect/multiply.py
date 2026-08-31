"""Multiplication from seeds — the press's second generation core (B3.2).

docs/e2e-research/60 §3's Mimic contract, run on B3.1's instruments:
a GENERATED candidate is a kept seed episode's actions plus bounded
noise, from a varied initial state, under a fresh dynamics draw — an
open-loop action sequence, so W candidates ride one batched device
rollout (`batched_success`) and only the survivors pay for the CPU
reference (`cpu_execute`), which renders the dataset's frames and
issues the shipping verdict in the same pass. A candidate the device
keeps and the CPU refuses is a FILTER FALSE-POSITIVE: counted in the
result, never written to disk.

What this core does NOT do (recorded in docs/07): Arena-style
retargeting — transforming the action segment by the object's pose
delta. B3.1 measured open-loop kitting replay surviving ±5 mm of spawn
jitter and dying at ±30 mm, so a task adapter's `variant_fn` must vary
initial states within the seeds' own basin (bounded offsets around a
seed's spawn — the Mimic `offset_range` element), not across the whole
band. Amplification here = seeds x local variation x dynamics draws,
and the manifest says which seed each episode came from.

Batching note: dynamics vary per ROUND (one rescaled scene per batched
call — the engine runs one model across its worlds); initial states
and action noise vary per WORLD. Rounds sweep the dynamics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rq_pipeline.collect.kitting_export import write_episode
from rq_pipeline.collect.press import EpisodeManifest
from rq_pipeline.collect.press_batch import (
    Seed,
    batched_success,
    cpu_execute,
    expand_controls,
)

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass(frozen=True)
class Variant:
    """One candidate's variation, chosen by the task adapter: which
    seed to multiply, the varied initial state, and the draw values
    the manifest records."""

    seed_index: int
    initial_state: Any
    draws: dict[str, Any]


# The task adapter: given the rng and the seeds, one Variant.
VariantFn = Callable[["Generator", "list[Seed]"], Variant]
# The dynamics draw: one (rescaled spec, dynamics dict, basis) per round.
DynamicsFn = Callable[["Generator"], tuple[Any, dict[str, float], str]]


@dataclass(frozen=True)
class MultiplyPlan:
    """Every knob of the multiplication loop, named (the press's
    `press()` habit). `engine` goes to `MJXWarpBackend` verbatim."""

    variant_fn: VariantFn
    dynamics_fn: DynamicsFn
    episodes: int
    worlds: int
    action_noise: float = 0.005  # rad std-dev on position targets (Arena's knob)
    max_rounds: int = 25  # the Mimic abort: give up rather than spin
    frame_every: int = 5  # control ticks between saved frames
    engine: dict[str, Any] | None = None


@dataclass(frozen=True)
class MultiplyResult:
    """The accounting a chain log and a docs entry quote."""

    out: Path
    wanted: int
    kept: int
    candidates: int
    device_kept: int
    false_positives: int
    rounds: int

    @property
    def complete(self) -> bool:
        return self.kept >= self.wanted


def nearest_seed(seeds: list[Seed], point: Any, key_fn: Callable[[Seed], Any]) -> int:
    """The Mimic nearest-neighbour source selection: the index of the
    seed whose `key_fn` value is closest (Euclidean) to `point`."""
    import numpy as np  # noqa: PLC0415 - sim extra

    point = np.asarray(point, dtype=float).ravel()
    distances = [
        float(np.linalg.norm(np.asarray(key_fn(seed), dtype=float).ravel() - point))
        for seed in seeds
    ]
    return int(np.argmin(distances))


def multiply(  # noqa: PLR0913 - every knob of the loop, named
    out: Path,
    *,
    task: Any,
    seeds: list[Seed],
    plan: MultiplyPlan,
    seed: int,
    instrument: str,
    control_hz: int,
    first_episode: int = 0,
    say: Callable[[str], None] = print,
) -> MultiplyResult:
    """Multiply `seeds` into `plan.episodes` kept episodes under `out`.

    Per round: one dynamics draw rescales the scene; `plan.worlds`
    candidates (varied starts, noised actions) ride one batched device
    rollout; the device's keepers are re-executed on the CPU reference,
    which renders frames and issues the shipping verdict. Episodes land
    in `DemoLayout` with an `EpisodeManifest` whose `draws` name the
    source episode and the variation.
    """
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    interval = task.protocol.control_interval
    steps = max(s.states.shape[0] for s in seeds)
    camera = task.cameras[0] if getattr(task, "cameras", None) else None

    kept = candidates = device_kept = false_positives = rounds = 0
    while kept < plan.episodes and rounds < plan.max_rounds:
        rounds += 1
        spec, dynamics, basis = plan.dynamics_fn(rng)
        variants = [plan.variant_fn(rng, seeds) for _ in range(plan.worlds)]
        controls = np.stack(
            [
                expand_controls(
                    seeds[v.seed_index].actions
                    + rng.normal(
                        0.0, plan.action_noise, seeds[v.seed_index].actions.shape
                    ),
                    control_interval=interval,
                    steps=steps,
                )
                for v in variants
            ]
        )
        initials = np.stack([v.initial_state for v in variants])
        successes, _ = batched_success(
            task, initials, controls, engine=plan.engine, spec=spec
        )
        candidates += len(variants)
        device_kept += int(successes.sum())

        shipped_this_round = 0
        for world in np.flatnonzero(successes):
            if kept >= plan.episodes:
                break
            frames: list[tuple[int, Any]] = []
            holder: list[Any] = []  # the lazily built renderer
            every = plan.frame_every * interval

            # frames/holder/every bound as defaults — the late-binding
            # closure lesson from the kitting press (ruff B023).
            def snap(
                tick: int,
                model: Any,
                live: Any,
                *,
                sink=(frames, holder, every),
            ) -> None:
                frames, holder, every = sink
                if camera is None or tick % every != 0:
                    return
                if not holder:
                    holder.append(
                        mujoco.Renderer(model, height=camera.height, width=camera.width)
                    )
                holder[0].update_scene(live, camera=camera.camera_name)
                frames.append((tick, holder[0].render().copy()))

            ok, states, sensors = cpu_execute(
                task, initials[world], controls[world], spec=spec, on_step=snap
            )
            if holder:
                holder[0].close()
            if not ok:
                false_positives += 1
                continue
            variant = variants[world]
            source = seeds[variant.seed_index].manifest.get("expert", "seed")
            manifest = EpisodeManifest(
                seed=seed,
                attempt=candidates - len(variants) + int(world) + 1,
                task=getattr(task, "stamp", type(task).__name__),
                expert=f"multiplied:{source}",
                instrument=instrument,
                dynamics=dynamics,
                draws={
                    "multiplied_from": int(variant.seed_index),
                    "action_noise_std": plan.action_noise,
                    **variant.draws,
                },
                retries=[],
                control_hz=control_hz,
                frame_every_control_ticks=plan.frame_every,
                dynamics_basis=basis,
                verdict="success (task referee, CPU reference)",
            )
            write_episode(
                out,
                first_episode + kept,
                states=states,
                sensors=sensors,
                actions=controls[world][::interval],
                frames=frames,
                manifest=manifest,
            )
            kept += 1
            shipped_this_round += 1
        say(
            f"round {rounds}: dynamics {dynamics}, device keeps "
            f"{int(successes.sum())}/{len(variants)}, shipped {shipped_this_round}, "
            f"kept {kept}/{plan.episodes}"
        )

    return MultiplyResult(
        out=out,
        wanted=plan.episodes,
        kept=kept,
        candidates=candidates,
        device_kept=device_kept,
        false_positives=false_positives,
        rounds=rounds,
    )
