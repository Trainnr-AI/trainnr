"""The batched open-loop verifier — the press at MJX-Warp scale (B3.1).

docs/e2e-research/60 §3's "batch scale", scoped to what actually
batches: an OPEN-LOOP candidate — an action sequence known up front
(a multiplied seed, a replayed demonstration, a swept variant) — is one
row of a batched rollout, W worlds in lockstep on the device, judged by
the task's own referee per world on the returned rows. The adaptive
scripted experts stay on the CPU where their retry loops live; this
module is the engine for everything whose actions exist before the
physics runs.

The division of instruments (docs/e2e-research/49's habit, applied to
data): the DEVICE filters — cheap parallel physics decides which
candidates are worth anything — and the CPU EXECUTES the keepers:
`cpu_execute` re-runs a kept candidate on CPU MuJoCo, which yields the
camera frames a dataset needs AND the metrology-grade verdict in one
pass, because the CPU is the reference instrument and float32 device
physics is not (a keeper the CPU refuses is a filter false-positive,
recorded, never shipped).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainnr.collect.kitting_export import DemoLayout

# The batched engine's sizing for busy scenes: the kitting bundle's
# measured pair (the log 2026-08-27); a small scene ignores them.
KITTING_SIZING = {"naconmax": 4096, "njmax": 8192}


@dataclass(frozen=True)
class Seed:
    """One kept episode read back from `DemoLayout`: the trajectory the
    press wrote and the sidecar it stamped."""

    states: Any
    sensors: Any
    actions: Any
    manifest: dict[str, Any]
    # The episode directory this seed was read from. Multiplied episodes
    # cite THIS, never a position in a sorted list — re-sharding a seeds
    # directory used to silently repoint every provenance stamp at a
    # different episode (review 2026-09-01).
    episode: str = ""

    @classmethod
    def read(cls, episode_dir: Path) -> Seed:
        import numpy as np  # noqa: PLC0415 - sim extra

        episode_dir = Path(episode_dir)
        with np.load(episode_dir / DemoLayout.TRAJECTORY_FILE) as data:
            states = data[DemoLayout.STATES].astype(float)
            sensors = data[DemoLayout.SENSORS].astype(float)
            actions = data[DemoLayout.ACTIONS].astype(float)
        manifest = json.loads((episode_dir / DemoLayout.MANIFEST_FILE).read_text())
        return cls(
            states=states,
            sensors=sensors,
            actions=actions,
            manifest=manifest,
            episode=episode_dir.name,
        )


def expand_controls(actions: Any, *, control_interval: int, steps: int) -> Any:
    """Per-control-tick actions to per-physics-step controls: each action
    held for `control_interval` steps (the stepper's own zero-order
    hold), the last held to `steps` — the shape a batched rollout eats."""
    import numpy as np  # noqa: PLC0415

    actions = np.asarray(actions, dtype=float)
    per_step = np.repeat(actions, control_interval, axis=0)
    if per_step.shape[0] >= steps:
        return per_step[:steps]
    pad = np.repeat(per_step[-1:], steps - per_step.shape[0], axis=0)
    return np.concatenate([per_step, pad], axis=0)


def batched_success(
    task: Any,
    initial_states: Any,
    controls: Any,
    *,
    engine: dict[str, Any] | None = None,
    spec: Any = None,
) -> tuple[Any, Any]:
    """W open-loop candidates through the batched engine, judged by the
    task's referee: `(successes (W,), states (W, steps, nstate))`.

    The referee is called with `sensors=None`: the batched rollout
    returns FULLPHYSICS rows only, and a referee that needs sensors will
    say so loudly — kitting's judges states alone, which is the case
    this slice serves. The verdicts are the FILTER's, on the device
    instrument; a shipped episode is judged again by `cpu_execute`.

    `engine` is the `MJXWarpBackend` constructor's knobs verbatim
    (`impl`, and the sizing pair for busy scenes — `KITTING_SIZING`).
    `spec` overrides `task.spec` — a seed pressed under drawn dynamics
    replays under its OWN rescaled scene, not the nominal one.
    """
    import numpy as np  # noqa: PLC0415

    from trainnr.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415

    backend = MJXWarpBackend(**(engine or {}))
    backend.load_spec(task.spec if spec is None else spec)
    states = backend.rollout(initial_states, controls)
    successes = np.array(
        [
            bool(task.protocol.success(states[world], None))
            for world in range(states.shape[0])
        ]
    )
    return successes, states


def cpu_execute(
    task: Any,
    initial_state: Any,
    controls: Any,
    *,
    spec: Any = None,
    on_step: Any = None,
) -> tuple[bool, Any, Any]:
    """One kept candidate on CPU MuJoCo — the reference instrument's
    verdict, and the states/sensors a dataset writer renders frames
    from: `(succeeded, states, sensors)`. `controls` are per physics
    step (the `expand_controls` shape). `on_step(tick, model, data)`,
    when given, sees the live model/data after every step — the frame
    renderer's hook."""
    import numpy as np  # noqa: PLC0415

    from trainnr.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

    controls = np.asarray(controls, dtype=float)
    backend = MuJoCoBackend()
    backend.load_spec(task.spec if spec is None else spec)
    stepper = backend.stepper(initial_state, controls.shape[0])
    for tick, row in enumerate(controls):
        stepper.advance(row, 1)
        if on_step is not None:
            on_step(tick, backend.model, stepper.data)
    succeeded = bool(task.protocol.success(stepper.states, stepper.sensors))
    return succeeded, stepper.states, stepper.sensors
