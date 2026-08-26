"""Our tasks as a gymnasium environment — the ecosystem's face of the harness.

Written 2026-08-26 (docs/32-evaluation-layer.md). One class for any task
a builder returns: its compiled model, its protocol's paired starts, its
cameras and its referee, behind the interface every evaluator speaks
(docs/e2e-research/46 §3): `reset(seed)` → observation, `step(action)` →
observation, reward, terminated, truncated, info. Observations use
LeRobot's raw names — `{"pixels": {<camera key>: uint8 (H, W, 3)},
"agent_pos": float32 (state_width,)}` — which
`lerobot.envs.utils.preprocess_observation` renames to
`observation.images.<key>` / `observation.state`; GR00T's and openpi's
prefixes are an adapter over the same three things.

Three things a stock env does not carry, kept on purpose:

- **Paired starts through the seed.** `reset(seed=k)` starts trial
  `k % trials` of the protocol — `perturb(trial, home)`, never a draw —
  so policy A's episode k and policy B's begin identically, and
  `lerobot-eval` (which seeds episode i with `seed + i`) pairs for free.
- **Success on the LAST tick only, under both names.** The referee
  judges the hold window at the end of the fixed-length episode; the
  episode ends by `truncated`, never `terminated = success` (LIBERO's
  conflation, which gymnasium's own text forbids). `info["is_success"]`
  is what LeRobot reads, `info["success"]` what GR00T, SimplerEnv and
  robomimic read; both carry the same bool.
- **The census gate at construction.** Actuators, sensors, geoms and
  cameras are counted before an episode is spent, and `source` must be
  a `name@hash` stamp — the harness's two rules, moved to where the env
  is born.

Renderers are created on first use, not in `__init__`: under
`AsyncVectorEnv`'s forkserver an EGL context made in the parent is stale
in the worker (LIBERO defers for the same reason).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - the helpful error
    raise ImportError(
        "the gymnasium env needs the 'sim' extra: uv sync --extra sim"
    ) from error

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.evaluate.harness import events_for, home_state
from rq_pipeline.evaluate.records import (
    EpisodeRecord,
    append_records,
    protocol_fields,
    protocol_hash,
)
from rq_pipeline.evaluate.variations import Variation, describe, draw_all
from rq_pipeline.physics.mujoco_backend import MuJoCoBackend, Stepper
from rq_pipeline.robot.model_checks import assert_model_alive
from rq_pipeline.tasks.aloha2 import BUNDLE_XML, build_kitting, build_transfer_cube
from rq_pipeline.tasks.so101 import (
    DEFAULT_ARM_XML,
    build_insert,
    build_lift,
    build_reach,
    build_stack,
)

RGB_CHANNELS = 3
PIXEL_MAX = 255
GYM_ID_PREFIX = "robotiq"  # gym.make("robotiq/<task>-v0")
GYM_ID_VERSION = "v0"
OFFSET_COMPONENTS = 3


class _Nominal:
    """The model's identified values, copied once, restored every reset
    — so a variation is always nominal x draw and never compounds
    (Arena's snapshot-then-write rule)."""

    def __init__(self, model: Any) -> None:
        self.dof_damping = model.dof_damping.copy()
        self.gain = model.actuator_gainprm[:, 0].copy()
        self.bias = model.actuator_biasprm[:, 1].copy()
        self.body_mass = model.body_mass.copy()
        self.body_inertia = model.body_inertia.copy()
        self.cam_pos = model.cam_pos.copy()
        self.light_diffuse = model.light_diffuse.copy()

    def restore(self, model: Any) -> None:
        model.dof_damping[:] = self.dof_damping
        model.actuator_gainprm[:, 0] = self.gain
        model.actuator_biasprm[:, 1] = self.bias
        model.body_mass[:] = self.body_mass
        model.body_inertia[:] = self.body_inertia
        model.cam_pos[:] = self.cam_pos
        model.light_diffuse[:] = self.light_diffuse


def _named(model: Any, kind: Any, name: str, what: str) -> int:
    import mujoco  # noqa: PLC0415 - sim extra, present if we got here

    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"variation names {what} {name!r}, which the model lacks")
    return index


def apply_variation(model: Any, nominal: _Nominal, key: str, value: Any) -> None:
    """Write one drawn value into the compiled model, from its nominal.

    The vocabulary: `joints.damping_scale`, `actuators.gain_scale`
    (BOTH kp terms of a position servo — scaling `gainprm[0]` alone
    moves the setpoint, the 2026-08-26 lesson in `scale_dynamics`),
    `<body>.mass_scale` (inertia scaled with it), `<camera>.offset_m`
    (three components, metres), `lights.diffuse_scale`. Anything else
    is refused, at construction, before a trial is spent.
    """
    import mujoco  # noqa: PLC0415 - sim extra, present if we got here

    host, _, name = key.rpartition(".")
    if key == "joints.damping_scale":
        model.dof_damping[:] = nominal.dof_damping * float(value)
    elif key == "actuators.gain_scale":
        model.actuator_gainprm[:, 0] = nominal.gain * float(value)
        model.actuator_biasprm[:, 1] = nominal.bias * float(value)
    elif key == "lights.diffuse_scale":
        model.light_diffuse[:] = nominal.light_diffuse * float(value)
    elif name == "mass_scale":
        body = _named(model, mujoco.mjtObj.mjOBJ_BODY, host, "body")
        model.body_mass[body] = nominal.body_mass[body] * float(value)
        model.body_inertia[body] = nominal.body_inertia[body] * float(value)
    elif name == "offset_m":
        camera = _named(model, mujoco.mjtObj.mjOBJ_CAMERA, host, "camera")
        offset = np.asarray(value, dtype=float)
        if offset.shape != (OFFSET_COMPONENTS,):
            raise ValueError(f"{key} needs three components, got {value!r}")
        model.cam_pos[camera] = nominal.cam_pos[camera] + offset
    else:
        raise ValueError(
            f"unknown variation {key!r}; the env knows joints.damping_scale, "
            "actuators.gain_scale, lights.diffuse_scale, <body>.mass_scale, "
            "<camera>.offset_m"
        )


@dataclass(frozen=True)
class TaskEntry:
    """A builder and the bundle it composes, whose hash stamps the source."""

    build: Callable[[], Any]
    bundle_dir: Path


# The tasks anyone with gymnasium can construct by id, on the nominal
# bundles. State width, cameras and instruction are the Task's own.
TASKS: dict[str, TaskEntry] = {
    "transfer_cube": TaskEntry(build_transfer_cube, BUNDLE_XML.parent),
    "kitting": TaskEntry(build_kitting, BUNDLE_XML.parent),
    "reach": TaskEntry(build_reach, DEFAULT_ARM_XML.parent),
    "lift": TaskEntry(build_lift, DEFAULT_ARM_XML.parent),
    "block_stack": TaskEntry(build_stack, DEFAULT_ARM_XML.parent),
    "tool_insert": TaskEntry(build_insert, DEFAULT_ARM_XML.parent),
}


@functools.cache
def bundle_source(bundle_dir: Path = BUNDLE_XML.parent) -> str:
    """A bundle's `name@hash` — hashed once per process."""
    return stamp(bundle_dir.name, bundle_dir)


class RobotiqEnv(gym.Env):
    """A `Task` (spec + protocol + cameras + state width + instruction)
    as a gymnasium env.

    The task's `state_width` is the `agent_pos` the policy sees and its
    `instruction` is exposed as `task_description` for evaluators that
    read it. With `record_to`, the env appends one `EpisodeRecord` per
    finished episode — verdict, milestones, seed, stamps — so a runner
    that keeps only a success list (LeRobot's) still leaves our full row
    behind; `policy_name` is what the row calls the policy, since the
    env never sees it.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": 0,  # per instance, from the model's timestep
    }

    def __init__(  # noqa: PLR0913 - keyword-only identity, each part of the row
        self,
        task: Any,
        *,
        source: str,
        render_mode: str = "rgb_array",
        record_to: Path | None = None,
        policy_name: str = "policy",
        variations: tuple[Variation, ...] = (),
    ) -> None:
        state_width = task.state_width
        instruction = task.instruction
        if "@" not in source:
            raise ValueError(
                f"source must be a name@hash stamp, got {source!r} — the same "
                "rule certify() enforces, applied before episodes are spent"
            )
        backend = MuJoCoBackend()
        backend.load_spec(task.spec)
        counts = backend.counts()
        assert_model_alive(
            counts.actuators,
            counts.sensors,
            counts.geoms,
            source=source,
            cameras=counts.cameras,
        )
        self._model = backend.model
        self.instrument = backend.instrument
        self.protocol = task.protocol
        self.cameras = tuple(task.cameras)
        self.state_width = state_width
        self.source = source
        self.task = task.name
        self.task_description = instruction
        self.render_mode = render_mode
        self._home = home_state(backend, task.protocol)
        interval = task.protocol.control_interval
        # Control ticks per episode; the last tick may hold fewer physics
        # steps if the protocol's budget is not a multiple.
        self._max_episode_steps = -(-task.protocol.steps // interval)
        self.metadata = {
            **type(self).metadata,
            "render_fps": round(1.0 / (self._model.opt.timestep * interval)),
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        camera.key: spaces.Box(
                            0,
                            PIXEL_MAX,
                            (camera.height, camera.width, RGB_CHANNELS),
                            np.uint8,
                        )
                        for camera in self.cameras
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, (state_width,), np.float32),
            }
        )
        limited = self._model.actuator_ctrllimited.astype(bool)
        ctrlrange = self._model.actuator_ctrlrange
        self.action_space = spaces.Box(
            np.where(limited, ctrlrange[:, 0], -np.inf).astype(np.float32),
            np.where(limited, ctrlrange[:, 1], np.inf).astype(np.float32),
            dtype=np.float32,
        )
        self._renderers: dict[tuple[int, int], Any] = {}
        self._stepper: Stepper | None = None
        self._trial = 0
        self._seed: int | None = None
        self._episodes = 0
        self._last_pixels: dict[str, Any] = {}
        self._record_to = Path(record_to) if record_to is not None else None
        self._policy_name = policy_name
        # The variation space is part of the protocol's identity: its
        # description enters the fields, so the hash — and every draw —
        # changes when a knob does.
        self.variations = tuple(variations)
        self._protocol_fields = {
            **protocol_fields(task.protocol),
            "variations": describe(self.variations),
        }
        self._protocol_hash = protocol_hash(self._protocol_fields)
        self._nominal = _Nominal(self._model)
        self._values: dict[str, Any] = {}
        # Every key must be applicable NOW: a misspelt body or camera is
        # refused before a trial is spent, not at the first reset.
        for key, value in draw_all(self.variations, 0, self._protocol_hash).items():
            apply_variation(self._model, self._nominal, key, value)
        self._nominal.restore(self._model)
        import mujoco  # noqa: PLC0415 - sim extra, present if we got here

        # One MjData for the env's whole life: a passive viewer binds to
        # the object, so resets seat into it rather than replacing it.
        self._data = mujoco.MjData(self._model)

    # -- for tools (viewers, mirrors): read-only views of the live sim ----

    @property
    def model(self) -> Any:
        return self._model

    @property
    def data(self) -> Any:
        """The live MjData — bind a viewer to it; never write to it."""
        return self._data

    @property
    def states(self) -> Any:
        """Per-physics-step FULLPHYSICS rows of the current episode so far
        (what the referee reads); rows past `physics_step` are unset."""
        if self._stepper is None:
            raise RuntimeError("reset() before reading states")
        return self._stepper.states

    @property
    def physics_step(self) -> int:
        return 0 if self._stepper is None else self._stepper.step

    # -- the contract ----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        index = self._episodes if seed is None else seed
        self._trial = index % self.protocol.trials
        self._seed = seed
        self._episodes += 1
        # Nominal first, then this trial's draw: identical for every
        # policy on trial k, never compounding across resets.
        self._nominal.restore(self._model)
        self._values = draw_all(self.variations, self._trial, self._protocol_hash)
        for key, value in self._values.items():
            apply_variation(self._model, self._nominal, key, value)
        initial = self.protocol.perturb(self._trial, self._home)
        self._stepper = Stepper(
            self._model, initial, self.protocol.steps, data=self._data
        )
        return self._observe(), self._info(False)

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._stepper is None:
            raise RuntimeError("reset() before step()")
        self._stepper.advance(action, self.protocol.control_interval)
        done = self._stepper.done
        ok = False
        if done:
            states, sensors = self._stepper.states, self._stepper.sensors
            ok = bool(self.protocol.success(states, sensors))
            if self._record_to is not None:
                append_records(
                    self._record_to,
                    [
                        EpisodeRecord(
                            source=self.source,
                            policy=self._policy_name,
                            trial=self._trial,
                            success=ok,
                            steps=len(states),
                            instrument=self.instrument,
                            protocol=self._protocol_fields,
                            seed=self._seed,
                            events=events_for(self.protocol, states, sensors),
                            variations=dict(self._values),
                        )
                    ],
                )
        return self._observe(), float(ok), False, done, self._info(ok)

    def render(self) -> Any:
        """The first camera's latest frame — what evaluators stack into video."""
        if not self._last_pixels:
            self._observe()
        return self._last_pixels[self.cameras[0].key]

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    # -- internals -------------------------------------------------------

    def _info(self, ok: bool) -> dict[str, Any]:
        return {"is_success": ok, "success": ok, "trial": self._trial}

    @property
    def drawn(self) -> dict[str, Any]:
        """This episode's variation values, keyed `host.name`."""
        return dict(self._values)

    def _renderer(self, height: int, width: int) -> Any:
        key = (height, width)
        if key not in self._renderers:
            import mujoco  # noqa: PLC0415 - sim extra, present if we got here

            self._renderers[key] = mujoco.Renderer(
                self._model, height=height, width=width
            )
        return self._renderers[key]

    def _observe(self) -> dict[str, Any]:
        assert self._stepper is not None
        data = self._stepper.data
        pixels: dict[str, Any] = {}
        for camera in self.cameras:
            renderer = self._renderer(camera.height, camera.width)
            renderer.update_scene(data, camera=camera.camera_name)
            pixels[camera.key] = renderer.render().copy()
        self._last_pixels = pixels
        return {
            "pixels": pixels,
            "agent_pos": np.asarray(
                data.sensordata[: self.state_width], dtype=np.float32
            ).copy(),
        }


def make_env(
    task: str,
    *,
    render_mode: str = "rgb_array",
    record_to: Path | str | None = None,
    policy_name: str = "policy",
    variations: tuple[Variation, ...] = (),
) -> RobotiqEnv:
    """`gym.make`'s entry point: a task by name on its nominal bundle."""
    if task not in TASKS:
        raise KeyError(f"no task {task!r}; the env knows {sorted(TASKS)}")
    entry = TASKS[task]
    return RobotiqEnv(
        entry.build(),
        source=bundle_source(entry.bundle_dir),
        render_mode=render_mode,
        record_to=Path(record_to) if record_to is not None else None,
        policy_name=policy_name,
        variations=variations,
    )


for _name in TASKS:
    # Registered at import, gymnasium's convention: `gym.make("robotiq/kitting-v0")`
    # works for any gymnasium client, LeRobot or not.
    gym.register(
        id=f"{GYM_ID_PREFIX}/{_name}-{GYM_ID_VERSION}",
        entry_point="rq_pipeline.envs.robotiq:make_env",
        kwargs={"task": _name},
    )
