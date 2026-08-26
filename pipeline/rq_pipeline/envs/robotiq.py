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
)
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

    def __init__(
        self,
        task: Any,
        *,
        source: str,
        render_mode: str = "rgb_array",
        record_to: Path | None = None,
        policy_name: str = "policy",
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
        self._protocol_fields = protocol_fields(task.protocol)
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
) -> RobotiqEnv:
    """`gym.make`'s entry point: a task by name on the nominal ALOHA 2 bundle."""
    if task not in TASKS:
        raise KeyError(f"no task {task!r}; the env knows {sorted(TASKS)}")
    entry = TASKS[task]
    return RobotiqEnv(
        entry.build(),
        source=bundle_source(entry.bundle_dir),
        render_mode=render_mode,
        record_to=Path(record_to) if record_to is not None else None,
        policy_name=policy_name,
    )


for _name in TASKS:
    # Registered at import, gymnasium's convention: `gym.make("robotiq/kitting-v0")`
    # works for any gymnasium client, LeRobot or not.
    gym.register(
        id=f"{GYM_ID_PREFIX}/{_name}-{GYM_ID_VERSION}",
        entry_point="rq_pipeline.envs.robotiq:make_env",
        kwargs={"task": _name},
    )
