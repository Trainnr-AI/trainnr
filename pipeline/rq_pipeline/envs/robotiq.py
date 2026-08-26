"""Our tasks as a gymnasium environment — the ecosystem's face of the harness.

Written 2026-08-26 (docs/32-evaluation-layer.md). One class for any task
a builder returns: its compiled model, its protocol's paired starts, its
cameras and its referee, behind the interface every evaluator speaks
(docs/e2e-research/46 §3): `reset(seed)` → observation, `step(action)` →
observation, reward, terminated, truncated, info. Observations use
LeRobot's raw names (`envs/contract.py`), which
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
  conflation, which gymnasium's own text forbids). The flag is emitted
  under the name LeRobot reads and the one GR00T, SimplerEnv and
  robomimic read.
- **The census gate at construction.** Actuators, sensors, geoms and
  cameras are counted before an episode is spent, and `source` must be
  a `name@hash` stamp — the harness's two rules, moved to where the env
  is born.

Renderers are created on first use, not in `__init__`: under
`AsyncVectorEnv` each worker process must build its own GL context — one
made in the parent is stale or absent in the child (LIBERO defers for
the same reason).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

try:
    import gymnasium as gym
    import mujoco
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - the helpful error
    raise ImportError(
        "the gymnasium env needs the 'sim' extra: uv sync --extra sim"
    ) from error

from rq_pipeline.bundles.hashing import require_stamp, stamp
from rq_pipeline.envs.contract import (
    DEFAULT_POLICY_NAME,
    PIXEL_MAX,
    RENDER_MODE,
    RGB_CHANNELS,
    InfoKeys,
    ObservationKeys,
    gym_id,
)
from rq_pipeline.evaluate.harness import home_state
from rq_pipeline.evaluate.records import EpisodeRecord, append_records, protocol_hash
from rq_pipeline.evaluate.variations import Variation, describe, draw_all
from rq_pipeline.physics.mujoco_backend import MuJoCoBackend, Stepper
from rq_pipeline.physics.variations import apply_key, restore_all, snapshot_all
from rq_pipeline.protocol import events_for, protocol_fields
from rq_pipeline.robot.model_checks import assert_model_alive
from rq_pipeline.tasks.registry import resolve, tasks

PROTOCOL_VARIATIONS_FIELD = "variations"


@functools.cache
def bundle_source(bundle_dir: Path) -> str:
    """A bundle's `name@hash` — hashed once per process. No default: the
    bundle is the task's (`Task.bundle_dir`), never the module's."""
    return stamp(bundle_dir.name, bundle_dir)


class RobotiqEnv(gym.Env):
    """A `Task` (spec + protocol + cameras + state width + instruction)
    as a gymnasium env.

    The task's `state_width` is the `agent_pos` the policy sees and its
    `instruction` is exposed as `task_description` for evaluators that
    read it. With `record_to`, the env appends one `EpisodeRecord` per
    finished episode — verdict, milestones, seed, draws, stamps — so a
    runner that keeps only a success list (LeRobot's) still leaves our
    full row behind; `policy_name` is what the row calls the policy,
    since the env never sees it. `variations` are drawn by trial index
    and applied through the engine's applier registry
    (`physics/variations.py`); an unknown knob is refused here, before a
    trial is spent.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [RENDER_MODE],
        "render_fps": 0,  # per instance, from the task's control rate
    }

    def __init__(  # noqa: PLR0913 - keyword-only identity, each part of the row
        self,
        task: Any,
        *,
        source: str,
        render_mode: str = RENDER_MODE,
        record_to: Path | None = None,
        policy_name: str = DEFAULT_POLICY_NAME,
        variations: tuple[Variation, ...] = (),
    ) -> None:
        require_stamp(source)
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
        self.state_width = task.state_width
        self.source = source
        self.task = task.name
        self.task_description = task.instruction
        self.render_mode = render_mode
        self._home = home_state(backend, task.protocol)
        # Public for tools; LeRobot's rollout reads the underscored name.
        self.max_episode_steps = task.protocol.control_ticks
        self._max_episode_steps = self.max_episode_steps
        self.metadata = {**type(self).metadata, "render_fps": task.control_hz}
        self.observation_space = spaces.Dict(
            {
                ObservationKeys.PIXELS: spaces.Dict(
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
                ObservationKeys.AGENT_POS: spaces.Box(
                    -np.inf, np.inf, (self.state_width,), np.float32
                ),
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
            PROTOCOL_VARIATIONS_FIELD: describe(self.variations),
        }
        self._protocol_hash = protocol_hash(self._protocol_fields)
        self._nominal = snapshot_all(self._model)
        self._values: dict[str, Any] = {}
        # Every key must be applicable NOW: a misspelt body or camera is
        # refused before a trial is spent, not at the first reset.
        for key, value in draw_all(self.variations, 0, self._protocol_hash).items():
            apply_key(self._model, self._nominal, key, value)
        restore_all(self._model, self._nominal)
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

    @property
    def drawn(self) -> dict[str, Any]:
        """This episode's variation values, keyed `host.name`."""
        return dict(self._values)

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
        restore_all(self._model, self._nominal)
        self._values = draw_all(self.variations, self._trial, self._protocol_hash)
        for key, value in self._values.items():
            apply_key(self._model, self._nominal, key, value)
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
                append_records(self._record_to, [self._record(ok, states, sensors)])
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
        return {
            InfoKeys.IS_SUCCESS: ok,
            InfoKeys.SUCCESS: ok,
            InfoKeys.TRIAL: self._trial,
        }

    def _record(self, ok: bool, states: Any, sensors: Any) -> EpisodeRecord:
        return EpisodeRecord(
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

    def _renderer(self, height: int, width: int) -> Any:
        key = (height, width)
        if key not in self._renderers:
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
            ObservationKeys.PIXELS: pixels,
            ObservationKeys.AGENT_POS: np.asarray(
                data.sensordata[: self.state_width], dtype=np.float32
            ).copy(),
        }


def make_env(
    task: str,
    *,
    render_mode: str = RENDER_MODE,
    record_to: Path | str | None = None,
    policy_name: str = DEFAULT_POLICY_NAME,
    variations: tuple[Variation, ...] = (),
    **builder_kwargs: Any,
) -> RobotiqEnv:
    """`gym.make`'s entry point: a registered task by id (`robotiq/kitting`,
    or bare `kitting` for a built-in); extra keywords reach the builder
    (`look=...`)."""
    built = resolve(task).build(**builder_kwargs)
    return RobotiqEnv(
        built,
        source=bundle_source(built.bundle_dir),
        render_mode=render_mode,
        record_to=Path(record_to) if record_to is not None else None,
        policy_name=policy_name,
        variations=variations,
    )


def register_gym_ids() -> tuple[str, ...]:
    """Every registered task as a gymnasium id, idempotently — so
    `gym.make("robotiq/kitting-v0")` (or a plugin's `acme/pour-v0`) works
    for any gymnasium client, LeRobot or not."""
    ids = []
    for task_id in tasks():
        env_id = gym_id(task_id)
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point="rq_pipeline.envs.robotiq:make_env",
                kwargs={"task": task_id},
            )
        ids.append(env_id)
    return tuple(ids)


register_gym_ids()
