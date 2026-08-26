"""LeRobot's door to our tasks: the `EnvConfig` that `lerobot-eval` and
`lerobot-train` load.

    LD_LIBRARY_PATH=/usr/lib/wsl/lib GALLIUM_DRIVER=d3d12 MUJOCO_GL=egl \\
    .venv-train/bin/lerobot-eval --env.type=robotiq --env.task=kitting \\
        --env.discover_packages_path=rq_pipeline.envs \\
        --policy.path=runs/t5-act-kitting/checkpoints/020000/pretrained_model \\
        --seed=1000 --eval.n_episodes=4 --eval.batch_size=1 \\
        --eval.use_async_envs=false

`--env.discover_packages_path` (lerobot/configs/parser.py) imports every
module of the named PACKAGE before the CLI is parsed — a module path is
refused, it walks `__path__` — which runs the registration decorator
below; a distribution named `lerobot_env_*` is imported the same way with
no flag.
Everything LeRobot needs beyond gymnasium lives here and nowhere else:
the feature declarations a policy config reads (`features`,
`features_map`) and the vector-env factory. Physics, cameras, referee and
pairing are the gymnasium env's (rq_pipeline.envs.robotiq) — GR00T,
openpi or a bare `gym.make("robotiq/kitting-v0")` reach the same env
without this module.

Pairing survives the runner: `lerobot-eval` seeds episode i with
`seed + i`, and the env starts trial `seed % trials` — so every policy's
episode i begins from the same state, and the recorded `seed` lets the
per-episode rows fold into paired counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import gymnasium as gym
from gymnasium.vector import AutoresetMode
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from rq_pipeline.envs.robotiq import RGB_CHANNELS, TASKS, make_env

DEFAULT_TASK = "kitting"
CONTROL_HZ = 50  # the ALOHA protocols: 500 Hz physics, control every 10 steps


@EnvConfig.register_subclass("robotiq")
@dataclass
class RobotiqEnvConfig(EnvConfig):
    """`--env.type=robotiq --env.task=<name>`; features come from the task."""

    task: str | None = DEFAULT_TASK
    fps: int = CONTROL_HZ
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"no task {self.task!r}; the env knows {sorted(TASKS)}")
        entry = TASKS[self.task]
        # The action is one command per servo and the state one reading
        # per servo: the same width on both rigs.
        width = entry.state_width
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(width,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(width,)),
        }
        self.features_map = {ACTION: ACTION, "agent_pos": OBS_STATE}
        # Cameras are the task's; building the spec (no compile) is cheap.
        for camera in entry.build().cameras:
            key = f"pixels/{camera.key}"
            self.features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(camera.height, camera.width, RGB_CHANNELS),
            )
            self.features_map[key] = f"{OBS_IMAGES}.{camera.key}"

    @property
    def gym_kwargs(self) -> dict:
        return {"task": self.task}

    def create_envs(
        self, n_envs: int, use_async_envs: bool = False
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """`{suite: {task_id: VectorEnv}}`, the shape `lerobot.envs.make_env`
        documents. Async copies use forkserver so each worker builds its
        own renderer (the env defers renderer creation to first use)."""
        factories = [partial(make_env, self.task)] * n_envs
        if use_async_envs and n_envs > 1:
            vec: gym.vector.VectorEnv = gym.vector.AsyncVectorEnv(
                factories,
                context="forkserver",
                autoreset_mode=AutoresetMode.SAME_STEP,
            )
        else:
            vec = gym.vector.SyncVectorEnv(
                factories, autoreset_mode=AutoresetMode.SAME_STEP
            )
        return {self.type: {0: vec}}
